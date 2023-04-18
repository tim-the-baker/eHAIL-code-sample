import math
import torch
import numpy as np
from SCython.SNG_torch import SNG, RNS, PCC, MSG
import SCython.SNG as np_SNG
import SCython.Circuits.stream_adders as np_adders

class HardwiredAdder:
    def __init__(self, tree_height, weights, data_rns, data_pcc, share_r, msg, full_corr, bipolar, device, wire_map=None):
        self.tree_height = tree_height
        self.weights = weights
        self.data_rns = data_rns
        self.data_pcc = data_pcc
        self.share_r = share_r
        self.msg = msg
        self.full_corr = full_corr
        self.bipolar = bipolar
        self.device = device

        # Unipolar (non-bipolar) hardwired muxes can only implement weighted addition using positive weights
        assert self.bipolar or (self.weights >= 0).all()

        # Tree height should be equal to RNS precision.
        assert tree_height == data_rns.n

        # weights should be two dimensional (first dim is batch, second dim is input size
        assert weights.dim() == 2
        N, K = weights.shape

        self.num_inputs = K
        self.inv_mask = weights < 0  # need to know which weights are negative for inverter array

        # wire maps
        if wire_map is None:
            self.wire_map, self.quant_norm_weights = HardwiredAdder.get_wire_map_and_qnw(weights, tree_height, device)
            self.scale_factors = 1 / self.quant_norm_weights.abs().sum(dim=-1)
        else:
            assert (self.weights.square() == 1).all(), "Giving a wiremap to adder should only be used when weights are +/-1"
            self.wire_map = wire_map
            self.quant_norm_weights = None

        # sng_mask is passed to the data_sng's gen_SN method to implement full correlation if full correlation is enabled.
        self.sng_mask = weights < 0 if self.full_corr else None


    @classmethod
    def get_wire_map_and_qnw(cls, weights, tree_height, device):
        N, K = weights.shape
        weights = weights.cpu()
        num_tree_inputs = int(2**tree_height)

        abs_weights = torch.abs(weights)
        abs_normalized_weights = abs_weights / torch.sum(abs_weights, dim=-1, keepdim=True)
        target_numerators = abs_normalized_weights * num_tree_inputs
        quantized_numerators = torch.round(target_numerators).to(torch.int)

        # rounding to nearest integer does not guarantee that exactly all 2^m mux inputs are used. need to fix this.
        mask = quantized_numerators.sum(dim=-1) > num_tree_inputs  # shape: (N,)
        while mask.any():
            diffs = quantized_numerators[mask] - target_numerators[mask]  # (N, K)
            index = torch.argmax(diffs, dim=-1)  # (N,)
            quantized_numerators[mask, index] -= 1
            mask = quantized_numerators.sum(dim=-1) > num_tree_inputs  # shape: (N,)

        mask = quantized_numerators.sum(dim=-1) < num_tree_inputs  # shape: (N,)
        while mask.any():
            diffs = -(quantized_numerators[mask] - target_numerators[mask])  # (N, K)
            index = torch.argmax(diffs, dim=-1)  # (N,)
            quantized_numerators[mask, index] += 1
            mask = quantized_numerators.sum(dim=-1) < num_tree_inputs  # shape: (N,)

        '''
        - wire_map's i-th value tells us the index of the data input SN that is connected to the i-th mux tree slot
        - Ex: wire_map[i] = j says that the j-th input SN, X_j, is connected to i-th mux input slot
        -     in other words, X_j will will be sampled when the mux's select input is i.
        '''
        wire_map = torch.zeros((N, num_tree_inputs), dtype=torch.long) #, device=device)
        for N_idx in range(N):
            next_available_tree_slot = 0
            for idx in range(K):
                end_slot = next_available_tree_slot + quantized_numerators[N_idx, idx]
                wire_map[N_idx, next_available_tree_slot:end_slot] = idx
                next_available_tree_slot = end_slot
            # check to make sure we used up all 2^m mux slots
            assert next_available_tree_slot == num_tree_inputs, f"{next_available_tree_slot}, {num_tree_inputs}"

        # we also compute the quantized, normalized weights
        quantized_numerators = torch.zeros_like(weights)
        for N_idx in range(N):
            for i in range(num_tree_inputs):
                quantized_numerators[N_idx, wire_map[N_idx, i]] += 1
        quant_norm_weights = quantized_numerators / num_tree_inputs
        quant_norm_weights[weights < 0] *= -1  # don't forgot to undo the absolute value operation

        return wire_map.to(device), quant_norm_weights.to(device)

    def forward(self, input_values, SN_length):
        raise NotImplementedError


class HardwiredMux(HardwiredAdder):
    def forward(self, input_values, SN_length):
        assert self.data_rns.n == math.log2(SN_length)
        assert input_values.shape == self.weights.shape, \
        f"Value shape:{input_values.shape} should match weights shape{self.weights.shape}"
        N, K = self.weights.shape

        # Quantize input values first
        quant_n = self.data_pcc.n
        input_values = SNG.q_floor(input_values, quant_n, self.bipolar)
        ps = (input_values+1) / 2 if self.bipolar else input_values

        # Check to make sure that all probabilities are valid
        assert (ps >= 0).all() and (ps <= 1).all(), \
            f"Error, input values should be between [0,1] (unipolar) or [-1,1] (bipolar):\n{input_values}"

        # Generate the tree's select inputs
        msg_share = self.msg.must_share()
        selects = self.msg.gen_selects(SN_length, shape=(N,), share=msg_share)  # (N, L)

        # transform the selects into SN_indexes by using the hardwired mux wire_map
        # wiremap.shape: (N, 2^h); selects.shape: (N, L)
        SN_indexes = torch.gather(self.wire_map, dim=1, index=selects) # (N, L)

        # Use the SN_indexes to pick out which input value is selected each clock cycle
        # ps.shape: (N, K); SN_indexes.shape (N, L)
        selected_ps = torch.gather(ps, dim=1, index=SN_indexes)  # (N, L)
        # selected_ps = ps[SN_indexes]  # shape: (N, L)

        # Use the RNS to generate the random inputs and pick out which R is valid each clock cycle.
        Rs = self.data_rns.gen_RN(SN_length, ps.shape, self.share_r, inv_mask=self.sng_mask) # (N, K, L)
        selected_Rs = torch.gather(Rs, dim=1, index=SN_indexes[:,None])  # (N, 1, L)
        selected_Rs = selected_Rs.squeeze() # (N, L)

        # Quantize both the Rs and convert everything to an integer
        # If bipolar is used then we have to drop the precision by 1
        if self.data_pcc.n != self.data_rns.n:
            selected_Rs = torch.bitwise_right_shift(selected_Rs, self.data_rns.n - self.data_pcc.n)
        selected_Bs = torch.bitwise_left_shift(selected_ps, self.data_pcc.n).to(torch.int)

        # Use PCC to convert the selected_Rs and input_values to SNs
        output_SN = self.data_pcc.forward(selected_Rs, selected_Bs)

        # Invert the appropriate SN bits (those who have corresponding negative weights).
        # self.inv_mask.shape = (N, K); SN_indexes.shape = (N, L)
        selected_mask = torch.gather(self.inv_mask, dim=1, index=SN_indexes).to(torch.bool)
        # selected_mask = self.inv_mask[SN_indexes].to(torch.bool)
        output_SN[selected_mask] = torch.logical_not(output_SN[selected_mask])

        return output_SN.to(torch.int)

    def forward_SN(self, SNs):
        N, K, L = SNs.shape
        assert self.weights.shape == SNs.shape[0:-1]

        # Generate the tree's select inputs
        msg_share = self.msg.must_share()
        selects = self.msg.gen_selects(L, shape=(N,), share=msg_share)  # (N, L)

        # transform the selects into SN_indexes by using the hardwired mux wire_map
        # wiremap.shape: (N, 2^h); selects.shape: (N, L)
        SN_indexes = torch.gather(self.wire_map, dim=1, index=selects)  # (N, L)
        output_SNs = torch.gather(SNs, dim=1, index=SN_indexes[:, None, :])

        if self.bipolar:  # handle da xnor array
            # self.inv_mask.shape: (N, K)
            inv_mask = torch.gather(self.inv_mask[..., None].expand(N, K, L), dim=1, index=SN_indexes[:, None, :])
            output_SNs[inv_mask] = torch.logical_not(output_SNs[inv_mask])  # invert selected bits

        return output_SNs.to(torch.int).squeeze()


def test1():
    N = 32
    K = 100
    h = 8
    device = torch.device('cuda:0')
    weights = torch.rand((N,K), device=device)
    weights_np = weights.cpu().numpy()
    wire_maps, quant_norm_weights = \
        HardwiredAdder.get_wire_map_and_qnw(weights, tree_height=h, device=device)

    np_wiremaps, np_qnw = wire_maps.cpu().numpy().copy(), quant_norm_weights.cpu().numpy().copy()
    for N_idx in range(N):
        np_wiremaps[N_idx], np_qnw[N_idx], _ = \
            np_adders.HardwiredAdder.get_wire_map_and_quant_norm_weights(weights_np[N_idx], tree_height=h, use_ddg_tree=False)

    print((np_wiremaps == wire_maps.cpu().numpy()).all())
    print((np_qnw == quant_norm_weights.cpu().numpy()).all())


def test2():
    n = 8
    L = int(2**n)
    N = 32
    K = 100
    h = 8
    H = int(2**h)

    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    weights = torch.rand((N,K), device=device)
    pxs = torch.rand((N,K), device=device)

    weights_np = weights.cpu().numpy()
    pxs_np = pxs.cpu().numpy()
    import time
    start = time.time()
    cemux = HardwiredMux(h, weights, RNS.VDC_RNS(n, device), PCC.Comparator(n), share_r=True,
                           msg=MSG.Long_Counter_Exact_MSG(H, device), full_corr=True, bipolar=False, device=device)
    Zs = cemux.forward(pxs, L)
    end1 = time.time()
    Zs_np = np.empty_like(Zs.cpu().numpy())
    for N_idx in range(N):
        cemux_np = np_adders.CeMux(h, weights_np[N_idx], np_SNG.PCC.Comparator(n), ddg_tree=False, bipolar=False)
        Zs_np[N_idx] = cemux_np.forward(pxs_np[N_idx], L)
    end2 = time.time()
    print(Zs[0, 0:20], Zs_np[0, 0:20])
    print((Zs_np == Zs.cpu().numpy()).all())
    print(end1 - start)
    print(end2 - end1)


if __name__ == '__main__':
    test2()