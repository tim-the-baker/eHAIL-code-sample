import torch

class PCC:
    """ Abstract class for stochastic computing probability conversion circuits """
    def __init__(self, precision):
        """
        :param precision:
        :param can_generate_all_1s:
        """
        assert precision <= 16
        self.n = precision
        self.name: str = ""
        self.legend: str = ""

    def forward(self, Rs, Cs):
        """
        Abstract method for implementation of the PCC.
        :param torch.tensor Rs: the random number inputs to the PCC.
        :param torch.tensor Cs: the control inputs to the PCC.
        :return: the generated SNs
        """
        raise NotImplementedError

    def __str__(self):
        return self.name

    def info(self):
        return f"### PCC Info: name={self.name} n={self.n}"

class WBG(PCC):
    def __init__(self, precision):
        super().__init__(precision)
        self.name = "WBG"
        self.legend = "WBG"

    def forward(self, Rs, Bs):
        # SNs = np.full(Rs.shape, -1, dtype=int)
        # np.log2(Rs,  where=(Rs != 0), out=SNs, casting='unsafe')
        # SNs = np.bitwise_and(np.right_shift(Bs.T, SNs.T), 1).T

        SNs = torch.full_like(Rs, -1, dtype=torch.float32)
        SNs[Rs!=0] = torch.log2(Rs[Rs!=0])
        SNs = SNs.int()
        SNs = torch.bitwise_and(torch.bitwise_right_shift(Bs[..., None], SNs), 1)
        # the following line handles the case when the input value is = 1. In real hardware, a real WBG cannot handle
        # this situation. For instance, a standard 4-bit WBG can generate unipolar SNs with value 0, 1/16, 2/16, ...,
        # 15/16, but not with value 16/16 because an extra input bit would be needed. However, relatively little hardware
        # modification would needed to handle the value = 16/16 case (just 1 OR gate would be needed).
        SNs[Bs == pow(2, self.n)] = 1

        return SNs

class Comparator(PCC):
    """
    Class that implements comparator PCC.
    """
    def __init__(self, precision):
        super().__init__(precision)
        self.name = "Comparator"
        self.legend = "CMP"

    def forward(self, Rs, Bs):
        # SNs = (Rs.T < Bs.T).T.float()
        if Rs.dim() == Bs.dim():
            SNs = Rs < Bs
        else:
            SNs = (Rs < Bs[..., None])
        return SNs

# TODO update this class from NumPy
class Mux_maj_chain(PCC):
    def __init__(self, precision, num_mux, num_maj, invert_maj_R):
        super().__init__(precision)
        assert precision == num_mux + num_maj
        self.num_mux = num_mux
        self.num_maj = num_maj
        self.invert_maj_R = invert_maj_R

        self.is_maj_list = torch.ones(precision)
        self.is_maj_list[0:num_mux] = 0
        self.list_idx = int(sum([bit * 2**idx for idx, bit in enumerate(self.is_maj_list)]))

        self.name = f"MUX-MAJ-chain-k{num_maj}"
        self.legend = rf"MMC $k={self.num_maj}$ MAJ"

    def forward(self, Rs, Bs):
        mask = int(2**self.num_mux) - 1

        lower_Rs, lower_Bs = torch.bitwise_and(Rs, mask), torch.bitwise_and(Bs, mask)
        upper_Rs, upper_Bs = Rs >> self.num_mux, Bs >> self.num_mux

        # Do the WBG on lower bits
        WBG_outs = torch.full_like(lower_Rs, -1, dtype=torch.float32)
        WBG_outs[lower_Rs != 0] = torch.log2(lower_Rs[lower_Rs != 0])
        WBG_outs = WBG_outs.int()
        WBG_outs = torch.bitwise_and(torch.bitwise_right_shift(lower_Bs[..., None], WBG_outs), 1)

        # the following line handles the case when the input value is = 1. In real hardware, a real WBG cannot handle
        # this situation. For instance, a standard 4-bit WBG can generate unipolar SNs with value 0, 1/16, 2/16, ...,
        # 15/16, but not with value 16/16 because an extra input bit would be needed. However, relatively little hardware
        # modification would needed to handle the value = 16/16 case (just 1 OR gate would be needed).
        WBG_outs[Bs == pow(2, self.n)] = 1

        # do the comparator (i.e., the MAJ gates)
        if upper_Bs.ndim == upper_Rs.ndim - 1:
            upper_Bs = upper_Bs[..., None]
        if self.invert_maj_R:
            upper_Rs = pow(2, self.num_maj) - 1 - upper_Rs

        SNs = ((upper_Rs - upper_Bs - WBG_outs) < 0).int()

        return SNs


def get_pcc_class_from_name(name):
    for cls in PCC.__subclasses__():
        if cls.__name__ == name:
            return cls


if __name__ == '__main__':
    precision = 4
    SN_length = int(2**precision)
    device = torch.device('cpu')
    from SCython.SNG_torch import SNG, RNS
    import SCython.SNG.RNS as RNS2
    import SCython.SNG.PCC as PCC2
    import SCython.SNG.SNG as SNG2
    import numpy as np

    rns = RNS.Counter_RNS(precision, device)
    rns2 = RNS2.Counter_RNS(precision)
    invert = False
    for k in range(precision):
        print(k)
        pcc = Mux_maj_chain(precision, num_mux=precision-k, num_maj=k, invert_maj_R=invert)
        pcc2 = PCC2.Mux_maj_chain(precision, num_mux=precision-k, num_maj=k, invert_maj_R=invert)
        sng = SNG.SNG(rns, pcc)
        sng2 = SNG2.SNG(rns2, pcc2)

        pxs = torch.arange(0, SN_length+1)/SN_length
        pxs2 = np.arange(0, SN_length+1)/SN_length
        SNs = sng.gen_SN(pxs, SN_length, bipolar=False, share_RNS=True, RNS_mask=None, full_corr=None)
        SNs2 = sng2.gen_SN(pxs2, SN_length, bipolar=False, share_RNS=True, RNS_mask=None)
        for px, SN, SN2 in zip(pxs, SNs, SNs2):
            # print("\t", px, SN, SN.float().mean())
            print(f"\tPx correct: {px == SN.float().mean()}  SNs match:{(SN.numpy()==SN2).all()}\n"
                  f"SN Comp: {SN.numpy()==SN2}\n")
            assert (px == SN.float().mean()) and (SN.numpy()==SN2).all()
        print("")
