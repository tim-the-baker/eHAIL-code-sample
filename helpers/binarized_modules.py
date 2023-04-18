# Adapted from https://github.com/itayhubara/BinaryNet.pytorch/blob/master/models/binarized_modules.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from SCython.Utilities.SN_operations import get_SCC, get_SN_value

eps = 1e-10

def Binarize(tensor, quant_mode='det'):
    # det stands for deterministic.
    if quant_mode == 'det':
        #  the sign bit is mapped to +/- 1
        return tensor.sign()
    else:
        raise NotImplementedError
        # The sign bit is stochastically mapped to +/- 1
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


def Quantize(tensor, bitwidth, signed=True):
    # this method should probably only be used on the input layer.
    # Quantize the vector to (bitwidth-int(signed))-bit precision (1 bit is reserved for the sign bit if signed=True)
    tensor = tensor.mul(2**(bitwidth-int(signed))).round().div(2**(bitwidth-int(signed)))

    return tensor


class BinarizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        assert self.bias is None, "I haven't checked that bias terms work for BinarizedLinear module"

    def forward(self, input, input_prec=None):
        # Handle input precision
        if input_prec is None:  # don't modify input
            pass
        elif input_prec == 1:  # binarize input
            input.data = Binarize(input.data)
        elif input_prec > 1:  # quantize input
            input.data = Quantize(input.data, input_prec)
        else:
            raise ValueError(f"Input precision: {input_prec} must be >= to 1.")

        # save the original weights into a instance variable
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        # now binarize the weights for inference
        self.weight.data = Binarize(self.weight.org)
        out = F.linear(input, self.weight, self.bias)

        return out


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        assert self.bias is None, "I haven't checked that bias terms work for BinarizedCNN module"


    def forward(self, input, input_prec=None):
        # Handle input precision
        if input_prec is None:  # don't modify input
            pass
        elif input_prec == 1:  # binarize input
            input.data = Binarize(input.data)
        elif input_prec > 1:  # quantize input
            input.data = Quantize(input.data, input_prec)
        else:
            raise ValueError(f"Input precision: {input_prec} must be >= to 1.")

        # store a copy of the weights, if not done so already.
        if not hasattr(self.weight,'org'):
            self.weight.org = self.weight.data.clone()

        # now binarize the weights and run the layer
        self.weight.data = Binarize(self.weight.org)

        out = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out


class SCMuxLinear(nn.Linear):
    def __init__(self, msg, split_gradients, *kargs, **kwargs):
        super(SCMuxLinear, self).__init__(*kargs, **kwargs)
        self.msg = msg
        self.split_gradients = split_gradients

        assert self.bias is None, f"Bias is not currently implemented for SCMuxLinear."

    def forward(self, SNs, full_corr=False):
        """
        Input SN shape is expected to be (N, K, L)
        self.weight is expected to be (M, K)
        N: batch size, K: input nodes , L: SN length, M: output nodes
        """
        # check, selects = self.forward2(SNs, full_corr)
        # torch.cuda.empty_cache()

        if full_corr:
            assert SNs.ndim == 4, "Input SN shape is expected to be (N, K, 2, L)"
            N, K, _, L = SNs.shape
        else:
            assert SNs.ndim == 3, "Input SN shape is expected to be (N, K, L)"
            N, K, L = SNs.shape
        M = self.out_features

        # save the original weights into a instance variable
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        # now binarize the weights for inference
        self.weight.data = Binarize(self.weight.org).float()  # shape: (M, K)

        # if full corr, we need to grab the right SNs for each input
        if full_corr:
            SNs = SNs.transpose(dim0=1, dim1=3)  # (N, L, 2, K)
            # expand mask and grab the SNs
            mask = self.weight < 0
            mask = mask[None, None].long().expand(N, L, M, K)  # (N, L, M, K)
            SNs = SNs.gather(dim=2, index=mask)  # shape: (N, L, M, K)
            del mask

        else:
            SNs = SNs.transpose(dim0=1, dim1=2)  # (N, L, K)
            SNs = SNs[:, :, None].expand(N, L, M, K)

        # we do mux selection before multiplication to reduce memory overhead and hopefully improve speed
        # at this point SNs is shape (N, L, M, K) and self.weight is shape (M, K)
        selects = self.msg.gen_selects_NN(batch_size=N, num_outputs=self.out_features, SN_length=L) # shape: (N, L, M, 1)

        # gather over dim corresponding to K
        SNs = torch.gather(SNs, dim=-1, index=selects) # (N, L, M, L)
        weights = torch.gather(self.weight[None].expand(L, M, K), dim=-1, index=selects[0])  # (L, M, 1)

        # multiply weights by SN bits
        SNs = torch.mul(SNs, weights).char()  # out shape: (N, L, M, 1)

        # sum over dim corresponding to L
        SNs = SNs.sum(dim=1)  # (N, M, 1)

        return SNs.squeeze()


    def forward_old(self, SNs, full_corr=False):
        """
        Input SN shape is expected to be (N, K, L)
        self.weight is expected to be (M, K)
        N: batch size, K: input nodes , L: SN length, M: output nodes
        """
        if full_corr:
            assert SNs.ndim == 4, "Input SN shape is expected to be (N, K, 2, L)"
            N, K, _, L = SNs.shape
        else:
            assert SNs.ndim == 3, "Input SN shape is expected to be (N, K, L)"
            N, K, L = SNs.shape
        M = self.out_features

        # save the original weights into a instance variable
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
            self.inv_mask = self.weight.data < 0

        # now binarize the weights for inference
        self.weight.data = Binarize(self.weight.org).half()  # shape: (M, K)

        # if full corr, we need to grab the right SNs for each input
        if full_corr:
            SNs = SNs.transpose(dim0=1, dim1=3)  # (N, L, 2, K)
            # expand mask and grab the SNs
            mask = self.weight < 0
            mask = mask[None, None].long().expand(N, L, M, K)  # (N, L, M, K)
            SNs = SNs.gather(dim=2, index=mask)  # shape: (N, L, M, K)
            del mask
            torch.cuda.empty_cache()

            # multiply weights by SN bits
            SNs = torch.mul(SNs, self.weight).char() # out shape: (N, L, M, K)
        else:
            SNs = SNs.transpose(dim0=1, dim1=2)  # (N, L, K)
            SNs = SNs[:, :, None].expand(N, L, M, K)
            # multiply weights by SN bits
            SNs = torch.mul(SNs, self.weight).char() # out shape: (N, L, M, K)

        # account for mux and its select gen
        selects = self.msg.gen_selects_NN(batch_size=N, num_outputs=self.out_features, SN_length=L) # shape: (N, L, M, 1)
        SNs = torch.gather(SNs, dim=-1, index=selects)  # gather over dim corresponding to K
        SNs = SNs.sum(dim=1)  #sum over dim correponding to L

        return SNs.squeeze()


# This was used in exp05_mux_exps.py. It only works with sampler height divides input size evenly (e.g., K=784
# can use height = 1, 2, 3, 4, 5 which changes input size to K = 784, 392, 196, 98, 49.
class SamplingLayer(nn.Linear):
    def __init__(self, msg_class, height, device, *kargs, **kwargs):
        super(SamplingLayer, self).__init__(*kargs, **kwargs)
        assert height == -1 or (self.in_features / (2**height)) % 1 == 0

        self.height = height # height of each sampling block
        self.block_size = self.in_features if height == -1 else int(2**height) # K is inputs per sampling block
        self.samp_blocks = self.in_features // self.block_size
        self.msg = msg_class(self.block_size, device)

        assert self.bias is None, f"Bias is not currently implemented for SamplingLayer."

    def forward(self, SNs, full_corr=False):
        """
        Input SN shape is expected to be (N, K, L)
        self.weight is expected to be (M, K)
        N: batch size, K: input nodes, L: SN length, M: output nodes
        S: number of sampling blocks, B SNs per sampling block
        """
        # check, selects = self.forward2(SNs, full_corr)
        # torch.cuda.empty_cache()

        if full_corr:
            assert SNs.ndim == 4, "Input SN shape is expected to be (N, K, 2, L)"
            N, K, _, L = SNs.shape
        else:
            assert SNs.ndim == 3, "Input SN shape is expected to be (N, K, L)"
            N, K, L = SNs.shape
        M, B, S = self.out_features, self.block_size, self.samp_blocks

        # save the original weights into a instance variable
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        # now binarize the weights for inference
        self.weight.data = Binarize(self.weight.org).float()  # shape: (M, K)

        # if full corr, we need to grab the right SNs for each input
        if full_corr:
            SNs = SNs.transpose(dim0=1, dim1=3)  # (N, L, 2, K)
            # expand mask and grab the SNs
            mask = self.weight < 0
            mask = mask[None, None].long().expand(N, L, M, K)  # (N, L, M, K)
            SNs = SNs.gather(dim=2, index=mask)  # shape: (N, L, M, K)
            del mask

        else:
            SNs = SNs.transpose(dim0=1, dim1=2)  # (N, L, K)
            SNs = SNs[:, :, None].expand(N, L, M, K)

        # Now we need to reshape SNs so that we can do sampling properly
        SNs = SNs.view(N, L, M, S, B)  # (N, L, M, S, B)

        # we do mux selection before multiplication to reduce memory overhead and hopefully improve speed
        # at this point SNs is shape (N, L, M, K) and self.weight is shape (M, K)
        selects = self.msg.gen_selects_NN(batch_size=N, num_outputs=self.out_features, SN_length=L) # shape: (N, L, M, 1)
        selects = selects[...,None,:].expand(N, L, M, S, 1)  # (N, L, M, S, 1)

        # gather over dim corresponding to K
        SNs = torch.gather(SNs, dim=-1, index=selects) # (N, L, M, S, 1)
        weights = torch.gather(self.weight[None].expand(L, M, K).view(L, M, S, B), dim=-1, index=selects[0])  # (L, M, S, 1)

        # multiply weights by SN bits
        SNs = torch.mul(SNs, weights).char()  # out shape: (N, L, M, S, 1)

        # sum over dim corresponding to L
        SNs = SNs.sum(dim=(1, -2, -1))  # (N, M)

        return SNs


    def forward_explicit(self, SNs, full_corr=False):
        """
        Input SN shape is expected to be (N, K, L)
        self.weight is expected to be (M, K)
        N: batch size, K: input nodes , L: SN length, M: output nodes
        """
        if full_corr:
            assert SNs.ndim == 4, "Input SN shape is expected to be (N, K, 2, L)"
            N, K, _, L = SNs.shape
        else:
            assert SNs.ndim == 3, "Input SN shape is expected to be (N, K, L)"
            N, K, L = SNs.shape
        M = self.out_features

        # now binarize the weights for inference
        self.weight.data = Binarize(self.weight).half()  # shape: (M, K)

        # if full corr, we need to grab the right SNs for each input
        if full_corr:
            SNs = SNs.transpose(dim0=1, dim1=3)  # (N, L, 2, K)
            # expand mask and grab the SNs
            mask = self.weight < 0
            mask = mask[None, None].long().expand(N, L, M, K)  # (N, L, M, K)
            SNs = SNs.gather(dim=2, index=mask)  # shape: (N, L, M, K)
            del mask

            # multiply weights by SN bits
            SNs = torch.mul(SNs, self.weight).char() # out shape: (N, L, M, K)
        else:
            SNs = SNs.transpose(dim0=1, dim1=2)  # (N, L, K)
            SNs = SNs[:, :, None].expand(N, L, M, K)
            # multiply weights by SN bits
            SNs = torch.mul(SNs, self.weight).char() # out shape: (N, L, M, K)

        # account for mux and its select gen
        selects = self.msg.gen_selects_NN(batch_size=N, num_outputs=self.out_features, SN_length=L) # shape: (N, L, M, 1)
        SNs = torch.gather(SNs, dim=-1, index=selects)  # gather over dim corresponding to K
        SNs = SNs.sum(dim=1)  #sum over dim correponding to L

        return SNs.squeeze()


# This class generalizes SamplingLayer to work with arbitrary height by putting leftover SNs int their own group.
class PSA_Linear(nn.Linear):
    def __init__(self, height, msg, rem_mode, sep_bip, *kargs, **kwargs):
        super(PSA_Linear, self).__init__(*kargs, **kwargs)

        self.height = height
        self.msg = msg
        self.block_size = int(2**height)
        self.samp_blocks = self.in_features // self.block_size
        self.rem = int((self.in_features % self.block_size))
        self.sep_bip = sep_bip
        self.is_apc = self.height == 0

        assert self.bias is None, f"Bias is not currently implemented for SamplingLayer."


    def forward(self, SNs, full_corr=False):
        """
        Input SN shape is expected to be (N, K, L)
        self.weight is expected to be (M, K)
        N: batch size, K: input nodes, L: SN length, M: output nodes
        S: number of sampling blocks, B: SNs per sampling block
        """
        if self.is_apc:
            return self.forward_apc(SNs)

        if full_corr:
            assert SNs.ndim == 4, "Input SN shape is expected to be (N, K, 2, L)"
            N, K, _, L = SNs.shape
        else:
            assert SNs.ndim == 3, "Input SN shape is expected to be (N, K, L)"
            N, K, L = SNs.shape

        M, B, S, R = self.out_features, self.block_size, self.samp_blocks, self.rem

        # save the original weights into a instance variable
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        # now binarize the weights for inference
        self.weight.data = Binarize(self.weight.org).float()  # shape: (M, K)

        if self.sep_bip:
            return self.forward_sep_bip(SNs, full_corr)

        # if full corr, we need to grab the right SNs for each input
        if full_corr:
            SNs = SNs.transpose(dim0=1, dim1=3)  # (N, L, 2, K)
            # expand mask and grab the SNs
            mask = self.weight < 0
            mask = mask[None, None].long().expand(N, L, M, K)  # (N, L, M, K)
            SNs = SNs.gather(dim=2, index=mask)  # shape: (N, L, M, K)
            del mask

        else:
            SNs = SNs.transpose(dim0=1, dim1=2)  # (N, L, K)
            SNs = SNs[:, :, None].expand(N, L, M, K)  # (N, L, M, K)


        # Split based on remainder
        SNs_main, SNs_rem = SNs.split((B*S, R), dim=-1)  # (N, L, M, B*S) and (N, L, M, R)
        del SNs
        weights_main, weights_rem = self.weight.split((B*S, R), dim=-1)  # (M, B*S)  and (M, R)

        # Now we need to reshape SNs so that we can do sampling properly
        SNs_main = SNs_main.view(N, L, M, S, B)

        # we do mux selection before multiplication to reduce memory overhead and hopefully improve speed
        # at this point SNs is shape (N, L, M, K) and self.weight is shape (M, K)
        selects = self.msg.gen_selects_NN(batch_size=N, num_outputs=self.out_features, SN_length=L) # shape: (N, L, M, 1)
        selects = selects[...,None,:].expand(N, L, M, S, 1)  # (N, L, M, S, 1)

        # gather over dim corresponding to K
        SNs_main = torch.gather(SNs_main, dim=-1, index=selects) # (N, L, M, S, 1)
        weights_main = torch.gather(weights_main[None].expand(L, M, S*B).view(L, M, S, B), dim=-1, index=selects[0])  # (L, M, S, 1)

        # multiply weights by SN bits
        SNs_main = torch.mul(SNs_main, weights_main)  # out shape: (N, L, M, S, 1)
        SNs_main = SNs_main.sum(dim=(1, -2, -1), dtype=torch.float)/L*self.block_size  # (N, M)

        # Handle remainders
        if R > 0:
            SNs_rem = torch.mul(SNs_rem, weights_rem).short()  # out shape: (N, L, M, R)
            SNs_rem = SNs_rem.sum(dim=(1, -1)) / L  # (N, M)
            SNs_main = (SNs_main + SNs_rem)

        return SNs_main


    def forward_apc(self, SNs):
        """
        Input SN shape is expected to be (N, K, L)
        self.weight is expected to be (M, K)
        N: batch size, K: input nodes, L: SN length, M: output nodes
        S: number of sampling blocks, B: SNs per sampling block
        """
        assert SNs.ndim == 3, "Input SN shape is expected to be (N, K, L)"
        N, K, L = SNs.shape

        # save the original weights into a instance variable
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        # now binarize the weights for inference
        self.weight.data = Binarize(self.weight.org).float()  # shape: (M, K)

        SNs = SNs.transpose(dim0=1, dim1=2).float()  # (N, L, K)
        SNs = F.linear(SNs, self.weight, self.bias)  # (N, L, M)
        SNs = SNs.sum(dim=1) / L

        return SNs


    def forward_sep_bip(self, SNs, full_corr):
        assert not self.training
        if full_corr:
            N, K, _, L = SNs.shape
        else:
            N, K, L = SNs.shape
        M, B, = self.out_features, self.block_size

        # Split into positive and negative
        out = torch.zeros((N, M), device='cuda:0')
        for w_idx in range(M):
            # Get weight mask
            curr_weights = self.weight[w_idx]  # (K,)
            weight_mask = curr_weights < 0  # (K,)

            # sep SNs into positive and negative
            if full_corr:
                neg_SNs = SNs[:, weight_mask, 1]  # (N, Q, L)
                pos_SNs = SNs[:, ~weight_mask, 0]  # (N, P, L)
            else:
                neg_SNs = SNs[:, weight_mask]  # (N, Q, L)
                pos_SNs = SNs[:, ~weight_mask]  # (N, P, L)

            # split based on remainder
            Q, P = neg_SNs.shape[1], pos_SNs.shape[1]
            assert Q + P == K
            S_neg, R_neg = Q // self.block_size, Q % self.block_size  # negative SNs num samp blocks and remainder
            S_pos, R_pos = P // self.block_size, P % self.block_size  # positive SNs num samp blocks and remainder

            neg_SNs_main, neg_SNs_rem = neg_SNs.split((B*S_neg, R_neg), dim=1)  # (N, B*S_neg, L) and (N, R_neg, L)
            pos_SNs_main, pos_SNs_rem = pos_SNs.split((B*S_pos, R_pos), dim=1)  # (N, B*S_pos, L) and (N, R_pos, L)

            # Change SN and weight views so we can do sampling :)
            neg_SNs_main = neg_SNs_main.view(N, S_neg, B, L)  # (N, S_neg, B, L)
            pos_SNs_main = pos_SNs_main.view(N, S_pos, B, L)  # (N, S_pos, B, L)

            # we do mux selection before multiplication to reduce memory overhead and hopefully improve speed
            selects = self.msg.gen_selects(SN_length=L, shape=None, share=True)  # (L,)
            selects_neg = selects.expand(N, S_neg, 1, L)  # (N, S_neg, 1, L)
            selects_pos = selects.expand(N, S_pos, 1, L)  # (N, S_pos, 1, L)

            # gather SNs and weights
            neg_SNs_main = torch.gather(neg_SNs_main, dim=-2, index=selects_neg) # (N, S_neg, 1, L)
            pos_SNs_main = torch.gather(pos_SNs_main, dim=-2, index=selects_pos) # (N, S_pos, 1, L)

            # all weights have an absolute value of 1 so we can just sum the SN bits :-)
            neg_SNs_main = neg_SNs_main.sum(dim=(-3, -2, -1))/L*self.block_size  # (N, )
            pos_SNs_main = pos_SNs_main.sum(dim=(-3, -2, -1))/L*self.block_size  # (N, )

            # Handle remainders
            if R_pos > 0:
                pos_SNs_rem =  pos_SNs_rem.sum(dim=(-1, -2))/L  # (N,)
                pos_SNs_main = pos_SNs_main + pos_SNs_rem
            if R_neg > 0:
                neg_SNs_rem = neg_SNs_rem.sum(dim=(-1, -2))/L  # (N,)
                neg_SNs_main = neg_SNs_main + neg_SNs_rem

            out[:, w_idx] = pos_SNs_main - neg_SNs_main

        return out


    def forward_explicit(self, SNs, full_corr=False):
        """
        Input SN shape is expected to be (N, K, L)
        self.weight is expected to be (M, K)
        N: batch size, K: input nodes , L: SN length, M: output nodes
        """
        if full_corr:
            assert SNs.ndim == 4, "Input SN shape is expected to be (N, K, 2, L)"
            N, K, _, L = SNs.shape
        else:
            assert SNs.ndim == 3, "Input SN shape is expected to be (N, K, L)"
            N, K, L = SNs.shape
        M = self.out_features

        # now binarize the weights for inference
        self.weight.data = Binarize(self.weight).half()  # shape: (M, K)

        # if full corr, we need to grab the right SNs for each input
        if full_corr:
            SNs = SNs.transpose(dim0=1, dim1=3)  # (N, L, 2, K)
            # expand mask and grab the SNs
            mask = self.weight < 0
            mask = mask[None, None].long().expand(N, L, M, K)  # (N, L, M, K)
            SNs = SNs.gather(dim=2, index=mask)  # shape: (N, L, M, K)
            del mask

            # multiply weights by SN bits
            SNs = torch.mul(SNs, self.weight).char() # out shape: (N, L, M, K)
        else:
            SNs = SNs.transpose(dim0=1, dim1=2)  # (N, L, K)
            SNs = SNs[:, :, None].expand(N, L, M, K)
            # multiply weights by SN bits
            SNs = torch.mul(SNs, self.weight).char() # out shape: (N, L, M, K)

        # account for mux and its select gen
        selects = self.msg.gen_selects_NN(batch_size=N, num_outputs=self.out_features, SN_length=L) # shape: (N, L, M, 1)
        SNs = torch.gather(SNs, dim=-1, index=selects)  # gather over dim corresponding to K
        SNs = SNs.sum(dim=1)  #sum over dim correponding to L

        return SNs.squeeze()
