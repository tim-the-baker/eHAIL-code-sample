# TODO update docstrings
import numpy as np
import torch
from typing import Union
from SCython.SNG_torch import RNS, PCC


#### Quantization Functions ###
def q_floor(x, precision, signed=False) -> torch.tensor:
    """
    truncates x to a specified precision
    :param torch.Tensor x:
    :param int precision:
    :param bool signed:
    :return:
    """
    #  truncates x to specified precision
    pow2n = 2**(precision - int(signed))
    return torch.floor(x*pow2n)/pow2n


def q_nearest(x, precision, signed=False) -> torch.tensor:
    """
    rounds x to specified precision (rounds to nearest)
    :param numpy.ndarray x:
    :param int precision:
    :param bool signed:
    :return:
    """
    pow2n = 2**(precision-int(signed))
    return torch.round(x*pow2n)/pow2n


class SNG:
    def __init__(self, rns, pcc):
        """
        :param RNS.RNS rns: RNS of the SNG (see SNG.RNS)
        :param PCC.PCC pcc: PCC of the SNG (see SNG.RNS)
        """
        assert rns.n >= pcc.n
        self.q_func = q_floor

        self.rns = rns
        self.pcc = pcc
        self.device = rns.device

    def gen_SN(self, values, SN_length, bipolar, share_RNS, RNS_mask, full_corr, Rs=None):
        """
        :param torch.tensor values:
        :param int SN_length:
        :param bool bipolar:
        :param bool share_RNS:
        :param Union[np.ndarray, None] RNS_mask:
        :param Union[np.ndarray, None] Rs:
        :return:
        """
        # quantize the input values
        values = self.q_func(values, self.pcc.n)

        # determine the SN probabilities based on values and whether SNs are bipolar or unipolar.
        ps = (values+1)/2 if bipolar else values

        assert (ps >= 0).all(), ps
        assert (ps <= 1).all(), ps
        assert values.device == self.device, f"{values.device} {self.device}"

        # transform the fixed point probability values to integer format
        Bs = (ps * (2**self.pcc.n)).short()

        # generate RNs
        if Rs is None:
            Rs = self.rns.gen_RN(SN_length, ps.shape, share_RNS, RNS_mask, full_corr)

        if full_corr and RNS_mask is None:
            Bs = Bs[..., None].expand(*Bs.shape, 2)

        # the Rs must also quantized to fit into the PCC if rns.n != pcc.n. Do this by truncating the MSBs of the Rs.
        # To accomplish MSB truncation, a 1s mask of length pcc.n can be bitwise AND'ed with Rs
        if self.pcc.n != self.rns.n:
            raise NotImplementedError
            # truncation_mask = int(2**self.pcc.n) - 1
            # Rs = np.bitwise_and(Rs, truncation_mask)

        SNs = self.pcc.forward(Rs, Bs)

        if full_corr:
            assert SNs.shape == (*values.shape, 2, SN_length),\
                f"SN Shape: {SNs.shape} was expected to be ({values.shape}, 2, {SN_length})."
        else:
            assert SNs.shape == (*values.shape, SN_length),\
                f"SN Shape: {SNs.shape} was expected to be ({values.shape}, {SN_length})."

        return SNs

    def gen_verilog(self):
        raise NotImplementedError

    def info(self):
        return f"{self.rns.info()}\n{self.pcc.info()}"


if __name__ == '__main__':
    pass
