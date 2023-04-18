# TODO DOC strings
import numpy as np
from SCython.SNG import RNS, PCC, SNG
from SCython.Utilities import seq_utils
import SCython.Utilities.SN_operations as SN_ops
import matplotlib.pyplot as plt
import logging
import sys

def gen_gray_counter(precision):
    pow2n = int(2**precision)
    counter = np.empty(pow2n)
    for i in range(pow2n):
        counter[i] = i ^ (i >> 1)
    # for i in range(1, precision):
    #     curr = int(2**i)
    #     counter[(curr-1)*pow2n//curr:] = list(reversed(counter[(curr-1)*pow2n//curr:]))
    return counter

# configure logger
# logging.basicConfig(level=logging.DEBUG, format='[%(levelname)8s]:  %(message)s', stream=sys.stdout)

def generation_test(rns, pcc, precision, bipolar, dimensions=1, **sng_kwargs):
    """
    This test checks if this RNS and PCC pair can generate SNs with the correct values for both bipolar and unipolar SNs.
    Both the RNS and PCC have the same precision.
    :param rns:
    :param pcc:
    :param precision:
    :param bipolar:
    :param dimensions
    :return:
    """
    assert 1 <= dimensions <= 3
    assert 4 <= precision <= 8

    pow2n = int(2**precision)
    input_probs = np.arange(pow2n+1)/pow2n
    input_values = 2*input_probs - 1 if bipolar else input_probs
    if dimensions == 2:
        input_values = np.array(np.meshgrid(input_values, input_values))[0]
    if dimensions == 3:
        input_values = np.array(np.meshgrid(input_values, input_values))

    sng = SNG.SNG(rns, pcc, rns_precision=precision, pcc_precision=precision, **sng_kwargs)
    Xs = sng.gen_SN(values=input_values, SN_length=pow2n, bipolar=bipolar, share_RNS=True, RNS_mask=None)
    X_hats = SN_ops.get_SN_value(Xs, bipolar)

    message = f"with rns:{sng.rns} and pcc:{sng.pcc} when using precision={precision}, bipolar={bipolar} and dimensions={dimensions}"
    if (X_hats == input_values).all():
        logging.info(f"generation_test passed {message}")
    else:
        logging.error(f"generation_test failed {message}\n{X_hats}")


def correlated_SCC_test(rns, pcc, precision, **sng_kwargs):
    """
    This test checks if this RNS and PCC pair can generate SNs with SCC = + 1 and SCC = -1 when the comparator is used.
    :param rns:
    :param pcc:
    :param precision:
    :return:
    """
    assert 3 <= precision <= 16

    pow2n = int(2**precision)
    input_probs = np.arange(pow2n)/pow2n
    # input_probs = gen_gray_counter(precision)/pow2n
    XY_values = np.array(np.meshgrid(input_probs, input_probs))

    share_RNS_mask_anti = np.array([np.zeros((pow2n, pow2n)), np.ones((pow2n, pow2n))], dtype=bool)
    sng = SNG.SNG(rns, pcc, **sng_kwargs)

    corr_SNs = sng.gen_SN(values=np.array(XY_values), SN_length=pow2n, bipolar=False, share_RNS=True, RNS_mask=None)
    anti_corr_SNs = sng.gen_SN(values=np.array(XY_values), SN_length=pow2n, bipolar=False, share_RNS=True,
                               RNS_mask=share_RNS_mask_anti)

    corr_SCCs = SN_ops.get_SCC(Xs=corr_SNs[0], Ys=corr_SNs[1], pxs=XY_values[0], pys=XY_values[1])
    anti_corr_SCCs = SN_ops.get_SCC(Xs=anti_corr_SNs[0], Ys=anti_corr_SNs[1], pxs=XY_values[0], pys=XY_values[1])

    plt.pcolor(input_probs+input_probs[1], input_probs+input_probs[1], corr_SCCs[1:, 1:], cmap="plasma", vmin=-1, vmax=1)
    plt.xlabel("$P_X$", fontsize=20)
    plt.ylabel("$P_Y$", fontsize=20)
    plt.xticks([0, 0.25, 0.5, 0.75, 1], fontsize=14)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=14)
    plt.tight_layout()
    plt.colorbar()
    plt.show()
    plt.pcolor(input_probs+input_probs[1], input_probs+input_probs[1], anti_corr_SCCs[1:,1:], cmap="plasma", vmin=-1, vmax=1)
    plt.xlabel("$P_X$", fontsize=16)
    plt.ylabel("$P_Y$", fontsize=16)
    plt.xticks([0, 0.25, 0.5, 0.75, 1], fontsize=10)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=10)
    plt.tight_layout()
    plt.colorbar()
    plt.show()


def uncorrelated_SCC_test(rns, pcc, precision, num_runs, **sng_kwargs):
    """
    This test checks if this RNS and PCC pair can generate SNs with SCC = + 1 and SCC = -1 when the comparator is used.
    :param rns:
    :param pcc:
    :param precision:
    :return:
    """
    assert 3 <= precision <= 16

    pow2n = int(2**precision)
    input_probs = np.arange(pow2n)/pow2n
    XY_values = np.array(np.meshgrid(input_probs, input_probs))

    sng = SNG.SNG(rns, pcc, rns_precision=precision, pcc_precision=precision, **sng_kwargs)

    SCCs = np.empty((*XY_values[0].shape, num_runs))
    for r_idx in range(num_runs):
        if r_idx % 100 == 0:
            print(f"Run: {r_idx}/{num_runs}")
        SNs = sng.gen_SN(values=np.array(XY_values), SN_length=pow2n, bipolar=False, share_RNS=False, RNS_mask=None)
        SCCs[..., r_idx] = SN_ops.get_SCC(Xs=SNs[0], Ys=SNs[1], pxs=XY_values[0], pys=XY_values[1], do_cov=True)

    SCCs = np.mean(SCCs, axis=-1)
    plt.imshow(SCCs, origin="lower", cmap="plasma")
    plt.colorbar()
    plt.show()


def run_series_of_generation_tests():
    precision = 8
    RNSs = [RNS.Hypergeometric_RNS]
    PCCs = [PCC.Comparator, PCC.WBG]

    for dimensions in [1, 2, 3]:
        for rns in RNSs:
            for pcc in PCCs:
                for bipolar in [False, True]:
                    generation_test(rns, pcc, precision=precision, bipolar=bipolar, dimensions=dimensions)


def run_SCC_test():
    precision = 8
    RNSs = [RNS.Hypergeometric_RNS]
    PCCs = [PCC.Comparator, PCC.WBG]
    vdc_seq = seq_utils.get_vdc(n=precision)

    for rns in RNSs:
        for pcc in PCCs:
            if rns is RNS.VDC_RNS:
                correlated_SCC_test(rns, pcc, precision=precision, vdc_seq=vdc_seq)
            else:
                correlated_SCC_test(rns, pcc, precision=precision)


if __name__ == '__main__':
    # run_series_of_generation_tests()
    # run_SCC_test()
    precision = 8
    rns = RNS.Hypergeometric_RNS(precision)
    pcc = PCC.WBG(precision)
    num_runs = 10000
    correlated_SCC_test(rns, pcc, precision)
    # uncorrelated_SCC_test(rns, pcc, precision, num_runs)
