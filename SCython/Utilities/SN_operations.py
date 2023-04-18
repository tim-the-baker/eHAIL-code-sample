import numpy as np

def get_SN_value(SN_array, bipolar):
    """
    Computes the value of each SN in an array. Assumes last dimension of SN_array corresponds to the SN bits.
    For instance, if SN_array is shape (5,2,32), then that means we have a 5x2 array of SNs of length 32.
    :param SN_array
    :param bipolar: whether or not bipolar format is used
    :return: SN values
    """
    if bipolar:
        values = 2*np.mean(SN_array, axis=-1) - 1
    else:
        values = np.mean(SN_array, axis=-1)

    return values

def get_SCC(Xs, Ys, pxs=None, pys=None, do_cov=False, do_corr=False, tie=0):
    """
    :param Xs: List of SNs to measure SCC.
    :param Ys: List of other SN to measure SCC (must be same size as Xs)
    :param pxs: Xs' values if known  (value will be estimated if not given)
    :param pys: Ys' values if known  (value will be estimated if not given)
    :param do_cov: If true, the covariance of the bit-streams are returned rather than their SCC.
    :param do_corr: If true, the correlation of the bit-streams are returned rather than their SCC.
    :return:
    """
    # Do some checking
    assert Xs.shape == Ys.shape
    assert not (do_cov and do_corr)
    if pxs is not None and pys is not None:
        assert pxs.shape == pys.shape
        assert pxs.shape == Xs.shape[:-1], f"pxs shape: {pxs.shape} and Xs shape: {Xs.shape[:-1]} must match"

    # Estimate SN probabilities if they were not given
    if pxs is None:
        pxs = np.mean(Xs, axis=-1)
    if pys is None:
        pys = np.mean(Ys, axis=-1)

    # SN_length is presumed to be last dimension
    SN_length = Xs.shape[-1]

    # first get the covariance, the numerator of SCC
    corrs = np.sum(Xs * Ys, axis=-1)/SN_length
    covariances = corrs - pxs*pys

    # SCC treats positive and negative covariances differently
    pos_covs = covariances > 0
    neg_covs = covariances < 0
    zero_covs = covariances == 0
    indet_covs = np.logical_or.reduce((pxs==1, pxs==0, pys==1, pys==0))

    # Compute the SCC by dividing the covariance by the proper SCC normalizing factor
    if not do_cov:
        if pxs.ndim > 1:
            covariances[pos_covs] /= np.minimum(pxs[pos_covs], pys[pos_covs]) - pxs[pos_covs]*pys[pos_covs]
            covariances[neg_covs] /= pxs[neg_covs]*pys[neg_covs] - np.clip(pxs[neg_covs] + pys[neg_covs] - 1, a_min=0, a_max=None)
            covariances[zero_covs] = 0
            covariances[indet_covs] = tie
        else:
            # raise ValueError("Please check this SCC code, it seems wrong")
            if pos_covs:
                covariances /= np.minimum(pxs, pys) - pxs*pys
            elif neg_covs:
                covariances /= pxs*pys - np.clip(pxs + pys - 1, a_min=0, a_max=None)
            elif indet_covs:
                covariances = tie
            else:
                covariances = 0

    if do_corr:
        return corrs
    else:
        return covariances

def check_SCC(Xs, pxs=None, diag_value=1, tie_value=1, verbose=False):
    assert len(Xs.shape) == 2, "Only implemented for a list of SNs. Please reshape or update this method <3"
    num_SNs = Xs.shape[0]
    SCCs = np.zeros((num_SNs, num_SNs))
    for idx in range(num_SNs):
        # SN is always +1 correlated with itself, but can overwrite if interested in checking other values
        SCCs[idx, idx] = diag_value
        for jdx in range(idx+1, num_SNs):
            px, py = (pxs[idx], pxs[jdx]) if pxs is not None else (None, None)
            SCC = get_SCC(Xs[idx], Xs[jdx], px, py, tie=tie_value)
            SCCs[idx, jdx] = SCC
            SCCs[jdx, idx] = SCC

    if verbose:
        print(f"Average SCCs: {np.mean(SCCs)}.")
    return SCCs

