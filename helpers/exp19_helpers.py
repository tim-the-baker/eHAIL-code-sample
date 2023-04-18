"""
Functions that facilitate running the simulations in exp19
"""

import enum
import torch
import numpy as np
import scipy.io as sp_io
from Code_examples.helpers import projects, exp13_helpers, general
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from SCython.SNG_torch import SNG, RNS, PCC, MSG
from SCython.Circuits import torch_stream_adders

FILTER_DIR = r"C:\Users\Tim\PycharmProjects\stochastic-computing\Data\filter_coefs_ecg"

@enum.unique
class C_TYPE(enum.IntEnum):
    PSA = 0
    TFF = 1
    AXPC = 2
    HAPC = 3
    # skip 4 because I think i used it for something in exp15.
    # also skip 5 because it was used in exp18 for HW_MUX with NUMPY rather than torch. Did not update that func here.
    HW_MUX = 6  # for use with torch
    CORSMC = 7  # COR-SMC differs from CeMux when implementing weighted addition


#####################
# General Functions #
#####################
def get_filter_weights(num_weights, device):
    assert 269 >= num_weights >= 5, f"Number of taps must be between 5 and 269. Given: {num_weights}"

    a = sp_io.loadmat(rf"{FILTER_DIR}\blo_order_4_268.mat")
    blo = a['blo']  # 2D array, row corresponds to filter order: 4, 6, ..., 266, 268.
    index = num_weights - 5
    weights = blo[index, 0:num_weights]
    assert (blo[index, num_weights:] == 0).all()  # Make sure we didn't mess up getting the coefficients
    weights = torch.tensor(weights, device=device)

    return weights


def get_MIT(record='118', db="mitdb", noise_level=0.1, show=True):
    filename = f"{FILTER_DIR}/rec{record}_{db}_noise{noise_level}_dac2021_preprocessed_signal.npy"
    noisy = np.load(filename, allow_pickle=True)
    return noisy


def get_pxs_and_weights(job):
    device = torch.device(job.doc.device)
    B, N, R, K = job.doc.num_batches, job.doc.batch_size, job.sp.sim_runs, job.sp.num_addends

    # get pxs:
    if job.sp.px_dist == 'rand':
        pxs = torch.rand(size=(R, K), device=device).reshape(B, N, K)
        pxs = 2*pxs-1 if job.sp.bipolar else pxs

    elif job.sp.px_dist == 'ecg':  # Noisy MIT ECG signal
        assert job.sp.bipolar, "ECG only works for bipolar! (negative weights)"

        signal = get_MIT()  # get the 65,000 element signal
        starts = torch.randint(0, len(signal)-K, (R,))  # get random starting points
        pxs = np.array([signal[starts[r_idx]:starts[r_idx]+K].copy() for r_idx in range(R)])
        pxs = torch.tensor(data=pxs, device=device).reshape(B, N, K)

    elif job.sp.px_dist == 'fmnist':  # FashionMNIST MLP
        assert job.sp.num_addends == 28*28, "FMNIST has exactly 784 input SNs"
        fmnist_data = datasets.FashionMNIST('data/datasets', train=True, download=True, transform=transforms.ToTensor())
        selected_data, _ = random_split(fmnist_data, [R, len(fmnist_data)-R])
        selected_data = DataLoader(selected_data, batch_size=N, shuffle=False)
        pxs = torch.empty((B, N, K), device=device)
        for b_idx, (images, labels) in enumerate(selected_data):
            pxs[b_idx] = images.view((N, K))
    else:
        raise NotImplementedError(f"px_dist: {job.sp.px_dist} not supported. Valid choices: ['rand', 'ecg', 'fmnist]")

    # get weights
    if job.sp.weight_dist == 'rand':
        weights = torch.rand(size=(job.sp.sim_runs, K), device=device).reshape(B, N, K)
        weights = 2*weights-1 if job.sp.bipolar else weights

    elif job.sp.weight_dist == 'ecg':
        weights = get_filter_weights(K, device)  # should be (K,) elements long
        weights = weights[None, None].expand(B, N, K)

    elif job.sp.weight_dist == 'fmnist':  # non-binarized weights
        job = general.check_and_load_job(projects.pr_exp13, sp={"net_type": 4, "epochs": 30}, allow_new=False)
        model = exp13_helpers.get_model(job)
        all_weights = model.fc1.weight.detach()
        idxs = torch.randint(0, len(all_weights), size=(R,))
        weights = all_weights[idxs][None].reshape(B, N, K)
        weights = weights.to(device)
        sf = max(weights.min().abs(), weights.max())
        # print(sf)
        # sf = 1/2**torch.ceil(torch.log2(sf))
        # print(sf)
        weights = weights / sf
    elif job.sp.weight_dist == 'fmnist-bnn':  # binarized weights
        # for job in projects.pr_exp13.find_jobs({"net_type": 1, }):
        #     print(job.sp)
        #     print(job.doc)
        # exit()
        job = general.check_and_load_job(projects.pr_exp13, sp={"net_type": 1, "epochs": 100}, allow_new=False)
        model = exp13_helpers.get_model(job)
        all_weights = model.fc1.weight.detach()
        idxs = torch.randint(0, len(all_weights), size=(R,))
        weights = all_weights[idxs][None].reshape(B, N, K)
        weights = weights.to(device)
    else:
        raise NotImplementedError(f"weight_dist: {job.sp.weight_dist} not supported. Valid choices: ['rand', 'ecg', 'fmnist]")

    return pxs, weights


def get_norm(job):
    bip_factor = 1+int(job.sp.bipolar)  # for unipolar this factor is 1, for bipolar it is 2
    if job.sp.type in [C_TYPE.PSA]:
        # shifting is used so output ranges from 0 to K or -K to K
        norm = job.sp.num_addends*bip_factor  # [0,K] for unipolar; [-K,K] for bipolar
    elif job.sp.type in [C_TYPE.AXPC, C_TYPE.HAPC]:
        # these designs use an APC with K/2 inputs
        norm = job.sp.num_addends / 2 * bip_factor  # [0, K/2] for unipolar. [-K/2, K/2] for bipolar
    elif job.sp.type in [C_TYPE.TFF, C_TYPE.HW_MUX, C_TYPE.CORSMC]:
        # these designs have bit-stream outputs
        norm = bip_factor  # [0, 1] for unipolar. [-1, 1] for bipolar
    return norm
#################
# PSA functions #
#################
def parallel_sampler_gen_SN(pxs, weights, data_sng, weight_sng, job):
    # if full_correlation, create an RNS_mask. When calling SNG, full_corr=False b/c mask handles the correlation
    RNS_mask = weights < 0 if job.sp.full_corr else None

    if job.sp.prune:
        N, K = pxs.shape
        # quantize weights:
        weights = SNG.q_nearest(weights, precision=8, signed=True)
        mask = weights != 0
        pxs, weights = pxs[mask].reshape(N, -1), weights[mask].reshape(N, -1)

    if job.sp.samp_share_rns or not job.sp.share_rns:  # if all SNs share a RNS or if no SNs share a RNS do this
        Xs = data_sng.gen_SN(pxs, job.sp.SN_length, job.sp.bipolar, job.sp.share_rns, RNS_mask, full_corr=False)
        Ws = weight_sng.gen_SN(weights, job.sp.SN_length, job.sp.bipolar, job.sp.share_rns, RNS_mask, full_corr=False)

    else:  # if SNs within a sampler share a RNS, but samplers don't share a RNS
        raise NotImplementedError
        # Xs = torch.tensor([data_sng.gen_SN(pxs[:, idx], job.sp.SN_length, job.sp.bipolar, job.sp.share_rns,
        #                                    RNS_mask, full_corr=False)] for idx in range(job.sp.samp_rate))

    Xs = Xs.to(torch.int)
    Ws = Ws.to(torch.int)
    Xs = 2*Xs-1 if job.sp.bipolar else Xs
    Ws = 2*Ws-1 if job.sp.bipolar else Ws

    return Xs, Ws


def parallel_sampler(SNs, weights, samp_height, msg_class, bipolar):
    """
    :param SNs: the input SNs. expected shape: (batch_size, num_addends, SN_length)
    :param weights: input weights. expected shape: (batch_size, num_addends, SN_length)
    :param msg: mux select input generator. should be set to None if the design is an APC
    :return: the output SN value(s)
    """
    assert SNs.ndim == 3, f"Input SN shape is expected to be (N, K, L): {SNs.shape}"
    N, K, L = SNs.shape
    assert weights.shape == (N, K, L), f"shape of weights: {weights.shape} be same as SN shape: {SNs.shape}"

    # initialize
    curr_h = samp_height
    R = K  # set remaining SNs equal to num_addends
    SNs_rem, w_rem = SNs, weights
    out = torch.zeros(size=(N,), device=SNs.device)

    # loop over decreasing small samplers
    while R != 0:
        if curr_h < 0:
            raise ValueError("Critical error. Something has gone wrong with the sampling.")
        elif curr_h == 0:  # when sampler height becomes zero, we have an APC. so don't sample
            SNs_rem = SNs_rem * w_rem  # (N, R, L)
            out += SNs_rem.sum(dim=(-1, -2))
            R = 0
        else:
            H = int(2**curr_h)
            B = R // H  # number of H-input samplers
            R = R % H  # number of leftover SNs that can't form another H-input sampler

            if B > 0:
                # separate out the full blocks
                SNs_main, SNs_rem = SNs_rem.split((B*H, R), dim=1)  # split over the K dimension
                w_main, w_rem = w_rem.split((B*H, R), dim=1)  # split over the K dimension

                # reshape SNs and weights
                SNs_main = SNs_main.view(N, B, H, L)
                w_main = w_main.view(N, B, H, L)

                # get the MSG and mux select inputs
                msg = msg_class(H, SNs.device)  # H is sampler size
                msg_share = msg.must_share()  # Only use MSG sharing across batch when necessary. Always share MSG for circuits
                selects = msg.gen_selects(SN_length=L, shape=(N,), share=msg_share)  # (N, L)
                selects = selects[:, None, None, :].expand(N, B, 1, L)  # (N, B, 1, L)

                # gather the SNs and weights
                SNs_main = torch.gather(SNs_main, -2, index=selects)  # (N, B, 1, L)
                w_main = torch.gather(w_main, -2, index=selects) # (N, B, 1, L)

                # multiply weights by SN bits
                SNs_main = torch.mul(SNs_main, w_main)  # (N, B, 1, L)
                SNs_main = SNs_main.sum(dim=(-1, -2, -3))  # accumulate SN bits; (N,)
                out += SNs_main.bitwise_left_shift(curr_h) # left shift to offset scale factors and accumulate

            # prepare next
            curr_h -= 1

    return out


def init_exp1_job_doc_PSA(job):
    job.doc.init = False

    # check to see if its a valid parallel sampler design
    job.doc.samp_size = 2**job.sp.samp_height
    assert job.doc.samp_size <= job.sp.num_addends, f"Sampler size can't exceed number of input SNs: {job.doc.samp_size} {job.sp.num_addends}"

    # check to see if sharing makes sense
    assert job.sp.bipolar or not job.sp.full_corr, "If unipolar SNs are used, then full_corr must be false"
    assert job.sp.share_rns or not job.sp.full_corr, "full_corr can only be used if SNs share an RNS"
    assert job.sp.share_rns or not job.sp.samp_share_rns, "Samplers can only share a common RNS if SNs within sampler share an RNS"

    job.doc.is_apc = (job.sp.samp_height == 0)
    assert not job.doc.is_apc or job.sp.msg_name is None

    job.doc.scale_factor = 1  # parallel samplers values are accumulated such that SF = 1 (i.e., with shifting).

    job.doc.batch_size = 32
    job.doc.num_batches = job.sp.sim_runs // job.doc.batch_size
    assert job.doc.batch_size * job.doc.num_batches == job.sp.sim_runs

    job.doc.device = 'cuda:0'
    job.doc.init = True

    return job.doc.init


def run_exp1_job_PSA(job, device, pxs, weights):
    assert job.doc.init
    L, K = job.sp.SN_length, job.sp.num_addends
    N, B = job.doc.batch_size, job.doc.num_batches

    # Set up SNGs
    prec = job.sp.rns_prec
    d_kwargs = {"seq_idxs": [job.sp.get("d_seq_idx")], "seed": job.sp.get("d_seed"), "reuse": job.sp.get("reuse"),
                "shuffle": job.sp.get("shuffle"), "flip": job.sp.get("flip")}
    w_kwargs = {"seq_idxs": [job.sp.get("w_seq_idx")], "seed": job.sp.get("w_seed"), "reuse": job.sp.get("reuse"),
                "shuffle": job.sp.get("shuffle"), "flip": job.sp.get("flip")}

    if job.sp.pcc_name == PCC.Mux_maj_chain.__name__:
        p_kwargs = {"num_mux": job.sp.rns_prec-job.sp.num_maj, "num_maj": job.sp.num_maj, "invert_maj_R": job.sp.invert_maj_R}
    else:
        p_kwargs = {}
    d_rns = RNS.get_rns_class_from_name(job.sp.d_rns_name)(prec, device, **d_kwargs)
    w_rns = RNS.get_rns_class_from_name(job.sp.w_rns_name)(prec, device, **w_kwargs)
    pcc = PCC.get_pcc_class_from_name(job.sp.pcc_name)(prec, **p_kwargs)
    data_sng = SNG.SNG(d_rns, pcc)
    weight_sng = SNG.SNG(w_rns, pcc)

    # set up data
    Z_stars = (pxs*weights).sum(dim=-1)*job.doc.scale_factor
    Z_stars = Z_stars.cpu()
    Z_hats = torch.zeros_like(Z_stars, dtype=torch.float, device=device)

    # set up MSG
    if job.doc.is_apc:
        msg_class = job.sp.msg_name  # should be None
    else:
        msg_class = MSG.get_msg_class_from_name(job.sp.msg_name)  # we can't store class in job.sp so we load like this

    weights = weights.view(B, N, K)  # (batch id, batch_size, num_addends)
    for b_idx in range(B):  # loop over batches
        Xs, Ws = parallel_sampler_gen_SN(pxs[b_idx], weights[b_idx], data_sng, weight_sng, job)  # (N, K, L)
        Z_hats[b_idx] = parallel_sampler(Xs, Ws, job.sp.samp_height, msg_class, job.sp.bipolar).float()
        Z_hats[b_idx] = Z_hats[b_idx] / L

    # combines batch and sim runs dimension into a single dimension
    Z_hats = Z_hats.view(job.sp.sim_runs)
    Z_stars = Z_stars.view(job.sp.sim_runs)
    scale_factors = torch.full_like(Z_stars, fill_value=job.doc.scale_factor)

    return Z_hats, Z_stars, scale_factors


#################
# TFF functions #
#################
def nonPSA_gen_SN(pxs, weights, data_sng, weight_sng, job):
    # if full_correlation, create an RNS_mask. When calling SNG, full_corr=False b/c mask handles the correlation
    RNS_mask = weights < 0 if job.sp.full_corr else None

    Xs = data_sng.gen_SN(pxs, job.sp.SN_length, job.sp.bipolar, job.sp.share_rns, RNS_mask, full_corr=False).to(torch.int)
    Ws = weight_sng.gen_SN(weights, job.sp.SN_length, job.sp.bipolar, job.sp.share_rns, RNS_mask, full_corr=False).to(torch.int)

    Xs = 2*Xs-1 if job.sp.bipolar else Xs
    Ws = 2*Ws-1 if job.sp.bipolar else Ws
    return Xs, Ws


def TFF_fast(Xs, Ys, bipolar, starts):
    """
    2-input TFF adder
    :param Xs: input 1. expected shape: (..., L) where L is SN_length
    :param Ys: input 2. expected shape: (..., L) where L is SN_length. Should be same shape as Xs.
    :param bipolar: whether SNs are currently unipolar (0, 1) or bipolar (-1, 1)
    :param starts: expected shape: (K) where K is Xs.shape[-2]=Ys.shape[-2]
    :return: TFF output
    """
    assert Xs.shape == Ys.shape, f"Xs and Ys shapes should match. X.shape:{Xs.shape}; Y.shape:{Ys.shape}"
    assert (Xs.shape[1:2] == starts.shape) or (Xs.dim() == 2 and starts.numel() == 1), f"{Xs.shape} {starts.shape}"

    if bipolar:  # convert to unipolar if bipolar to make it easier
        Xs, Ys = (Xs+1)//2, (Ys+1)//2

    xor = Xs ^ Ys
    cumsum = torch.cumsum(xor, dim=-1)   # cumulative sum over last dimension

    # when xor = 1, the circuit's memory, Ts, should toggle during the next clock cycle
    # therefore, when cumsum[i] is even, d[i+1]=0 and when cumsum[i] is odd, d[i+1]=1
    Ts = torch.zeros_like(Xs, dtype=torch.int)
    Ts[..., 1:] = cumsum[..., 0:-1] & 0x1   # d[i+1] = cumsum[i] % 2
    Ts[..., starts, :] = 1-Ts[..., starts, :]  # account for start state. 0 becomes 1 and 1 becomes 0 if start=True

    # the output is simply a majority between X, Y and T
    Zs = ((Xs + Ys + Ts) > 1).to(torch.int)

    # convert back
    if bipolar:
        Zs = 2*Zs-1

    return Zs


def TFF_tree_fast(d_SNs, w_SNs, height, bipolar, start_mode):
    """
    :param SNs: the input SNs. expected shape: (batch_size, num_addends, SN_length)
    :param weights: input weights. expected shape: (batch_size, num_addends, SN_length)
    :return: the output SN value(s)
    """
    assert d_SNs.ndim == 3, f"Input SN shape is expected to be (N, K, L): {d_SNs.shape}."
    N, K, L = d_SNs.shape
    assert w_SNs.shape == (N, K, L), f"shape of weights: {w_SNs.shape} should be same as SN shape: {d_SNs.shape}."

    SNs = d_SNs*w_SNs

    # account for start mode
    if start_mode == 'zero':
        starts = torch.zeros((K//2), dtype=torch.bool)
    elif start_mode == 'intra_alt':
        starts = torch.empty((K//2), dtype=torch.bool)
        starts[::2] = False
        starts[1::2] = True
    else:
        raise NotImplementedError

    # helper for non-power of two inputs
    helper = torch.zeros((N, L), dtype=torch.int)
    if bipolar:
        helper[:, ::2] = 1

    layer_outs = torch.zeros((N, K//2+(K%2), L), dtype=torch.int)

    curr_size = K//2
    rem  = K%2
    rem_count = 0
    for h_idx in range(height):
        if h_idx == 0:  # first tree layer
            # first handle the TFFs with two inputs
            layer_outs[:, 0:curr_size] = TFF_fast(SNs[:, 0:2*curr_size:2], SNs[:, 1:2*curr_size:2], bipolar, starts[:curr_size])

            # then handle any remaining TFFs
            if rem == 1:
                layer_outs[:, curr_size] = TFF_fast(SNs[:, 2*curr_size], helper, bipolar, starts[rem_count])
                rem_count += 1

        else:  # all other tree layers
            # first handle the TFFs with two inputs
            layer_outs[:, 0:curr_size] = TFF_fast(layer_outs[:, 0:2*curr_size:2], layer_outs[:, 1:2*curr_size:2], bipolar, starts[:curr_size])

            # then handle any remaining TFFs
            if rem == 1:
                layer_outs[:, curr_size] = TFF_fast(layer_outs[:, 2*curr_size], helper, bipolar, starts[rem_count])
                rem_count += 1
        temp = curr_size
        curr_size = (temp+rem) // 2
        rem = (temp+rem) % 2
    return layer_outs[:, 0].sum(-1)  # (N)


def init_exp1_job_doc_TFF(job):
    job.doc.init = False

    job.doc.height = int(np.ceil(np.log2(job.sp.num_addends)))
    job.doc.scale_factor = 1/int(2**job.doc.height)

    job.doc.batch_size = 32
    job.doc.num_batches = job.sp.sim_runs // job.doc.batch_size

    job.doc.device = 'cuda:0'
    job.doc.init = True

    # checks
    assert job.sp.bipolar or not job.sp.full_corr, "If unipolar SNs are used, then full_corr must be false"
    assert job.sp.share_rns or not job.sp.full_corr, "full_corr can only be used if SNs share an RNS"
    assert job.doc.height % 1 == 0
    assert job.doc.batch_size * job.doc.num_batches == job.sp.sim_runs

    return job.doc.init


def run_exp1_job_TFF(job, device, pxs, weights):
    assert job.doc.init
    L, K = job.sp.SN_length, job.sp.num_addends
    N, B = job.doc.batch_size, job.doc.num_batches

    # Set up SNG
    prec = job.sp.rns_prec
    d_rns = RNS.get_rns_class_from_name(job.sp.d_rns_name)(prec, device, seq_idxs=[job.sp.d_seq_idx])
    w_rns = RNS.get_rns_class_from_name(job.sp.w_rns_name)(prec, device, seq_idxs=[job.sp.w_seq_idx])
    pcc = PCC.get_pcc_class_from_name(job.sp.pcc_name)(prec)
    data_sng = SNG.SNG(d_rns, pcc)
    weight_sng = SNG.SNG(w_rns, pcc)

    # set up data
    Z_stars = (pxs*weights).sum(dim=-1)*job.doc.scale_factor
    Z_stars = Z_stars.cpu()
    Z_hats = torch.zeros_like(Z_stars, dtype=torch.float, device=device)

    weights = weights.view(B, N, job.sp.num_addends)
    for b_idx in range(B):  # loop over batches
        Xs, Ws = nonPSA_gen_SN(pxs[b_idx], weights[b_idx], data_sng, weight_sng, job)
        # note the divide by L after function call
        Z_hats[b_idx] = TFF_tree_fast(Xs, Ws, job.doc.height, job.sp.bipolar, job.sp.start)/L

    # combines batch and sim runs dimension into a single dimension
    Z_hats = Z_hats.view(job.sp.sim_runs)
    Z_stars = Z_stars.view(job.sp.sim_runs)
    scale_factors = torch.full_like(Z_stars, fill_value=job.doc.scale_factor)

    return Z_hats, Z_stars, scale_factors


##############################################
#  Torch Hardwired Mux (SUC, CeMux, Conv Mux #
##############################################
def init_exp1_job_doc_HW_mux(job):
    job.doc.init = False

    # check to see if its a valid parallel sampler design
    job.doc.tree_height = job.sp.rns_prec
    job.doc.tree_size = 2**job.doc.tree_height

    # check to see if sharing makes sense
    assert job.sp.bipolar or not job.sp.full_corr, "If unipolar SNs are used, then full_corr must be false"
    assert job.sp.share_rns or not job.sp.full_corr, "full_corr can only be used if SNs share an RNS"

    job.doc.batch_size = 1
    job.doc.num_batches = job.sp.sim_runs // job.doc.batch_size
    assert job.doc.batch_size * job.doc.num_batches == job.sp.sim_runs

    job.doc.device = 'cpu'
    job.doc.init = True

    return job.doc.init


def run_exp1_job_HW_mux(job, device, pxs, weights):
    assert job.doc.init

    N, B = job.doc.batch_size, job.doc.num_batches

    # Set up SNG
    prec = job.sp.rns_prec
    d_rns = RNS.get_rns_class_from_name(job.sp.d_rns_name)(prec, device)
    pcc = PCC.get_pcc_class_from_name(job.sp.pcc_name)(prec)
    data_sng = SNG.SNG(d_rns, pcc)
    if job.sp.msg_name == MSG.Sep_FSR_MSG.__name__:
        msg = MSG.get_msg_class_from_name(job.sp.msg_name)(int(2**prec), device, precision=prec)
    else:
        msg = MSG.get_msg_class_from_name(job.sp.msg_name)(int(2**prec), device)

    # set up data
    Z_stars = (pxs*weights).sum(dim=-1)
    Z_stars = Z_stars.cpu()
    Z_hats = torch.zeros_like(Z_stars, dtype=torch.float, device=device)
    scale_factors = torch.zeros_like(Z_stars, dtype=torch.float, device=device)

    weights = weights.view(B, N, job.sp.num_addends)
    if (job.sp.weight_dist == 'ecg') or (job.sp.weight_dist == 'fmnist-bnn'):
        hw_mux = torch_stream_adders.HardwiredMux(job.doc.tree_height, weights[0], d_rns, pcc, job.sp.share_rns,
                                                  msg, job.sp.full_corr, job.sp.bipolar, device)
    for b_idx in range(B):  # loop over batches
        if job.sp.weight_dist == 'fmnist-bnn':
            hw_mux = torch_stream_adders.HardwiredMux(job.doc.tree_height, weights[b_idx], d_rns, pcc, job.sp.share_rns,
                                                      msg, job.sp.full_corr, job.sp.bipolar, device, wire_map=hw_mux.wire_map)
        elif job.sp.weight_dist != 'ecg':
            if b_idx % 100 == 0:
                print(b_idx)
            hw_mux = torch_stream_adders.HardwiredMux(job.doc.tree_height, weights[b_idx], d_rns, pcc, job.sp.share_rns,
                                                      msg, job.sp.full_corr, job.sp.bipolar, device)

        # run HW mux. note full_corr is handled by RNS mask
        RNS_mask = hw_mux.inv_mask if job.sp.full_corr else None
        Xs = data_sng.gen_SN(pxs[b_idx], job.sp.SN_length, job.sp.bipolar, job.sp.share_rns, RNS_mask, full_corr=False).to(torch.bool)
        Zs = hw_mux.forward_SN(Xs)  # (N, L)
        Zs = 2*Zs-1 if job.sp.bipolar else Zs

        # update datas
        Z_hats[b_idx] = Zs.float().mean(dim=-1)
        scale_factors[b_idx] = 1/weights[b_idx].abs().sum(dim=-1)
        Z_stars[b_idx] = Z_stars[b_idx]*scale_factors[b_idx]

    # combines batch and sim runs dimension into a single dimension
    Z_hats = Z_hats.view(job.sp.sim_runs)
    Z_stars = Z_stars.view(job.sp.sim_runs)
    scale_factors = scale_factors.view(job.sp.sim_runs)

    return Z_hats, Z_stars, scale_factors


########################
# Approx APC functions #
########################
def AxPC_HAPC(d_SNs, w_SNs, use_and):
    """
    :param SNs: the input SNs. expected shape: (batch_size, num_addends, SN_length)
    :param weights: input weights. Assumed to be +/- 1. expected shape: (batch_size, num_addends, SN_length)
    :return: the output SN value(s)
    """
    assert d_SNs.ndim == 3, f"Input SN shape is expected to be (N, K, L): {d_SNs.shape}."
    N, K, L = d_SNs.shape
    assert w_SNs.shape == (N, K, L), f"shape of weight SNs: {w_SNs.shape} should match SN shape: {d_SNs.shape}."

    # handles weights
    SNs = d_SNs*w_SNs

    if use_and:
        approx_layer1 = SNs[:, 0:K:4] * SNs[:, 1:K:4]  # do AND gates
        approx_layer2 = SNs[:, 2:K:4] + SNs[:, 3:K:4] - SNs[:, 2:K:4]*SNs[:, 3:K:4]  # do OR gates
        outs = approx_layer1.sum(dim=(-1, -2)) + approx_layer2.sum(dim=(-1, -2))
    else:
        approx_layer = SNs[:, 0:K:2] + SNs[:, 1:K:2] - SNs[:, 0:K:2]*SNs[:, 1:K:2]  # do OR gates
        outs = approx_layer.sum(dim=(-1, -2))

    return outs


def init_exp1_job_doc_AxPC_HAPC(job):
    job.doc.init = False

    # check to see if sharing makes sense
    assert job.sp.bipolar or not job.sp.full_corr, "If unipolar SNs are used, then full_corr must be false"
    assert job.sp.share_rns or not job.sp.full_corr, "full_corr can only be used if SNs share an RNS"

    job.doc.use_and = job.sp.type == C_TYPE.AXPC
    assert job.doc.use_and or job.sp.type == C_TYPE.HAPC, "C_TYPE must either be AXPC or HAPC"
    assert not job.sp.bipolar, "AxPc/Hybrid APC are not work for bipolar"

    job.doc.scale_factor = 0.5 if job.sp.type == C_TYPE.AXPC else 1  # AxPC technically scales down by 1/2

    job.doc.batch_size = 32
    job.doc.num_batches = job.sp.sim_runs // job.doc.batch_size
    assert job.doc.batch_size * job.doc.num_batches == job.sp.sim_runs

    job.doc.device = 'cuda:0'
    job.doc.init = True

    return job.doc.init


def run_exp1_job_AxPC_HAPC(job, device, pxs, weights):
    assert job.doc.init
    L, K = job.sp.SN_length, job.sp.num_addends
    N, B = job.doc.batch_size, job.doc.num_batches

    # Set up SNG
    prec = job.sp.rns_prec
    d_rns = RNS.get_rns_class_from_name(job.sp.d_rns_name)(prec, device)
    w_rns = RNS.get_rns_class_from_name(job.sp.w_rns_name)(prec, device)
    pcc = PCC.get_pcc_class_from_name(job.sp.pcc_name)(prec)
    data_sng = SNG.SNG(d_rns, pcc)
    weight_sng = SNG.SNG(w_rns, pcc)

    # set up data
    Z_stars = (pxs*weights).sum(dim=-1)*job.doc.scale_factor
    Z_stars = Z_stars.cpu()
    Z_hats = torch.zeros_like(Z_stars, dtype=torch.float, device=device)

    weights = weights.view(B, N, job.sp.num_addends)

    for b_idx in range(B):  # loop over batches
        Xs, Ws = nonPSA_gen_SN(pxs[b_idx], weights[b_idx], data_sng, weight_sng, job)
        # note the divide by L after function call
        Z_hats[b_idx] = AxPC_HAPC(Xs, Ws, job.doc.use_and)/L

    # combines batch and sim runs dimension into a single dimension
    Z_hats = Z_hats.view(job.sp.sim_runs)
    Z_stars = Z_stars.view(job.sp.sim_runs)
    scale_factors = torch.full_like(Z_stars, fill_value=job.doc.scale_factor)

    return Z_hats, Z_stars, scale_factors


######################
#  COR-SMC functions #
######################
def init_exp1_job_doc_corsmc(job):
    job.doc.init = False

    # check to see if its a valid parallel sampler design
    job.doc.tree_height = job.sp.rns_prec
    job.doc.tree_size = 2**job.doc.tree_height

    # check to see if sharing makes sense
    assert not job.sp.bipolar, "COR-SMC only meant for bipolar"
    assert not job.sp.full_corr, "If unipolar SNs are used, then full_corr must be false"
    assert job.sp.share_rns or not job.sp.full_corr, "full_corr can only be used if SNs share an RNS"

    job.doc.scale_factor = 1/job.sp.num_addends

    job.doc.batch_size = 32
    job.doc.num_batches = job.sp.sim_runs // job.doc.batch_size
    assert job.doc.batch_size * job.doc.num_batches == job.sp.sim_runs

    job.doc.device = 'cpu'
    job.doc.init = True

    return job.doc.init


def run_exp1_job_corsmc(job, device, pxs, weights):
    assert job.doc.init

    N, B = job.doc.batch_size, job.doc.num_batches
    K = job.sp.num_addends

    # Set up SNG
    prec = job.sp.rns_prec
    d_rns = RNS.get_rns_class_from_name(job.sp.d_rns_name)(prec, device, seq_idxs=[job.sp.d_seq_idx])
    w_rns = RNS.get_rns_class_from_name(job.sp.w_rns_name)(prec, device, seq_idxs=[job.sp.w_seq_idx])
    pcc = PCC.get_pcc_class_from_name(job.sp.pcc_name)(prec)
    data_sng = SNG.SNG(d_rns, pcc)
    weight_sng = SNG.SNG(w_rns, pcc)

    if job.sp.msg_name == MSG.Sep_FSR_MSG.__name__:
        msg = MSG.get_msg_class_from_name(job.sp.msg_name)(int(2**prec), device, precision=prec)
    else:
        msg = MSG.get_msg_class_from_name(job.sp.msg_name)(int(2**prec), device)

    # set up data
    Z_stars = (pxs*weights).sum(dim=-1)*job.doc.scale_factor
    Z_stars = Z_stars.cpu()
    Z_hats = torch.zeros_like(Z_stars, dtype=torch.float, device=device)

    # set up mux (always all 1 weights bc we multiply with AND array)
    mux_weights = torch.ones((N, K), device=job.doc.device)
    hw_mux = torch_stream_adders.HardwiredMux(job.doc.tree_height, mux_weights, d_rns, pcc, job.sp.share_rns,
                                              msg, job.sp.full_corr, job.sp.bipolar, device)

    RNS_mask = None
    weights = weights.view(B, N, job.sp.num_addends)
    for b_idx in range(B):  # loop over batches
        Xs = data_sng.gen_SN(pxs[b_idx], job.sp.SN_length, job.sp.bipolar, job.sp.share_rns, RNS_mask, full_corr=False)
        Ws = weight_sng.gen_SN(weights[b_idx], job.sp.SN_length, job.sp.bipolar, job.sp.share_rns, RNS_mask, full_corr=False)
        Xs = Ws*Xs

        Zs = hw_mux.forward_SN(Xs)  # (N, L)
        Z_hats[b_idx] = Zs.float().mean(dim=-1)

    # combines batch and sim runs dimension into a single dimension
    Z_hats = Z_hats.view(job.sp.sim_runs)
    Z_stars = Z_stars.view(job.sp.sim_runs)
    scale_factors = torch.full_like(Z_stars, fill_value=job.doc.scale_factor)

    return Z_hats, Z_stars, scale_factors


if __name__ == '__main__':
    # code for previewing FMNIST images and ECG signals
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['figure.dpi'] = 150

    date2023 = False
    thesis = True
    save = True

    do_ecg = True
    do_fmnist = False

    if date2023:
        figsize = (3, 3.4)
        fs1 = 36
        bot_pad = 0.150
        if do_ecg:
            num_taps = 100
            delay = (num_taps-1) // 2
            points = 500
            pow2n = 128

            signal = get_MIT()
            coefs = get_filter_weights(num_taps, device=torch.device('cpu'))
            coefs = coefs.numpy()
            q_coefs = np.round(coefs*pow2n)/pow2n

            filtered = np.convolve(signal, coefs, mode='full')

            q_filtered = np.convolve(signal, q_coefs, mode='full')

            # plt.plot(coefs*pow2n)
            # plt.plot(q_coefs*pow2n)
            # plt.show(block=True)
            print(signal.shape)
            # figsize = (3, 3.2)
            # bot_pad = 0.1
            fig, ax = plt.subplots(figsize=figsize, dpi=300)
            ax.plot(signal[0:points], color='k', lw=1)
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            ax.set_xlabel("(b)", fontsize=fs1)
            fig.subplots_adjust(bottom=bot_pad, top=0.999, left=0.001, right=0.999)

            fig2, ax2 = plt.subplots(figsize=figsize, dpi=300)
            ax2.plot(q_filtered[delay:delay+points], color='k', lw=1)
            ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            ax2.set_xlabel("(c)", fontsize=fs1)
            fig2.subplots_adjust(bottom=bot_pad, top=0.999, left=0.001, right=0.999)
            # plt.show(block=True)

        if do_fmnist:
            torch.manual_seed(100003400)
            fmnist_data = datasets.FashionMNIST('data/datasets', train=False, download=True, transform=transforms.ToTensor())
            selected_data = DataLoader(fmnist_data, batch_size=1, shuffle=True)
            nrow, ncol = 2, 2
            fig3, ax3s = plt.subplots(figsize=figsize, nrows=nrow, ncols=ncol, dpi=300)
            prev_labels = []
            for b_idx, (image, label) in enumerate(selected_data):
                if len(prev_labels) == nrow*ncol:
                    break
                r_idx = len(prev_labels) // ncol
                c_idx = len(prev_labels) % ncol
                ax = ax3s[r_idx, c_idx]
                print(label)
                if label in prev_labels:
                    print('skip')
                    continue
                else:
                    prev_labels.append(label)
                ax.imshow(image.squeeze(), cmap='Greys_r', vmin=0, vmax=1)
                ax.axis('off')
            plt.tight_layout()
            fig3.text(0.5, 0.02, '(a)', ha='center', fontsize=fs1)
            plt.subplots_adjust(hspace=0.015, wspace=0.0, bottom=bot_pad, top=1, left=0, right=1)
        plt.show(block=True)

    if thesis:
        if do_ecg:
            # not modified yet compared to DATE
            figsize = (3, 3.4)
            fs1 = 36
            bot_pad = 0.150
            num_taps = 100
            delay = (num_taps - 1) // 2
            points = 500
            pow2n = 128

            signal = get_MIT()
            coefs = get_filter_weights(num_taps, device=torch.device('cpu'))
            coefs = coefs.numpy()
            q_coefs = np.round(coefs * pow2n) / pow2n

            filtered = np.convolve(signal, coefs, mode='valid')

            q_filtered = np.convolve(signal, q_coefs, mode='valid')

            mod_points = 100
            start = 125

            fs1 = 20
            fs2 = 16
            figsize = (4, 3.4)
            bot_pad = 0.25

            fig1, ax1 = plt.subplots(figsize=figsize)
            ax1.plot(signal[0:points], color='k', lw=1)
            ax1.axvline(start, color='r')
            ax1.axvline(start+mod_points, color='r')

            fig2, ax2 = plt.subplots(figsize=figsize)
            ax2.plot(np.arange(delay, delay+points), q_filtered[delay:delay + points], color='k', lw=1)

            fig3, ax3 = plt.subplots(figsize=figsize)
            ax3.plot(signal[start:start+mod_points], '-o', color='k', lw=1, ms=3)
            ax3.set_ylim(ax1.get_ylim())

            caps = ['a', 'c', 'b']
            axs = [ax1, ax2, ax3]
            figs = [fig1, fig2, fig3]
            ylabels = ['X_{t}', 'Y_{t}', "X_{t}"]
            for ax, y_label in zip(axs, ylabels):
                # ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                #                labelbottom=False, labelleft=False)
                ax.set_xlabel(f'Sample, $t$', fontsize=fs2)
                ax.set_ylabel(f"Amplitude, ${y_label}$", fontsize=fs2)
                ax.set_ylim(-1, 1)
                ax.grid()

            for cap, fig in zip(caps, figs):
                # fig.subplots_adjust(bottom=bot_pad, top=0.999, left=0.001, right=0.999)
                fig.text(x=0.5, y=0.015, s=f"({cap})", fontsize=fs1, transform=fig.transFigure)
                fig.tight_layout()
                fig.subplots_adjust(bottom=bot_pad)

            if save:
                dir = r"C:\Users\Tim\OneDrive - Umich\Documents\University of Michigan\Research\Thesis\Thesis\Figures\Ch5"
                for cap, fig in zip(caps, figs):
                    filename = f"28{cap}_ecg-input-ivd"
                    fig.savefig(rf"{dir}\{filename}.png", dpi=300)

        if do_fmnist:
            figsize = (3, 1.2)
            torch.manual_seed(100003400)
            fmnist_data = datasets.FashionMNIST('data/datasets', train=False, download=True,
                                                transform=transforms.ToTensor())
            selected_data = DataLoader(fmnist_data, batch_size=1, shuffle=True)
            nrow, ncol = 2, 5
            fig3, ax3s = plt.subplots(figsize=figsize, nrows=nrow, ncols=ncol, dpi=300)
            prev_labels = []
            for b_idx, (image, label) in enumerate(selected_data):
                if len(prev_labels) == nrow * ncol:
                    break
                r_idx = len(prev_labels) // ncol
                c_idx = len(prev_labels) % ncol
                ax = ax3s[r_idx, c_idx]
                print(label)
                if label in prev_labels:
                    print('skip')
                    continue
                else:
                    prev_labels.append(label)
                ax.imshow(image.squeeze(), cmap='Greys_r', vmin=0, vmax=1)
                ax.axis('off')
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0, top=1, left=0, right=1)

            for label in prev_labels:
                print(f"{datasets.FashionMNIST.classes[label]}, ", end='')
            print("")
            print(datasets.FashionMNIST.classes)
            print(prev_labels)

            if save:
                dir = r"C:\Users\Tim\OneDrive - Umich\Documents\University of Michigan\Research\Thesis\Thesis\Figures\Ch5"
                filename = "25_FMNIST-examples"
                fig3.savefig(rf"{dir}\{filename}.png", dpi=300)
        plt.show(block=True)