"""
Simulate adders with non-power-of-two operands and arbitrary weights.

----------------------
  Simulation params
----------------------
These are the simulation-level params used all circuits in the simulation:

   rns_prec   seed   SN_length(L)=2^rns_prec   num_addends(K)   sim_runs   bipolar   px_dist

- seed is the random seed for determining the experiment's weights and input values
- use RNS.get_rns_class_from_name(job.sp.rns_name)(args) to init a RNS object. Process is similar for PCC/MSG
- SN_length(L) means that SN_length is renamed as 'L' for brevity purposes (i.e., L = job.sp.SN_length)

---------------------
   Circuit params
---------------------
These are circuit-level params that all circuits use
   type: [C_TYPE.MUX, C_TYPE.PSA, C_TYPE.TFF, C_TYPE.AXPC]
   rns   pcc   share_rns   full_corr

These are 'mux' circuit-specific params. The 'mux' circuit type covers conventional mux adders:
    msg

These are 'psa' circuit specific params. The PSA circuit type covers the CeMux, PSAs, APCs and SUC designs:
    msg   samp_rate(S)   samp_share_rns

There are no circuit-specific params for the "tff", or "axpc" circuit types
   - The "tff" circuit type covers the TFF and CEASE adders (which are functionally equivalent)
   - The "axpc" circuit type covers the approximate parallel counter design with 1 layer of approximation

share_rns is True if an RNS is shared within a sampler or across the whole circuit for non-PSA designs
samp_share_rns is True if all samplers share a common RNS (share_rns must also be True)

-------------------
   Job Statepoint
-------------------
The job statepoint is the concatenation of the simulation params with the circuit params.

-------------------
   Job Document
-------------------
saved  scale_factor(sf)  init

- 'saved' means that the job finished and we successfully saved the data
- 'scale_factor' comes from using muxes and is determined based on sp.num_addends and sp.samp_rate. sf = 1/(K/S) = S/K.
- 'init' means that we successfully initialized the job doc and made sure statepoint was valid. check job.doc.init before
every experiment!

--------------
   Job Data
--------------
Z_hats   Z_stars

- Z_hats is the circuit's output
- Z_stars is the circuit's target output.
- Z_stars should be determined based on job.sp.num_addends and job.sp.sim_runs (so we can sync input values across jobs)
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Code_examples.helpers import projects, general, exp19_helpers
from SCython.SNG_torch import RNS, PCC, MSG

pr_19 = projects.pr_exp19
ERROR_DIR = "./" # directory for saving error data
C_TYPE = exp19_helpers.C_TYPE  # import C_Type from exp19_helpers for brevity

#######################
# Experiment 1 Set up #
#######################
def get_circuit_templates(use_cmp):
    """
    Used to load pre-defined circuit templates.
    :param use_cmp: True if CMP PCCs are used and False otherwise
    :return: the pre-defined circuit templates.
    """
    templates = {}
    pcc_name = PCC.Comparator.__name__ if use_cmp else PCC.WBG.__name__

    templates["psa_base"] = {
        "type": C_TYPE.PSA.value,
        "pcc_name": pcc_name,
        "d_rns_name": RNS.SOBOL_RNS.__name__,
        "w_rns_name": RNS.SOBOL_RNS.__name__,
        "share_rns": True,
        "full_corr": False,
        "msg_name": MSG.Long_Counter_Exact_MSG.__name__
    }

    templates["psa_base_mod"] = {
        "type": C_TYPE.PSA.value,
        "pcc_name": pcc_name,
        "d_rns_name": RNS.VDC_RNS.__name__,
        "w_rns_name": RNS.Counter_RNS.__name__,
        "share_rns": True,
        "full_corr": True,
        "msg_name": MSG.HYPER_EXACT_MSG.__name__
    }

    templates["tff_intra_alt"] = {
        "type": C_TYPE.TFF.value,
        "d_rns_name": RNS.SOBOL_RNS.__name__,
        "w_rns_name": RNS.SOBOL_RNS.__name__,
        "pcc_name": pcc_name,
        "share_rns": True,
        "full_corr": False,
        "start": 'intra_alt'
    }

    templates["tff_intra_alt_flip"] = {
        "type": C_TYPE.TFF.value,
        "d_rns_name": RNS.SnF_SOBOL_RNS.__name__,
        "w_rns_name": RNS.SnF_SOBOL_RNS.__name__,
        "pcc_name": pcc_name,
        "share_rns": False,
        "full_corr": False,
        "start": 'intra_alt',
        "d_seed": 5389,
        "w_seed": 1231,
        "d_seq_idx": 0,
        "w_seq_idx": 2,
        "reuse": True,
        "shuffle": False,
        "flip": True
    }

    templates["cemux"] = {
        "type": C_TYPE.HW_MUX,
        "d_rns_name": RNS.VDC_RNS.__name__,
        "pcc_name": pcc_name,
        "msg_name": MSG.Long_Counter_Exact_MSG.__name__,
        "share_rns": True,
        "full_corr": True,
    }

    templates["mux_nlfsr"] = {
        "type": C_TYPE.HW_MUX,
        "d_rns_name": RNS.NLFSR_RNS.__name__,
        "pcc_name": pcc_name,
        "msg_name": MSG.Sep_FSR_MSG.__name__,
        "share_rns": True,
        "full_corr": False
    }

    templates["suc"] = {
        "type": C_TYPE.HW_MUX,
        "rns_name": RNS.NLFSR_RNS.__name__,
        "d_rns_name": RNS.NLFSR_RNS.__name__,
        "pcc_name": pcc_name,
        "msg_name": MSG.Long_Counter_Exact_MSG.__name__,
        "share_rns": True,
        "full_corr": False
    }

    templates["axpc"] = {
        "type": C_TYPE.AXPC.value,
        "d_rns_name": RNS.Hypergeometric_RNS.__name__,
        "w_rns_name": RNS.Hypergeometric_RNS.__name__,
        "pcc_name": pcc_name,
        "share_rns": True,
        "full_corr": False
    }

    templates["hapc"] = {
        "type": C_TYPE.HAPC.value,
        "d_rns_name": RNS.Hypergeometric_RNS.__name__,
        "w_rns_name": RNS.Hypergeometric_RNS.__name__,
        "pcc_name": pcc_name,
        "share_rns": True,
        "full_corr": False
    }

    templates['corsmc'] = {
        "type": C_TYPE.CORSMC.value,
        "d_rns_name": RNS.SOBOL_RNS.__name__,
        "w_rns_name": RNS.SOBOL_RNS.__name__,
        "pcc_name": pcc_name,
        "share_rns": True,
        "full_corr": False,
        "msg_name": MSG.Long_Counter_Exact_MSG.__name__
    }

    templates["psa_base_flip"] = {
        "type": C_TYPE.PSA.value,
        "pcc_name": pcc_name,
        "d_rns_name": RNS.SnF_SOBOL_RNS.__name__,
        "w_rns_name": RNS.SnF_SOBOL_RNS.__name__,
        "share_rns": False,
        "full_corr": False,
        "msg_name": MSG.Long_Counter_Exact_MSG.__name__,
        "d_seed": 5389,
        "w_seed": 1231,
        "d_seq_idx": 0,
        "w_seq_idx": 2,
        "reuse": True,
        "shuffle": False,
        "flip": True
    }


    return templates


def get_exp1_params(version):
    """

    :param version: version controls which circuit designs, circuit sizes and SN lengths are simulated:
        - version=1 generic unipolar test
        - version=2 generic bipolar test
        - version=3 fmnist test
        - if version is negative, then do PSA only
    :return: tuple containing the: circuit precision, number of operands, circuit statepoints, circuit string labels
    """
    seq1, seq2 = 0, 2  # For parallel samplers
    use_cmp, snf_psa, bipolar = None, None, None
    prune = False

    if version in [1, 2]:  # all adders, CMP, no flip PSA. v1: unipolar; v2: bipolar
        ns = torch.tensor([8, 10])
        Ks = torch.arange(50, 300, 50)
        # Ks = torch.tensor([512])
        use_cmp, snf_psa = True, False
        bipolar = version == 2
    elif version in [-1, -2]:  # PSA only, CMP, no flip PSA. v1: unipolar; v2: bipolar
        ns = torch.arange(4, 11)
        Ks = torch.tensor([512])
        use_cmp, snf_psa = True, False
        bipolar = version == -2
    elif version in [-3, 3]:  # fmnist, CMP, no flip PSA. v3: all adders; v-3: PSA only
        ns = torch.arange(4, 9)
        Ks = torch.tensor([784])
        use_cmp, snf_PSA = True, False
        bipolar = True
    elif version in [4, 5]:  # all adders, CMP, FLIP PSA. v4: unipolar; v5: bipolar
        ns = torch.tensor([8, 10])
        Ks = torch.arange(50, 300, 50)
        use_cmp, snf_psa = True, True
        bipolar = version == 5
    elif version in [6, 7]:  # all adders, CMP, FLIP PSA. FMNIST. 6=use_cmp, 7=not use_cmp
        ns = torch.arange(4, 9)
        Ks = torch.tensor([784])
        snf_psa = True
        use_cmp = version == 6
        bipolar = True

    templates = get_circuit_templates(use_cmp)
    tff_circ = templates["tff_intra_alt_flip"] if snf_psa else templates["tff_intra_alt"]
    # tff_circ = templates["tff_intra_alt"]

    if version < 0:  # PSA only
        circuits, circuit_labels = [], []
    elif bipolar:  # bipolar circuits
        circuits = [templates["mux_nlfsr"], templates["cemux"], tff_circ]
        circuit_labels = ['Conv. mux', 'CeMux', 'TFF tree']
    else:  # unipolar circuits
        circuits = [templates["hapc"], templates["axpc"], templates["corsmc"],
                    templates["mux_nlfsr"], templates["suc"], templates["cemux"], tff_circ]
        circuit_labels = ['Hybrid APC', 'AxPC', 'COR-SMC', 'Conv mux', 'SUC', 'CeMux', 'TFF tree']
    assert len(circuits) == len(circuit_labels)

    # duplicate non PSA circuits for each n and K
    circuits = [[circuits.copy() for _ in Ks] for _ in ns]
    circuit_labels = [[circuit_labels.copy() for _ in Ks] for _ in ns]

    # add PSAs and also update each circuit's bipolar param
    for n_idx, n in enumerate(ns):
        for K_idx, K in enumerate(Ks):
            # add parallel samplers to circuit list
            max_height = torch.floor(np.log2(K)).to(torch.int).item()
            for h in range(max_height+1):
                psa = templates["psa_base_flip"].copy() if snf_psa else templates[f"psa_base"].copy()
                psa["samp_height"] = h
                psa["samp_share_rns"] = not snf_psa
                psa["prune"] = prune
                if h == 0:  # APCs don't use MSGs. MSG name is otherwise set from template
                    psa["msg_name"] = None
                circuits[n_idx][K_idx].append(psa)
                if h == 0:
                    circuit_labels[n_idx][K_idx].append(f"APC")
                else:
                    # circuit_labels[n_idx][K_idx].append(f"PSA{int(2**h)}")
                    circuit_labels[n_idx][K_idx].append(f"$G={int(2**h)}$")

            # now update every circuit's bipolar and seq params
            for circuit in circuits[n_idx][K_idx]:
                circuit["bipolar"] = bipolar
                if circuit["d_rns_name"] == RNS.SOBOL_RNS.__name__:
                    circuit["d_seq_idx"] = seq1
                if circuit.get("w_rns_name", None) is not None:
                    if circuit["w_rns_name"] == RNS.SOBOL_RNS.__name__:
                        circuit["w_seq_idx"] = seq2
                if not bipolar:  # full_corr isn't a thing for bipolar
                    circuit["full_corr"] = False

    return ns, Ks, circuits, circuit_labels


############################
#  Experiment 1 Execution  #
############################
def run_exp1_job(job, save):
    """
    Executes a simulation job
    :param job: the simulation job to be executed
    :param save: True if you'd like to save the job and False otherwise
    :return: a tuple containing the simulation results: (circuit outputs, circuit target outputs, circuit scale factors)
    """

    # First, initialize the simulation job and grab the experiment file
    if job.sp.type == C_TYPE.PSA:
        exp19_helpers.init_exp1_job_doc_PSA(job)
        exp_func = exp19_helpers.run_exp1_job_PSA
    elif job.sp.type == C_TYPE.TFF:
        exp19_helpers.init_exp1_job_doc_TFF(job)
        exp_func = exp19_helpers.run_exp1_job_TFF
    elif job.sp.type in [C_TYPE.AXPC, C_TYPE.HAPC]:
        exp19_helpers.init_exp1_job_doc_AxPC_HAPC(job)
        exp_func = exp19_helpers.run_exp1_job_AxPC_HAPC
    elif job.sp.type == C_TYPE.HW_MUX:
        exp19_helpers.init_exp1_job_doc_HW_mux(job)
        exp_func = exp19_helpers.run_exp1_job_HW_mux
    elif job.sp.type == C_TYPE.CORSMC:
        exp19_helpers.init_exp1_job_doc_corsmc(job)
        exp_func = exp19_helpers.run_exp1_job_corsmc
    else:
        exp_func = None

    assert job.doc.init
    device = torch.device(job.doc.device)

    # random seed
    torch.random.manual_seed(job.sp.seed)
    torch.cuda.manual_seed(job.sp.seed)

    # Get input values and weights
    pxs, weights = exp19_helpers.get_pxs_and_weights(job)

    # Run the simulation
    Z_hats, Z_stars, scale_factors = exp_func(job, device, pxs, weights)

    if save:
        with job.data as f:
            f.Z_hats = Z_hats.cpu().numpy()
            f.Z_stars = Z_stars.cpu().numpy()
            f.scale_factors = scale_factors.cpu().numpy()

        job.doc.saved = True

    return Z_hats, Z_stars, scale_factors


def run_exp1_batch(version, px_dist, weight_dist, save=True, redo=False, version_func=None):
    """
    Run a batch of simulation jobs
    :param version: determines the experimental parameters
    :param px_dist: how input values are sampled
    :param weight_dist: how weight values are sampled
    :param save: True if the simulation results should be saved.
    :param redo: True if the simulation should be rerun.
    If redo=False then simulation is only run if it doesn't already have saved data
    :param version_func: sloppy, but sometimes I want to run a different version function.
    :return: None
    """
    if version_func is None:
        ns, Ks, circuits, labels = get_exp1_params(version)
    else:
        ns, Ks, circuits, labels = version_func(version)

    seed = 1231
    sim_runs = 32 * 313  # ~10,000

    # set some params
    sp_template = {"seed": seed,
                   "sim_runs": sim_runs,
                   "px_dist": px_dist,
                   "weight_dist": weight_dist}

    for n_idx, n in enumerate(ns):
        sp_template["rns_prec"] = n.item()
        sp_template["SN_length"] = int(2**n)
        for k_idx, K in enumerate(Ks):
            sp_template["num_addends"] = int(K.item())
            for c_idx, circuit in enumerate(circuits[n_idx][k_idx]):
                sp = sp_template.copy()
                sp.update(circuit)  # add circuit items to statepoint
                job = general.check_and_load_job(pr_19, sp)
                if not job.doc.saved or redo:
                    print(f"Working on n={n} L={int(2 ** n)} K={K} C={c_idx + 1}/{len(circuits[n_idx][k_idx])} "
                          f"{labels[n_idx][k_idx][c_idx]}")
                    run_exp1_job(job, save)


#########################
# Experiment 1 Plotting #
#########################
# !!! Note to e-HAIL staff. The following code is very sloppy. I don't share this code with anyone, so I rarely improve
# the quality of plotting functions once they create the result I want. I left these in as examples of my visualization
# experience. !!!
def load_data_fixL(version, n, px_dist, weight_dist):
    sim_runs = 32*313
    seed = 1231

    ns, Ks, circuits, circ_labels = get_exp1_params(version)
    n_idx = torch.argwhere(ns == n).item()
    num_circuit = len(circuits[n_idx][-1])
    num_psa = int(np.floor(np.log2(Ks[-1])))  # no +1 b/c we don't wanna count APC
    labels = circ_labels[n_idx][-1]

    # set some params
    sp_template = {"seed": seed,
                   "sim_runs": sim_runs,
                   "px_dist": px_dist,
                   "weight_dist": weight_dist,
                   "rns_prec": n,
                   "SN_length": int(2**n),
                   }

    # and now this code gets the data
    Z_hats, Z_stars, scale_factors, norms = np.full((4, len(Ks), num_circuit, sim_runs), np.nan)
    bipolar = None
    for k_idx, K in enumerate(Ks):
        sp_template["num_addends"] = K.item()
        for c_idx, circuit in enumerate(circuits[n_idx][k_idx]):
            # this check ensures we didnt mess up circuit ordering. All PSAs should go at end of circuit list bc there's
            # a variable number of PSA designs
            assert circ_labels[n_idx][k_idx][c_idx] == labels[
                c_idx], f"{circ_labels[n_idx][k_idx][c_idx]} {labels[c_idx]}"

            sp = sp_template.copy()
            sp.update(circuit)  # add circuit items to statepoint
            job = general.check_and_load_job(pr_19, sp)
            assert job.doc.saved, job.sp
            if bipolar is None:
                bipolar = job.sp.bipolar
            else:
                assert job.sp.bipolar == bipolar, "This function assumes all circuits are unipolar or bipolar."

            with job.data as f:
                Z_hats[k_idx, c_idx] = f.Z_hats[:]
                Z_stars[k_idx, c_idx] = f.Z_stars[:]
                scale_factors[k_idx, c_idx] = f.scale_factors[:]
                norms[k_idx, c_idx] = exp19_helpers.get_norm(job)
    return Z_hats, Z_stars, scale_factors, norms, Ks, bipolar, num_circuit, labels, num_psa


def load_data_fixK(version, K, px_dist, weight_dist):
    sim_runs = 32*313
    seed = 1231

    ns, Ks, circuits, circ_labels = get_exp1_params(version)
    K_idx = torch.argwhere(Ks == K).item()
    num_circuit = len(circuits[0][K_idx])
    num_psa = int(np.ceil(np.log2(K))) + 1
    labels = circ_labels[0][K_idx]

    # set some params
    sp_template = {"seed": seed,
                   "sim_runs": sim_runs,
                   "px_dist": px_dist,
                   "weight_dist": weight_dist,
                   "num_addends": K,
                   }

    # and now this code gets the data
    Z_hats, Z_stars, scale_factors, norms = np.full((4, len(ns), num_circuit, sim_runs), np.nan)
    bipolar = None
    for n_idx, n in enumerate(ns):
        sp_template["rns_prec"] = n.item()
        sp_template["SN_length"] = int(2**n.item())
        for c_idx, circuit in enumerate(circuits[n_idx][K_idx]):
            # this check ensures we didnt mess up circuit ordering. All PSAs should go at end of circuit list bc there's
            # a variable number of PSA designs
            assert circ_labels[n_idx][K_idx][c_idx] == labels[
                c_idx], f"{circ_labels[n_idx][K_idx][c_idx]} {labels[c_idx]}"

            sp = sp_template.copy()
            sp.update(circuit)  # add circuit items to statepoint
            job = general.check_and_load_job(pr_19, sp)
            assert job.doc.saved
            if bipolar is None:
                bipolar = job.sp.bipolar
            else:
                assert job.sp.bipolar == bipolar, "This function assumes all circuits are unipolar or bipolar."
            with job.data as f:
                Z_hats[n_idx, c_idx] = f.Z_hats[:]
                Z_stars[n_idx, c_idx] = f.Z_stars[:]
                scale_factors[n_idx, c_idx] = f.scale_factors[:]
                norms[n_idx, c_idx] = exp19_helpers.get_norm(job)

    return Z_hats, Z_stars, scale_factors, norms, ns, bipolar, num_circuit, labels, num_psa


def save_errors(version, n, px_dist, weight_dist):
    Z_hats, Z_stars, Ks, bipolar, num_circuit, labels, num_psa = load_data_fixL(version, n, px_dist, weight_dist)
    Ks = Ks.cpu().numpy()

    # compute errors. Z_hats and Z_stars HAVE BEEN NORMALIZED BY SCALE FACTOR when loaded :)
    errors = (Z_hats - Z_stars)
    mses = np.mean(np.square(errors), axis=-1)
    rmses = np.sqrt(mses)
    norm = Ks * (1+bipolar)
    percent_errors = np.mean(np.abs(errors / norm[:, None, None]), axis=-1)

    data = [[labels[c_idx], Ks[K_idx], percent_errors[K_idx, c_idx]] for c_idx in range(num_circuit) for K_idx in range(len(Ks))]

    df = pd.DataFrame(data, columns=["Circuit", "Num Addends", "NED"])

    df.to_csv(rf"{ERROR_DIR}\n{n}_{px_dist}_{weight_dist}_NED.csv")
    print(df)


# v1 plots K on x-axis and plots each design as a separate line
def plot_exp1_fixL_varyK_v1(version, n, px_dist, weight_dist, logy, do_percent, sep, sep_legend, dup_apc, same_ylim, save_plot, captions):
    assert not dup_apc or sep, "Only makes sense to duplicate APC if we are using seperate plots"
    Z_hats, Z_stars, scale_factors, norms, Ks, bipolar, num_circuit, labels, num_psa =\
        load_data_fixL(version, n, px_dist, weight_dist)

    num_non_psa = num_circuit - num_psa

    # compute errors. Z_hats and Z_stars
    errors = (Z_hats - Z_stars)

    thesis = True
    y_lab = "Root BMSE" if thesis else "RMSE"
    y_lab = "Error" if do_percent else y_lab


    # compute aggregate metrics
    mses = np.mean(np.square(errors/scale_factors), axis=-1)
    rmses = np.sqrt(mses)
    percent_errors = np.mean(np.abs(errors / norms), axis=-1)
    sfs = np.mean(scale_factors, axis=-1)
    for label, rmse, pe, norm, sf in zip(labels, rmses.T, percent_errors.T, norms[:, :, 0].T, sfs.T):
        print(f"{label}\nRMSE:{rmse}\nPE:{pe}\nNorm:{norm}\nSf:{sf}\n")

    # assign x and y
    x = Ks
    y = percent_errors if do_percent else rmses*np.sqrt(2**n)

    combined = not sep

    fs1 = 16
    fs2 = 14
    lw = 2.5
    markers = [('o', 8), ('P',8), ('>',8), ('<',8), ('^',8), ('s',8), ('*',12), ('d',8),
               ('v', 8), ('X', 8), ('p', 10), ('h', 8)]
    non_psa_colors = ['#1b9e77', '#d95f02', '#e7298a', '#7570b3', '#66a61e', '#e6ab02', '#a6761d', '#666666']
    non_psa_colors = non_psa_colors[:num_non_psa]
    psa_colors = plt.cm.plasma(np.linspace(0, 0.85, num_psa + int(dup_apc)))
    if dup_apc:
        non_psa_colors[-1] = psa_colors[0]
    comb_colors = non_psa_colors
    comb_colors.extend(psa_colors)
    fig_labels = ['\n(b)', '\n(a)'] if sep else ['', '']

    psa_fig = plt.figure(figsize=(8, 3.68))
    psa_ax = psa_fig.gca()
    if sep:
        non_psa_figsize = (8, 4.2) if num_non_psa > 4 else (8, 2.75)
        non_psa_fig = plt.figure(figsize=non_psa_figsize)
        non_psa_ax = non_psa_fig.gca()
    else:
        non_psa_fig = psa_fig
        non_psa_ax = psa_ax

    for c_idx in range(num_non_psa):
        marker, ms = markers[0:num_non_psa][-c_idx-1]
        color = comb_colors[c_idx] if combined else non_psa_colors[c_idx]
        non_psa_ax.plot(x, y[:, c_idx], lw=lw, ms=ms, label=labels[c_idx], color=color, marker=marker)

    for p_idx in range(num_psa+int(dup_apc)):
        c_idx = num_non_psa + p_idx - int(dup_apc)
        lab_mod = " ($G=1$)" if p_idx == 0 and dup_apc else ""
        marker, ms = markers[c_idx] if combined else markers[p_idx]
        color = comb_colors[c_idx] if combined else psa_colors[p_idx]
        psa_ax.plot(x, y[:, c_idx], lw=lw, ms=ms, label=labels[c_idx]+lab_mod, color=color, marker=marker)

    # handle log scale
    if logy:
        psa_ax.set_yscale('log', base=10)
        non_psa_ax.set_yscale('log', base=10)
        # y_lab += " (log scale)"

    # figure out ylims
    if same_ylim:
        psa_ylim = psa_ax.get_ylim()
        non_psa_ylim = non_psa_ax.get_ylim()
        ylim = [min(psa_ylim[0], non_psa_ylim[0]), max(psa_ylim[1], non_psa_ylim[1])]

    # handle labels
    figs = [psa_fig, non_psa_fig]
    axs = [psa_ax, non_psa_ax]
    leg_titles = ["Group size, $G$", "Adder"]
    x_label_base = "Number of addends, $M$" if weight_dist != 'ecg' else "Number of filter taps, $M$"
    for ax_idx, ax in enumerate(axs):
        ax.set_xlabel(f"{x_label_base}", fontsize=fs1)
        if captions[ax_idx] is not None:
            ax.text(x=0.5, y=0, s=captions[ax_idx], ha='center', va='bottom', transform=plt.gcf().transFigure, fontsize=fs1+2)
        ax.set_ylabel(y_lab, fontsize=fs1)
        ax.tick_params(axis='both', which='major', labelsize=fs2)
        ax.grid()
        ax.legend(title=leg_titles[ax_idx], fontsize=fs2)
        figs[ax_idx].tight_layout()
        if same_ylim:
            ax.set_ylim(ylim)
        # handle legend:
        if sep or ax_idx == 0:
            # Shrink current axis by 20%
            if sep_legend:
                handles, labs = ax.get_legend_handles_labels()
                order = np.arange(len(handles))
                if sep and ax_idx == 0:
                    order = order[::-1]
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])

                # Put a legend to the right of the current axis
                legend = ax.legend([handles[idx] for idx in order], [labs[idx] for idx in order], title=leg_titles[ax_idx],
                                    ncol=1, fontsize=fs2 - 2, loc='center left', bbox_to_anchor=(1, 0.5))
                plt.setp(legend.get_title(), fontsize=fs2)

    if save_plot:
        assert bipolar, "Fix figure number if you wanna use unipolar (i.e., add new fig numbers)"
        adjusts = [{"top": 0.959, "bottom": 0.275, "left": 0.106, "right": 0.776, "hspace": 0.2, "wspace": 0.2},
                   {"top": 0.945, "bottom": 0.369, "left": 0.106, "right": 0.786, "hspace": 0.2, "wspace": 0.2}]

        if px_dist == 'rand':
            fig_num = 20
        elif px_dist == 'ecg':
            fig_num = 21
        if version in [4, 5, 6]:  # using flip APC and PSAs
            fig_num += 2
        for fig, fig_cap, fig_name, adjust in zip([psa_fig, non_psa_fig], ["b", "a"], ["psa", "adder"], adjusts):
            flip_str = "-flip" if version in [4, 5, 6] else ""

            fig.tight_layout()
            fig.subplots_adjust(**adjust)
            dir = r"C:\Users\Tim\OneDrive - Umich\Documents\University of Michigan\Research\Thesis\Thesis\Figures\Ch5"
            filename = f"{fig_num}{fig_cap}_{fig_name}-error-{px_dist}-n{n}-varyCK-bip{int(bipolar)}{flip_str}"
            fig.savefig(rf"{dir}\{filename}.png", dpi=300)
    plt.show(block=True)


# v2 plots design on x-axis and then varies K as separate lines (this looked bad)
def plot_exp1_fixL_varyK_v2(version, n, logy, do_percent, px_dist, weight_dist):
    Z_hats, Z_stars, bipolar, Ks, num_circuit, labels, num_psa = load_data_fixL(version, n, px_dist, weight_dist)

    # compute errors. Z_hats and Z_stars HAVE BEEN NORMALIZED BY SCALE FACTOR when loaded :)
    errors = (Z_hats - Z_stars)
    mses = np.mean(np.square(errors), axis=-1)
    rmses = np.sqrt(mses)
    norm = Ks.cpu().numpy() * (1 + bipolar)
    percent_errors = np.mean(np.abs(errors / norm[:, None, None]), axis=-1)

    # set fig size and other params
    plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (8, 4.5)

    # assign x and y
    x = np.arange(num_circuit)
    y = percent_errors if do_percent else rmses
    y_lab = "NMED" if do_percent else 'RMSE'

    lw = 2.5
    ms = 8
    m_idx = 0
    colors = list(reversed(['#1b9e77', '#377eb8', '#7570b3', '#e7298a']))
    colors += ['#ffffe5', '#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506']
    for K_idx in range(len(Ks)):
        plt.plot(x, y[K_idx], 'o', lw=lw, ms=ms, label=f"$K={Ks[K_idx]}$", color=colors[K_idx])
        m_idx += 1

    fs1 = 16
    fs2 = 14
    if logy:
        plt.semilogy(base=10)
        y_lab += " (log scale)"
    plt.xlabel("Number of addend inputs", fontsize=fs1)
    plt.ylabel(y_lab, fontsize=fs1)
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.legend(title="Number Addends", fontsize=fs2)


    plt.grid(color='k')
    sc_format = "bipolar" if bipolar else " unipolar"
    plt.title(f"{sc_format.title()} with SN length: {int(2 ** n)}")
    plt.tight_layout()

    # handle legend:
    # Shrink current axis by 20%
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])
    # Put a legend to the right of the current axis
    legend = plt.legend(title="Number Addends", ncol=1, fontsize=fs2 - 2, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.setp(legend.get_title(), fontsize=fs2)

    plt.xticks(ticks=np.arange(num_circuit), labels=labels, rotation=30)
    plt.tight_layout()
    plt.show(block=True)


# plots group size on x-axis and plots each SN length for a fixed num inputs K
def plot_exp1_fixK_varyL_varyG_v1(version, K, logx, logy, do_percent, sep_legend, px_dist, weight_dist, save_plot, caption=None):
    Z_hats, Z_stars, scale_factors, norms, ns, bipolar, num_circuit, labels, num_psa = load_data_fixK(version, K, px_dist, weight_dist)

    # get rid of the non_psa designs b/c we are plotting versus group size.
    start_idx = num_circuit - num_psa
    end_idx = -2

    # get group sizes
    Gs = 2**np.arange(num_psa+end_idx)
    xticklabels = [f"$2^{{{i}}}$" for i in range(num_psa+end_idx)]
    xticklabels[0] += "\n(APC)"

    # compute errors. Z_hats and Z_stars HAVE BEEN NORMALIZED BY SCALE FACTOR when loaded :)
    errors = (Z_hats - Z_stars)
    mses = np.mean(np.square(errors/scale_factors), axis=-1)
    rmses = np.sqrt(mses)
    percent_errors = np.mean(np.abs(errors / norms), axis=-1)

    # assign x and y
    x = Gs
    y = percent_errors if do_percent else rmses

    plt.rcParams["figure.figsize"] = (8, 3.2)
    fs1 = 16
    fs2 = 14
    lw = 2.5
    markers = [('o', 8), ('P',8), ('>',8), ('<',8), ('^',8), ('s',8), ('*',12), ('d',8),
               ('v', 8), ('X', 8), ('p', 10), ('h', 8)]

    cmap = plt.cm.plasma if bipolar else plt.cm.viridis
    colors = cmap(np.linspace(0, 0.85, num_psa))
    colors = list(reversed(colors))
    for n_idx, n in enumerate(ns):
        print(n)
        marker, ms = markers[n_idx]
        color = colors[n_idx]
        plt.plot(x, y[n_idx, start_idx:end_idx], lw=lw, ms=ms, label=f'$L={int(2**n)}$', color=color, marker=marker)

    if logx:
        plt.semilogx(base=2)
        plt.xticks(Gs, xticklabels)
    if logy:
        plt.semilogy(base=10)
        # y_lab += " (log scale)"

    thesis = True
    xlabel = "Sampling group size, $G$"
    ylabel = "Root BMSE" if thesis else "RMSE"
    ylabel = "NMED" if do_percent else ylabel

    if caption is not None:
        xlabel += "\n"
        label_pad = -fs2
        plt.gca().text(x=0.5, y=0, s=caption, ha='center', va='bottom', transform=plt.gcf().transFigure, fontsize=fs1+2)
    else:
        label_pad = 0
    plt.xlabel(xlabel, fontsize=fs1, labelpad=label_pad)
    plt.ylabel(ylabel, fontsize=fs1)
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.legend(title="SN Length, $L$", fontsize=fs2)
    plt.grid()
    plt.tight_layout()

    # handle legend:
    order = np.arange(len(ns))
    ax = plt.gca()
    if sep_legend:
        handles, labs = plt.gca().get_legend_handles_labels()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])

        # Put a legend to the right of the current axis
        legend = plt.legend([handles[idx] for idx in order], [labs[idx] for idx in order], title="SN Length, $L$",
                            ncol=1, fontsize=fs2 - 2, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.setp(legend.get_title(), fontsize=fs2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.925, bottom=0.301, left=0.12, right=0.798, hspace=0.2, wspace=0.2)
    if save_plot:
        dir = r"C:\Users\Tim\OneDrive - Umich\Documents\University of Michigan\Research\Thesis\Thesis\Figures\Ch5"
        fig_num = "19" if bipolar else "18"
        filename = f"{fig_num}_psa-error-{px_dist}-K{K}-varyGL-bip{int(bipolar)}"
        plt.savefig(rf"{dir}\{filename}.png", dpi=300)
    plt.show(block=True)


# plots circuit on x-axis and plots each SN length for a fixed num inputs K
def plot_exp1_fixK_varyL_varyC_v1(version, K, logy, do_percent, sep_legend, px_dist, weight_dist):
    Z_hats, Z_stars, scale_factors, norms, ns, bipolar, num_circuit, labels, num_psa =\
        load_data_fixK(version, K, px_dist, weight_dist)

    # compute errors. Z_hats and Z_stars
    errors = (Z_hats - Z_stars)
    # norms = np.nanmax(np.abs(Z_stars), axis=-1, keepdims=True)
    norms = scale_factors

    mses = np.mean(np.square(errors/scale_factors), axis=-1)
    rmses = np.sqrt(mses)
    percent_errors = np.nanmean(np.abs(errors / norms), axis=-1)

    # assign x and y
    x = np.arange(num_circuit)
    y = percent_errors if do_percent else rmses
    y_lab = "NMED" if do_percent else 'RMSE'

    plt.rcParams["figure.figsize"] = (8, 3.4)
    fs1 = 16
    fs2 = 14
    lw = 2.5
    markers = [('o', 8), ('P',8), ('^',8), ('s',8), ('*',12), ('X', 8), ('p', 10)]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(ns)))
    colors = list(reversed(colors))
    fig_label = '\n(c)' if bipolar else '\n(b)'
    for n_idx, n in enumerate(ns):
        print(n)
        marker, ms = markers[n_idx]
        color = colors[n_idx]
        # plt.plot(x, y[n_idx, :], 'o', label=f'$L={int(2**n)}$', color=color)
        plt.plot(x, y[n_idx, :], f'{marker}', label=f'$L={int(2**n)}$', color=color, ms=ms)

    # plt.xticks(x, labels, rotation=-30, ha='left')
    plt.xticks(x, labels, rotation=-30, rotation_mode='anchor', ha='left')
    if logy:
        plt.semilogy(base=10)
        y_lab += " (log scale)"

    plt.ylabel(y_lab, fontsize=fs1)
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.legend(title="SN Length, $L$", fontsize=fs2)
    plt.grid()
    plt.tight_layout()

    # handle legend:
    order = np.arange(len(ns))
    ax = plt.gca()
    if sep_legend:
        handles, labs = plt.gca().get_legend_handles_labels()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])

        # Put a legend to the right of the current axis
        legend = plt.legend([handles[idx] for idx in order], [labs[idx] for idx in order], title="SN Length, $L$",
                            ncol=1, fontsize=fs2 - 2, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.setp(legend.get_title(), fontsize=fs2)

    plt.tight_layout()
    plt.show(block=True)


if __name__ == '__main__':
    # v1: Unipolar rand/ecg | v2: bipolar rand/ecg | v3: bipolar FMNIST/BNN | v4: Unipolar rand/ecg with flip PSAs
    # v5: bipolar rand/ecg with flip PSAs | v6: bipolar FMNIST/BNN with flip PSAs
    # position versions go through all circuits, negative versions only do PSAs
    version = -2
    do_ecg = False  # for use when version is not [3, 6, 7] (i.e., when version is not MNIST related).
    redo = False  # whether redo the simulations
    save = True  # whether to save the simulations
    save_plot = True  # whether to save any plots made
    varyL = True  # if not varyL (SN length) then we varyK (inputs)
    captions = ['(b)', '(b)']  # for some plotting functions

    if version in [3, 6, 7, -3, -6, -7]:
        px_dist, weight_dist = 'fmnist', 'fmnist-bnn'
        K = 28*28
        print(version)
        # run_exp1_batch(version, px_dist, weight_dist, save=save, redo=redo)
        plot_exp1_fixK_varyL_varyC_v1(version, K, logy=True, do_percent=False, sep_legend=True, px_dist=px_dist, weight_dist=weight_dist)
    else:
        n = 8 if version < 0 else 10  # if we are running PSA only baselines then use 8 for thesis else use 10
        n = None if varyL else n
        K = 512 if varyL else None
        px_dist, weight_dist = ('ecg', 'ecg') if do_ecg else ('rand', 'rand')
        print(version, n, K, px_dist, weight_dist)

        # run this if you need to generate more data:
        run_exp1_batch(version, px_dist, weight_dist, save=save, redo=redo)

        # run this when you want to vary number of inputs and fix SN length. For thesis ECG case study (n=10) and baseline (n=8)
        if not varyL:
            plot_exp1_fixL_varyK_v1(version, n, px_dist, weight_dist, logy=True, do_percent=False, sep=True, sep_legend=True, dup_apc=True, same_ylim=False, save_plot=save_plot,
                                    captions=captions)

        # run this when you want to vary SN length L and group size G. For PSA baseline results (i.e., unipolar/bipolar rand)
        if varyL:
            plot_exp1_fixK_varyL_varyG_v1(version, K, logx=True, logy=True, do_percent=False, sep_legend=True, px_dist=px_dist, weight_dist=weight_dist, save_plot=save_plot,
                                          caption=captions[0])

