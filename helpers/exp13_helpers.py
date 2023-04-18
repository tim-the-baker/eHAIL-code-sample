"""
File for storing the sometimes long list of parser arguments
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from Code_examples.models03_updated_FMNIST import NN_MLP2, BNN_MLP2, PSA_MLP2
from SCython.SNG_torch import SNG, RNS, PCC, MSG
from Code_examples.helpers.bcolors import bcolors
from Code_examples.helpers import general, projects
import enum

project = projects.pr_exp13
seed1 = 2918263
#############
# Utilities #
#############
@enum.unique
class NET_TYPE(enum.IntEnum):
    # Don't forget to update ALL methods below when you add a new net!
    BNN_MLP2 = 1
    APC_MLP = 2
    PSA_MLP = 3
    NN_MLP2 = 4

    def is_Binarized(self):
        return self in [NET_TYPE.BNN_MLP2, NET_TYPE.APC_MLP, NET_TYPE.PSA_MLP]

    def is_SC(self):
        return self in [NET_TYPE.APC_MLP, NET_TYPE.PSA_MLP]

    def is_mux(self):
        return self in [NET_TYPE.PSA_MLP]

    def is_cnn(self):
        return self in []


def load_pretrained_weights(model, job, addt_filters=None):
    if addt_filters is None:
        addt_filters = {}

    if job.sp.pretrain_net is None:
        print(f"Pre-trained net not found!")
        return model

    filters = {"net_type": job.sp.pretrain_net, **addt_filters}
    best_acc = 0
    best_job = None

    # first find the net that achieved best validation accuracy
    for curr_job in project.find_jobs(filters):
        if curr_job.doc.saved:
            if curr_job.doc.val_acc > best_acc:
                best_job = curr_job
                best_acc = curr_job.doc.val_acc

    # then grab and load its weights
    if best_job is not None:
        model.load_state_dict(torch.load(best_job.fn("state_dict.pt")))
        model.pretrained = True
        print(f"Loaded model weights from job with val_acc {best_acc} and statepoint:\n{best_job.sp}")
    else:
        print(f"Could not find pretrained model with filters matching: {filters}")
        raise
    return model


def get_data_loaders(val_split, batch_size, test_batch_size):
    assert 0 <= val_split <= 1

    train_set = datasets.FashionMNIST('data/datasets', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_len, val_len = int(len(train_set)*(1-val_split)), int(len(train_set)*val_split)
    train_set, val_set = random_split(train_set, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    if val_split > 0:
        val_loader = DataLoader(val_set, batch_size=test_batch_size, shuffle=True)
    else:
        val_loader = None
    test_loader = DataLoader(datasets.FashionMNIST('data/datasets', train=False,transform=transforms.Compose([transforms.ToTensor()])),
                             batch_size=test_batch_size, shuffle=True)
    return train_loader, val_loader, test_loader


def get_model(job):
    device = torch.device('cuda:0') if job.sp.cuda else torch.device('cpu')
    rns, pcc, sng, msg = None, None, None, None

    if NET_TYPE(job.sp.net_type).is_SC():
        rns = RNS.get_rns_class_from_name(job.sp.rns_name)(job.sp.rns_prec, device)
        if job.sp.pcc_name == PCC.Mux_maj_chain.__name__:
            pcc = PCC.Mux_maj_chain(job.sp.rns_prec, num_mux=job.sp.rns_prec-job.sp.num_maj, num_maj=job.sp.num_maj, invert_maj_R=False)
        else:
            pcc = PCC.get_pcc_class_from_name(job.sp.pcc_name)(job.sp.rns_prec)
        sng = SNG.SNG(rns, pcc)

    if NET_TYPE(job.sp.net_type).is_mux():
        msg = MSG.get_msg_class_from_name(job.sp.msg_name)(int(2**job.sp.samp_height), device)

    # Grab the net
    if job.sp.net_type == NET_TYPE.NN_MLP2:
        model = NN_MLP2(job.sp.input_prec, job.sp.bias, job.sp.affine, job.sp.dropout_p)
    elif job.sp.net_type == NET_TYPE.BNN_MLP2:
        model = BNN_MLP2(job.sp.input_prec, job.sp.bias, job.sp.affine, job.sp.dropout_p)
    elif job.sp.net_type == NET_TYPE.PSA_MLP:
        model = PSA_MLP2(sng, msg, job.sp.SN_length, job.sp.samp_height, job.sp.bias, job.sp.bipolar, job.sp.full_corr, job.sp.sep_bip)
    else:
        raise NotImplementedError(f"Could not find model in get_model method {job.sp.net_type}.")

    if job.doc.saved:
        model.load_state_dict(torch.load(job.fn("state_dict.pt")))
        print(f"{bcolors.WARNING}Saved model found! Loaded Weights!{bcolors.ENDC}")
    return model


#########
# Setup #
#########
def get_PSA_job(rns_prec, samp_height, pretrain_net, sep_bip, **kwargs):
    net_type = NET_TYPE.PSA_MLP
    seed = seed1
    bipolar = not sep_bip

    # SC Parameters that shouldn't change often
    SC_sp = {
        "rns_prec": rns_prec,
        "samp_height": samp_height,
        "sep_bip": sep_bip,
        "bipolar": bipolar,
        "rns_name": kwargs.pop("rns_name", RNS.VDC_RNS.__name__),
        "msg_name": kwargs.pop("msg_name", MSG.Long_Counter_Exact_MSG.__name__),
        "pcc_name": kwargs.pop("pcc_name", PCC.Comparator.__name__),
        "full_corr": kwargs.pop("full_corr", True),
        "bip_to_uni": kwargs.pop("bip_to_uni", False),
        "SN_length": int(2**rns_prec)
    }

    num_maj = kwargs.pop("num_maj", None)  # for use if PCC = MMC
    if SC_sp["pcc_name"] == PCC.Mux_maj_chain.__name__:
        SC_sp["num_maj"] = num_maj

    lr = kwargs.pop("lr", None)
    if lr is None:
        lr = 1e-3 if pretrain_net is None else 1e-2
    NN_sp = {
        "net_type": net_type.value,
        "batch_size": kwargs.pop("batch_size", 32),
        "test_batch_size": kwargs.pop("test_batch_size", 32),
        "pretrain_net": pretrain_net,
        "epochs": kwargs.pop("epochs", 20),
        "dropout_p": kwargs.pop("dropout_p", 0.20),
        "momentum": 0.5,
        "val_split": 0.1,
        "cuda": torch.cuda.is_available(),
        "seed": seed1,
        "bias": False,
        "lr": lr
    }
    sp = {**SC_sp, **NN_sp}

    # set-up random
    if seed is not None:
        print(f"{bcolors.WARNING}THIS EXPERIMENT HAS BEEN SEEDED!{bcolors.ENDC}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # init the job
    # check to see if job exists
    job = general.check_and_load_job(project, sp, verbose=False)

    # Fixed stuff
    return job


def get_NN_or_BNN_job(pretrain_net, affine, net_type, **kwargs):
    assert net_type in [NET_TYPE.NN_MLP2, NET_TYPE.BNN_MLP2]
    seed = seed1

    lr = kwargs.pop("lr", None)
    if lr is None:
        lr = 1e-3 if pretrain_net is None else 1e-2
    sp = {
        "net_type": net_type.value,
        "input_prec": kwargs.pop("input_prec", 9),
        "batch_size": kwargs.pop("batch_size", 32),
        "test_batch_size": kwargs.pop("test_batch_size", 32),
        "pretrain_net": pretrain_net,
        "epochs": kwargs.pop("epochs", 20),
        "momentum": 0.5,
        "val_split": 0.1,
        "dropout_p": kwargs.pop("dropout_p", 0.20),
        "cuda": torch.cuda.is_available(),
        "seed": seed1,
        "bias": False,
        "lr": lr,
        "affine":affine
    }

    # set-up random
    if seed is not None:
        print(f"{bcolors.WARNING}THIS EXPERIMENT HAS BEEN SEEDED!{bcolors.ENDC}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # init the job
    # check to see if job exists
    job = general.check_and_load_job(project, sp, verbose=False)

    # Fixed stuff
    return job

