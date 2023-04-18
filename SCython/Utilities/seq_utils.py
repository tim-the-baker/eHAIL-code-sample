import numpy as np
import torch
import scipy.io as sp_io
import os

# Useful directories
SOBOL_SEQ_DIR = r'C:\Users\Tim\Documents\Michigan Research Data\Sobol Seq Data'
LFSR_SEQ_DIR = r'C:\Users\Tim\Documents\Michigan Research Data\LFSR Seq Data'
LFSR_FEED_DIR = r'C:\Users\Tim\Documents\Michigan Research Data\LFSR Feed Data'

def get_vdc(n, verbose=True):
    """
    Generates the VDC sequence according to a given precision
    :param n: the desired precision, in bits
    :param verbose: when True, a message is printed that indicates this method is run. This method is slow and should
    only be run once during the entire simulation rather than once per simulation run.
    :return: the n-bit VDC sequence
    """
    if verbose:
        print(f"Generating {n}-bit VDC sequence")
    pow2n = int(2**n)
    seq = np.zeros(pow2n, dtype=np.int)
    for idx in range(pow2n):
        bin_rep = np.binary_repr(idx, n)
        seq[idx] = sum([int(2**b_idx * int(bit)) for b_idx, bit in enumerate(bin_rep)])
    return seq


def get_Sobol_seqs(precision, verbose=True):
    """
    Retrieves four Sobol sequences that were generated using MATLAB. Converts them to given precision.
    :param precision: the desired precision of the sequences.
    :param verbose: when True, a message is printed that indicates this method is run. This method is slow and should
    only be run once during the entire simulation rather than once per simulation run.
    :return: the n-bit integer Sobol sequence
    """
    assert precision <= 16, "Sobol sequences only go up to 16 bits."
    if verbose:
        print(f"Retrieving {precision}-bit Sobol sequences")

    # Load the sequences. seq's shape is (4, 2^(16)
    seqs = sp_io.loadmat(f"{SOBOL_SEQ_DIR}\sobol_4x65536.mat")['ans'].T

    # Truncate the lists to the correct length
    seqs = seqs[:, :int(2**precision)]

    # Convert the lists to integers of the desired precision
    seqs = (seqs*int(2**precision)).astype(int)

    return seqs


def get_vdc_torch(n, device, verbose=True):
    seq = get_vdc(n, verbose)
    seq = torch.tensor(seq, dtype=torch.int, device=device)
    return seq


def get_LFSR_seqs(n, feedback, extended=False, verbose=True):
    """
    :param n: bit-width of LFSR
    :param feedback: LFSR feedback type. 'e': external feedback, 'i': internal feedback, 'a': all/both feedback types.
    :param extended: if True, use a LFSR that has the all 0 state inserted. LFSR becomes a NLFSR.
    :return: corresponding list of LFSR seqs.
    """
    if verbose:
        print(f"Loading {n}-bit {feedback} {'N'*int(extended)}LFSR sequences")
    all_seqs = np.load(LFSR_SEQ_DIR + r'\n%d_e%d_p%d.npy' % (n, extended, 0))
    all_seqs = torch.tensor(all_seqs, dtype=torch.int)
    if feedback == 'a' or extended:
        return all_seqs
    elif feedback == 'e':
        return all_seqs[len(all_seqs)//2:]  # second half are the external LFSR seqs
    elif feedback == 'i':
        return all_seqs[0:len(all_seqs)]  # first half are the internal LFSR seqs
    else:
        raise ValueError(f"feedback should be 'e' (external) 'i' (internal) or 'a' (all). Given feedback: {feedback}")


def get_LFSR_feeds(n):
    feeds = np.load(LFSR_FEED_DIR + fr'\\n{n}.npy')
    return feeds


if __name__ == '__main__':
    prec = 8
    num = 2
    seq_idxs = np.random.permutation(4)[0:num]

    rng = np.random.default_rng()
    perms = rng.permuted(np.arange(prec)[None].repeat(num, axis=0), axis=1).T  # (prec, num)
    invs = rng.integers(low=0, high=2, size=(prec, num), dtype=bool)  # (prec, num)

    seqs = get_Sobol_seqs(precision=prec).astype(np.uint8)
    print(seqs.shape)
    Rs = seqs[seq_idxs]
    print(Rs.shape)
    Rs_bits = np.unpackbits(Rs[None], axis=0)
    print(Rs.shape)
    perm_Rs_bits = np.take_along_axis(Rs_bits, perms[..., None], axis=0)
    print(perm_Rs_bits.shape)
    inv_Rs_bits = np.where(invs[..., None], 1-perm_Rs_bits, perm_Rs_bits)
    print(inv_Rs_bits.shape)
    mod_Rs = np.packbits(inv_Rs_bits, axis=0)[0]
    print(mod_Rs.shape)
    for mod_R in mod_Rs:
        print(len(np.unique(mod_R)))
    exit()

    print(seqs[0])
    print(seqs[1])
    print(seqs[2])
    print(seqs[3])
    print(get_vdc(4))
