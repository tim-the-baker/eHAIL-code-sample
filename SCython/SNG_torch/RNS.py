import torch
import numpy as np
from SCython.Utilities import seq_utils
import math

class RNS:
    """ Abstract class for stochastic computing random number sources """
    def __init__(self, precision, device, **kwargs):
        """
        :param precision: bit-width of the RNS.
        :param kwargs: Extra parameters mainly for hardware RNSs. Current choices include:
            feedback: LFSR feedback type (internal: 'i', external: 'e' or all: 'a')
            seqs: Pre-loaded seqs (seqs should correspond to n-bit RNS). (Note this class now automatically loads seqs.
            seq_idx: index of specific seq in seqs that should be used; seqs should be specified when using this.
        """
        assert precision <= 16
        self._n: int = precision
        self.device = device
        self._max_N: int = -1
        self.name: str = ""
        self.legend: str = ""
        self.is_hardware: bool = False

    def _gen_RN(self, N, shape, share):
        """
        Helper function for gen_SN. This method must be overridden in the subclass and should implement the RNS.
        See the gen_RN method for more info or view some of the including RNS subclasses for examples.
        :param int N: SN length
        :param tuple shape: dimensions of random number array.
        :param bool share: whether or not to share the RNS amongst all SNs.
        :return: the random number array
        """
        raise NotImplementedError

    def gen_RN(self, N, shape, share, inv_mask=None, full_corr=False):
        """
        Generates an array of random numbers used for SN generation. Can implement sharing the RNS directly by setting
        share=True and can implement sharing the inverted RNS by using share=True and the inv_mask param. This
        method implements the part of RN generation that applies to all RNS while the helper _gen_RN method implements
        each RNS specific manner of creating random numbers.
        :param int N: SN length.
        :param tuple shape: dimension of the random number array.
        :param bool share: whether or not to share the RNS amongst all RNs in the array.
        :param np.ndarray inv_mask: inv_mask is a Boolean array whose size should match the shape param. Each entry in
        inv_mask corresponds to a random number in the final random number array. When the inv_mask entry is True, the
        corresponding shared random number is inverted and when inv_mask entry is False, the corresponding shared random
        number is left unchanged.
        :return: the random number array whose shape is determined by the N and shape params: (*shape, N).
        """
        assert not full_corr or inv_mask is None  # full_corr and inv_mask cannot be used together
        if inv_mask is not None or full_corr:
            assert share, f"Share needs to be true if inv_mask:{inv_mask} is not none if full_corr:{full_corr} is True"
        # assert share, "Need to double check shapes on non-share, but too lazy to do right now" # TODO

        # figure out how many times the RNS needs to repeat itself.
        # E.g. a 4-bit LFSR RNS with sequence length 15 needs to repeat its sequence thrice when SN length, N, is 32.
        repeats = math.ceil(N / self._max_N)


        # generate random numbers according to the RNS being used.
        if repeats > 1:
            assert False, "I haven't checked repeating RNS. This is probably just useful for LFSR class."
            # Rs = self._gen_RN(self._max_N, shape, share)
        else:
            Rs = self._gen_RN(N, shape, share)

        if full_corr:
            Rs = Rs.repeat(2, 1)
            Rs[1] = int(2**self.n) - 1 - Rs[1]
            Rs = Rs.expand(*shape, 2, N)
        elif share:
            Rs = Rs.expand(*shape, N)

        if inv_mask is not None:
            assert inv_mask.shape == shape, f"Mask shape: {inv_mask.shape} and given input shape: {shape} must match."
            # inverting all RNS bits is the same as doing Rs=(2^n -1-Rs)
            Rs = Rs.clone()  # need to fix underlying view mechanic :o
            Rs[inv_mask] = int(2**self.n) - 1 - Rs[inv_mask]


        # repeat the RNS if necessary (see first comment for what it means for the RNS to repeat itself).
        if repeats > 1:
            raise NotImplementedError
            Rs = Rs.repeat(repeats)[..., 0:N]
            Rs = np.tile(A=Rs, reps=repeats)[..., 0:N]

        return Rs

    def __str__(self) -> str:
        return self.name

    def info(self) -> str:
        """
        returns a string with basic information about the RNS.
        :return:
        """
        return f"### RNS Info: name={self.name} n={self._n} max_N={self._max_N}"

    @property
    def n(self):
        return self._n

    @property
    def max_N(self):
        return self._max_N


class Bernoulli_RNS(RNS):
    """
    Software implemented Bernoulli-type RNS. This RNS is mainly used to check theoretical derivations made using the
    Bernoulli model of SNs. It relies on NumPy's np.random.randint method whose RN quality is sufficiently high to
    imitate independent random numbers.
    """
    def __init__(self, precision, device):
        """
        :param precision: the precision, in bits, of the random numbers generated.
        """
        super().__init__(precision, device)
        self.name = "Bernoulli"
        self.legend = "Bern"  # string used in plot legends
        self.hardware = False

    def _gen_RN(self, N, shape, share):
        if share:
            Rs = torch.randint(low=0, high=int(2**self.n), size=(N,), device=self.device, dtype=torch.int16)
        else:
            Rs = torch.randint(low=0, high=int(2**self.n), size=(*shape, N), device=self.device, dtype=torch.int16)

        return Rs


class Hypergeometric_RNS(RNS):
    """
    Software implemented Hypergeometric-type RNS. This RNS is mainly used to check theoretical derivations made using the
    Hypergeometric model of SNs. It relies on NumPy's torch.randperm method whose RN quality is sufficiently high to
    imitate hypergeometric random numbers.
    """
    def __init__(self, precision, device):
        super().__init__(precision, device)
        self.name = "Hypergeometric"
        self.legend = "Hyper"
        self.is_hardware = False
        self._max_N = int(2**self.n)

    def _gen_RN(self, N, shape, share):
        assert N <= self._max_N
        if share:
            Rs = torch.randperm(self._max_N, device=self.device, dtype=torch.int16)
        else:
            Rs = torch.vstack([torch.randperm(self._max_N, device=self.device, dtype=torch.int16) for _ in range(math.prod(shape))]).reshape((*shape, N))
            # raise NotImplementedError
            # Rs = np.array([np.random.permutation(self._max_N)[:N] for _ in range(math.prod(shape))]).reshape((*shape, N))

        return Rs[..., 0:N]


class FSR_RNS(RNS):
    # hardware linear (or non-linear) feedback shift register RNG
    def __init__(self, precision, device, nonlinear, **kwargs):
        super().__init__(precision, device)

        self.nonlinear = nonlinear  # True if you want LFSR, False if you want NLFSR
        self.feedback = kwargs.get("feedback")  # FSR feedback type ('i'nternal, 'e'xternal or 'a'll)
        self.seq_idxs = kwargs.get("seq_idx")  # idx of seq from seqs to use
        self.seqs = kwargs.get("seqs")

        if self.feedback is None:
            print("Warning: No LFSR feedback given, using 'e' for external feedback")
            self.feedback = 'e'

        if self.seqs is None:  # load seqs if they were not given
            self.seqs = seq_utils.get_LFSR_seqs(precision, self.feedback, extended=self.nonlinear, verbose=True)

        self.name = "NLFSR" if self.nonlinear else "LFSR"
        self.legend = "NLFSR" if self.nonlinear else "LFSR"
        self.is_hardware = True
        self._max_N = int(2**precision) if self.nonlinear else int(2**precision) - 1
        assert (not self.nonlinear) or self.feedback == 'e', "NLFSRs are only implemented for external feedback"

    def _gen_RN(self, N, shape, share, verbose=True):
        assert N <= self._max_N
        pow2n = int(2**self._n)
        # Rs = torch.empty((runs, *shape, N))
        if share:
            seq_idx = torch.randint(len(self.seqs), (1,))
            start_idx = torch.randint(self.max_N, (1,))
            Rs = torch.roll(self.seqs[seq_idx], (-start_idx,))
        else:
            assert self.seq_idxs is None, "LFSR no-share when given seq_idxs is probably a bad idea"
            numel = math.prod(shape)

            # pick random sequences:
            seq_idxs = torch.randint(0, len(self.seqs), shape)  # (shape,)
            Rs = self.seqs[seq_idxs]  # (*shape, L)
            # pick random start states
            starts = torch.randint(0, len(self.seqs[0]), (numel,))  # (numel,)
            Rs = Rs.view((numel, N))
            for idx in range(numel):
                Rs[idx] = torch.roll(Rs[idx], (starts[idx].item(),))
            Rs = Rs.view((*shape, N))

        Rs = Rs.to(device=self.device)
        return Rs


class Counter_RNS(RNS):
    """
    RNS class for a counter random number source.
    """
    def __init__(self, precision, device, **kwargs):
        super().__init__(precision, device)
        pow2n = int(2 ** precision)
        self.name = "Counter"
        self.legend = "Counter"
        self.is_hardware = True
        self._max_N = pow2n
        self.seq = torch.arange(pow2n, device=device, dtype=torch.int16)

    def _gen_RN(self, N, shape, share):
        assert N <= self._max_N
        assert share or sum(shape) == 1

        return self.seq[0:N]


class VDC_RNS(RNS):
    """
    RNS class for a Van der Corput low discrepancy sequence source.
    """
    def __init__(self, precision, device, **kwargs):
        super().__init__(precision, device)

        pow2n = int(2 ** precision)

        self.name = "VDC"
        self.legend = "VDC"
        self.is_hardware = True
        self._max_N = pow2n
        self.seq = kwargs.get('vdc_seq')
        verbose = kwargs.get('verbose', True)
        if self.seq is None:
            self.seq = seq_utils.get_vdc_torch(precision, device, verbose=verbose).short()
        else:
            assert self.seq.device == self.device

    def _gen_RN(self, N, shape, share):
        assert N <= self._max_N
        assert share or sum(shape) == 1

        return self.seq[0:N]


class SOBOL_RNS(RNS):
    """
    RNS class for a Sobol low discrepancy sequence source.
    """
    def __init__(self, precision, device, **kwargs):
        super().__init__(precision, device)

        pow2n = int(2**precision)

        self.name = "Sobol"
        self.legend = "Sobol"
        self.is_hardware = True
        self._max_N = pow2n
        self.seqs = kwargs.get('seqs')
        if self.seqs is None:
            self.seqs = seq_utils.get_Sobol_seqs(precision, verbose=True)

        self.seqs = torch.tensor(self.seqs, device=device)
        self.seq_idxs = torch.tensor(kwargs.get('seq_idxs'), device=device)

    def _gen_RN(self, N, shape, share):
        shape_size = math.prod(shape)
        assert N <= self._max_N
        assert share or shape_size <= len(self.seqs)

        if share:
            # pick a random sequence if one wasn't given
            seq_idx = np.random.randint(len(self.seqs)) if self.seq_idxs is None else self.seq_idxs[0]
            Rs = self.seqs[seq_idx]
        else:
            seq_idxs = np.random.permutation(shape_size)[:shape_size] if self.seq_idxs is None else self.seq_idxs[:shape_size]
            Rs = self.seqs[seq_idxs].reshape(*shape, N)

        return Rs


class SnF_SOBOL_RNS(SOBOL_RNS):
    def __init__(self, precision, device, **kwargs):
        super().__init__(precision, device, **kwargs)
        self.seed = kwargs.get('seed')
        self.reuse = kwargs.get('reuse')
        self.shuffle = kwargs.get('shuffle')
        self.flip = kwargs.get('flip')
        self.rng = np.random.default_rng(self.seed)
        self.first = True
        self.seqs = self.seqs.cpu().numpy()

        assert precision <= 16

    def _gen_RN(self, N, shape, share):
        shape_size = math.prod(shape)
        assert N <= self._max_N

        # pick a random sequence if one wasn't given
        seq_idx = np.random.randint(len(self.seqs)) if self.seq_idxs is None else self.seq_idxs[0]

        # get Rs (can just return Rs if share is used
        Rs = self.seqs[seq_idx]  # shape: (pow2n)
        if not share:  # use perms and stuff to create new sequences
            Rs = Rs[None].repeat(shape_size, axis=0)  # shape (num_seqs, pow2n)
            if self.reuse and not self.first:
                assert shape_size == self.prev_shape_size
                Rs = self.prev_Rs
            else:
                # np.unpackbits and np.packbits assume 8-bit precision. Can modify to work with up to 16-bit prec.
                upper = np.right_shift(Rs, 8).astype(np.uint8)
                lower = Rs.astype(np.uint8)
                assert (Rs == (upper*256 + lower)).all()

                upper_bits, lower_bits = np.unpackbits(upper[None], axis=0), np.unpackbits(lower[None], axis=0)
                combined_bits = np.append(upper_bits, lower_bits, axis=0)  # combine bits into one array (16, num_seqs, pow2n)
                leading_zeros = combined_bits[:-self.n]  # ignore leading zeros shape: (16-n, num_seqs, pow2n)
                Rs_bits = combined_bits[-self.n:]  # take only n bits instead of all 16.  (n, num_seqs, pow2n)
                # Rs_bits = np.unpackbits(Rs[None], axis=0)  # shape: (n, num_seqs, pow2n)

                perms, invs = None, None
                if self.shuffle:
                    perms = self.rng.permuted(np.arange(self.n)[None].repeat(shape_size, axis=0), axis=1).T  # (prec, num)
                    # Rs_bits = np.take_along_axis(Rs_bits, perms[..., None], axis=0)  # apply perms to bits axis
                    Rs_bits = np.take_along_axis(Rs_bits, perms[..., None], axis=0)  # apply perms to bits axis

                if self.flip:
                    invs = np.zeros(shape=(self.n, shape_size))
                    invs[-self.n:] = self.rng.integers(low=0, high=2, size=(self.n, shape_size), dtype=bool)  # (prec, num)
                    Rs_bits = np.where(invs[..., None], 1 - Rs_bits, Rs_bits)  # apply invs to bits axis

                bits = np.append(leading_zeros, Rs_bits, axis=0)
                Rs = np.packbits(bits, axis=0).astype(int)   # shape: (2, num_seqs, pow2n)
                Rs = Rs[0]*256 + Rs[1]
                Rs = Rs.reshape(*shape, N)

                self.prev_shape_size = shape_size
                self.prev_Rs = Rs
                self.prev_perms, self.prev_invs = perms, invs
                self.first = False

        return torch.tensor(Rs, device=self.device)


class LFSR_RNS(FSR_RNS):
    def __init__(self, precision, device, **kwargs):
        kwargs['nonlinear'] = False
        super().__init__(precision, device, **kwargs)


class NLFSR_RNS(FSR_RNS):
    def __init__(self, precision, device, **kwargs):
        kwargs['nonlinear'] = True
        super().__init__(precision, device, **kwargs)


def get_rns_class_from_name(name):
    for cls in RNS.__subclasses__():
        if cls.__name__ == name:
            return cls
    for cls in FSR_RNS.__subclasses__():
        if cls.__name__ == name:
            return cls
    for cls in SOBOL_RNS.__subclasses__():
        if cls.__name__ == name:
            return cls

if __name__ == '__main__':
    n = 8
    N = 1000
    shape = (100, 100)
    Bs = torch.randint(0, int(2**n), shape).cuda()
    Bs_short = Bs.short().cuda()


    import time
    ### CUDA TIME TEST
    start = time.time()
    for i in range(10000):
        Rs = torch.cuda.IntTensor(N).random_(0, int(2 ** n))
        Rs = Rs.repeat((*shape, 1))
        Xs = (Rs < Bs[..., None]).char()
    end = time.time()
    print(f"torch.cuda.IntTensor with repeat runtime:{end-start}\n")
    del Rs

    device = torch.device('cuda')
    start = time.time()
    for i in range(10000):
        Rs = torch.randint(low=0, high=int(2**n), size=(N,), device=device)
        Rs = Rs.repeat((*shape, 1))
        Xs = (Rs < Bs[..., None]).char()
    end = time.time()
    print(f"torch.randint(..., device='cuda') with repeat runtime:{end-start}\n")
    del Rs

    start = time.time()
    for i in range(100):
        Rs = torch.randint(low=0, high=int(2**n), size=(N,))
        Rs = Rs.repeat((*shape, 1)).cuda()
        Xs = (Rs < Bs[..., None]).char()
    end = time.time()
    print(f"torch.randint->repeat->cuda runtime: {(end-start)*10000/100}\n")

    ### Testing expand now
    start = time.time()
    for i in range(10000):
        Rs = torch.cuda.IntTensor(N).random_(0, int(2 ** n))
        Rs = Rs.expand((*shape, N))
        Xs = (Rs < Bs[..., None]).char()
    end = time.time()
    print(end-start)
    print(f"torch.cuda.IntTensor with expand runtime:{end-start}\n")
    del Rs

    device = torch.device('cuda')
    start = time.time()
    for i in range(10000):
        Rs = torch.randint(low=0, high=int(2**n), size=(N,), device=device)
        Rs = Rs.expand((*shape, N))
        Xs = (Rs < Bs[..., None]).char()
    end = time.time()
    print(f"torch.randint(..., device='cuda') with expand runtime:{end-start}\n")
    del Rs

    # time test for int16 (all previous are for 64-bit integers)
    device = torch.device('cuda')
    start = time.time()
    for i in range(10000):
        Rs = torch.randint(low=0, high=int(2**n), size=(N,), device=device, dtype=torch.int16)
        Rs = Rs.expand((*shape, N))
        Xs = (Rs < Bs_short[..., None]).char()
    end = time.time()
    print(f"torch.randint(..., device='cuda') with expand runtime and 16-bit ints:{end-start}\n")
    del Rs
