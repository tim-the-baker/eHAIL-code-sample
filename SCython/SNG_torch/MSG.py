import torch
import math
from SCython.SNG_torch import RNS, PCC, SNG

class EXACT_MSG:
    """ Abstract class for exact mux select input generators (MSGs). """
    def __init__(self, num_inputs, device, is_precise):
        """
        :param int num_inputs:
        """
        self.num_inputs = num_inputs
        self.device = device
        self.is_precise = is_precise
        self.seq = None

    def gen_selects_NN(self, batch_size, num_outputs, SN_length):
        """
        Abstract method for generating a correlated set of mux select inputs based on SN length
        """
        raise NotImplementedError

    def gen_selects(self, SN_length, shape, share):
        """
        Abstract method for generating a correlated set of mux select inputs based on SN length
        """
        raise NotImplementedError

    @staticmethod
    def must_share():
        raise NotImplementedError


class HYPER_EXACT_MSG(EXACT_MSG):
    def __init__(self, num_inputs, device):
        super().__init__(num_inputs, device, is_precise=True)

    def gen_selects_NN(self, batch_size, num_outputs, SN_length):
        num_repeats = math.ceil(SN_length / self.num_inputs)
        seqs = torch.rand(self.num_inputs, num_outputs, device=self.device).argsort(dim=0)  # (S, M)
        seqs = seqs.repeat(num_repeats, 1)  # (>L, M)
        seqs = seqs[None, 0:SN_length, :, None].expand(batch_size, SN_length, num_outputs, 1) # (N, L, M, 1)

        return seqs

    def gen_selects(self, SN_length, shape, share):
        num_repeats = math.ceil(SN_length / self.num_inputs)
        if share:
            seqs = torch.randperm(self.num_inputs, device=self.device)
            seqs = seqs.repeat(num_repeats)[0:SN_length]
            seqs = seqs.expand(*shape, SN_length)
        else:
            seqs = torch.rand(*shape, self.num_inputs, device=self.device).argsort(dim=-1)
            helper = tuple([1 for _ in range(len(shape))])
            seqs = seqs.repeat(*helper, num_repeats)[..., 0:SN_length]

        return seqs

    @staticmethod
    def is_precise():
        return True

    @staticmethod
    def must_share():
        return False


class BERN_MSG(EXACT_MSG):
    def __init__(self, num_inputs, device):
        super().__init__(num_inputs, device, is_precise=False)

    def gen_selects_NN(self, batch_size, num_outputs, SN_length):
        seqs = torch.randint(low=0, high=self.num_inputs, size=(1, SN_length, num_outputs, 1), device=self.device)
        seqs = seqs.expand(batch_size, SN_length, num_outputs, 1)

        return seqs

    def gen_selects(self, SN_length, shape, share):
        if share:
            seqs = torch.randint(0, self.num_inputs, size=SN_length, device=self.device)
            seqs = seqs.expand((*shape, SN_length))
        else:
            seqs = torch.randint(0, self.num_inputs, size=(*shape, SN_length), device=self.device)

        return seqs

    @staticmethod
    def is_precise():
        return False

    @staticmethod
    def must_share():
        return False


class COUNTER_EXACT_MSG(EXACT_MSG):
    def __init__(self, num_inputs, device):
        super().__init__(num_inputs, device, is_precise=True)

    def gen_selects_NN(self, batch_size, num_outputs, SN_length):
        num_repeats = math.ceil(SN_length / self.num_inputs)
        seqs = torch.arange(0, self.num_inputs, step=1, device=self.device)  # (S,)
        seqs = seqs.repeat(num_repeats)
        seqs = seqs[None, 0:SN_length, None, None].expand(batch_size, SN_length, num_outputs, 1) # (N, L, M, 1)

        return seqs

    def gen_selects(self, SN_length, shape, share):
        assert share, "Only sharing makes sense with counter MSGs"
        num_repeats = math.ceil(SN_length / self.num_inputs)
        seqs = torch.arange(0, self.num_inputs, step=1, device=self.device)  # (S,)
        seqs = seqs.repeat(num_repeats)
        seqs = seqs[0:SN_length]
        seqs = seqs.expand(*shape, SN_length)

        return seqs

    @staticmethod
    def is_precise():
        return True

    @staticmethod
    def must_share():
        return True


class Long_Counter_Exact_MSG(EXACT_MSG):
    def __init__(self, num_inputs, device):
        super().__init__(num_inputs, device, is_precise=True)
        self.prev_SN_length = 0

    def gen_selects_NN(self, batch_size, num_outputs, SN_length):
        # seqs = self.gen_selects(SN_length, shape=(SN_length,), share=True)
        seqs = self.gen_selects(SN_length, shape=None, share=True)
        seqs = seqs[None, 0:SN_length, None, None].expand(batch_size, SN_length, num_outputs, 1) # (N, L, M, 1)

        return seqs

    def gen_selects(self, SN_length, shape, share):
        assert share, "Only sharing makes sense with counter MSGs"
        if self.prev_SN_length == SN_length:
            seqs = self.prev_seqs

        else:
            min_repeats = math.floor(SN_length / self.num_inputs)
            extra = SN_length - min_repeats*self.num_inputs

            if extra == 0:
                seqs = torch.arange(0, self.num_inputs, step=1, device=self.device)  # (S,)
                seqs = seqs.repeat_interleave(min_repeats)
            else:
                seq1 = [idx for idx in range(extra) for _ in range(min_repeats+1)]
                seq2 = [idx for idx in range(extra, self.num_inputs) for _ in range(min_repeats)]
                seqs = torch.tensor([*seq1, *seq2], device=self.device)
            assert len(seqs) == SN_length

            self.prev_SN_length = SN_length
            self.prev_seqs = seqs

        if shape is not None:
            seqs = seqs.expand(*shape, SN_length)

        return seqs

    @staticmethod
    def is_precise():
        return True

    @staticmethod
    def must_share():
        return True


class Sep_FSR_MSG(EXACT_MSG):
    def __init__(self, num_inputs, device, precision, nonlinear=True):
        assert math.log2(num_inputs) % 1 == 0, "FSR MSG only works with power-of-two inputs bc hardwired tree assumed"
        self.num_rns = int(math.log2(num_inputs))

        rns = RNS.FSR_RNS(precision, device, nonlinear=nonlinear, share_batch=True)
        pcc = PCC.Comparator(precision)
        self.sng = SNG.SNG(rns, pcc)

        self.precision = precision
        self.shifter = torch.arange(0, self.num_rns, device=device)[:, None]  # shape:(m, 1)
        self.max_L = int(2**precision) - 1 + int(nonlinear)
        super().__init__(num_inputs, device, is_precise=False)

    def gen_selects_NN(self, batch_size, num_outputs, SN_length):
        raise NotImplementedError

    def gen_selects(self, SN_length, shape, share):
        # create SNs
        if share:
            values = torch.full((self.num_rns,), 0.5, device=self.device) # generate one set of select inputs  # (1, L)
        else:
            values = torch.full((*shape, self.num_rns), 0.5, device=self.device) # generate many sets of select inputs
        SNs = self.sng.gen_SN(values, SN_length, bipolar=False, share_RNS=False, RNS_mask=None, full_corr=False)  # (*shape, L)
        SNs = SNs.expand(*shape, self.num_rns, SN_length)  # expand only changes things for share=True
        # then convert those SNs into integer select values
        SNs = torch.bitwise_left_shift(SNs, self.shifter)
        SNs = torch.sum(SNs, dim=(-2,))
        return SNs

    @staticmethod
    def is_precise():
        return False

    @staticmethod
    def must_share():
        return False




def get_msg_class_from_name(name):
    for cls in EXACT_MSG.__subclasses__():
        if cls.__name__ == name:
            return cls


if __name__ == '__main__':
    # test NLFSR_Sep_SNs
    shape = (2, 3)
    share = False

    precision = 4
    SN_length = int(2**precision)
    num_inputs = SN_length
    device = torch.device('cpu')

    msg = Sep_FSR_MSG(num_inputs, precision, device, nonlinear=True)

    selects = msg.gen_selects(SN_length, shape, share)
    print(selects)