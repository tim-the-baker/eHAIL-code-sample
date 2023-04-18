import numpy as np
import scipy.io as sp_io
from SCython.SNG import SNG
from enum import Enum, IntEnum, auto

DATA_DIR = r"C:\Users\Tim\PycharmProjects\stochastic-computing\Data"

class INPUT_MODE(IntEnum):
    RAND = 0
    DATA = 1


class WEIGHT_MODE(IntEnum):
   UNIFORM = 0
   UNIFORM_SIGNED = 1
   RAND = 2
   RAND_SIGNED = 3
   DATA = 4


class WEIGHT_SET(Enum):
    ECG_1 = auto()  # coefs from Susan used in the CeMux papers.

    def get_filename(self):
        if self is WEIGHT_SET.ECG_1:
            return r"filter_coefs_ecg\blo_order_4_268.mat"


def choose_input_values(num_values, bipolar, mode, runs=1, seed=None, precision=None, data=None, start=None):
    """
    Method for choosing input values for simulation of a stochastic circuit.
    :param num_values: number of input values.
    :param bipolar: whether the input values are bipolar (bipolar=True) or unipolar (bipolar=False).
    :param mode: determines how input values are selected. Choices include:
    'rand' (or 0): values are chosen uniformly randomly.
    'data' (or 1) values are taken from a random subsequence of given data. requires use of data param.
    :param runs: TODO
    :param seed: optional parameter. If given, seed is passed to np.random.seed before executing the rest of this method.
    This parameter is used so that results can be replicated.
    :param precision: optional parameter. If given, the sampled input values will be quantized down (i.e., with the
    floor operation) to the given precision.
    :param data: required parameter when mode='data' (or mode=1). This is the data from which a random sequence will be
    sampled from.
    :param start: optional parameter when mode='data' (or mode=1). If specified, start will used as the beginning of the
    sampled sequence rather than a random start value. Only works when runs=1 for now.
    :return:
    """
    if seed is not None:
        np.random.seed(seed)
    if mode == INPUT_MODE.RAND:
        input_values = np.random.random_sample((runs, num_values))
        if bipolar:
            input_values = 2*input_values - 1

    elif mode == INPUT_MODE.DATA:
        # check the required data parameter
        assert data is not None, "Error: Using choose_input_values method with mode='data', but no data given"
        if bipolar:
            assert (-1 <= data).all() and (data <= 1).all(), "Error: values in data given to choose_input values do not fall in [-1,1]"
        else:
            assert 0 <= data <= 1, "Error: values in data given to choose_input values do not fall in [0,1]"

        # Pick a random subset of the data of length "num"
        input_values = np.empty((runs, num_values))
        for r_idx in range(runs):
            if start is None:
                assert runs == 1
                start = np.random.randint(len(data) - num_values)
            input_values[r_idx] = data[start:start+num_values]

    else:
        raise ValueError

    if precision is not None:  # quantize the inputs if necessary
        input_values = SNG.q_floor(input_values, precision, signed=bipolar)

    return input_values.squeeze()


def choose_mux_weights(num_weights, mode, runs=1, seed=None, weight_set=None):
    """
    Method for choosing mux input weights for the simulation of a mux adder.
    :param num_weights: number of weights (should match the mux adder's input size)
    :param mode: determines how the weights are selected. Options include:
    'uniform' (or 0): every coefficient is 1/M
    'uniform signed' (or 1): every coefficient is randomly set to +/- 1/M
    'random' (or 2): weights are random over interval [0, 1]
    'random signed' (or 3): weights are random over interval [-1, 1]
    'random norm' (or 4): weights are random over [0, 1] but then normalized to 1.
    'random norm signed' (or 5): weights are random over [-1, 1] but then normalized to 1.
    'data' (or 6): weights are loaded from data. requires use of data parameter.
    :param runs: TODO
    :param seed: optional parameter. If given, seed is passed to np.random.seed before executing the rest of this method.
    This parameter is used so that results can be replicated.
    :param weight_set: required parameter when mode = 'data' (or 6). weight_set refers to which stored weights
    :return:
    """
    if seed is not None:
        np.random.seed(seed)

    if mode == WEIGHT_MODE.UNIFORM:
        weights = np.ones((runs, num_weights))/num_weights
    elif mode == WEIGHT_MODE.UNIFORM_SIGNED:
        weights = ((-1) ** np.random.randint(0, 2, (runs, num_weights)))/num_weights
    elif mode == WEIGHT_MODE.RAND:
        weights = np.random.rand(runs, num_weights)
    elif mode == WEIGHT_MODE.RAND_SIGNED:
        weights = 2*np.random.rand(runs, num_weights) - 1
    elif mode == WEIGHT_MODE.DATA:
        if weight_set is WEIGHT_SET.ECG_1:
            assert 269 >= num_weights >= 5, f"{num_weights}: must be odd and at most 269"

            a = sp_io.loadmat(fr"{DATA_DIR}\{weight_set.get_filename()}")
            blo = a['blo']  # 2D array, row corresponds to order: 4, 6, ..., 62, 64.
            index = num_weights - 5
            weights = blo[index, 0:num_weights]
            assert (blo[index, num_weights:] == 0).all()  # Make sure we didn't mess up getting the coefficients
            weights = np.tile(weights, runs).reshape((runs, num_weights))
    else:
        raise NotImplementedError

    return weights.squeeze()
