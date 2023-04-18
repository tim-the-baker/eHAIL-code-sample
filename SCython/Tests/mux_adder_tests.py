import numpy as np
import matplotlib.pyplot as plt
from SCython.SNG import RNS, PCC
from SCython.Circuits import stream_adders
from SCython.Utilities import seq_utils, input_value_chooser
import SCython.Utilities.SN_operations as SN_ops

def get_CeMux_variants(version):
    # circuit = (RNS, PCC, use_ddg, symmetric, label)
    # Correlation Enhanced
    CeMux = (RNS.VDC_RNS, PCC.Comparator, False, False, "CeMux")
    CeMux_WBG = (RNS.VDC_RNS, PCC.WBG, False, False, "CeMux with WBGs")

    if version == 1:
        circuits = [CeMux, CeMux_WBG]

    return circuits


if __name__ == '__main__':
    version = 1
    precision = 8
    input_mode, weight_mode = 'rand', 'rand'
    num_inputs = [2, 4, 8, 16, 32, 64, 128, 256]
    bipolar = True
    runs = 500

    tree_height, vdc_n, pcc_n = precision, precision, precision
    SN_length = int(2 ** precision)
    circuits = get_CeMux_variants(version)
    vdc_seq = seq_utils.get_vdc(precision)

    Z_stars = np.zeros((len(circuits), len(num_inputs), runs))
    Z_hats = np.zeros((len(circuits), len(num_inputs), runs))
    for M_idx, num_input in enumerate(num_inputs):
        for r_idx in range(runs):
            input_values = input_value_chooser.choose_input_values(num_input, bipolar, input_mode, precision=precision)
            weights = input_value_chooser.choose_mux_weights(num_input, weight_mode)

            Z_stars[:, M_idx, r_idx] = np.inner(input_values, weights)/np.sum(np.abs(weights))
            for c_idx, (rns, pcc, use_ddg, symmetric, label) in enumerate(circuits):
                cemux = stream_adders.CeMux(tree_height, weights, pcc, use_ddg, bipolar, symmetric, vdc_n, pcc_n, vdc_seq)
                Z = cemux.forward(input_values, SN_length)
                Z_hat = SN_ops.get_SN_value(Z, bipolar)
                Z_hats[c_idx, M_idx, r_idx] = Z_hat

    errors = Z_hats - Z_stars
    MSEs = np.mean(np.square(errors), axis=-1)
    RMSEs = np.sqrt(MSEs)
    norm_RMSEs = RMSEs*np.sqrt(SN_length)

    for c_idx in range(len(circuits)):
        plt.plot(num_inputs, norm_RMSEs[c_idx])
    plt.show()
