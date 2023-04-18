from SCython.Circuits import stream_adders
from SCython.SNG import RNS,PCC,MSG
from SCython.Utilities.verilog_module_generator import *
import SCython.Utilities.input_value_chooser as IVC

#################################
#   CeMux/CeMaj Related Files   #
#################################
def gen_verilog_cemux_file(precision, weights, bipolar=True, pcc_array_ID="", tree_ID="", output_ID="", latency_factor=None):
    cemux = stream_adders.CeMux(tree_height=latency_factor, weights=weights, data_pcc=PCC.Comparator(precision), bipolar=bipolar)
    invert_list = cemux.quant_norm_weights[cemux.quant_norm_weights != 0] < 0  # list for which inputs have neg weights
    pcc_input_size = len(invert_list)

    # begin the file
    file_string = ""
    file_string += "`timescale 1 ns / 1 ns\n\n"
    # generate verilog code for the filter's memory
    file_string += gen_verilog_filter_memory(cemux.quant_norm_weights, pcc_precision=precision,
                                             rns_precision=latency_factor, gated=True)

    # generate verilog code for a counter module that is used as the VDC RNS and for the mux select input
    file_string += gen_verilog_counter_with_rev(latency_factor)

    # generate verilog for a comparator module and for a comparator module that inverts its output.
    file_string += gen_verilog_comparator(precision, bipolar, invert_output=False)
    file_string += gen_verilog_comparator(precision, bipolar, invert_output=True)

    # generate verilog for an array of comparators used to generate CeMux's inputs.
    file_string += gen_verilog_comparator_array(precision, share_r=True, invert_list=invert_list, full_correlation=True,
                                                ID=pcc_array_ID)

    # generate verilog for cemux's hardwired tree
    file_string += gen_verilog_hardwired_mux_tree(cemux.quant_norm_weights, cemux.vhdl_wire_map,
                                                  tree_height=latency_factor, ID=tree_ID)

    # generate verilog for the output counter
    file_string += gen_verilog_output_counter(latency_factor, bipolar, ID=output_ID)

    # generate verilog for the filtercore and top-level CeMux module
    file_string += gen_verilog_cemux_filtercore(precision, pcc_input_size, pcc_array_ID, tree_ID, output_ID, latency_factor)
    file_string += gen_verilog_cemux_filter_toplevel(precision, pcc_input_size, latency_factor=latency_factor)
    return file_string


def gen_verilog_cemaj_file(precision, weights, bipolar=True, pcc_array_ID="", tree_ID="", output_ID="", latency_factor=None):
    latency_factor = precision if latency_factor is None else latency_factor
    assert latency_factor >= precision, "Latency factor must be >= precision"
    cemux = stream_adders.CeMux(tree_height=latency_factor, weights=weights, data_pcc=PCC.Comparator(precision), bipolar=bipolar)
    invert_list = cemux.quant_norm_weights[cemux.quant_norm_weights != 0] < 0  # list for which inputs have neg weights
    pcc_input_size = len(invert_list)

    # begin the file
    file_string = ""
    file_string += "`timescale 1 ns / 1 ns\n\n"
    # generate verilog code for the filter's memory
    file_string += gen_verilog_filter_memory(cemux.quant_norm_weights, pcc_precision=precision,
                                             rns_precision=latency_factor, gated=True)

    # generate verilog code for a counter module that is used as the VDC RNS and for the mux select input
    file_string += gen_verilog_counter_with_rev(latency_factor)

    # generate verilog for WBG P1, WBG P2 and for WBG P2 that inverts the output bit.
    file_string += gen_verilog_wbg_p1(precision)
    file_string += gen_verilog_wbg_p2(precision, bipolar, invert_output=False)
    file_string += gen_verilog_wbg_p2(precision, bipolar, invert_output=True)

    # generate verilog for an array of comparators used to generate CeMux's inputs.
    file_string += gen_verilog_wbg_array(precision, share_r=True, invert_list=invert_list, full_correlation=True,
                                                ID=pcc_array_ID)

    # generate verilog for cemux's hardwired tree
    file_string += gen_verilog_hardwired_maj_tree(cemux.quant_norm_weights, cemux.vhdl_wire_map,
                                                  tree_height=latency_factor, ID=tree_ID)

    # generate verilog for the output counter
    file_string += gen_verilog_output_counter(latency_factor, bipolar, ID=output_ID)

    # generate verilog for the filtercore and top-level CeMux module
    file_string += gen_verilog_cemaj_filtercore(precision, pcc_input_size, pcc_array_ID, tree_ID, output_ID, latency_factor)
    file_string += gen_verilog_cemux_filter_toplevel(precision, pcc_input_size, latency_factor)
    return file_string


# below method handles custom PCCs
def gen_verilog_cemux_mmc_filter_file(precision, weights, num_maj, explicit, bipolar=True,
                                      pcc_array_ID="", tree_ID="", output_ID="", latency_factor=None):
    # explicit=True means to explicitly use WBG and CMPs if num_maj = 0 or n-1. False means to always use MMC

    latency_factor = precision if latency_factor is None else latency_factor
    tree_height = latency_factor
    num_mux = precision - num_maj
    data_rns, data_pcc = RNS.VDC_RNS(precision,verbose=False), PCC.Mux_maj_chain(precision, num_mux, num_maj)
    share_r, full_corr = True, True
    msg = MSG.Counter_MSG(precision, latency_factor, weights)

    adder = stream_adders.HardwiredMux(tree_height, weights, data_rns, data_pcc, share_r, msg,full_corr, bipolar)
    invert_list = adder.quant_norm_weights[adder.quant_norm_weights != 0] < 0  # list for which inputs have neg weights
    pcc_input_size = len(invert_list)

    # begin the file
    file_string = ""
    file_string += "`timescale 1 ns / 1 ns\n\n"
    # generate verilog code for the filter's memory
    file_string += gen_verilog_filter_memory(adder.quant_norm_weights, pcc_precision=precision,
                                             rns_precision=latency_factor, gated=True)

    # generate verilog code for a counter module that is used as the VDC RNS and for the mux select input
    file_string += gen_verilog_counter_with_rev(latency_factor)

    # generate verilog for a PCC module, for a PCC module that inverts its output and for an array of modules
    if explicit and num_maj == 0:  # its a WBG
        file_string += gen_verilog_wbg_p1(precision)
        file_string += gen_verilog_wbg_p2(precision, bipolar, invert_output=False)
        file_string += gen_verilog_wbg_p2(precision, bipolar, invert_output=True)
        file_string += gen_verilog_wbg_array(precision, share_r, invert_list, full_corr, ID=pcc_array_ID)
        pcc_array_name = "compara_array"
    elif explicit and (num_maj == precision - 1 or num_maj == precision == num_maj): # it's a CMP
        file_string += gen_verilog_comparator(precision, bipolar, invert_output=False)
        file_string += gen_verilog_comparator(precision, bipolar, invert_output=True)
        file_string += gen_verilog_comparator_array(precision, share_r, invert_list, full_corr, ID=pcc_array_ID)
        pcc_array_name = "wbg_array"
    else:
        file_string += gen_verilog_mmc_chain(precision, num_mux, bipolar, invert_output=False)
        file_string += gen_verilog_mmc_chain(precision, num_mux, bipolar, invert_output=True)
        file_string += gen_verilog_mmc_chain_array(precision, share_r, invert_list, full_corr, ID=pcc_array_ID)
        pcc_array_name = "mmc_chain_array"

    # generate verilog for cemux's hardwired tree
    file_string += gen_verilog_hardwired_mux_tree(adder.quant_norm_weights, adder.vhdl_wire_map, tree_height, ID=tree_ID)

    # generate verilog for the output counter
    file_string += gen_verilog_output_counter(latency_factor, bipolar, ID=output_ID)

    # generate verilog for the filtercore and top-level CeMux module
    file_string += gen_verilog_cemux_filtercore(precision, pcc_input_size, pcc_array_ID, tree_ID, output_ID,
                                                latency_factor, pcc_array_name=pcc_array_name)
    file_string += gen_verilog_cemux_filter_toplevel(precision, pcc_input_size, latency_factor=latency_factor)
    return file_string


#################################
#    PCC (MMC) Related Files    #
#################################
def gen_verilog_mmc_file(pcc_precision, num_mux, num_SNs, bipolar=True):
    invert_list = np.zeros(num_SNs)

    # begin the file
    file_string = ""
    file_string += "`timescale 1 ns / 1 ns\n\n"

    # whether we use a CMP, WBG or a MMC, we must generate two modules: one that outputs SN bit and one that outputs inverted SN bit
    if 0 <= num_mux <= 1:  # if the file uses only MAJ, then it is a comparator
        file_string += gen_verilog_comparator(pcc_precision, bipolar, invert_output=False)
        file_string += gen_verilog_comparator(pcc_precision, bipolar, invert_output=True)

    elif 1 < num_mux < pcc_precision:  # if the file uses mux and maj chains, then generate the mcc_p2
        file_string += gen_verilog_wbg_p1(precision=num_mux)
        file_string += gen_verilog_wbg_p2(precision=num_mux, bipolar=False, invert_output=False)
        file_string += gen_verilog_mmc_p2(pcc_precision, num_mux, bipolar, invert_output=False)
        file_string += gen_verilog_mmc_p2(pcc_precision, num_mux, bipolar, invert_output=True)

    elif num_mux == pcc_precision:  # if the file uses only MUX, then it is a WBG
        file_string += gen_verilog_wbg_p1(pcc_precision)
        file_string += gen_verilog_wbg_p2(pcc_precision, bipolar, invert_output=False)
        file_string += gen_verilog_wbg_p2(pcc_precision, bipolar, invert_output=True)

    # generate verilog for an array of MMCs used to generate the input SNs
    file_string += gen_verilog_mmc_array(pcc_precision, num_mux, share_r=True, invert_list=invert_list, full_correlation=True, ID="")

    return file_string

def gen_verilog_mmc_chain_file(pcc_precision, num_mux, num_SNs, bipolar=True):
    invert_list = np.zeros(num_SNs)

    # begin the file
    file_string = ""
    file_string += "`timescale 1 ns / 1 ns\n\n"

    file_string += gen_verilog_mmc_chain(pcc_precision, num_mux, bipolar, invert_output=False)
    file_string += gen_verilog_mmc_chain(pcc_precision, num_mux, bipolar, invert_output=True)

    # generate verilog for an array of MMCs used to generate the input SNs
    file_string += gen_verilog_mmc_chain_array(pcc_precision, share_r=True, invert_list=invert_list, full_correlation=True)

    return file_string


#################################
#         Main Methods          #
#################################
def exp1_MMCs():
    pcc_precision = 6
    num_mux = 2
    num_SNs = 1

    # print(gen_verilog_mmc_file(pcc_precision, num_mux, num_SNs))
    print(gen_verilog_mmc_chain_file(pcc_precision, num_mux, num_SNs))

def exp2_cemux_ecg():
    precision = 8
    num_weights = 100
    bipolar = True

    explicit = True
    num_maj = precision//2

    weight_mode, weight_data = IVC.WEIGHT_MODE.DATA, IVC.WEIGHT_SET.ECG_1
    weights = IVC.choose_mux_weights(num_weights, weight_mode, weight_set=weight_data)
    print(gen_verilog_cemux_mmc_filter_file(precision, weights, num_maj, explicit, bipolar))

if __name__ == '__main__':
    exp2_cemux_ecg()