# TODO: Account for coefficients who weights are zero. Important for when flattened synthesis is *not* used.
import numpy as np
from SCython.SNG import RNS, PCC
from SCython.Circuits import stream_adders
from SCython.Utilities import seq_utils

####################################
#   Random Number Source Modules   #
####################################
def gen_verilog_counter_with_rev(precision, ID=""):
    """
    This verilog module is a counter that increments by 1 each clock cycle. It has two outputs, the current counter state
    and the current counter state bitwise reversed. This module is useful in CeMux as it implements the mux select input
    generator and the VDC RNS in one.
    :param precision: bit-width of counter.
    :return: verilog code for the module in string form.
    """
    n = precision
    file_string = f"module counter{ID} (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\toutput logic [{n-1}:0] state,\n" \
                  f"\toutput logic [{n-1}:0] rev_state\n);\n" \
                  f"\talways_comb begin\n" \
                  f"\t\tfor (int i = 0; i < {n}; i+=1) begin\n" \
                  f"\t\t\trev_state[i] = state[{n-1}-i];\n" \
                  f"\t\tend\n" \
                  f"\tend\n\n" \
                  f"\talways_ff @(posedge clock) begin\n" \
                  f"\t\tif (reset == 1) state <= 'b0; else\n" \
                  f"\t\t                state <= state + 1;\n" \
                  f"\tend\n" \
                  f"endmodule\n\n\n"
    return file_string


def gen_verilog_counter_rns(precision, ID=""):
    """
    This verilog module is a counter that increments by 1 each clock cycle. It has one output, its current state. It is
    useful as a "random" number source.
    :param precision: bit-width of counter.
    :return: verilog code for the module in string form.
    """
    n = precision
    file_string = f"module counter{ID} (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\toutput logic [{n-1}:0] state\n);\n" \
                  f"\talways_ff @(posedge clock) begin\n" \
                  f"\t\tif (reset == 1) state <= 'b0; else\n" \
                  f"\t\t                state <= state + 1;\n" \
                  f"\tend\n" \
                  f"endmodule\n\n\n"
    return file_string


def gen_verilog_lfsr(precision, feedback, ID="", init_state="1'b1", feed_type='e'):
    '''
    generate verilog for a LFSR

    :param precision: LFSR bitwidth
    :param f:
    :param feedback: feedback polynomial given as a list of {0,1} taps (len = bitwdith).
    feedback[i] = 1 means that the i-th indexed bit is part of feedback. feedback[0] (LSB) is always part of feedback
    :param ID: unique identifier for LFSR
    :param init_state: initial state of LFSR (what it resets to)
    :param feed_type: internal vs. extrenal feedback
    :return:
    '''
    assert feed_type == 'e', "Only implemented external feedback for simplicity!"
    n = precision
    file_string = ""
    file_string += f"module lfsr{ID}(\n" \
                   f"\tinput                 clock, reset,\n" \
                   f"\toutput logic [{n-1}:0]    state\n);\n"

    # Feedback bit
    file_string += "\tlogic feedback_bit;\n"
    feed_eq = f"state[0]"  # the LSB is always part of feedback
    for j in range(1, n):
        if feedback[j] == '1':
            feed_eq += f" ^ state[{j}]"
    file_string += f"\tassign feedback_bit = {feed_eq};\n\n"

    file_string += f"\talways_ff @(posedge clock) begin\n" \
                   f"\t\tif (reset == 1) state      <= {init_state};\n" \
                   f"\t\telse begin\n" \
                   f"\t\t                state[{n-2}:0] <= state[{n-1}:1];\n" \
                   f"\t\t                state[{n-1}]   <= feedback_bit;\n" \
                   f"\t\tend\n" \
                   f"\tend\n" \
                   f"endmodule\n\n\n"

    return file_string


def gen_verilog_fsr(precision, feedback, nonlinear, ID="", init_state="1'b1", feed_type='e'):
    """
    :param precision: LFSR bitwidth
    :param feedback: feedback polynomial given as a list of {0,1} taps (len = bitwdith).
    feedback[i] = 1 means that the i-th indexed bit is part of feedback. feedback[0] (LSB) is always part of feedback
    :param nonlinear: whether to make a nonlinear (True) or linear (False) feedback shift register
    :param ID: unique identifier for FSR
    :param init_state: initial state of FSR (what it resets to)
    :param feed_type: internal vs. external feedback (only external is implemented)

    :return:
    """
    assert feed_type=='e', "Only implemented external feedback for simplicity!"
    name = "nlfsr" if nonlinear else "lfsr"
    n = precision
    file_string = ""

    # Initialize a new FSR module
    file_string += f"module {name}{ID}(\n" \
                   f"\tinput                 clock, reset,\n" \
                   f"\toutput logic [{n-1}:0]    state\n);\n"

    #  Initialize xor feedback bit for FSR
    file_string += "\tlogic xor_feedback;\n"

    # Compute the assignment equation for feedback_bit
    feed_eq = f'state[0]'  # the LSB is always part of the feedback
    for j in range(1, n):
        if feedback[j] == '1':
            feed_eq += f' ^ state[{j}]'
    file_string += f"\tassign xor_feedback = {feed_eq};\n\n"

    # If we are constructing an NLFSR, then we need extra logic for the OR gate that inserts the all 0 state
    if nonlinear:
        file_string += "\tlogic zero_detect;\n"
        # OR together all bits but the LSB (want to figure out if they are all zero)
        zero_eq = f"state[{n-1}]"
        for j in range(n-2, 0, -1):
            zero_eq += f" | state[{j}]"
        file_string += f"\tassign zero_detect = {zero_eq};\n\n"

    # Create variable for FSR's feedback bit
    file_string += "\tlogic feedback_bit;\n"
    if nonlinear:  # if NLFSR then we need the logic for dealing with the all-0 state insertion
        # example: 0001 maps to 0000 which then maps to 1000
        file_string += f"\tassign feedback_bit = zero_detect ? xor_feedback : ~state[0];\n\n"
    else:  # if LFSR, then the feedback is just the xor feedback bit
        file_string += "\tassign feedback_bit = xor_feedback;\n\n"

    # Initialize logic for the state register and end the module
    file_string += f"\talways_ff @(posedge clock) begin\n" \
                   f"\t\tif (reset == 1) state      <= #1 {init_state};\n" \
                   f"\t\telse begin\n" \
                   f"\t\t                state[{n-2}:0] <= #1 state[{n-1}:1];\n" \
                   f"\t\t                state[{n-1}]   <= #1 feedback_bit;\n" \
                   f"\t\tend\n" \
                   f"\tend\n" \
                   f"endmodule\n\n\n"

    return file_string


def gen_verilog_many_fsrs(precision, nonlinear, IDs="", feedbacks=None, init_states=None, feed_type='e'):
    assert feed_type == 'e', "Only implemented external feedback for simplicity!"
    n = precision
    if feedbacks is None:
        all_feeds = seq_utils.get_LFSR_feeds(n)
        feedbacks = np.random.permutation((len(all_feeds)))[0:len(IDs)]
    else:
        assert len(feedbacks) == len(IDs)
    if init_states is None:
        init_states = ["1'b1" for _ in range(len(IDs))]
        print("Note: Using default init_state of 1 for ALL LFSRs and NLFSRs")
    else:
        assert len(init_states) == len(IDs)

    print(f"Chosen Feeds: {feedbacks}")

    file_string = ""
    for i, ID in enumerate(IDs):
        file_string += gen_verilog_fsr(precision, feedbacks[i], nonlinear, ID, init_states[i], feed_type)

    return file_string


######################################
#   Probability Conversion Modules   #
######################################
def gen_verilog_comparator(precision, bipolar, invert_output):
    """
    Generates verilog code for a comparator module
    :param precision: bit-width of comparator PCC
    :param bipolar: whether this PCC generates a bipolar SN or not. If this PCC is used to generate a bipolar SN, then
    the MSB of the control input is inverted before comparison.
    :param invert_output: whether the output of the comparator should be inverted. This is useful in some MUX circuits.
    :return: verilog code for the module in string form.
    """
    n = precision
    module_name = "compara_inv" if invert_output else "compara"
    file_string = f"module {module_name} (\n" \
                  f"\tinput  logic [{n-1}:0] r,\n" \
                  f"\tinput  logic [{n-1}:0] p,\n" \
                  f"\toutput logic       SN_bit\n);\n"

    if bipolar:  # invert MSB of control input
        p_str = "p2"
        file_string += f"\tlogic [{n - 1}:0] p2;\n"
        file_string += f"\tassign p2[{n-1}] = ~p[{n-1}];\n" \
                       f"\tassign p2[{n-2}:0] = p[{n-2}:0];\n"
    else:
        p_str = "p"

    if invert_output:
        file_string += f"\tassign SN_bit = ~(r < {p_str});\n"
    else:
        file_string += f"\tassign SN_bit = r < {p_str};\n"

    file_string += f"endmodule\n\n\n"
    return file_string


def gen_verilog_comparator_array(precision, share_r, invert_list, full_correlation, ID=""):
    """
    Generates verilog code for an array of comparators.
    :param precision: bit-width of the comparators.
    :param share_r: Boolean value that is True when the comparators share a random number input (must be True for now)
    :param invert_list: list of Boolean values where invert_list[i]=True means that the i-th comparator should invert its
    output while invert_list[i]=False means that the i-th comparator should not invert its output.
    :param full_correlation: whether the SNs should be generate in a manner such that SCC(X1,X2) = +1 for all X1,X2. Only
    works when share_r is True.
    :param ID: identification number or string to append to the end of the module's name. This is used when you need more than
    one logically distinct comparator arrays in a single verilog file.
    :return: verilog code for the module in string form.
    """
    assert share_r, "Only sharing R is implemented at the moment."
    assert not full_correlation or share_r, "When full_correlation=True, share_r must also be True"
    n = precision
    M = len(invert_list)  # M is the number of control inputs to the comparator array and the number of outputs.

    file_string = f"module compara_array{ID} (\n" \
                  f"\tinput  logic [{n-1}:0] in [{M-1}:0],\n" \
                  f"\tinput  logic [{n-1}:0] r,\n" \
                  f"\toutput logic       SNs [{M-1}:0]\n);\n"

    if full_correlation:
        file_string += f"\tlogic [{n-1}:0] r_inv;\n" \
                       f"\tassign r_inv = ~r;\n\n"
        r_neg = "r_inv"
        spacing = " "*4
    else:
        r_neg = "r"
        spacing = ""

    for j in range(M):
        if invert_list[j] == 1:
            file_string += f"\tcompara_inv comp{j}(.r({r_neg}), .p(in[{j}]), .SN_bit(SNs[{j}]));\n"
        else:
            file_string += f"\tcompara     comp{j}(.r(r), {spacing}.p(in[{j}]), .SN_bit(SNs[{j}]));\n"

    file_string += f"endmodule\n\n\n"

    return file_string


def gen_verilog_wbg_p1(precision):
    n = precision
    file_string = f"module wbg_p1  (\n" \
                  f"\tinput  logic [{n-1}:0]     r,\n" \
                  f"\toutput logic [{n-1}:0]     out\n);\n"
    inverts = ""
    for idx in range(n-1, -1, -1):  # loop backwards
        file_string += f"\tassign out[{idx}] = r[{idx}]{inverts};\n"
        inverts += f" & ~r[{idx}]"
    file_string += "endmodule\n\n\n"
    return file_string


def gen_verilog_wbg_p2(precision, bipolar, invert_output):
    n = precision
    module_name = "wbg_p2_inv" if invert_output else "wbg_p2"
    file_string = f"module {module_name} (\n" \
                  f"\tinput  logic [{n-1}:0]     w,\n" \
                  f"\tinput  logic [{n-1}:0]     p,\n" \
                  f"\toutput logic           out\n);\n"

    if bipolar:  # invert MSB of control input
        p_str = "p2"
        file_string += f"\tlogic [{n - 1}:0] p2;\n"
        file_string += f"\tassign p2[{n-1}] = ~p[{n-1}];\n" \
                       f"\tassign p2[{n-2}:0] = p[{n-2}:0];\n"
    else:
        p_str = "p"

    file_string += f"\tlogic [{n-1}:0] u;\n"
    file_string += f"\tassign u = w & {p_str};\n"

    if invert_output:
        file_string += f"\tassign out = ~(|u);\n"
    else:
        file_string += f"\tassign out = |u;\n"

    file_string += f"endmodule\n\n\n"
    return file_string


def gen_verilog_wbg_array(precision, share_r, invert_list, full_correlation, ID=""):
    assert share_r, "Only sharing R is implemented at the moment."
    n = precision
    M = len(invert_list)

    file_string = f"module wbg_array{ID}  (\n" \
                  f"\tinput  logic [{n - 1}:0]     in  [{M-1}:0],\n" \
                  f"\tinput  logic [{n - 1}:0]     r,\n" \
                  f"\toutput logic           SNs  [{M-1}:0]\n);\n"

    file_string += f"\tlogic [{n-1}:0] w;\n"
    file_string += f"\twbg_p1 wbgp1(.r(r), .out(w));\n\n"

    if full_correlation:
        # Handle inverted WBG P1 with inverted R input if corr_adj is used
        file_string += f"\tlogic [{n-1}:0] w_neg;\n"
        file_string += f"\twbg_p1 wbgp1_inv(.r(~r), .out(w_neg));\n\n"
        w_neg = "w_neg"
        spacing = " " * 4
    else:
        w_neg = "w"
        spacing = ""

    for j in range(M):
        if invert_list[j] == 1:
            file_string += f"\twbg_p2_inv wbg{j}(.w({w_neg}), .p(in[{j}]), .out(SNs[{j}]));\n"
        else:
            file_string += f"\twbg_p2     wbg{j}(.w(w), {spacing}.p(in[{j}]), .out(SNs[{j}]));\n"

    file_string += f"endmodule\n\n\n"
    return file_string


def gen_verilog_mmc_p2(precision, num_mux, bipolar, invert_output):
    """
    Generates verilog code for a mux-maj-chain module. implements WBG_p2 and comparator part of mux-maj-chain
    :param precision: bit-width of comparator PCC
    :param bipolar: whether this PCC generates a bipolar SN or not. If this PCC is used to generate a bipolar SN, then
    the MSB of the control input is inverted before comparison.
    :param invert_output: whether the output of the comparator should be inverted. This is useful in some MUX circuits.
    :return: verilog code for the module in string form.
    """
    # set up useful parameters
    n = precision
    n_wbg = num_mux
    n_cmp = precision - num_mux
    file_string = ""
    inv_str = "_inv" if invert_output else ""
    module_name = f"mmc_p2{inv_str}"

    # write file
    file_string += f"module {module_name} (\n" \
                   f"\tinput  logic [{n_cmp-1}:0] r,\n"
    if n_wbg > 1:
        file_string += f"\tinput  logic [{n_wbg-1}:0] w,\n"
    file_string += f"\tinput  logic [{n-1}:0] p,\n" \
                   f"\toutput logic       SN_bit\n);\n"

    if bipolar:  # invert MSB of control input
        p_str = "p2"
        file_string += f"\tlogic [{n-1}:0] p2;\n"
        file_string += f"\tassign p2[{n-1}] = ~p[{n-1}];\n" \
                       f"\tassign p2[{n-2}:0] = p[{n-2}:0];\n"
    else:
        p_str = "p"

    # implement logic of mux-maj-chain
    # start with initializing intermediate variables
    file_string += "\tlogic wbg_out;\n"
    file_string += "\tlogic cmp_out;\n"

    # then initialize the wbg_p2
    file_string += f"\twbg_p2 wbg(.w(w), .p({p_str}[{n_wbg-1}:0]), .out(wbg_out));\n"

    # then implement the MAJ chain.... but how to do this??
    # first we can try the multiplexer approach
    file_string += f"\tassign cmp_out = wbg_out ? r <= {p_str}[{n-1}:{n_wbg}] : r < {p_str}[{n-1}:{n_wbg}];\n"

    # implement inversion if necessary
    if invert_output:
        file_string += f"\tassign SN_bit = ~(cmp_out);\n"
    else:
        file_string += f"\tassign SN_bit = cmp_out;\n"

    file_string += f"endmodule\n\n\n"
    return file_string


def gen_verilog_mmc_array(precision, num_mux, share_r, invert_list, full_correlation, ID=""):
    assert 0 <= num_mux <= precision
    assert share_r, "Only sharing R is implemented at the moment."
    n = precision
    M = len(invert_list)
    n_wbg = num_mux
    n_cmp = precision - num_mux

    file_string = f"module mmc_array{ID}  (\n" \
                  f"\tinput  logic [{n-1}:0]     in  [{M-1}:0],\n" \
                  f"\tinput  logic [{n-1}:0]     r,\n" \
                  f"\toutput logic           SNs  [{M-1}:0]\n);\n"

    # handle edge cases of when this module is a comparator array and when this module is a WBG array
    if 0 <= num_mux <= 1:  # this is a CMP
        if full_correlation and invert_list.any():
            file_string += f"\tlogic [{n - 1}:0] r_inv;\n" \
                           f"\tassign r_inv = ~r;\n\n"
            r_neg = "r_inv"
            spacing = " " * 4
        else:
            r_neg = "r"
            spacing = ""

        for j in range(M):
            if invert_list[j] == 1:
                file_string += f"\tcompara_inv comp{j}(.r({r_neg}), .p(in[{j}]), .SN_bit(SNs[{j}]));\n"
            else:
                file_string += f"\tcompara     comp{j}(.r(r), {spacing}.p(in[{j}]), .SN_bit(SNs[{j}]));\n"

        file_string += f"endmodule\n\n\n"

    elif num_mux == precision:  # this is a WBG
        file_string += f"\tlogic [{n - 1}:0] w;\n"
        file_string += f"\twbg_p1 wbgp1(.r(r), .out(w));\n\n"

        if full_correlation and invert_list.any():
            # Handle inverted WBG P1 with inverted R input if corr_adj is used
            file_string += f"\tlogic [{n - 1}:0] w_neg;\n"
            file_string += f"\twbg_p1 wbgp1_inv(.r(~r), .out(w_neg));\n\n"
            w_neg = "w_neg"
            spacing = " " * 4
        else:
            w_neg = "w"
            spacing = ""

        for j in range(M):
            if invert_list[j] == 1:
                file_string += f"\twbg_p2_inv wbg{j}(.w({w_neg}), .p(in[{j}]), .out(SNs[{j}]));\n"
            else:
                file_string += f"\twbg_p2     wbg{j}(.w(w), {spacing}.p(in[{j}]), .out(SNs[{j}]));\n"

        file_string += f"endmodule\n\n\n"

    else:  # this is a MUX-MAJ-chain
        # initialize wires
        file_string += f"\tlogic [{n_wbg-1}:0] w;\n"
        file_string += f"\tlogic [{n_wbg-1}:0] lower_r;\n"
        file_string += f"\tlogic [{n-1}:{n_wbg}] upper_r;\n\n"

        # assign wires
        file_string += f"\tassign lower_r = r[{n_wbg-1}:0];\n"
        file_string += f"\tassign upper_r = r[{n-1}:{n_wbg}];\n"

        # implement a shared wbg_p1
        if num_mux > 1:
            file_string += f"\twbg_p1 wbgp1(.r(lower_r), .out(w));\n\n"

        if full_correlation and invert_list.any():
            # Handle inverted WBG P1 with inverted R input if corr_adj is used
            file_string += f"\tlogic [{n_wbg-1}:0] w_neg;\n"
            file_string += f"\twbg_p1 wbgp1_inv(.r(~lower_r), .out(w_neg));\n\n"
            w_neg = "w_neg"
        else:
            w_neg = "w"

        for j in range(M):
            if invert_list[j] == 1:
                file_string += f"\tmmc_p2_inv mmc{j}(.w({w_neg}), .r(upper_r), .p(in[{j}]), .SN_bit(SNs[{j}]));\n"
            else:
                file_string += f"\tmmc_p2 mmc{j}(.w({w_neg}), .r(upper_r), .p(in[{j}]), .SN_bit(SNs[{j}]));\n"

        file_string += f"endmodule\n\n\n"

    return file_string


def gen_verilog_mmc_chain(precision, num_mux, bipolar, invert_output):
    """
    :param bipolar: whether this PCC generates a bipolar SN or not. If this PCC is used to generate a bipolar SN, then
    the MSB of the control input is inverted before comparison.
    :param invert_output: whether the output of the comparator should be inverted. This is useful in some MUX circuits.
    :return: verilog code for the module in string form.
    """
    n = precision
    module_name = "mmc_chain_inv" if invert_output else "mmc_chain"
    file_string = f"module {module_name} (\n" \
                  f"\tinput  logic [{n-1}:0] r,\n" \
                  f"\tinput  logic [{n-1}:0] p,\n" \
                  f"\toutput logic       SN_bit\n);\n"

    if bipolar:  # invert MSB of control input
        p_str = "p2"
        file_string += f"\tlogic [{n-1}:0] p2;\n"
        file_string += f"\tassign p2[{n-1}] = ~p[{n-1}];\n" \
                       f"\tassign p2[{n-2}:0] = p[{n-2}:0];\n"
    else:
        p_str = "p"

    # initialize the intermediate outputs
    file_string += f"\tlogic [{n}:0] c;\n" \
                   f"\tassign c[0] = 0;\n"

    for i in range(precision):
        if i < num_mux:
            file_string += f"\tassign c[{i+1}] = r[{i}] ? {p_str}[{i}] : c[{i}];\n"
        else:
            file_string += f"\tassign c[{i+1}] = (r[{i}]&{p_str}[{i}]) | (r[{i}]&c[{i}]) | ({p_str}[{i}]&c[{i}]);\n"

    if invert_output:
        file_string += f"\tassign SN_bit = ~c[{n}];\n"
    else:
        file_string += f"\tassign SN_bit = c[{n}];\n"

    file_string += f"endmodule\n\n\n"
    return file_string


def gen_verilog_mmc_chain_array(precision, share_r, invert_list, full_correlation, ID=""):
    """
    Generates verilog code for an array of mmc_chains.
    :param precision: bit-width of the comparators.
    :param share_r: Boolean value that is True when the comparators share a random number input (must be True for now)
    :param invert_list: list of Boolean values where invert_list[i]=True means that the i-th comparator should invert its
    output while invert_list[i]=False means that the i-th comparator should not invert its output.
    :param full_correlation: whether the SNs should be generate in a manner such that SCC(X1,X2) = +1 for all X1,X2. Only
    works when share_r is True.
    :param ID: identification number or string to append to the end of the module's name. This is used when you need more than
    one logically distinct comparator arrays in a single verilog file.
    :return: verilog code for the module in string form.
    """
    assert share_r, "Only sharing R is implemented at the moment."
    assert not full_correlation or share_r, "When full_correlation=True, share_r must also be True"
    n = precision
    M = len(invert_list)  # M is the number of control inputs to the comparator array and the number of outputs.

    file_string = f"module mmc_chain_array{ID} (\n" \
                  f"\tinput  logic [{n-1}:0] in [{M-1}:0],\n" \
                  f"\tinput  logic [{n-1}:0] r,\n" \
                  f"\toutput logic       SNs [{M-1}:0]\n);\n"

    if full_correlation:
        file_string += f"\tlogic [{n-1}:0] r_inv;\n" \
                       f"\tassign r_inv = ~r;\n\n"
        r_neg = "r_inv"
        spacing = " "*4
    else:
        r_neg = "r"
        spacing = ""

    for j in range(M):
        if invert_list[j] == 1:
            file_string += f"\tmmc_chain_inv mmc{j}(.r({r_neg}), .p(in[{j}]), .SN_bit(SNs[{j}]));\n"
        else:
            file_string += f"\tmmc_chain     mmc{j}(.r(r), {spacing}.p(in[{j}]), .SN_bit(SNs[{j}]));\n"

    file_string += f"endmodule\n\n\n"

    return file_string



#############################
#   MUX/MAJ Adder Modules   #
#############################
# TODO Doc string
def gen_verilog_hardwired_mux_tree(quant_norm_weights, hdl_wire_map, tree_height, ID=""):
    M = (quant_norm_weights != 0).sum()
    m = tree_height

    file_string = f"module hw_tree{ID}  (\n" \
                  f"\tinput  logic           data_SNs  [{M-1}:0],\n" \
                  f"\tinput  logic [{m-1}:0]     mux_select,\n" \
                  f"\toutput logic           out_SN\n);\n"

    # Initialize an array of wires for every level in the mux tree. (for internal signals)
    num_mux = int(2 ** m)
    for level in range(m):
        num_mux //= 2
        file_string += f"\tlogic level{level}  [{num_mux - 1}:0];\n"

    # Assign the just initialized wires
    num_mux = int(2 ** m)
    for level in range(m):
        file_string += "\n"
        num_mux //= 2
        if level == 0:
            for mux_idx in range(num_mux):
                file_string += f"\tassign level{level}[{mux_idx}] = mux_select[{level}] ? data_SNs[{hdl_wire_map[2*mux_idx]}] :" \
                               f" data_SNs[{hdl_wire_map[2*mux_idx+1]}];\n"
        else:
            for mux in range(num_mux):
                file_string += f"\tassign level{level}[{mux}] = mux_select[{level}] ? level{level-1}[{2*mux}] :" \
                               f" level{level-1}[{2*mux+1}];\n"
    file_string += "\n"
    file_string += f"\tassign out_SN = level{m-1}[0];\n"
    file_string += f"endmodule\n\n\n"
    return file_string


def gen_verilog_hardwired_maj_tree(quant_norm_weights, hdl_wire_map, tree_height, ID=""):
    M = (quant_norm_weights != 0).sum()
    m = tree_height

    file_string = f"module hw_tree{ID}  (\n" \
                  f"\tinput  logic           data_SNs  [{M-1}:0],\n" \
                  f"\tinput  logic [{m-1}:0]     select_SN,\n" \
                  f"\toutput logic           out_SN\n);\n"

    # Initialize an array of wires for every level in the mux tree. (for internal signals)
    num_maj = int(2**m)
    for level in range(m):
        num_maj //= 2
        file_string += f"\tlogic level{level}  [{num_maj-1}:0];\n"

    # Assign the just initialized wires
    num_maj = int(2 ** m)
    for level in range(m):
        file_string += "\n"
        num_maj //= 2
        if level == 0:
            for maj_idx in range(num_maj):
                data_SN1 = f"data_SNs[{hdl_wire_map[2*maj_idx]}]"
                data_SN2 = f"data_SNs[{hdl_wire_map[2*maj_idx+1]}]"
                select = f"select_SN[{level}]"
                file_string += f"\tassign level{level}[{maj_idx}] = ({select}&{data_SN1}) | ({select}&{data_SN2}) | ({data_SN1}&{data_SN2});\n"
        else:
            for maj_idx in range(num_maj):
                data_SN1 = f"level{level-1}[{2*maj_idx}]"
                data_SN2 = f"level{level-1}[{2*maj_idx+1}]"
                select = f"select_SN[{level}]"
                file_string += f"\tassign level{level}[{maj_idx}] = ({select}&{data_SN1}) | ({select}&{data_SN2}) | ({data_SN1}&{data_SN2});\n"
    file_string += "\n"
    file_string += f"\tassign out_SN = level{m-1}[0];\n"
    file_string += f"endmodule\n\n\n"
    return file_string


################################
#   Parallel Sampler Modules   #
################################
# TODO: docstring
def gen_verilog_mux_sampler(num_inputs, ID=""):
    M = num_inputs
    select_n = int(np.ceil(np.log2(num_inputs)))

    # Header
    file_string = f"module mux_sampler{ID}  (\n" \
                  f"\tinput  logic       data_SNs  [{M-1}:0],\n" \
                  f"\tinput  logic [{select_n-1}:0] mux_select,\n" \
                  f"\toutput logic       out_SN\n);\n"


    ### construct APC tree  ###

    # first initiate intermediate signals
    total_levels = select_n
    curr_in = num_inputs
    for level in range(total_levels):
        remainder = curr_in % 2
        curr_in = (curr_in)//2 + remainder

        file_string += f"\tlogic level{level}  [{curr_in-1}:0];\n"
    file_string += "\n"

    # then assign intermediate signals
    curr_in = num_inputs
    for level in range(total_levels):
        remainder = curr_in % 2
        curr_in = (curr_in)//2

        for mux_idx in range(curr_in):
            if level == 0:
                in1 = f"data_SNs[{2*mux_idx}]"
                in2 = f"data_SNs[{2*mux_idx+1}]"
            else:
                in1 = f"level{level-1}[{2*mux_idx}]"
                in2 = f"level{level-1}[{2*mux_idx+1}]"
            file_string += f"\tassign level{level}[{mux_idx}] = mux_select[{level}] ? {in1} : {in2};\n"
        for leftover in range(remainder):
            if level == 0:
                in1 = f"data_SNs[{2*curr_in+leftover}]"
            else:
                in1 = f"level{level-1}[{2*curr_in+leftover}]"
            file_string += f"\tassign level{level}[{curr_in+leftover}] = {in1};\n"
        curr_in += remainder
        file_string += "\n"

    file_string += f"\tassign out_SN = level{total_levels-1}[0];\n"
    file_string += f"endmodule\n\n\n"
    return file_string


# TODO: docstring
def gen_verilog_apc(num_inputs, bipolar, ID=""):
    M = num_inputs
    out_n = int(np.ceil(np.log2(num_inputs+1))) + 2*int(bipolar)
    sign = " signed" if bipolar else ""

    # Header
    file_string = f"module apc{ID}  (\n" \
                  f"\tinput  logic        data_SNs  [{M-1}:0],\n" \
                  f"\toutput logic{sign} [{out_n-1}:0]  out\n);\n"

    # first convert bipolar SN bits to two's complement if necessary
    if bipolar:
        file_string += f"\tlogic signed [1:0] converted_SNs [{M-1}:0];\n"
        for idx in range(M):
            file_string += f"\tbipolar_converter conv{idx}(.in_SN(data_SNs[{idx}]), .out(converted_SNs[{idx}]));\n"
        prefix = "converted_"
        file_string += "\n"
    else:
        prefix = "data_"

    ### construct APC tree  ###

    # first initiate intermediate signals
    total_levels = int(np.ceil(np.log2(num_inputs)))
    curr_in = num_inputs
    curr_n = 1 + int(bipolar)
    for level in range(total_levels):
        remainder = curr_in % 2
        curr_in = (curr_in)//2 + remainder
        curr_n += 1

        file_string += f"\tlogic{sign} [{curr_n-1}:0] level{level}  [{curr_in-1}:0];\n"
    file_string += "\n"

    # then assign intermediate signals
    curr_in = num_inputs
    for level in range(total_levels):
        remainder = curr_in % 2
        curr_in = (curr_in)//2

        for pair in range(curr_in):
            if level == 0:
                in1 = f"{prefix}SNs[{2*pair}]"
                in2 = f"{prefix}SNs[{2*pair+1}]"
            else:
                in1 = f"level{level-1}[{2*pair}]"
                in2 = f"level{level-1}[{2*pair+1}]"
            file_string += f"\tassign level{level}[{pair}] = {in1} + {in2};\n"
        for leftover in range(remainder):
            if level == 0:
                in1 = f"{prefix}SNs[{2*curr_in+leftover}]"
            else:
                in1 = f"level{level-1}[{2*curr_in+leftover}]"
            file_string += f"\tassign level{level}[{curr_in+leftover}] = {in1};\n"
        curr_in += remainder
        file_string += "\n"

    file_string += f"\tassign out = level{total_levels-1}[0];\n"
    file_string += f"endmodule\n\n\n"
    return file_string


# TODO: docstring
def gen_verilog_bipolar_converter(ID=""):
    file_string = f"module bipolar_converter{ID}  (\n" \
                  f"\tinput  logic        in_SN,\n" \
                  f"\toutput logic signed [1:0]  out\n);\n"
    file_string += f"\tassign out[1] = ~in_SN;\n"
    file_string += f"\tassign out[0] = 1'b1;\n"
    file_string += f"endmodule\n\n\n"

    return file_string


# TODO: docstring
def gen_verilog_parallel_sampler(num_inputs, sampling_rate, bipolar, ID="", mux_ID="", apc_ID=""):
    K, S = num_inputs, sampling_rate
    assert K % S == 0, f"The number of inputs:{K} should be divisible by the sampling rate:{S}. Current ratio: {K/S}"

    inputs_per_sampler = K//S
    out_n = int(np.ceil(np.log2(sampling_rate+1))) + int(bipolar)
    select_n = max(int(np.ceil(np.log2(inputs_per_sampler))), 1)  # the max op makes this value 1 when pure apc is used.

    file_string = f"module parallel_sampler{ID}  (\n" \
                  f"\tinput  logic        data_SNs  [{num_inputs-1}:0],\n" \
                  f"\tinput  logic [{select_n-1}:0]  mux_select,\n" \
                  f"\toutput logic [{out_n-1}:0]  out\n);\n"

    # instantiate the parallel sampler outputs
    file_string += f"\tlogic samp_outs[{sampling_rate-1}:0];\n"
    file_string += "\n"

    # instantiate the parallel mux samplers if necessary and assign the parallel sampler outputs
    if S != K:  # we have mux gates
        for samp_idx in range(sampling_rate):
            start_SN_idx = samp_idx*inputs_per_sampler
            end_SN_idx = (samp_idx+1)*inputs_per_sampler
            file_string += f"\tmux_sampler{mux_ID} samp{samp_idx}(.data_SNs(data_SNs[{end_SN_idx-1}:{start_SN_idx}]), .mux_select(mux_select), .out_SN(samp_outs[{samp_idx}]));\n"
    else:  # we don't have mux gates
        for samp_idx in range(sampling_rate):
            file_string += f"\tassign samp_outs[{samp_idx}] = data_SNs[{samp_idx}];\n"
    file_string += "\n"

    # instantiate the APC if necessary
    if S > 1:  # we have an APC
        file_string += f"\tapc apc{apc_ID}(.data_SNs(samp_outs), .out(out));\n"
    elif bipolar:  # we have no APC and we are using bipolar format
        file_string += f"\tbipolar_converter conv(.in_SN(samp_outs), .out(out));\n"
    else:  # we have no APC and we are using unipolar format
        file_string += "\tassign out = samp_outs[0];\n"
    file_string += f"endmodule\n\n\n"

    return file_string


# TODO: docstring
def gen_verilog_parallel_sampler_uneven(num_inputs, tree_height, bipolar, ID="", mux_ID="", apc_ID=""):
    # assert not bipolar, "Haven't really checked this for bipolar"
    group_size = int(2**tree_height)
    K, G = num_inputs, group_size
    num_groups = K // G
    R = K % G

    inputs_per_sampler = G
    out_n = int(np.ceil(np.log2(num_groups + R + 1))) + int(bipolar)
    select_n = max(int(np.ceil(np.log2(inputs_per_sampler))), 1)  # the max op makes this value 1 when pure apc is used.

    file_string = f"module parallel_sampler{ID}  (\n" \
                  f"\tinput  logic        data_SNs  [{num_inputs-1}:0],\n" \
                  f"\tinput  logic [{select_n-1}:0]  mux_select,\n" \
                  f"\toutput logic [{out_n-1}:0]  out\n);\n"

    # instantiate the parallel sampler mux outputs and apc inputs
    file_string += f"\tlogic mux_outs[{num_groups-1}:0];\n"
    file_string += f"\tlogic apc_ins[{num_groups+R-1}:0];\n"
    file_string += "\n"

    # instantiate the parallel mux samplers if necessary and assign the parallel sampler outputs
    if G > 1:  # we have mux gates
        for samp_idx in range(num_groups):
            start_SN_idx = samp_idx*inputs_per_sampler
            end_SN_idx = (samp_idx+1)*inputs_per_sampler
            file_string += f"\tmux_sampler{mux_ID} samp{samp_idx}(.data_SNs(data_SNs[{end_SN_idx-1}:{start_SN_idx}]), .mux_select(mux_select), .out_SN(mux_outs[{samp_idx}]));\n"
    else:  # we don't have mux gates
        for samp_idx in range(num_groups):
            file_string += f"\tassign mux_outs[{samp_idx}] = data_SNs[{samp_idx}];\n"
    file_string += "\n"

    file_string += f"\tassign apc_ins[{num_groups-1}:0] = mux_outs;\n"
    if R > 0:
        file_string += f"\tassign apc_ins[{num_groups+R-1}:{num_groups}] = data_SNs[{num_inputs-1}:{num_inputs-R}];\n"

    # instantiate the APC if necessary
    if G != K:  # we have an APC
        file_string += f"\tapc{apc_ID} apc(.data_SNs(apc_ins), .out(out));\n"
    elif bipolar:  # we have no APC and we are using bipolar format
        file_string += f"\tbipolar_converter conv(.in_SN(apc_ins), .out(out));\n"
    else:  # we have no APC and we are using unipolar format
        file_string += "\tassign out = apc_ins[0];\n"
    file_string += f"endmodule\n\n\n"

    return file_string


# TODO: docstring
def gen_verilog_pos_neg_splitter(precision, weights, ID=""):
    # splits values, not SNs!
    num_inputs = len(weights)
    inv_list = weights < 0
    num_neg = len(weights[inv_list])
    num_pos = len(weights[~inv_list])

    assert num_neg + num_pos == num_inputs
    assert (num_neg > 0) and (num_pos > 0)
    n = precision

    file_string = f"module pos_neg_splitter{ID}  (\n" \
                  f"\tinput  logic [{n-1}:0]   in  [{num_inputs-1}:0],\n" \
                  f"\toutput logic [{n-1}:0]   neg_out  [{num_neg-1}:0],\n" \
                  f"\toutput logic [{n-1}:0]   pos_out  [{num_pos-1}:0]\n);\n"

    pos_idx, neg_idx = 0, 0
    for idx in range(num_inputs):
        if inv_list[idx]:  # negative
            file_string += f"\tassign neg_out[{neg_idx}] = in[{idx}];\n"
            neg_idx += 1
        else:  # positive
            file_string += f"\tassign pos_out[{pos_idx}] = in[{idx}];\n"
            pos_idx += 1

    file_string += "endmodule\n\n\n"
    assert neg_idx == num_neg
    assert pos_idx == num_pos

    return file_string


# TODO: docstring
def gen_verilog_para_samp_toplevel_single(rns_bitwidth, num_inputs, samp_rate, bipolar, use_vdc, pcc_array_name=None,
                                          pcc_array_ID="", samp_ID="", out_cnt_ID=""):
    rns_n, K, S = rns_bitwidth, num_inputs, samp_rate
    assert K % S == 0, f"Sampling rate: {S} must evenly divide number of inputs: {K}. Current ratio: {K/S}"


    inputs_per_samp = num_inputs//samp_rate
    select_n = max(int(np.ceil(np.log2(inputs_per_samp))), 1)  # the max op makes this value 1 when apc is used!
    samp_out_n = int(np.ceil(np.log2(samp_rate+1))) + int(bipolar)
    out_n = rns_n + samp_out_n

    file_string = f"module toplevel (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\tinput  logic [{rns_n-1}:0]  data [{K-1}:0],\n" \
                  f"\toutput logic [{out_n-1}:0]  out\n);\n"

    # instantiate the intermediate signals
    file_string += f"\tlogic [{rns_n-1}:0] r;\n" \
                   f"\tlogic [{rns_n-1}:0] s;\n" \
                   f"\tlogic [{select_n-1}:0] mux_select;\n" \
                   f"\tlogic       data_SNs [{K-1}:0];\n" \
                   f"\tlogic [{samp_out_n-1}:0] sampler_out;\n\n"

    # instantiate the RNS (which is either a counter or VDC generator)
    if use_vdc:
        file_string += f"\tcounter cnt(.clock(clock), .reset(reset), .rev_state(r), .state(s));\n"
    else:
        file_string += f"\tcounter cnt(.clock(clock), .reset(reset), .state(r));\n"
        file_string += f"\tassign s = r;\n"

    file_string += f"\tassign mux_select = s[{select_n-1}:0];\n"
    file_string += "\n"

    # instantiate the PCCs
    pcc_array_name = f"compara_array{pcc_array_ID}" if pcc_array_name is None else f"{pcc_array_name}{pcc_array_ID}"
    file_string += f"\t{pcc_array_name} pccs{pcc_array_ID}(.in(data), .r(r[{rns_n-1}:0]), .SNs(data_SNs));\n"

    # instantiate the parallel sampler
    file_string += f"\tparallel_sampler{samp_ID} sampler{samp_ID}(.data_SNs(data_SNs), .mux_select(mux_select), .out(sampler_out));\n"

    # instantiate the output accumulator
    file_string += f"\taccumulator{out_cnt_ID} accum{out_cnt_ID}(.clock(clock), .reset(reset), .data_in(sampler_out), .out(out));\n"
    file_string += "endmodule\n\n\n"

    return file_string


def gen_verilog_PSA_neuron_toplevel_single(rns_bitwidth, num_neg, num_pos, tree_height, use_vdc, pcc_array_name=None):
    num_inputs = num_neg + num_pos
    assert (num_neg > 0) and (num_pos > 0), "Not implemented for other cases yet"
    rns_n = rns_bitwidth
    select_n = max(tree_height, 1)  # the max op makes this value 1 when apc is used!
    inputs_per_samp = int(2**tree_height)

    pos_groups, pos_rem = num_pos // inputs_per_samp, num_pos % inputs_per_samp
    pos_psa_out_n = int(np.ceil(np.log2(pos_groups+pos_rem+1)))
    pos_acum_n = pos_psa_out_n + rns_n
    neg_groups, neg_rem = num_neg // inputs_per_samp, num_neg % inputs_per_samp
    neg_psa_out_n = int(np.ceil(np.log2(neg_groups+neg_rem+1)))
    neg_acum_n = neg_psa_out_n + rns_n
    diff_n = max(pos_acum_n, neg_acum_n) + 1

    file_string = f"module toplevel (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\tinput  logic [{rns_n-1}:0]  data [{num_inputs-1}:0],\n" \
                  f"\toutput logic       out\n);\n"

    # instantiate the intermediate signals
    file_string += f"\tlogic [{rns_n-1}:0] r;\n" \
                   f"\tlogic [{rns_n-1}:0] s;\n" \
                   f"\tlogic [{select_n-1}:0] mux_select;\n" \
                   f"\tlogic [{rns_n-1}:0] neg_data [{num_neg-1}:0];\n" \
                   f"\tlogic [{rns_n-1}:0] pos_data [{num_pos-1}:0];\n" \
                   f"\tlogic       pos_SNs [{num_pos-1}:0];\n" \
                   f"\tlogic       neg_SNs [{num_neg-1}:0];\n" \
                   f"\tlogic [{neg_psa_out_n-1}:0] neg_psa_out;\n" \
                   f"\tlogic [{pos_psa_out_n-1}:0] pos_psa_out;\n" \
                   f"\tlogic [{neg_acum_n-1}:0] neg_acum_out;\n" \
                   f"\tlogic [{pos_acum_n-1}:0] pos_acum_out;\n" \
                   f"\tlogic signed [{neg_acum_n}:0] sign_neg_acum_out;\n" \
                   f"\tlogic signed [{pos_acum_n}:0] sign_pos_acum_out;\n" \
                   f"\tlogic signed [{diff_n-1}:0] diff;\n\n"

    # instantiate the RNS (which is either a counter or VDC generator)
    if use_vdc:
        file_string += f"\tcounter cnt(.clock(clock), .reset(reset), .rev_state(r), .state(s));\n"
    else:
        file_string += f"\tcounter cnt(.clock(clock), .reset(reset), .state(r));\n"
        file_string += f"\tassign s = r;\n"

    file_string += f"\tassign mux_select = s[{select_n-1}:0];\n\n" \
                   f"\tpos_neg_splitter splitter(.in(data), .neg_out(neg_data), .pos_out(pos_data));\n"

    # instantiate the PCCs
    pcc_array_name = f"compara_array" if pcc_array_name is None else f"{pcc_array_name}"
    labs = ["neg", "pos"]
    nums = [num_neg, num_pos]
    for lab, num in zip(labs, nums):
        file_string += f"\t{pcc_array_name}_{lab} pccs_{lab}(.in({lab}_data), .r(r[{rns_n-1}:0]), .SNs({lab}_SNs));\n"

        # instantiate the parallel sampler
        file_string += f"\tparallel_sampler_{lab} sampler_{lab}(.data_SNs({lab}_SNs), .mux_select(mux_select), .out({lab}_psa_out));\n"

        # instantiate the output accumulator
        file_string += f"\taccumulator_{lab} accum_{lab}(.clock(clock), .reset(reset), .data_in({lab}_psa_out), .out({lab}_acum_out));\n\n"

    # now do the end logic
    file_string += f"\tassign sign_neg_acum_out[{neg_acum_n-1}:0] = neg_acum_out;\n"
    file_string += f"\tassign sign_pos_acum_out[{pos_acum_n-1}:0] = pos_acum_out;\n"
    file_string += f"\tassign sign_neg_acum_out[{neg_acum_n}] = 0;\n"
    file_string += f"\tassign sign_pos_acum_out[{pos_acum_n}] = 0;\n\n"

    file_string += f"\tassign diff = sign_pos_acum_out - sign_neg_acum_out;\n" \
                   f"\tassign out = diff[{diff_n-1}];\n"
    file_string += "endmodule\n\n\n"

    return file_string


def gen_verilog_conv_mux_toplevel(rns_bitwidth, num_inputs, bipolar, pcc_array_name=None,
                                          pcc_array_ID="", samp_ID="", out_cnt_ID="", rns_ID="", mux_select_IDs=None):
    rns_n, K = rns_bitwidth, num_inputs
    select_n = int(np.ceil(np.log2(num_inputs)))
    out_n = rns_n + int(bipolar)

    file_string = f"module toplevel (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\tinput  logic [{rns_n-1}:0]  data [{K-1}:0],\n" \
                  f"\toutput logic [{out_n-1}:0]  out\n);\n"

    # instantiate the intermediate signals
    file_string += f"\tlogic [{rns_n-1}:0] r;\n" \
                   f"\tlogic [{rns_n-1}:0] lfsr_states [{select_n-1}:0];\n" \
                   f"\tlogic [{select_n-1}:0] mux_select;\n" \
                   f"\tlogic       data_SNs [{K-1}:0];\n" \
                   f"\tlogic       tree_out;\n\n"

    # instantiate the LFSR RNS
    file_string += f"\tlfsr{rns_ID} lfsr0(.clock(clock), .reset(reset), .state(r));\n"

    # instantiate the mux select input
    for i in range(select_n):
        file_string += f"\tlfsr{mux_select_IDs[i]} lfsr{i+1}(.clock(clock), .reset(reset), .state(lfsr_states[{i}]));\n"
    file_string += "\n"
    for i in range(select_n):
        file_string += f"\tassign mux_select[{i}] = lfsr_states[{i}][{select_n-1}];\n"
    file_string += "\n"

    # instantiate the PCCs
    pcc_array_name = f"compara_array{pcc_array_ID}" if pcc_array_name is None else f"{pcc_array_name}{pcc_array_ID}"
    file_string += f"\t{pcc_array_name} pccs{pcc_array_ID}(.in(data), .r(r), .SNs(data_SNs));\n"

    # instantiate the mux
    file_string += f"\tmux_sampler{samp_ID} sampler{samp_ID}(.data_SNs(data_SNs), .mux_select(mux_select), .out_SN(tree_out));\n"

    # instantiate the output accumulator
    file_string += f"\taccumulator{out_cnt_ID} accum{out_cnt_ID}(.clock(clock), .reset(reset), .data_in(tree_out), .out(out));\n"
    file_string += "endmodule\n\n\n"

    return file_string


def gen_verilog_weighted_PSA_uneven_toplevel(rns_bitwidth, num_inputs, tree_height, bipolar, use_mults):
    # assert bipolar, "focus on bipolar. probs have to update some 'signed' declarations"
    rns_n = rns_bitwidth
    select_n = max(tree_height, 1)  # the max op makes this value 1 when apc is used!
    inputs_per_samp = int(2**tree_height)

    num_groups, num_rem = num_inputs // inputs_per_samp, num_inputs % inputs_per_samp
    psa_out_n = int(np.ceil(np.log2(num_groups+num_rem+1)))
    acum_n = psa_out_n + rns_n

    if use_mults:  # include weights if using multipliers
        file_string = f"module toplevel (\n" \
                      f"\tinput  clock, reset,\n" \
                      f"\tinput  logic [{rns_n-1}:0]  data [{num_inputs-1}:0],\n" \
                      f"\tinput  logic [{rns_n-1}:0]  weights [{num_inputs-1}:0],\n" \
                      f"\toutput logic [{acum_n-1}:0] out\n);\n"
    else:
        file_string = f"module toplevel (\n" \
                      f"\tinput  clock, reset,\n" \
                      f"\tinput  logic [{rns_n-1}:0]  data [{num_inputs-1}:0],\n" \
                      f"\toutput logic [{acum_n-1}:0] out\n);\n"

    # instantiate the intermediate signals
    file_string += f"\tlogic [{rns_n-1}:0] r;\n" \
                   f"\tlogic [{rns_n-1}:0] s;\n" \
                   f"\tlogic [{select_n-1}:0] mux_select;\n"
    if use_mults:
        file_string += f"\tlogic [{rns_n-1}:0] weight_r;\n"

    file_string += f"\tlogic       input_SNs [{num_inputs-1}:0];\n" \
                   f"\tlogic       prod_SNs [{num_inputs-1};0];\n"
    if use_mults:
        file_string += f"\tlogic       weight_SNs [{num_inputs-1}:0];\n"

    file_string += f"\tlogic [{psa_out_n-1}:0] psa_out;\n" \
                   f"\tlogic [{acum_n-1}:0] acum_out;\n" \

    # instantiate the RNS (which is either a counter or VDC generator)
    file_string += f"\tcounter cnt(.clock(clock), .reset(reset), .rev_state(r), .state(s));\n"
    if use_mults:
        file_string += f"\tcounter cnt2(.clock(clock), .rev_state(weight_r));\n"

    file_string += f"\tassign mux_select = s[{select_n-1}:0];\n\n" \

    # instantiate the PCCs
    file_string += f"\tcompara_array pccs(.in(data), .r(r[{rns_n-1}:0]), .SNs(SNs));\n"
    if use_mults:
        file_string += f"\tcompara_array weight_pccs(.in(data), .r(weight_r[{rns_n-1}:0]), .SNs(weight_SNs));\n"
        file_string += f"\tmult_array mults(.Xs(SNs), .Ws(weight_SNs), .prods(prod_SNs));\n\n"
    else:
        file_string += f"\tassign prod_SNs = SNs;\n\n"

    # instantiate the parallel sampler
    file_string += f"\tparallel_sampler sampler(.data_SNs(prod_SNs), .mux_select(mux_select), .out(psa_out));\n"

    # instantiate the output accumulator
    file_string += f"\taccumulator accum(.clock(clock), .reset(reset), .data_in(psa_out), .out(out));\n\n"

    # now do the end logic
    file_string += "endmodule\n\n\n"

    return file_string


def gen_verilog_weighted_APC_MMC_uneven_toplevel(rns_bitwidth, num_inputs, bipolar, use_mults):
    rns_n = rns_bitwidth
    apc_out_n = int(np.ceil(np.log2(num_inputs+1))) + 2*int(bipolar)
    acum_n = apc_out_n + rns_n
    sign = " signed" if bipolar else ""

    if use_mults:  # include weights if using multipliers
        weight_input = f"\tinput  logic{sign} [{rns_n-1}:0]  weights [{num_inputs-1}:0],\n"
        weight_SNs = f"\tlogic weight_SNs [{num_inputs-1}:0];\n"
    else:
        weight_input = f"\tinput  logic        weights [{num_inputs-1}:0],\n"
        weight_SNs = f"\tlogic weight_SNs [{num_inputs-1}:0];\n"

    file_string = f"module toplevel (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\tinput  logic{sign} [{rns_n-1}:0]  data [{num_inputs-1}:0],\n" \
                  f"{weight_input}" \
                  f"\toutput logic{sign} [{acum_n-1}:0] out\n);\n"

    # instantiate the intermediate signals
    file_string += f"\tlogic [{rns_n-1}:0] r;\n" \
                   f"\tlogic [{rns_n-1}:0] s;\n\n" \
                   f"\tlogic input_SNs [{num_inputs-1}:0];\n" \
                   f"{weight_SNs}" \
                   f"\tlogic prod_SNs [{num_inputs-1}:0];\n\n" \
                   f"\tlogic{sign} [{apc_out_n-1}:0] apc_out;\n"

    # instantiate the VDC RNS (which is either a counter or VDC generator)
    file_string += f"\tcounter cnt(.clock(clock), .reset(reset), .rev_state(r), .state(s));\n"

    # instantiate the PCCs
    file_string += f"\tmmc_chain_array pccs(.in(data), .r(r), .SNs(input_SNs));\n"
    if use_mults:
        file_string += f"\tmmc_chain_array weight_pccs(.in(weights), .r(s), .SNs(weight_SNs));\n"
        # file_string += f"\tmult_array mults(.Xs(input_SNs), .Ws(weight_SNs), .prods(prod_SNs));\n\n"
    else:
        # file_string += f"\tassign prod_SNs = input_SNs;\n\n"
        file_string += f"\tassign weight_SNs = weights;\n\n"

    file_string += f"\tmult_array mults(.Xs(input_SNs), .Ws(weight_SNs), .prods(prod_SNs));\n\n"
    # instantiate the APC
    file_string += f"\tapc apc(.data_SNs(prod_SNs), .out(apc_out));\n"

    # instantiate the output accumulator
    file_string += f"\taccumulator accum(.clock(clock), .reset(reset), .data_in(apc_out), .out(out));\n\n"

    # now do the end logic
    file_string += "endmodule\n\n\n"

    return file_string


#############################
#   CEASE Related Modules   #
#############################
def gen_verilog_CEASE_adder(invert_out):
    """
    This is a verilog module for a TFF adder from "Near Sensor Computing" SC paper.
    The TFF takes in two SN inputs and outputs a SN
    :param invert_out whether the TFF output should be inverted (helps improve quant error)
    :return verilog code for the module in string form.
    """

    if not invert_out:  # plain CEASE adder
        file_string = f"module cease (\n" \
                      f"\tinput  clock, reset,\n" \
                      f"\tinput  logic x,\n" \
                      f"\tinput  logic y,\n" \
                      f"\toutput logic out\n);\n" \
                      f"\tlogic state;\n" \
                      f"\tlogic next_state;\n\n" \
                      f"\tassign next_state = x ^ y ^ state;\n" \
                      f"\tassign out = (x&y) | (x&state) | (y&state);\n\n" \
                      f"\talways_ff @(posedge clock) begin\n" \
                      f"\t\tif (reset == 1) state <= 'b0; else\n" \
                      f"\t\t                state <= next_state;\n" \
                      f"\tend\n" \
                      f"endmodule\n\n\n"
    else:  # CEASE where the TFF output is inverted (helps reduce bias). only output logic changes
        file_string = f"module cease_inv (\n" \
                      f"\tinput  clock, reset,\n" \
                      f"\tinput  logic x,\n" \
                      f"\tinput  logic y,\n" \
                      f"\toutput logic out\n);\n" \
                      f"\tlogic state;\n" \
                      f"\tlogic next_state;\n\n" \
                      f"\tassign next_state = x ^ y ^ state;\n" \
                      f"\tassign out = (x&y) | (x&(~state)) | (y&(~state));\n\n" \
                      f"\talways_ff @(posedge clock) begin\n" \
                      f"\t\tif (reset == 1) state <= 'b0; else\n" \
                      f"\t\t                state <= next_state;\n" \
                      f"\tend\n" \
                      f"endmodule\n\n\n"

    return file_string


def gen_verilog_CEASE_tree(num_inputs, starts, ID=""):
    M = num_inputs

    # Header
    file_string = f"module cease_tree{ID}  (\n" \
                  f"\tinput  logic       data_SNs  [{M-1}:0],\n" \
                  f"\toutput logic       out_SN\n);\n"

    # first initiate intermediate signals
    total_levels = int(np.ceil(np.log2(num_inputs)))
    curr_in = num_inputs
    for level in range(total_levels):
        remainder = curr_in % 2
        curr_in = (curr_in)//2 + remainder

        file_string += f"\tlogic level{level}  [{curr_in-1}:0];\n"
    file_string += "\n"

    # then assign intermediate signals
    cease_names = ["cease    ", "cease_inv"]
    curr_in = num_inputs
    for level in range(total_levels):
        remainder = curr_in % 2
        curr_in = (curr_in)//2

        for adder_idx in range(curr_in):
            if level == 0:
                in1 = f"data_SNs[{2*adder_idx}]"
                in2 = f"data_SNs[{2*adder_idx+1}]"
            else:
                in1 = f"level{level-1}[{2*adder_idx}]"
                in2 = f"level{level-1}[{2*adder_idx+1}]"
            file_string += f"\t{cease_names[starts[level][adder_idx]]} adder{level}_{adder_idx}(.x({in1}), .y({in2})," \
                           f"  .out(level{level}[{adder_idx}]));\n"
        curr_in += remainder
        file_string += "\n"

    file_string += f"\tassign out_SN = level{total_levels-1}[0];\n"
    file_string += f"endmodule\n\n\n"
    return file_string


def gen_verilog_CEASE_toplevel(rns_bitwidth, num_inputs, bipolar, use_vdc, pcc_array_name=None,
                                          pcc_array_ID="", tree_ID="", out_cnt_ID=""):
    rns_n, K = rns_bitwidth, num_inputs
    out_n = rns_n + int(bipolar)

    file_string = f"module toplevel (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\tinput  logic [{rns_n-1}:0]  data [{K-1}:0],\n" \
                  f"\toutput logic [{out_n-1}:0]  out\n);\n"

    # instantiate the intermediate signals
    file_string += f"\tlogic [{rns_n-1}:0] r;\n" \
                   f"\tlogic [{rns_n-1}:0] s;\n" \
                   f"\tlogic       data_SNs [{K-1}:0];\n" \
                   f"\tlogic       tree_out;\n"
    if bipolar:
        file_string += f"\tlogic [1:0] conv_out;\n"
    else:
        file_string += "\n"

    # instantiate the RNS (which is either a counter or VDC generator)
    if use_vdc:
        file_string += f"\tcounter cnt(.clock(clock), .reset(reset), .rev_state(r), .state(s));\n"
    else:
        file_string += f"\tcounter cnt(.clock(clock), .reset(reset), .state(r));\n"
        file_string += f"\tassign s = r;\n"

    # instantiate the PCCs
    pcc_array_name = f"compara_array{pcc_array_ID}" if pcc_array_name is None else f"{pcc_array_name}{pcc_array_ID}"
    file_string += f"\t{pcc_array_name} pccs{pcc_array_ID}(.in(data), .r(r[{rns_n-1}:0]), .SNs(data_SNs));\n"

    # instantiate the CEASE tree
    file_string += f"\tcease_tree{tree_ID} tree{tree_ID}(.data_SNs(data_SNs), .out_SN(tree_out));\n"

    # instantiate the output accumulator
    if bipolar:
        file_string = f"bipolar_converter bip_conv(.in_SN(tree_out), .out(conv_out));\n"
        file_string += f"\taccumulator{out_cnt_ID} accum{out_cnt_ID}(.clock(clock), .reset(reset), .data_in(conv_out), .out(out));\n"
    else:
        file_string += f"\taccumulator{out_cnt_ID} accum{out_cnt_ID}(.clock(clock), .reset(reset), .data_in(tree_out), .out(out));\n"

    file_string += "endmodule\n\n\n"

    return file_string


#################################
#   AxPC HAPC Related Modules   #
#################################
def gen_verilog_axpc_hapc_layer(num_inputs, which, ID=""):
    M = num_inputs
    O = M//2 + M%2

    if which == 'axpc':
        ops = ['&', '|']
    elif which == 'hapc':
        ops = ["|", "|"]
    else:
        raise NotImplementedError

    # Header
    file_string = f"module approx_layer{ID}  (\n" \
                  f"\tinput  logic data_SNs [{M-1}:0],\n" \
                  f"\toutput logic out_SNs  [{O-1}:0]\n);\n"

    for i in range(0, M, 4):
        for j in range(2):
            file_string += f"\tassign out_SNs[{i//2+j}] = data_SNs[{i+2*j}] {ops[j]} data_SNs[{i+2*j+1}];\n"

    remaining = M % 4
    if remaining == 1:  # one extra
        file_string += f"\tassign out_SNs[{O-1}] = data_SNs[{M-1}];\n"
        assert False, "check this (coded but never tested)"
    elif remaining == 2:  # two extra
        file_string += f"\tassign out_SNs[{O-1}] = data_SNs[{M-2}] {ops[0]} data_SNs[{M-1}];\n"
        assert False, "check this (coded but never tested)"
    elif remaining == 3:  # three extra
        file_string += f"\tassign out_SNs[{O-2}] = data_SNs[{M-3}] {ops[0]} data_SNs[{M-2}];\n"
        file_string += f"assign out_SNs[{O-1}] = data_SNs[{M-1}];\n"
        assert False, "check this (coded but never tested)"

    file_string += f"endmodule\n\n\n"
    return file_string


def gen_verilog_axpc_hapc_toplevel(rns_bitwidth, num_inputs, bipolar, use_vdc, pcc_array_name=None,
                               pcc_array_ID="", approx_ID="", apc_ID="", out_cnt_ID=""):
    rns_n, K = rns_bitwidth, num_inputs
    approx_out = num_inputs//2 + num_inputs%2
    apc_n = int(np.ceil(np.log2(num_inputs//2+1))) + int(bipolar)
    out_n = apc_n + rns_n

    file_string = f"module toplevel (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\tinput  logic [{rns_n-1}:0]  data [{K-1}:0],\n" \
                  f"\toutput logic [{out_n-1}:0]  out\n);\n"

    # instantiate the intermediate signals
    file_string += f"\tlogic [{rns_n-1}:0] r;\n" \
                   f"\tlogic [{rns_n-1}:0] s;\n" \
                   f"\tlogic       data_SNs [{K-1}:0];\n" \
                   f"\tlogic       approx_out [{approx_out-1}:0];\n" \
                   f"\tlogic [{apc_n-1}:0] apc_out;\n\n"

    # instantiate the RNS (which is either a counter or VDC generator)
    if use_vdc:
        file_string += f"\tcounter cnt(.clock(clock), .reset(reset), .rev_state(r), .state(s));\n"
    else:
        file_string += f"\tcounter cnt(.clock(clock), .reset(reset), .state(r));\n"
        file_string += f"\tassign s = r;\n"

    # instantiate the PCCs
    pcc_array_name = f"compara_array{pcc_array_ID}" if pcc_array_name is None else f"{pcc_array_name}{pcc_array_ID}"
    file_string += f"\t{pcc_array_name} pccs{pcc_array_ID}(.in(data), .r(r[{rns_n-1}:0]), .SNs(data_SNs));\n"

    # instantiate the approx layer
    file_string += f"\tapprox_layer{approx_ID} approx{approx_ID}(.data_SNs(data_SNs), .out_SNs(approx_out));\n"

    file_string += f"\tapc apc{apc_ID}(.data_SNs(approx_out), .out(apc_out));\n"
    # instantiate the output accumulator
    file_string += f"\taccumulator{out_cnt_ID} accum{out_cnt_ID}(.clock(clock), .reset(reset), .data_in(apc_out), .out(out));\n"
    file_string += "endmodule\n\n\n"

    return file_string


##########################
#   SUC Related Modules  #
##########################
def gen_verilog_SUC_weight_generator(rns_precision, weights, use_cmp):
    num_inputs = len(weights)
    assert (weights == 1).all(), "this function is only implemented for averaging"
    assert np.log2(num_inputs) % 1 == 0, "only works for power of two inputs (for now)"
    assert 2**rns_precision >= num_inputs, "RNS sequence length should probably be >= num inputs"
    assert use_cmp, "only implemented for cmp pccs (could do wbg, but no reason to atm)"

    rns_n, M = rns_precision , num_inputs

    # header
    file_string = f"module suc_weight_gen (\n" \
                  f"\tinput  logic [{rns_n-1}:0]  r,\n" \
                  f"\toutput logic        out [{M-1}:0]\n);\n"

    # intermediate signals
    file_string += f"\tlogic weight_SNs [{M-2}:0];\n\n"

    # generate the weight SNs
    for i in range(1, num_inputs):  # SNs with value 1/M, 2/M, ..., M-1/M
        if use_cmp:
            file_string += f"\tassign weight_SNs[{i-1}] = r < {rns_n}'d{i};\n"

    # make weight SNs disjoint and the same value. first and last SNs are a bit different (bc all 0s or all 1s)
    file_string += f"\n\tassign out[0] = weight_SNs[0];\n"  # first SN is just 1/M
    for i in range(1, num_inputs-1):  # out[1], out[2], ... out[M-2]
        file_string += f"\tassign out[{i}] = weight_SNs[{i}] ^ weight_SNs[{i-1}];\n"  # xor i/M with (i-1)/M
    file_string += f"\tassign out[{M-1}] = ~weight_SNs[{M-2}];\n"  # last SNs is just inverse of M-1/M

    # end module
    file_string += "endmodule\n\n\n"

    return file_string


def gen_verilog_SUC_mac(num_inputs):
    M = num_inputs
    # basically, just AND the weight SNs by the input SNs
    file_string = f"module suc_mac (\n" \
                  f"\tinput  logic input_SNs [{M-1}:0],\n" \
                  f"\tinput  logic weight_SNs [{M-1}:0],\n" \
                  f"\toutput logic out_SN\n);\n"

    # intermediate product SNs
    file_string += f"\tlogic [{M-1}:0] product_SNs;\n\n"
    for i in range(num_inputs):
        file_string += f"\tassign product_SNs[{i}] = input_SNs[{i}] & weight_SNs[{i}];\n"

    # output is just an OR of the products
    file_string += "\n\tassign out_SN = |product_SNs;\n" \
                   "endmodule\n\n\n"

    return file_string


def gen_verilog_SUC_toplevel(rns_bitwidth, weights, bipolar, pcc_array_name=None, pcc_array_ID="", out_cnt_ID=""):
    rns_n = rns_bitwidth
    M = len(weights)
    out_n = rns_n + int(bipolar)

    assert not bipolar, "Only unipolar implemented for the moment"

    # header
    file_string = f"module toplevel (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\tinput  logic [{rns_n-1}:0]  data [{M-1}:0],\n" \
                  f"\toutput logic [{out_n-1}:0]  out\n);\n"

    # intermediate signals
    file_string += f"\tlogic [{rns_n-1}:0] data_r;\n" \
                   f"\tlogic [{rns_n-1}:0] weight_r;\n" \
                   f"\tlogic       data_SNs   [{M-1}:0];\n" \
                   f"\tlogic       weight_SNs [{M-1}:0];\n" \
                   f"\tlogic       out_SN;\n\n"

    # data and weight RNSs
    file_string += f"\tlfsr0   lfsr(.clock(clock), .reset(reset), .state(data_r));\n"
    file_string += f"\tcounter cnt(.clock(clock), .reset(reset), .state(weight_r));\n"

    # data and weight SNGs
    pcc_array_name = f"compara_array{pcc_array_ID}" if pcc_array_name is None else f"{pcc_array_name}{pcc_array_ID}"
    file_string += f"\t{pcc_array_name} pccs{pcc_array_ID}(.in(data), .r(data_r), .SNs(data_SNs));\n" \
                   f"\tsuc_weight_gen wgen(.r(weight_r), .out(weight_SNs));\n"

    # suc mac unit
    file_string += "\tsuc_mac mac(.input_SNs(data_SNs), .weight_SNs(weight_SNs), .out_SN(out_SN));\n"

    # output counter
    if bipolar:
        file_string = f"bipolar_converter bip_conv(.in_SN(tree_out), .out(conv_out));\n"
        file_string += f"\taccumulator{out_cnt_ID} accum{out_cnt_ID}(.clock(clock), .reset(reset), .data_in(out_SN), .out(out));\n"
    else:
        file_string += f"\taccumulator{out_cnt_ID} accum{out_cnt_ID}(.clock(clock), .reset(reset), .data_in(out_SN), .out(out));\n"

    file_string += "endmodule\n\n\n"
    return file_string


##############################
#   Output Related Modules   #
##############################
# TODO Doc string
def gen_verilog_output_counter(precision, bipolar, ID="", override=False):
    n = precision
    if bipolar:
        file_string = f"module en_counter{ID} (\n" \
                      f"\tinput  clock, reset,\n" \
                      f"\tinput  logic en,\n" \
                      f"\toutput logic signed [{n}:0] out\n);\n" \
                      f"\talways_ff @(posedge clock) begin\n" \
                      f"\t\tif      (reset == 1)   out <= 'b0;\n" \
                      f"\t\telse if (en == 1)      out <= out + 1;\n" \
                      f"\t\telse                   out <= out - 1;\n" \
                      f"\tend\n" \
                      f"endmodule\n\n\n"
    else:
        file_string = f"module en_counter(\n" \
                      f"\tinput  clock, reset,\n" \
                      f"\tinput  logic en,\n" \
                      f"\toutput logic [{n-1}:0] out\n);\n" \
                      f"\talways_ff @(posedge clock) begin\n" \
                      f"\t\tif (reset == 1) out <= 'b0; else\n" \
                      f"\t\t                out <=  out + en;\n" \
                      f"\tend\n" \
                      f"endmodule\n\n\n"

    return file_string


# TODO: docstring
def gen_verilog_output_accumulator(in_prec, out_prec, bipolar, ID=""):
    n_in = in_prec
    n_out =out_prec
    if bipolar:
        file_string = f"module accumulator{ID} (\n" \
                      f"\tinput  clock, reset,\n" \
                      f"\tinput  logic signed [{n_in-1}:0] data_in,\n" \
                      f"\toutput logic signed [{n_out-1}:0] out\n);\n" \
                      f"\talways_ff @(posedge clock) begin\n" \
                      f"\t\tif      (reset == 1)   out <= 'b0;\n" \
                      f"\t\telse                   out <= out + data_in;\n" \
                      f"\tend\n" \
                      f"endmodule\n\n\n"
    else:
        file_string = f"module accumulator{ID}(\n" \
                      f"\tinput  clock, reset,\n" \
                      f"\tinput  logic [{n_in-1}:0] data_in,\n" \
                      f"\toutput logic [{n_out-1}:0] out\n);\n" \
                      f"\talways_ff @(posedge clock) begin\n" \
                      f"\t\tif (reset == 1) out <= 'b0; else\n" \
                      f"\t\t                out <=  out + data_in;\n" \
                      f"\tend\n" \
                      f"endmodule\n\n\n"

    return file_string


##########################################
#    FIR Filter and Filterbank Modules   #
##########################################
def gen_verilog_filter_memory(quant_norm_coefs, pcc_precision, rns_precision, gated, ID=""):
    """
    Generate verilog code for an FIR filter's memory (control) module.
    :param quant_norm_coefs: FIR filter's quantized, normalized coefficients
    :param pcc_precision: bit-width of the filter's PCCs
    :param rns_precision: bit-width of the filter's RNS
    :param gated: whether to clock gate the flip flops (keep this as True for best performance)
    :param ID: identification number or string to append to the end of the module's name. This is used when you need more than
    one logically distinct comparator arrays in a single verilog file.
    :return: verilog code for the module in string form.
    """
    rns_n, pcc_n = rns_precision, pcc_precision
    # The filter's memory input size is the number of weights minus the number of 0 weights at end of filter.
    memory_size = len(quant_norm_coefs)
    while quant_norm_coefs[memory_size - 1] == 0:
        memory_size -= 1
    adjusted_quant_norm_weights = quant_norm_coefs[0:memory_size]

    # the output of filter memory is the input to PCC array. Only output the memory elements who have nonzero weights
    output_size = np.sum(adjusted_quant_norm_weights != 0)

    file_string = f"module memory{ID} (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\tinput  logic [{pcc_n - 1}:0] in,\n" \
                  f"\tinput  logic [{rns_n - 1}:0] count,\n" \
                  f"\toutput logic [{pcc_n - 1}:0] out [{output_size - 1}:0]\n);\n"
    file_string += f"\tlogic [{pcc_n - 1}:0] registers [{memory_size - 1}:0];\n"

    out_idx = 0
    for mem_idx in range(len(quant_norm_coefs)):
        if quant_norm_coefs[mem_idx] != 0:
            file_string += f"\tassign out[{out_idx}] = registers[{mem_idx}];\n"
            out_idx += 1
    assert out_idx == output_size

    # Set up the sequential logic
    if not gated:
        count_string = '1' * rns_n
        file_string += f"\n\talways_ff @(posedge clock) begin\n" \
                       f"\t\tif (reset == 1) begin\n" \
                       f"\t\t\tfor(int i=0; i<{memory_size}; i=i+1) registers[i] <= 'b0;\n" \
                       f"\t\tend\n" \
                       f"\t\telse begin\n" \
                       f"\t\t\tif (count == {rns_n}'b{count_string}) begin\n" \
                       f"\t\t\t\tregisters[{memory_size- 1}:1] <= registers[{memory_size- 2}:0];\n" \
                       f"\t\t\t\tregisters[0] <= in;\n" \
                       f"\t\t\tend\n" \
                       f"\t\tend\n" \
                       f"\tend\n" \
                       f"endmodule\n\n\n"
    else:
        file_string += f"\n\talways_ff @(posedge clock) begin\n" \
                       f"\t\tif (reset == 1) begin\n" \
                       f"\t\t\tfor(int i=0; i<{memory_size}; i=i+1) registers[i] <= 'b0;\n" \
                       f"\t\tend\n" \
                       f"\t\telse begin\n" \
                       f"\t\t\tregisters[{memory_size-1}:1] <= registers[{memory_size-2}:0];\n" \
                       f"\t\t\tregisters[0] <= in;\n" \
                       f"\t\tend\n" \
                       f"\tend\n" \
                       f"endmodule\n\n\n"

    return file_string


# TODO: Docstring
def gen_verilog_cemux_filtercore(precision, pcc_input_size, pcc_array_ID, tree_ID, output_ID, latency_factor=None,
                                 pcc_array_name=None, bipolar=True):
    latency_factor = precision if latency_factor is None else latency_factor

    sign_str = "signed " if bipolar else ""
    assert latency_factor >= precision, "Latency factor must be at least as large as the precision"
    rns_n, pcc_n = latency_factor, precision
    M = pcc_input_size

    file_string = f"module filter_core (\n" \
                   f"\tinput  clock, reset,\n" \
                   f"\tinput  logic [{pcc_n-1}:0]  pcc_in [{M-1}:0],\n" \
                   f"\tinput  logic [{rns_n-1}:0]  mux_select,\n" \
                   f"\tinput  logic [{rns_n-1}:0]  r,\n" \
                   f"\toutput logic {sign_str}[{rns_n-1+int(bipolar)}:0]  out\n);\n"

    # Initialize wires
    file_string += f"\tlogic       data_SNs         [{M-1}:0];\n" \
                   f"\tlogic       tree_out;\n\n"

    # Write the PCC array
    pcc_array_name = f"compara_array{pcc_array_ID}" if pcc_array_name is None else f"{pcc_array_name}{pcc_array_ID}"
    file_string += f"\t{pcc_array_name} pccs{pcc_array_ID}(.in(pcc_in), .r(r[{rns_n-1}:{rns_n-pcc_n}]), .SNs(data_SNs));\n"

    # Write the mux tree
    file_string += "\n"
    file_string += f"\thw_tree{tree_ID} tree{tree_ID}(.data_SNs(data_SNs), .mux_select(mux_select), .out_SN(tree_out));\n"

    # Write the output counter
    file_string += f"\ten_counter{output_ID} est{output_ID}(.clock(clock), .reset(reset), .en(tree_out), .out(out));\n"
    file_string += "endmodule\n\n\n"
    return file_string


def gen_verilog_cemaj_filtercore(precision, pcc_input_size, pcc_array_ID, tree_ID, output_ID, latency_factor=None):
    latency_factor = precision if latency_factor is None else latency_factor
    assert latency_factor >= precision, "Latency factor must be at least as large as the precision"
    rns_n, pcc_n = latency_factor, precision

    M = pcc_input_size
    file_string = f"module filter_core (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\tinput  logic [{pcc_n-1}:0]  pcc_in [{M-1}:0],\n" \
                  f"\tinput  logic [{rns_n-1}:0]  mux_select,\n" \
                  f"\tinput  logic [{rns_n-1}:0]  r,\n" \
                  f"\toutput logic [{rns_n-1}:0]  out\n);\n"

    # Initialize wires
    file_string += f"\tlogic       data_SNs         [{M-1}:0];\n" \
                   f"\tlogic       tree_out;\n\n"

    # Write the PCC array
    pcc_array_name = f"wbg_array{pcc_array_ID}"
    file_string += f"\t{pcc_array_name} pccs{pcc_array_ID}(.in(pcc_in), .r(r[{rns_n-1}:{rns_n-pcc_n}]), .SNs(data_SNs));\n"

    # Write the mux tree
    file_string += "\n"
    file_string += f"\thw_tree{tree_ID} tree{tree_ID}(.data_SNs(data_SNs), .select_SN(mux_select), .out_SN(tree_out));\n"

    # Write the output counter
    file_string += f"\ten_counter{output_ID} est{output_ID}(.clock(clock), .reset(reset), .en(tree_out), .out(out));\n"
    file_string += "endmodule\n\n\n"
    return file_string


# TODO: Docstring
def gen_verilog_cemux_filter_toplevel(precision, pcc_input_size, gated=True, latency_factor=None, bipolar=True):
    latency_factor = precision if latency_factor is None else latency_factor
    assert latency_factor >= precision, "Latency factor must be at least as large as the precision"
    rns_n, pcc_n = latency_factor, precision
    sign_str = " signed" if bipolar else ""
    # Generate the verilog for the top level module
    file_string = f"module filter (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\tinput  logic [{pcc_n-1}:0]  in,\n" \
                  f"\toutput logic{sign_str} [{rns_n-1+int(bipolar)}:0]  stored_out\n);\n"

    # Initialize wires
    file_string += f"\tlogic [{pcc_n-1}:0] pcc_in [{pcc_input_size-1}:0];\n" \
                   f"\tlogic [{rns_n-1}:0] mux_select;\n" \
                   f"\tlogic [{rns_n-1}:0] r;\n" \
                   f"\tlogic{sign_str} [{rns_n-1+int(bipolar)}:0] out;\n"
    if gated:
        file_string += f"\tlogic gated_clock;\n\n" \
                       f"\tassign gated_clock = clock & (&mux_select);\n"
    else:
        file_string += "\n"

    # Write the submodules
    # Write the RNS and MSG
    file_string += f"\tcounter cnt(.clock(clock), .reset(reset), .rev_state(r), .state(mux_select));\n"

    # Write the memory module
    if gated:
        file_string += f"\tmemory mem(.clock(gated_clock), .reset(reset), .in(in), .out(pcc_in), .count(mux_select));\n"
    else:
        file_string += f"\tcontrol ctrl(.clock(clock), .reset(reset), .in(in), .out(pcc_in), .count(mux_select));\n"

    # Write the filterbank core module
    file_string += f"\tfilter_core core(.clock(clock), .reset(reset), .mux_select(mux_select), .r(r), .pcc_in(pcc_in), .out(out));\n"

    # Write the output register
    file_string += "\n"

    if gated:
        file_string += f"\talways_ff @(posedge gated_clock) begin\n" \
                       f"\t\tif (reset == 1) begin\n" \
                       f"\t\t\tstored_out <= 'b0;\n" \
                       f"\t\tend\n" \
                       f"\t\telse begin\n" \
                       f"\t\t\tstored_out <= out;\n" \
                       f"\t\tend\n" \
                       f"\tend\n" \
                       f"endmodule"

    else:
        ones_string = '1' * rns_n
        file_string += f"\talways_ff @(posedge clock) begin\n" \
                       f"\t\tif (reset == 1) begin\n" \
                       f"\t\t\tstored_out <= 'b0;\n" \
                       f"\t\tend\n" \
                       f"\t\telse begin\n" \
                       f"\t\t\tif (mux_select == {rns_n}'b{ones_string}) begin\n" \
                       f"\t\t\t\tstored_out <= out;\n" \
                       f"\t\t\tend\n" \
                       f"\t\tend\n" \
                       f"\tend\n" \
                       f"endmodule"
    return file_string


###############
#    Other    #
###############
def gen_verilog_mult_array(num_inputs, bipolar):
    op = '~^' if bipolar else '&'
    M = num_inputs

    # header
    file_string = f"module mult_array(\n" \
                 f"\tinput  logic Xs    [{M-1}:0],\n" \
                 f"\tinput  logic Ws    [{M-1}:0],\n" \
                 f"\toutput logic prods [{M-1}:0]\n);\n"
    # logic
    for i in range(num_inputs):
        file_string += f"\tassign prods[{i}] = Xs[{i}] {op} Ws[{i}];\n"
    file_string += "endmodule\n\n\n"
    return file_string


# never tested (APC does bip conversion automatically)
def gen_verilog_bipolar_converter_array(num_inputs):
    M = num_inputs

    # header
    file_string = f"module bipolar_converter_array(\n" \
                  f"\tinput  logic              SNs [{M-1}:0],\n" \
                  f"\toutput logic signed [1:0] outs [{M-1}:0]\n);\n"
    # logic
    for i in range(num_inputs):
        file_string += f"\tbipolar_converter bip_conv{i}(.in_SN(SNs[{i}]), .out(outs[{i}]);\n"
    file_string += "endmodule;\n\n\n"
    return file_string


if __name__ == '__main__':
    # print(gen_verilog_mult_array(10, bipolar=True))
    # print(gen_verilog_mult_array(10, bipolar=False))
    print(gen_verilog_weighted_PSA_uneven_toplevel(8, 100, tree_height=4, bipolar=False, use_mults=True))
