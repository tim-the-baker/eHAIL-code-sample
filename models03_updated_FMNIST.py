"""
The models in this file follow the structure from "Stochastic Computing for Hardware Implementation
of Binarized Neural Networks." The input is a grayscale image and output is a classification. The network has 2
hidden layers of size 1024. There are three implementations: a BNN, a hybrid SC net (SC_BNN) and a SC net (SCNN).
"""
import torch.nn as nn
from Code_examples.helpers.binarized_modules \
    import BinarizeLinear, Binarize, SCMuxLinear, BinarizeConv2d, Quantize, PSA_Linear


# Global parameters
EPS = 1e-5

class NN_MLP2(nn.Module):
    """
    This model is a full precision neural network.
    """
    def __init__(self, input_prec, bias, affine, dropout_p):
        super(NN_MLP2, self).__init__()
        self.w, self.h = 28, 28
        self.input_prec, self.bias, self.affine = input_prec, bias, affine
        self.dropout_p = dropout_p
        assert input_prec == 9, "Only implemented for full precision!"

        # hyper-parameters
        self.hidden_layer_size = 1024
        self.output_size = 10
        self.pretrained = False

        # input layer to hidden layer 1
        self.fc1 = nn.Linear(self.h*self.w, self.hidden_layer_size, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(self.hidden_layer_size, affine=affine)

        # hidden layer 1 to hidden layer 2
        self.fc2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size, bias=self.bias)
        self.bn2 = nn.BatchNorm1d(self.hidden_layer_size, affine=affine)

        # hidden layer 2 to output layer
        self.fc3 = nn.Linear(self.hidden_layer_size, self.output_size, bias=self.bias)
        self.bn3 = nn.BatchNorm1d(self.output_size, affine=False)

        # classification
        self.logsoftmax = nn.LogSoftmax()

        # other layers
        self.Htanh = nn.Hardtanh()
        self.dropout = nn.Dropout(p=self.dropout_p)


    def forward(self, x):
        x = x.view(-1, self.h*self.w)  # flatten

        # input to layer 1
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.Htanh(x)

        # layer 1 to layer 2
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.Htanh(x)

        # layer 2 to output
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)

        return self.logsoftmax(x)


class BNN_MLP2(nn.Module):
    """
    This model is a binarized neural network.
    """
    def __init__(self, input_prec, bias, affine, dropout_p):
        super(BNN_MLP2, self).__init__()
        self.w, self.h = 28, 28
        self.input_prec, self.bias, self.affine = input_prec, bias, affine
        self.dropout_p = dropout_p

        # hyper-parameters
        self.hidden_layer_size = 1024
        self.output_size = 10
        self.pretrained = False

        # input layer to hidden layer 1
        self.fc1 = BinarizeLinear(self.h*self.w, self.hidden_layer_size, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(self.hidden_layer_size, affine=affine)

        # hidden layer 1 to hidden layer 2
        self.fc2 = BinarizeLinear(self.hidden_layer_size, self.hidden_layer_size, bias=self.bias)
        self.bn2 = nn.BatchNorm1d(self.hidden_layer_size, affine=affine)

        # hidden layer 2 to output layer
        self.fc3 = BinarizeLinear(self.hidden_layer_size, self.output_size, bias=self.bias)
        self.bn3 = nn.BatchNorm1d(self.output_size, affine=False)

        # classification
        self.logsoftmax = nn.LogSoftmax()

        # other layers
        self.Htanh = nn.Hardtanh()
        self.dropout = nn.Dropout(p=self.dropout_p)


    def forward(self, x):
        x = x.view(-1, self.h*self.w)  # flatten

        # input to layer 1
        x = self.dropout(x)
        x = self.fc1(x, input_prec=self.input_prec)
        x = self.bn1(x)
        x = self.Htanh(x) if self.training else Binarize(x)

        # layer 1 to layer 2
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.Htanh(x) if self.training else Binarize(x)

        # layer 2 to output
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)

        return self.logsoftmax(x)


class PSA_MLP2(nn.Module):
    def __init__(self, sng, msg, SN_length, samp_height, bias, bipolar, full_corr, sep_bip):
        super(PSA_MLP2, self).__init__()
        assert samp_height <= sng.rns.n
        assert (bipolar ^ sep_bip)
        # assert bipolar or not full_corr, "full_corr cannot be used with unipolar"

        self.w, self.h = 28, 28
        self.sng, self.msg = sng, msg
        self.SN_length, self.samp_height = SN_length, samp_height
        self.bipolar = bipolar
        self.full_corr = full_corr
        self.sep_bip = sep_bip

        self.hidden_layer_size = 1024
        self.output_size = 10
        self.pretrained = False

        # input layer to hidden layer 1
        self.fc1 = PSA_Linear(samp_height, msg, rem_mode='apc', sep_bip=self.sep_bip, in_features=self.h*self.w,
                              out_features=self.hidden_layer_size, bias=bias)
        self.bn1 = nn.BatchNorm1d(self.hidden_layer_size, affine=False)

        # hidden layer 1 to hidden layer 2
        self.fc2 = BinarizeLinear(self.hidden_layer_size, self.hidden_layer_size, bias=bias)
        self.bn2 = nn.BatchNorm1d(self.hidden_layer_size, affine=False)

        # hidden layer 2 to output layer
        self.fc3 = BinarizeLinear(self.hidden_layer_size, self.output_size, bias=bias)
        self.bn3 = nn.BatchNorm1d(self.output_size, affine=False)

        # classification
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # other layers
        self.Htanh = nn.Hardtanh()
        self.dropout = nn.Dropout()

    def forward(self, x, SN_length=None):
        x = x.view(-1, self.h*self.w)  # flatten

        if SN_length is None:
            SN_length = self.SN_length

        # x starts off as shape (N, K) where
        # N: batch size, K: input nodes

        # generate L-bit SNs and convert to bipolar {-1, 1} representation
        x = self.sng.gen_SN(x, SN_length, bipolar=self.bipolar, share_RNS=True, full_corr=self.full_corr, RNS_mask=None)  # (N, K, L) or (N, K, 2, L)
        x = x.short()
        x = 2*x-1 if self.bipolar else x

        x = self.fc1(x, self.full_corr)
        x = x*self.fc1.in_features/SN_length if self.pretrained else x.float()

        x = self.bn1(x)
        x = self.Htanh(x) if self.training else Binarize(x)

        # layer 1 to layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.Htanh(x) if self.training else Binarize(x)

        # layer 2 to output
        x = self.fc3(x)
        x = self.bn3(x)

        return self.logsoftmax(x)


    def freeze_batchnorm(self):
        self.bn1.eval()


