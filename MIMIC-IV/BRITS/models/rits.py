import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
# import data_loader

# from ipdb import set_trace
from sklearn import metrics


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

    
def ordered_logistic_loss_with_probas(f_x, y_true, 
                                      cutpoints=torch.tensor([-10^6, 0.25, 0.5, 0.75, 10^6]),
                                      reduce=False):
    
    n_classes = len(cutpoints)-1
    
    log_p_y_NC = torch.zeros((len(f_x), n_classes))
    
    
    for ii in range(n_classes):
        z_1 = (cutpoints[ii+1]-f_x)
        z_2 = (cutpoints[ii]-f_x)
        log_p_y_NC[:, ii] = torch.squeeze(torch.log(torch.sigmoid(z_1) - torch.sigmoid(z_2)))
    
    # sum across all classes assuming independence
    log_p_y_N = log_p_y_NC.sum(axis=1)
    
    if not reduce:
        return -log_p_y_N
    else :
        return -log_p_y_N.sum()
    

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Model(nn.Module):
    def __init__(self, 
                 rnn_hid_size=64, 
                 impute_weight=0.3, 
                 label_weight=1.0):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight

        self.build()

    def build(self):
        input_size = 41
        self.rnn_cell = nn.LSTMCell(input_size * 2, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size = input_size, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = input_size, output_size = input_size, diag = True)

        self.hist_reg = nn.Linear(self.rnn_hid_size, input_size)
        self.feat_reg = FeatureRegression(input_size)

        self.weight_combine = nn.Linear(input_size * 2, input_size)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(self.rnn_hid_size, 1)

    def forward(self, data, direct):
        # Original sequence with 24 time steps
#         values = data[direct]['values']
#         masks = data[direct]['masks']
#         deltas = data[direct]['deltas']
        values, labels, masks, deltas = data

#         evals = data[direct]['evals']
#         eval_masks = data[direct]['eval_masks']

#         labels = data['labels'].view(-1, 1)
#         is_train = data['is_train'].view(-1, 1)
        D = values.size()[2]
        N = values.size()[0]
        SEQ_LEN = values.size()[1]
        h = Variable(torch.zeros((N, self.rnn_hid_size)))
        c = Variable(torch.zeros((N, self.rnn_hid_size)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)
            
            h = h * gamma_h
            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c =  m * x +  (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        y_h = self.out(h)
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)
        y_loss = torch.sum(y_loss) / (N + 1e-5)

        y_h = F.sigmoid(y_h)
                
        return {'loss': x_loss * self.impute_weight + y_loss * self.label_weight, 'predictions': y_h,\
                'imputations': imputations, 'labels': labels}

    def run_on_batch(self, data, optimizer, epoch = None):
        
        ret = self(data, direct = 'forward')
        
        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
    
class OrdinalModel(nn.Module):
    def __init__(self, 
                 rnn_hid_size=64, 
                 impute_weight=0.3, 
                 label_weight=1.0):
        super(OrdinalModel, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        
        # create non trainable cutpoints
        self.cutpoints=torch.tensor([-1000, 0.25, 0.5, 0.75, 1000], requires_grad=False)
        
#         self.cutpoints=torch.tensor([-10., -5., 0., 5., 10.], requires_grad=True)
        self.build()

    def build(self):
        input_size = 41
        self.rnn_cell = nn.LSTMCell(input_size * 2, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size = input_size, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = input_size, output_size = input_size, diag = True)

        self.hist_reg = nn.Linear(self.rnn_hid_size, input_size)
        self.feat_reg = FeatureRegression(input_size)

        self.weight_combine = nn.Linear(input_size * 2, input_size)

        self.dropout = nn.Dropout(p = 0.1)
        self.out = nn.Linear(self.rnn_hid_size, 1)

    def forward(self, data, direct):
        # Original sequence with 24 time steps
#         values = data[direct]['values']
#         masks = data[direct]['masks']
#         deltas = data[direct]['deltas']
        values, labels, masks, deltas = data

#         evals = data[direct]['evals']
#         eval_masks = data[direct]['eval_masks']

#         labels = data['labels'].view(-1, 1)
#         is_train = data['is_train'].view(-1, 1)
        D = values.size()[2]
        N = values.size()[0]
        SEQ_LEN = values.size()[1]
        h = Variable(torch.zeros((N, self.rnn_hid_size)))
        c = Variable(torch.zeros((N, self.rnn_hid_size)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)
            
            h = h * gamma_h
            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c =  m * x +  (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)
        
        y_h = self.out(h)
        y_h = F.sigmoid(y_h)
        
        
        y_loss = ordered_logistic_loss_with_probas(y_h, labels, 
                                                   cutpoints=self.cutpoints, 
                                                   reduce=False)
        y_loss = torch.sum(y_loss) / (N + 1e-5)
        
        return {'loss': x_loss * self.impute_weight + y_loss * self.label_weight, 'predictions': self.predict(y_h),\
                'imputations': imputations, 'labels': labels}

    def run_on_batch(self, data, optimizer, epoch = None):
        
        ret = self(data, direct = 'forward')
        
        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
    
    def predict(self, f_x):
        
        cutpoints = self.cutpoints
        n_classes = len(cutpoints)-1
        p_y_NC = torch.zeros((len(f_x), n_classes))


        for ii in range(n_classes):
            z_1 = (cutpoints[ii+1]-f_x)
            z_2 = (cutpoints[ii]-f_x)
            p_y_NC[:, ii] = torch.squeeze(torch.sigmoid(z_1) - torch.sigmoid(z_2))
        
        return p_y_NC
