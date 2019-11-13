from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
from numba import njit, prange
import numpy as np
from local.torch_basics import *
from local.test import *
from local.basics import *
from local.data.all import *
from local.vision.core import *
from local.notebook.showdoc import show_doc
from local.audio.core import *
from local.audio.augment import *
from local.vision.learner import *
from local.vision.models.xresnet import *
from local.metrics import *
from local.callback.schedule import *

@njit
def generate_kernels(input_length, num_kernels, candidate_lengths=np.array((7, 9, 11)), stride=5):
    # initialise kernel parameters
    strides = np.ones(num_kernels, dtype = np.int32) * stride
    weights = np.zeros((num_kernels, candidate_lengths.max())) # see note
    lengths = np.zeros(num_kernels, dtype = np.int32) # see note
    biases = np.zeros(num_kernels)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)
    # note: only the first *lengths[i]* values of *weights[i]* are used
    for i in range(num_kernels):
        length = np.random.choice(candidate_lengths)
        _weights = np.random.normal(0, 1, length)
        bias = np.random.uniform(-1, 1)
        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) // (length - 1)))
        padding = ((length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        weights[i, :length] = _weights - _weights.mean()
        lengths[i], biases[i], dilations[i], paddings[i] = length, bias, dilation, padding
    return weights, lengths, biases, dilations, paddings, strides

@njit(fastmath=True)
def apply_kernel(X, weights, length, bias, dilation, padding, stride):
    # zero padding
    if padding > 0:
        _input_length = len(X)
        _X = np.zeros(_input_length + (2 * padding))
        _X[padding:(padding + _input_length)] = X
        X = _X
    input_length = len(X)
    output_length = (input_length - ((length - 1) * dilation))//stride
    _ppv = 0 # "proportion of positive values"
    _max = np.NINF
    for i in range(output_length):
        _sum = bias
        for j in range(length):
            _sum += weights[j] * X[i*stride + (j * dilation)]
        if _sum > 0:
            _ppv += 1
        if _sum > _max:
            _max = _sum
    return _ppv / output_length, _max

@njit(parallel=True, fastmath=True)
def apply_kernels(X, kernels):
    weights, lengths, biases, dilations, paddings, strides = kernels
    num_examples = len(X)
    num_kernels = len(weights)
    # initialise output
    _X = np.zeros((num_examples, num_kernels * 2)) # 2 features per kernel
    for i in prange(num_examples):
        for j in range(num_kernels):
            _X[i, (j * 2):((j * 2) + 2)] = \
            apply_kernel(X[i], weights[j][:lengths[j]], lengths[j], biases[j], dilations[j], paddings[j], strides[j])
    return _X

class ROCKETMAX(nn.Module):
    def __init__(self, c_in, seq_len, n_kernels=100, kss=[7, 9, 11], stride=7):
        '''
        ROCKET is a Pytorch implementation of the ROCKET methods generate_kernels and apply_kernels that can be used 
        with univariate and multivariate time series.
        Input: is a 3d torch tensor of type torch.float32. When used with univariate TS, make sure you transform
        the 2d to 3d by adding unsqueeze(1)
        c_in: number of channels in (features). For univariate c_in is 1.
        seq_len: sequence length (is the last dimension of the input)
        '''
        super().__init__()
        kss = [ks for ks in kss if ks < seq_len]
        convs = nn.ModuleList()
        for i in range(n_kernels):
            ks = np.random.choice(kss)
            dilation = 2**np.random.uniform(0, np.log2((seq_len - 1) // (ks - 1)))
            padding = int((ks - 1) * dilation // 2) if np.random.randint(2) == 1 else 0
            weight = torch.normal(0, 1, (1, c_in, ks))
            weight -= weight.mean()
            bias = 2 * (torch.rand(1) - .5)
            layer = nn.Conv1d(c_in, 1, ks, stride=stride, padding=2 * padding, dilation=int(dilation), bias=True)
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
            convs.append(layer)
        self.convs = convs
        self.n_kernels = n_kernels
        self.kss = kss

    def forward(self, x):
        for i in range(self.n_kernels):
            NUM_PPV = 5
            MIN_BIN = -2
            MAX_BIN = 2
            # THIS IS A HACK: Catches when labels are passed to rocket, rocket needs to be a proper transform!
            if(len(x.shape) == 1): return(x)
            x = x.float()
            out = self.convs[i](x)
            _max = out.max(dim=-1).values
            bs,_,y = out.shape
            ppvs = torch.gt(out.permute(0,2,1).expand(bs,y,NUM_PPV), torch.linspace(MIN_BIN, MAX_BIN, NUM_PPV).cuda()).sum(dim=1).float()/ out.shape[-1]
            cat = torch.cat((_max, ppvs), dim=-1)
            
            output = cat if i == 0 else torch.cat((output, cat), dim=-1)
        #This is a hack to get the right dims
        return output.unsqueeze(1)
    
class ROCKET(nn.Module):
    def __init__(self, c_in, seq_len, n_kernels=100, kss=[7, 9, 11], stride=7):
        '''
        ROCKET is a Pytorch implementation of the ROCKET methods generate_kernels and apply_kernels that can be used 
        with univariate and multivariate time series.
        Input: is a 3d torch tensor of type torch.float32. When used with univariate TS, make sure you transform
        the 2d to 3d by adding unsqueeze(1)
        c_in: number of channels in (features). For univariate c_in is 1.
        seq_len: sequence length (is the last dimension of the input)
        '''
        super().__init__()
        kss = [ks for ks in kss if ks < seq_len]
        convs = nn.ModuleList()
        for i in range(n_kernels):
            ks = np.random.choice(kss)
            dilation = 2**np.random.uniform(0, np.log2((seq_len - 1) // (ks - 1)))
            padding = int((ks - 1) * dilation // 2) if np.random.randint(2) == 1 else 0
            weight = torch.normal(0, 1, (1, c_in, ks))
            weight -= weight.mean()
            bias = 2 * (torch.rand(1) - .5)
            layer = nn.Conv1d(c_in, 1, ks, stride=stride, padding=2 * padding, dilation=int(dilation), bias=True)
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
            convs.append(layer)
        self.convs = convs
        self.n_kernels = n_kernels
        self.kss = kss

        
    def forward(self, x):
        for i in range(self.n_kernels):
            if(len(x.shape) == 1): return(x)
            x = x.float()
            out = self.convs[i](x)
            _max = out.max(dim=-1).values
            _ppv = torch.gt(out, 0).sum(dim=-1).float() / out.shape[-1]
            cat = torch.cat((_max, _ppv), dim=-1)
            output = cat if i == 0 else torch.cat((output, cat), dim=-1)
        #This is a hack to get the right dims
        return output.unsqueeze(1)