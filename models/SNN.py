import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import *
import numpy as np
thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.2 # decay constants
b_delta = 1
alif_thresh = 0.5

class ActFun_changeable(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, input, b):
        ctx.save_for_backward(input)
        ctx.b = b
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        original_bp = False
        if original_bp:
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            temp = abs(input - thresh) < lens
        else:
            input, = ctx.saved_tensors
            device = input.device
            grad_input = grad_output.clone()
            # temp = abs(input - thresh) < lens
            b = torch.tensor(ctx.b,device=device)
            temp = (1-torch.tanh(b*(input-0.5))**2)*b/2/(torch.tanh(b/2))
            temp[input<=0]=0
            temp[input>=1]=0
        return grad_input * temp.float(), None

class SNN_2d_SSAM(nn.Module):

    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3, decay=0.2):
        super(SNN_2d_SSAM, self).__init__()
        
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_c)
        self.act_fun = ActFun_changeable().apply
        self.b = b
        self.mem = None

        self.input_c = input_c
        self.output_c = output_c
        self.decay = decay
        self.sparsity = 0

    def forward(self, input, is_first): #20
        device = input.device
        if not self.bn.training:
            conv_bn = fuse_conv_bn_eval(self.conv1, self.bn)
            output = conv_bn(input)
        else:
            output = self.bn(self.conv1(input))
        if self.mem is None:
            self.mem = torch.zeros_like(output, device=device)
        if is_first == 1 :
            self.mem = torch.zeros_like(output, device=device)
            is_first = 0

        if is_first == 0:
            self.mem += output
            spike = self.act_fun(self.mem, self.b) 
            self.mem = self.mem * self.decay * (1. - spike)  
        self.sparsity = spike.sum()/np.prod(list(spike.shape))
        return spike

class SNN_2d_1(nn.Module):

    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3):
        super(SNN_2d_1, self).__init__()
        
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_c)
        self.act_fun = ActFun_changeable().apply
        self.b = b
        self.mem = None
        self.input_c = input_c
        self.output_c = output_c

    def forward(self, input, is_first): #20
        device = input.device
        if self.mem is None:
            self.mem = torch.zeros_like(self.conv1(input), device=device)
        if is_first == 1 :
            self.mem = torch.zeros_like(self.conv1(input), device=device)
            is_first = 0

        if is_first == 0:
            self.mem += self.conv1(input)
            spike = self.act_fun(self.mem, self.b) 
            self.mem = self.mem * decay * (1. - spike)  
        return spike

class SNN_2d(nn.Module):

    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3):
        super(SNN_2d, self).__init__()
        
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_c)
        self.act_fun = ActFun_changeable().apply
        self.b = b
        self.mem = None
        self.input_c = input_c
        self.output_c = output_c

    def forward(self, input, is_first): #20
        device = input.device
        if not self.bn.training:
            conv_bn = fuse_conv_bn_eval(self.conv1, self.bn)
            output = conv_bn(input)
        else:
            output = self.bn(self.conv1(input))
        if self.mem is None:
            self.mem = torch.zeros_like(output, device=device)
        if is_first == 1 :
            self.mem = torch.zeros_like(output, device=device)
            is_first = 0

        if is_first == 0:
            self.mem += output
            spike = self.act_fun(self.mem, self.b) 
            self.mem = self.mem * decay * (1. - spike)  
        return spike

class Spike(nn.Module):

    def __init__(self, b=3):
        super(Spike, self).__init__()
        self.act_fun = ActFun_changeable().apply
        self.b = b
        self.mem = None
        self.sparsity = 0
    def forward(self, input, is_first): 
        device = input.device
        if self.mem is None:
            self.mem = torch.zeros_like(input, device=device)
        if is_first == 1 :
            self.mem = torch.zeros_like(input, device=device)
            is_first = 0
        if is_first == 0:
            self.mem += input
            spike = self.act_fun(self.mem, self.b) 
            self.mem = self.mem * decay * (1. - spike)  
        self.sparsity = spike.sum()/np.prod(list(spike.shape))
        return spike

class ActFun_lsnn(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, input, b, v_th):
        ctx.save_for_backward(input)
        ctx.b = b
        ctx.v_th = v_th
        return input.gt(v_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        device = input.device
        grad_input = grad_output.clone()
        b = torch.tensor(ctx.b,device=device)
        v_th = ctx.v_th.clone().detach()
        temp = (1-torch.tanh(b*(input-v_th))**2)*b/2/(torch.tanh(b/2))
        temp[input<=0]=0
        temp[input>=1]=0
        return grad_input * temp.float(), None,  - grad_input * temp.float()


class SNN_2d_lsnn(nn.Module):

    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3, dilation=1, act='spike'):
        super(SNN_2d_lsnn, self).__init__()
        
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act = act
        self.act_fun = ActFun_lsnn().apply
        self.b = b
        self.mem = None
        self.bn = nn.BatchNorm2d(output_c)
        self.relu = nn.LeakyReLU()
        
        
        self.a = None
        self.thresh = 0.3
        self.beta = 0.07
        self.rho = nn.Parameter(0.87*torch.ones(output_c,1,1)).requires_grad_(True)
        
    def forward(self, input, param): #20
        B, C, H, W = input.shape
        if not self.bn.training:
            conv_bn = fuse_conv_bn_eval(self.conv1,self.bn)
            mem_this = conv_bn(input)
        else:
            mem_this = self.bn(self.conv1(input))

        if param['mixed_at_mem']:
            return mem_this
        
        device = input.device
        if param['is_first']:
            self.mem = torch.zeros_like(mem_this, device=device)
            self.a = torch.zeros_like(self.mem, device=device)
            self.rho.data.clamp_(0.64,1.1)
        
        A = self.thresh + self.beta*self.a
        self.mem = self.mem + mem_this
        spike = self.act_fun(self.mem, self.b, A)
        self.mem = self.mem * decay * (1. - spike) 
        self.a = torch.exp(-1/self.rho)*self.a + spike
        return spike

class SNN_2d_lsnn23_spike(nn.Module):

    def __init__(self, output_c, b=3 , act='spike'):
        super(SNN_2d_lsnn23_spike, self).__init__()
        self.act = act
        self.act_fun = ActFun_lsnn().apply
        self.b = b
        self.mem = None
        self.relu = nn.LeakyReLU()
        self.a = None
        self.thresh = 0.3
        self.beta = 0.07
        self.rho = nn.Parameter(0.87*torch.ones(output_c,1,1)).requires_grad_(True)
        
    def forward(self, input, is_first): #20
        mem_this = input
        device = input.device
        if is_first:
            self.mem = torch.zeros_like(mem_this, device=device)
            self.a = torch.zeros_like(self.mem, device=device)
            self.rho.data.clamp_(0.64,2.3)
        
        A = self.thresh + self.beta*self.a
        self.mem = self.mem + mem_this
        spike = self.act_fun(self.mem, self.b, A)
        self.mem = self.mem * decay * (1. - spike) 
        self.a = torch.exp(-1/self.rho)*self.a + spike
        return spike

class SNN_2d_ASPP_bn(nn.Module):

    def __init__(self, input_c, output_c, kernel_size=3, stride=1, dilation=1,padding=0, bias=False, momentum = 0.1,b=3, withbn=1 ):
        super(SNN_2d_ASPP_bn, self).__init__()
        
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride,dilation=dilation,padding=padding,bias=bias)
        self.bn = nn.BatchNorm2d(output_c,momentum = momentum)
        self.act_fun = ActFun_changeable().apply
        self.b = b
        self.mem = None

        self.input_c = input_c
        self.output_c = output_c
        self.withbn = withbn
        self.sparsity = 0
        
    def forward(self, input, is_first): #20
        device = input.device
        if not self.bn.training:
            conv_bn = fuse_conv_bn_eval(self.conv1, self.bn)
            output = conv_bn(input)
        else:
            if self.withbn==0:
                output = self.conv1(input)
            else:
                output = self.bn(self.conv1(input))
        if is_first == 1 :
            self.mem = torch.zeros_like(output, device=device)
            is_first = 0

        if is_first == 0:
            self.mem += output
            spike = self.act_fun(self.mem, self.b) 
            self.mem = self.mem * decay * (1. - spike)  
        self.sparsity = spike.sum()/np.prod(list(spike.shape))
        return spike

class SNN_AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output, b=3):
        super(SNN_AdaptiveAvgPool2d, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(output)
        self.act_fun = ActFun_changeable().apply
        self.b = b
        self.mem = None
    def forward(self, input, is_first): #20
        device = input.device
        if is_first == 1 :
            self.mem = torch.zeros_like(self.pooling(input), device=device)
            is_first = 0

        if is_first == 0:
            self.mem += self.pooling(input)
            spike = self.act_fun(self.mem, self.b) 
            self.mem = self.mem * decay * (1. - spike)  
        return spike


# new for tnnls thresh 0.1-1 internal = 0.1
class SNN_2d_lsnn_thresh(nn.Module):

    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3, dilation=1, act='spike'):
        super(SNN_2d_lsnn_thresh, self).__init__()
        
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act = act
        self.act_fun = ActFun_lsnn().apply
        self.b = b
        self.mem = None
        self.bn = nn.BatchNorm2d(output_c)
        self.relu = nn.LeakyReLU()
        
        
        self.a = None
        self.thresh = alif_thresh
        self.beta = 0.07
        self.rho = nn.Parameter(0.87*torch.ones(output_c,1,1)).requires_grad_(True)
        
    def forward(self, input, param): #20
        B, C, H, W = input.shape
        if not self.bn.training:
            conv_bn = fuse_conv_bn_eval(self.conv1,self.bn)
            mem_this = conv_bn(input)
        else:
            mem_this = self.bn(self.conv1(input))

        if param['mixed_at_mem']:
            return mem_this
        
        device = input.device
        if param['is_first']:
            self.mem = torch.zeros_like(mem_this, device=device)
            self.a = torch.zeros_like(self.mem, device=device)
            self.rho.data.clamp_(0.64,1.1)
        
        A = self.thresh + self.beta*self.a
        self.mem = self.mem + mem_this
        spike = self.act_fun(self.mem, self.b, A)
        self.mem = self.mem * decay * (1. - spike) 
        self.a = torch.exp(-1/self.rho)*self.a + spike
        return spike

