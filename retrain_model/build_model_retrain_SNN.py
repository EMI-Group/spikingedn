import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from time import time
# from retrain_model.build_autodeeplab_SNN import Retrain_Autodeeplab_SNN
from retrain_model.build_autodeeplab_SNN import Retrain_Autodeeplab_SNN
import numpy as np

class AutoRetrain(nn.Module):
    def __init__(self, args):
        super(AutoRetrain, self).__init__()
        self.timestep = args.timestep
        self.retrain_autodeeplab = Retrain_Autodeeplab_SNN(args)
        self.burning_time = args.burning_time
        
    def forward(self, x): 
        # x: B, S, C, H, W
        x_outs = []
        cost_all = []
        # for GRB image ,convert [B,C,H,W] to [B, S, C,H,W]
        if len(x.shape) == 4:
            S = self.timestep
            is_DVS = 0    # for input
            if self.initial_channels  != x.shape[1]:
                x = x[:,:self.initial_channels]
        else:
            S = x.shape[1]
            is_DVS = 1
        for i in range(S): # B,15,5,260,346 # TODO preframe = 10 # TODO batch = 10
            if i == 0:
                is_first = 1
            else:
                is_first = 0
            if is_DVS == 0:    # rgb
                if i == 0:  
                    x_out = self.retrain_autodeeplab(x,is_first)
                else:
                    x_out += self.retrain_autodeeplab(x,is_first)
            else:   # dvs
                x_out = self.retrain_autodeeplab(x[:,i],is_first)
                cost_all.append(x_out.unsqueeze(1))
        if is_DVS == 0:
            cost = x_out
        else:
            cost = torch.cat(cost_all,1)
            cost = cost[:,int(self.burning_time):]
        return cost
