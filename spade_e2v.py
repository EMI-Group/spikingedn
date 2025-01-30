from telnetlib import PRAGMA_HEARTBEAT
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SNN import Spike, SNN_2d_SSAM
import numpy as np

class SPADE_snn_v2(nn.Module):
    def __init__(self, norm_nc, label_nc, nhidden=64,h_channel=5,tau_SSAM=0.2):    # norm_nc=out channel label_nc=
        super().__init__()

        # instance normalization
        self.h_channel = h_channel # if v3_type =1 or 12, channel can change
        self.tau_SSAM = tau_SSAM
        self.nhidden = nhidden
        ks = 3
        pw = ks // 2
        self.mlp_shared = SNN_2d_SSAM(label_nc, nhidden, kernel_size=ks, padding=pw,decay=self.tau_SSAM)

        self.conv2 = nn.Conv2d(nhidden,nhidden//4,kernel_size=3,padding=1,dilation=1)
        self.conv3 = nn.Conv2d(nhidden,nhidden//4,kernel_size=3,padding=2,dilation=2)
        self.conv4 = nn.Conv2d(nhidden,nhidden//4,kernel_size=3,padding=3,dilation=3)
        self.conv5 = nn.Conv2d(nhidden,nhidden//4,kernel_size=3,padding=4,dilation=4)

        self.spike2 = Spike(b=3)
        self.mlp_beta = nn.Conv2d(nhidden, self.h_channel, kernel_size=ks, padding=pw)
        self.sparsity = 0
        

    def forward(self, segmap,x, is_first):

        normalized = x
        segmap = F.interpolate(segmap, size=x.size()[-2:], mode='nearest') 
        actv = self.mlp_shared(segmap, is_first)
        img_features_2 = self.conv2(actv)
        img_features_3 = self.conv3(actv)
        img_features_4 = self.conv4(actv)
        img_features_5 = self.conv5(actv)
        actv = torch.cat((img_features_2,img_features_3,img_features_4,img_features_5),dim=1)
        out = actv
        # self.sparsity = out.sum()/np.prod(list(out.shape))
        # print("sparsity of spade:",self.sparsity)
        # print("sparsity of spade:",self.sparsity)
        # print("sparsity of spade:",self.sparsity)
        # print("sparsity of spade:",self.sparsity)
        # print("sparsity of spade:",self.sparsity)
        # a=b
        return out

