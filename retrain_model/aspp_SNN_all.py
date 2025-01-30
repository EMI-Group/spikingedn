import torch
import torch.nn as nn
import numpy as np
from operations_snn import NaiveBN
from models.SNN import SNN_2d_ASPP_bn, SNN_AdaptiveAvgPool2d

class ASPP_all(nn.Module):
    def __init__(self, C, depth, num_classes, conv=nn.Conv2d, norm=NaiveBN, momentum=0.0003, mult=1):
        super(ASPP_all, self).__init__()
        self._C = C
        self._depth = depth
        self._num_classes = num_classes
        self.global_pooling = SNN_AdaptiveAvgPool2d(1)
        self.aspp1 = SNN_2d_ASPP_bn(C, depth, kernel_size=1, stride=1, bias=False, momentum = momentum)     # dilation=1
        self.aspp2 = SNN_2d_ASPP_bn(C, depth, kernel_size=3, stride=1, 
                            dilation=int(6 * mult), padding=int(6 * mult),
                            bias=False, momentum = momentum)    
        self.aspp3 = SNN_2d_ASPP_bn(C, depth, kernel_size=3, stride=1, 
                            dilation=int(12 * mult), padding=int(12 * mult),
                            bias=False, momentum = momentum)  
        self.aspp4 = SNN_2d_ASPP_bn(C, depth, kernel_size=3, stride=1, 
                            dilation=int(18 * mult), padding=int(18 * mult),
                            bias=False, momentum = momentum)
        self.aspp5 = SNN_2d_ASPP_bn(C, depth, kernel_size=1, stride=1, 
                            bias=False, momentum = momentum)
        self.conv2 = SNN_2d_ASPP_bn(depth * 5, depth,  kernel_size=1, stride=1, 
                    bias=False, momentum = momentum)
        self._init_weight()

    def forward(self, x, is_first):
        x1 = self.aspp1(x,is_first)
        x2 = self.aspp2(x,is_first)
        x3 = self.aspp3(x,is_first)
        x4 = self.aspp4(x,is_first)
        x5 = self.global_pooling(x,is_first)
        x5 = self.aspp5(x5,is_first)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='nearest')(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x,is_first)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
