import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operations_snn import NaiveBN
from models.SNN import SNN_2d_ASPP_bn

class Decoder_all(nn.Module):
    def __init__(self, num_classes, filter_multiplier, BatchNorm=NaiveBN, args=None, last_level=0):
        super(Decoder_all, self).__init__()
        low_level_inplanes = filter_multiplier
        C_low = 48
        self.conv1 = SNN_2d_ASPP_bn(low_level_inplanes, C_low, 1, bias=False)
        self.conv2 = SNN_2d_ASPP_bn(304,256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout1 = nn.Dropout(0.5)
        self.conv3 = SNN_2d_ASPP_bn(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout2 = nn.Dropout(0.1)
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self._init_weight()

    def forward(self, x, low_level_feat, is_first):
        low_level_feat = self.conv1(low_level_feat, is_first)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='nearest')
        x = torch.cat((x, low_level_feat), dim=1)
        x= self.conv2(x, is_first)
        x=self.dropout1(x)
        x= self.conv3(x, is_first)
        x=self.dropout2(x)
        x = self.final_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()