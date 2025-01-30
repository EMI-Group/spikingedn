import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch

from operations_snn import NaiveBN, ABN
from retrain_model.aspp_SNN_all import ASPP_all
from retrain_model.decoder_SNN_all import Decoder_all
from retrain_model.new_model_SNN import get_default_arch, newModel
from spade_e2v import SPADE_snn_v2


class Retrain_Autodeeplab_SNN(nn.Module):
    def __init__(self, args):
        super(Retrain_Autodeeplab_SNN, self).__init__()
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        BatchNorm2d = ABN if args.use_ABN else NaiveBN
        if (not args.dist and args.use_ABN) or (args.dist and args.use_ABN and dist.get_rank() == 0):
            print("=> use ABN!")
        if args.net_arch is not None and args.cell_arch is not None and args.net_path is not None:
            network_arch, cell_arch, network_path = np.load(args.net_arch), np.load(args.cell_arch).astype('uint8'), np.load(args.net_path)  # space\cell\backbone
        else:
            network_arch, cell_arch, network_path = get_default_arch()
        self.encoder = newModel(network_arch, cell_arch, args.num_classes, args.num_layer, args.filter_multiplier, BatchNorm=BatchNorm2d, args=args)
        self.spade_type=args.spade_type  # 0: event only; 1: ssam (e+f)
        self.aps_channel = args.aps_channel
        self.h_channel = args.h_channel
        self.tau_SSAM = args.tau_SSAM
        
        self.aspp = ASPP_all(args.filter_multiplier * args.block_multiplier * filter_param_dict[network_path[-1]],
                        256, args.num_classes, conv=nn.Conv2d, norm=BatchNorm2d)
        self.decoder = Decoder_all(args.num_classes, filter_multiplier=args.filter_multiplier * args.block_multiplier * filter_param_dict[network_path[2]],
                                args=args, last_level=network_path[-1])
        if self.spade_type == 1:
            self.spade_bn = args.spade_bn
            self.multi_gamma = args.multi_gamma
            self.spade_v3_type = args.spade_v3_type
            self.SSAM_type = args.SSAM_type
            self.SSAM_ALIF = args.SSAM_ALIF
            self.Spade = SPADE_snn_v2(int(args.initial_channels)-int(args.aps_channel),args.aps_channel,64,self.h_channel,self.tau_SSAM) # norm_nc(events shannel)  ,label_nc (image channel),nhidden
           
    def forward(self, x,is_first):
        if self.spade_type==1:  # ssam (e+f)
            x_spade = self.Spade(x[:,:self.aps_channel], x[:,self.aps_channel:], is_first)   # segmap,x
            encoder_output, low_level_feature = self.encoder(x[:,self.aps_channel:],x_spade, is_first)
        else: # event_only
            x_spade = None
            encoder_output, low_level_feature = self.encoder(x, x_spade, is_first)
            # encoder_output, low_level_feature = self.encoder(x, is_first)
        high_level_feature = self.aspp(encoder_output, is_first)
        decoder_output = self.decoder(high_level_feature, low_level_feature, is_first)
        return nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)(decoder_output)
