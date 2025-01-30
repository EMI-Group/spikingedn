import torch
import torch.nn as nn
import numpy as np
from genotypes import PRIMITIVES1
from operations_snn import *
from models.SNN import  SNN_2d, SNN_2d_1, ActFun_changeable,Spike, SNN_2d_lsnn23_spike,SNN_2d_lsnn_thresh
import torch.nn.functional as F
from modeling.decoder import *
import matplotlib.pyplot as plt

class Cell(nn.Module):

    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, cell_arch, network_arch,
                 filter_multiplier, downup_sample, args=None):
        super(Cell, self).__init__()
        self.cell_arch = cell_arch

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2
        self.cell_arch = self.cell_arch[torch.sort(self.cell_arch[:,0],dim=0)[1]].to(torch.uint8)
        if self.block_multiplier == 5:      # node=5
            for x in self.cell_arch:
                primitive = PRIMITIVES1[x[1]]
                if x[0] in [0,2,5,9,14]:
                    op = OPS_C2B_ANN_BR[primitive](self.C_prev_prev, self.C_out, stride=1, c2b_sig=1)
                elif x[0] in [1,3,6,10,15]:
                    op = OPS_C2B_ANN_BR[primitive](self.C_prev, self.C_out, stride=1, c2b_sig=1)
                else:
                    op = OPS_C2B_ANN_BR[primitive](self.C_out, self.C_out, stride=1, c2b_sig=1)

                self._ops.append(op)
        elif self.block_multiplier == 3:  # node = 3
            for x in self.cell_arch:
                primitive = PRIMITIVES1[x[1]]
                if x[0] in [0,2,5]:
                    op = OPS_C2B_ANN_BR[primitive](self.C_prev_prev, self.C_out, stride=1, c2b_sig=1)
                elif x[0] in [1,3,6]:
                    op = OPS_C2B_ANN_BR[primitive](self.C_prev, self.C_out, stride=1, c2b_sig=1)
                else:
                    op = OPS_C2B_ANN_BR[primitive](self.C_out, self.C_out, stride=1, c2b_sig=1)

                self._ops.append(op)
        elif self.block_multiplier == 4:  # node = 4
            for x in self.cell_arch:
                primitive = PRIMITIVES1[x[1]]
                if x[0] in [0,2,5,9]:
                    op = OPS_C2B_ANN_BR[primitive](self.C_prev_prev, self.C_out, stride=1, c2b_sig=1)
                elif x[0] in [1,3,6,10]:
                    op = OPS_C2B_ANN_BR[primitive](self.C_prev, self.C_out, stride=1, c2b_sig=1)
                else:
                    op = OPS_C2B_ANN_BR[primitive](self.C_out, self.C_out, stride=1, c2b_sig=1)

                self._ops.append(op)
        self.mem = None
        self.act_fun = ActFun_changeable().apply


    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input, is_first):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_h = self.scale_dimension(s1.shape[2], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[3], self.scale)
            s1 = F.interpolate(s1, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3]),
                                            mode='bilinear', align_corners=True)
        device = prev_input.device
        states = [s0, s1]
        offset = 0
        ops_index = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue

                    if isinstance(self._ops[ops_index],Identity):
                        new_state = self._ops[ops_index](h)
                    else:
                        new_state = self._ops[ops_index](h)
                    # TODO
                    if self.mem == None:
                        self.mem = [torch.zeros_like(new_state,device=device)]*self._steps

                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)

            self.mem[i] = self.mem[i] + s
            spike = self.act_fun(self.mem[i],3)
            self.mem[i] = self.mem[i] * decay * (1. - spike) 
            offset += len(states)
            states.append(spike)
        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
        self.sparsity = concat_feature.sum()/np.prod(list(concat_feature.shape))
        return prev_input, concat_feature

class newModel(nn.Module):
    def __init__(self, network_arch, cell_arch, num_classes, num_layers, filter_multiplier=20, block_multiplier=5, step=5, cell=Cell,
                 BatchNorm=NaiveBN, args=None):
        super(newModel, self).__init__()
        self.args = args
        self._step = args.step
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._block_multiplier = args.block_multiplier
        self._filter_multiplier = args.filter_multiplier
        self.use_ABN = args.use_ABN
        self.initial_channels = args.initial_channels
        self.spade_type = args.spade_type
        self.aps_channel = args.aps_channel
        self.h_channel = args.h_channel
        self.spade_channel_1 = self.h_channel
        self.fix_tau = args.fix_tau

        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial / 2)
        
        # stem retrain snn: stem0 ---lif or alif, stem2 ---lif
        if self.spade_type > 0:
            self.stem0 = ConvBR(int(self.initial_channels)-int(self.aps_channel), self.spade_channel_1 ,kernel_size=1, stride=1, padding=0)
            self.stem2 = SNN_2d(self.spade_channel_1, f_initial * self._block_multiplier,kernel_size=3, stride=2, padding=1,b=3)
            self.spike_module = SNN_2d_lsnn23_spike(self.spade_channel_1,b=3)
        else:
            self.stem0 = SNN_2d_lsnn_thresh(self.initial_channels, half_f_initial * self._block_multiplier,kernel_size=1, stride=1, padding=0,b=3)
            self.stem2 = SNN_2d_1(half_f_initial * self._block_multiplier, f_initial * self._block_multiplier,kernel_size=3, stride=2, padding=1,b=3)

        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)     # which col
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()      #  1
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()
            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                if self.spade_type > 0:
                    _cell = cell(self._step, self._block_multiplier, self.spade_channel_1/args.block_multiplier,
                                self._filter_multiplier,
                                self.cell_arch, self.network_arch[i],
                                self._filter_multiplier *
                                filter_param_dict[level],
                                downup_sample, self.args)
                else:
                    _cell = cell(self._step, self._block_multiplier, self._filter_multiplier/2,
                                self._filter_multiplier,
                                self.cell_arch, self.network_arch[i],
                                self._filter_multiplier *
                                filter_param_dict[level],
                                downup_sample, self.args)
            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier,
                                self._filter_multiplier,
                                self._filter_multiplier * filter_param_dict[prev_level],
                                self.cell_arch, self.network_arch[i],
                                self._filter_multiplier *
                                filter_param_dict[level],
                                downup_sample, self.args)
                else:
                    _cell = cell(self._step, self._block_multiplier,
                                 self._filter_multiplier * filter_param_dict[prev_prev_level],
                                 self._filter_multiplier *
                                 filter_param_dict[prev_level],
                                 self.cell_arch, self.network_arch[i],
                                 self._filter_multiplier *
                                 filter_param_dict[level], downup_sample, self.args)

            self.cells += [_cell]

    def forward(self, x, x_spade, is_first):
        param = {'mixed_at_mem':False, 'is_first':False}
        if is_first == 1:
            param['is_first'] = True
        
        if self.spade_type > 0:
            stem0 = self.stem0(x)
            stem_spade = stem0 + x_spade
            spike_stem_spade = self.spike_module(stem_spade, is_first)
            stem0 = spike_stem_spade
            stem1 = self.stem2(spike_stem_spade, is_first)
        else:
            stem0 = self.stem0(x, param)
            stem1 = self.stem2(stem0, is_first)
        two_last_inputs = (stem0, stem1)
        for i in range(self._num_layers):
            two_last_inputs = self.cells[i](two_last_inputs[0], two_last_inputs[1], is_first)
            if i == 2:
                low_level_feature = two_last_inputs[1]
        last_output = two_last_inputs[-1]
        return last_output, low_level_feature

def network_layer_to_space(net_arch):
    for i, layer in enumerate(net_arch):
        if i == 0:
            space = np.zeros((1, 4, 3))
            space[0][layer][0] = 1
            prev = layer
        else:
            if layer == prev + 1:
                sample = 0
            elif layer == prev:
                sample = 1
            elif layer == prev - 1:
                sample = 2
            space1 = np.zeros((1, 4, 3))
            space1[0][layer][sample] = 1
            space = np.concatenate([space, space1], axis=0)
            prev = layer
    """
        return:
        network_space[layer][level][sample]:
        layer: 0 - 12
        level: sample_level {0: 1, 1: 2, 2: 4, 3: 8}
        sample: 0: down 1: None 2: Up
    """
    return space

def get_default_cell():
    cell = np.zeros((10, 2))
    cell[0] = [0, 7]
    cell[1] = [1, 4]
    cell[2] = [2, 4]
    cell[3] = [3, 6]
    cell[4] = [5, 4]
    cell[5] = [8, 4]
    cell[6] = [11, 5]
    cell[7] = [13, 5]
    cell[8] = [19, 7]
    cell[9] = [18, 5]
    return cell.astype('uint8')

def get_default_arch():
    backbone = [1, 0, 0, 1, 2, 1, 2, 2, 3, 3, 2, 1]
    cell_arch = get_default_cell()
    return network_layer_to_space(backbone), cell_arch, backbone

def get_default_net(args=None):
    filter_multiplier = args.filter_multiplier if args is not None else 20
    path_arch, cell_arch, backbone = get_default_arch()
    return newModel(path_arch, cell_arch, 19, args.num_layer, filter_multiplier=filter_multiplier, args=args)
