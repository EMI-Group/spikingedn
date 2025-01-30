import argparse


def obtain_retrain_evaluate_autodeeplab_args():
    parser = argparse.ArgumentParser(description="PyTorch SpikingEDN Training  and evaluate")
    parser.add_argument('--train', action='store_true', default=True, help='training mode')
    parser.add_argument('--exp', type=str, default='bnlr7e-3', help='name of experiment')
    parser.add_argument('--gpu', type=str, default='0', help='test time gpu device id')
    parser.add_argument('--backbone', type=str, default='autodeeplab', help='resnet101')
    parser.add_argument('--dataset', type=str, default='cityscapes', help='pascal or cityscapes')
    parser.add_argument('--groups', type=int, default=None, help='num of groups for group normalization')
    parser.add_argument('--epochs', type=int, default=4000, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=14, help='batch size')
    parser.add_argument('--base_lr', type=float, default=0.05, help='base learning rate')    # evaluate blr=0.00025 no use
    parser.add_argument('--warmup_start_lr', type=float, default=5e-6, help='warm up learning rate')
    parser.add_argument('--lr-step', type=float, default=None)
    parser.add_argument('--warmup-iters', type=int, default=1000)
    parser.add_argument('--min-lr', type=float, default=None)
    parser.add_argument('--last_mult', type=float, default=1.0, help='learning rate multiplier for last layers')
    parser.add_argument('--scratch', action='store_true', default=False, help='train from scratch')
    parser.add_argument('--freeze_bn', action='store_true', default=False, help='freeze batch normalization parameters')
    parser.add_argument('--weight_std', action='store_true', default=False, help='weight standardization')
    parser.add_argument('--beta', action='store_true', default=False, help='resnet101 beta')
    parser.add_argument('--crop_size', type=int, default=769, help='image crop size')    # evaluate 513 no use
    parser.add_argument('--resize', type=int, default=769, help='image crop size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    # parser.add_argument('--filter_multiplier', type=int, default=32)
    parser.add_argument('--filter_multiplier', type=int, default=8)
    parser.add_argument('--dist', type=bool, default=False)
    parser.add_argument('--autodeeplab', type=str, default='train')
    parser.add_argument('--block_multiplier', type=int, default=5)
    parser.add_argument('--use-ABN', default=True, type=bool, help='whether use ABN')
    parser.add_argument('--affine', default=False, type=bool, help='whether use affine in BN')
    parser.add_argument('--port', default=6000, type=int)
    parser.add_argument('--max-iteration', default=1000000, type=bool)
    parser.add_argument('--net_arch', default=None, type=str)
    parser.add_argument('--cell_arch', default=None, type=str)
    parser.add_argument('--net_path', default=None, type=str)
    parser.add_argument('--criterion', default='Ohem', type=str)
    parser.add_argument('--initial-fm', default=None, type=int)
    parser.add_argument('--mode', default='poly', type=str, help='how lr decline')
    parser.add_argument('--local_rank', dest='local_rank', type=int, default=-1, )
    parser.add_argument('--train_mode', type=str, default='iter', choices=['iter', 'epoch'])
    parser.add_argument('--eval_scales', default=(0.25,0.5,0.75,1,1.25,1.5),
                        type=bool, help='eval_scale in evaluate')
    # evaluation option
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no_val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--num_layer', type=int, default=6,
                        help='search num layer')
    parser.add_argument('--timestep', type=int, default=2,
                         help='time step for snn')
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--randomseed', action='store_true', default=False,
                        help='whether random seed experiment')
    parser.add_argument('--initial_channels', type=int, default=3,
                         help='the num of dvs input channel')
    parser.add_argument('--aps_only', action='store_true', default=False,
                        help='whether use aps only')

    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--base_size', type=int, default=320,
                        help='base image size')
    parser.add_argument('--burning_time', type=int, default=2,
                         help='burning time + sequence = timestep')
    parser.add_argument('--sequence', type=int, default=5,
                         help='time step for ddd17 input')
    parser.add_argument('--is_LR_milestone', type=int, default=0,
                        help="1 use milestone, 0 use others")
    parser.add_argument('--use_ltc', type=int, default=0,
                         help='0 is not use, 1 is use 1 ltc')
    parser.add_argument('--is_resize', type=int, default=0,
                         help='0 is not resize, 1 is use resize')
    parser.add_argument('--is_allsnn', type=int, default=0,
                        help='0 means stem and decoder not snn, 1 is all snn')
    parser.add_argument('--aps_channel', type=int, default=1,
                         help='the num of aps input channel 1 or 3')
    parser.add_argument('--spade_type',type=int, default=0,
                        help='if 0,no spade; if 1,spade as input; if 2,spade before decode; if 3, spade in decode')
    parser.add_argument('--spade_snn',type=int, default=0,
                        help='if 0 ann spade; if 1,spade is snn')
    parser.add_argument('--evdata_type',type=int, default=0,
                        help='if 2 first 2 channels; if 4, second 2 channels; if 6 last 2 channels')
    parser.add_argument('--spade_bn',type=int, default=1,
                        help='if 1, with bn ; if 0 ,without bn')
    parser.add_argument('--multi_gamma',type=int, default=1,
                        help='if 1, with multi_gamma ; if 0 ,without multi_gamma')
    parser.add_argument('--spade_spike',type=int, default=0,
                        help='if 1, final spade spike ; if 0 , final not spike')
    parser.add_argument('--dvs_spade', action='store_true', default=False,
                        help='whether to use dvs spade (default: True)')
    parser.add_argument('--fix_tau',type=float, default=0.2 ,
                        help='fix tau')
    parser.add_argument('--spade_v3_type',type=int, default=0,
                        help='if 0,no spade; if 1,spade as skip; if 2,spade skip spike;if 12,skip and spike')
    parser.add_argument('--use_sbt', action='store_true', default=False,
                        help='whether to use sbt sign(default: True)')
    parser.add_argument('--h_channel', type=int, default=5,
                         help='the num of aps(h) channel 5 or larger')
    parser.add_argument('--use50ms', action='store_true', default=False,
                        help='whether to use 50ms to change last frame(default: True)')
    parser.add_argument('--SSAM_type',type=int, default=1,
                        help='if 1,SSAM1 default; if 2,SSAM2:NO SNN_2D;if 3, SSAM3: NO BETA')
    parser.add_argument('--tau_SSAM', type=float, default=0.2, help=' tau of snn_2d SSAM')
    parser.add_argument('--SSAM_ALIF',type=int, default=31,
                        help='if 31 ssam3 ALIF 1; 32 ssam3 ALIF 2 ; if 21 ssam2 ALIF 1;if 22 ssam2 ALIF 2')
    args = parser.parse_args()
    return args
