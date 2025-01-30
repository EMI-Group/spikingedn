import os
import pdb
from turtle import pen
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torch.optim as optim
import torch.nn.functional as F
import dataloaders
from utils.utils import AverageMeter
from utils.loss import build_criterion
from utils.step_lr_scheduler import Iter_LR_Scheduler
from retrain_model.build_model_retrain_SNN import AutoRetrain
from dataloaders.datasets.ddd17 import DDD17Segmentation
from dataloaders.datasets.ddd17_images import DDD17_images_Segmentation
from config_utils.snn_re_train_evaluate import obtain_retrain_evaluate_autodeeplab_args
from torchinfo import summary
import matplotlib
import matplotlib.image as img
from mypath import Path
from utils.utils import AverageMeter, inter_and_union
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from torch.utils.data.dataloader import DataLoader
import random

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def convert_str2index(this_str, is_b=False, is_wight=False, is_cell=False, is_aspp=False, is_decoder =False):
    if is_wight:
        this_str = this_str.split('.')[:-1] + ['conv1','weight']
    elif is_b:
        this_str = this_str.split('.')[:-1] + ['snn_optimal','b']
    elif is_cell:
        this_str = this_str.split('.')[:5]
    elif is_aspp:
        this_str = this_str.split('.')[:4]
    else:
        this_str = this_str.split('.')
    new_index = []
    for i, value in enumerate(this_str):
        if value.isnumeric():
            new_index.append('[%s]'%value)
        else:
            if i == 0:
                new_index.append(value)
            else:
                new_index.append('.'+value)
    return ''.join(new_index)

def reset_mem(model,mem_keys):
    for key in mem_keys:
        exec('model.%s.mem=None'%key)
    return model

def main():
    warnings.filterwarnings('ignore')
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    args = obtain_retrain_evaluate_autodeeplab_args()
    model_fname = 'logs/retrain/retrain_best_model/trained_model/{0}_{1}_epoch%d.pth'.format(args.dataset, args.exp)
    print("model_fname:",model_fname)
    if args.dataset == 'ddd17':
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        dataset_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
        dataset_val = DDD17Segmentation(args, root=Path.db_root_dir(args.dataset), split='test')
        args.num_classes = num_classes
    elif args.dataset == 'ddd17_images':
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        dataset_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
        dataset_val = DDD17_images_Segmentation(args, root=Path.db_root_dir(args.dataset), split='test')
        args.num_classes = num_classes
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    
    set_seed(args.seed)
    # use ABN
    args.use_ABN = False
    # fix seed
    set_seed(args.seed)

    if args.backbone == 'autodeeplab':
        model = AutoRetrain(args)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    if args.criterion == 'Ohem':
        args.thresh = 0.7
        args.crop_size = [args.crop_size, args.crop_size] if isinstance(args.crop_size, int) else args.crop_size
        args.n_min = int((args.batch_size / len(args.gpu) * args.crop_size[0] * args.crop_size[1]) // 16)

    criterion = build_criterion(args)
    model = nn.DataParallel(model).cuda()
    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    # optimizer = optim.SGD(model.module.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.module.parameters(), lr=args.base_lr, amsgrad=True, weight_decay=5e-4)

    model_all_keys = [name for name, value in model.named_parameters()]
    all_keys = [convert_str2index(name,is_cell=True) for name, value in model.named_parameters() if '_ops' in name] 
    all_keys = list(set(all_keys))
    mem_keys = list()
    for key in all_keys:
        try:
            eval('model.%s.mem'%key)
            mem_keys.append(key)
        except:
            print(key)
            pass

    # for save grad
    model_all_keys = [name for name, value in model.named_parameters()]
    exp_num = args.net_arch[args.net_arch.find("experiment_")+11:-23]
    retrain_evaluator = Evaluator(num_classes)
    if not args.no_val:
        evaluator = Evaluator(num_classes)
        val_dataloader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)
    max_iteration = len(dataset_loader) * 101
    # max_iteration = len(dataset_loader) * args.epochs
    scheduler = Iter_LR_Scheduler(args, max_iteration, len(dataset_loader))

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('=> no checkpoint found at {0}'.format(args.resume))

    for epoch in range(start_epoch, args.epochs):
        retrain_evaluator.reset()
        losses = AverageMeter()
        retrain_loss = 0.0
        model.train()
        for i, sample in enumerate(dataset_loader):
            cur_iter = epoch * len(dataset_loader) + i
            scheduler(optimizer, cur_iter)
            inputs = sample['image'].cuda()
            target = sample['label'].cuda()
            reset_mem(model,mem_keys)
            outputs = model(inputs)
            out_retrain = outputs.reshape(-1,outputs.shape[2],outputs.shape[3],outputs.shape[4])
            tar_retrain = target[:,int(args.burning_time):].reshape(-1,target.shape[2],target.shape[3])
            loss = criterion(out_retrain, tar_retrain)
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                pdb.set_trace()
            losses.update(loss.item(), args.batch_size)
            loss.backward()
            retrain_evaluator.add_batch(tar_retrain.cpu().numpy(), np.argmax(out_retrain.data.cpu().numpy(),axis=1))
            optimizer.step()
            optimizer.zero_grad()
            retrain_loss += loss.item()
            print('epoch: {0}\t''iter: {1}/{2}\t''lr: {3:.6f}\t''loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                epoch + 1, i + 1, len(dataset_loader), scheduler.get_lr(optimizer), loss=losses))
        # if epoch < args.epochs - 50:
        retrain_mIoU = retrain_evaluator.Mean_Intersection_over_Union()
        print('retrain_mIoU:',retrain_mIoU,' reset local total loss!')

        if args.epochs <= 101:
            save_epoch = 10
        else:
            save_epoch = 50
        if epoch % save_epoch == 0 or epoch == args.epochs:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_fname % (epoch + 1))
            if not args.no_val:
                model.eval()
                evaluator.reset()
                inter_meter = AverageMeter()
                union_meter = AverageMeter()
                for i, sample in enumerate(val_dataloader):
                    inputs, target = sample['image'], sample['label']
                    N, S, H, W = target.shape
                    total_outputs = torch.zeros((N, dataset_val.NUM_CLASSES, H, W)).cuda()
                    # if(i<31):
                    with torch.no_grad():
                        inputs = inputs.cuda()
                        reset_mem(model,mem_keys)
                        outputs = model(inputs)
                        out_retrain = outputs.reshape(-1,outputs.shape[2],outputs.shape[3],outputs.shape[4])
                        tar_retrain = target[:,int(args.burning_time):].reshape(-1,target.shape[2],target.shape[3])
                        _, pred = torch.max(out_retrain, 1)
                        pred = pred.detach().cpu().numpy().squeeze().astype(np.uint8)
                        mask = tar_retrain.numpy().astype(np.uint8)
                        print('eval: {0}/{1}'.format(i + 1, len(val_dataloader)))
                        # evaluate method in training process
                        target_new = tar_retrain.cpu().numpy()
                        pred_new = out_retrain.data.cpu().numpy()
                        pred_new = np.argmax(pred_new, axis=1)
                        evaluator.add_batch(target_new, pred_new)
                        inter, union = inter_and_union(pred, mask, args.num_classes)
                        inter_meter.update(inter)
                        union_meter.update(union)
                iou = inter_meter.sum / (union_meter.sum + 1e-10)
                miou = 'epoch: {0} Mean IoU: {1:.2f}'.format(epoch, iou.mean() * 100)
                miou_new= evaluator.Mean_Intersection_over_Union()
                miou_e_nan = 'epoch: {0} Mean IoU: {1:.2f}'.format(epoch, miou_new * 100)
                f_mIOU = open('log/result_mIOU_'+ args.dataset + '_'+ args.exp +'.txt','a')
                f_mIOU.write('\n')
                f_mIOU.write(miou_e_nan)
                f_mIOU.write('\n')
                f_mIOU.close()

if __name__ == "__main__":
    main()
