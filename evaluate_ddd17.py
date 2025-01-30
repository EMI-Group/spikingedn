import os
from dataloaders.datasets.ddd17_images import DDD17_images_Segmentation
import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from utils.metrics import Evaluator
from PIL import Image
from mypath import Path
from utils.utils import AverageMeter, inter_and_union
from config_utils.snn_evaluate_args import obtain_evaluate_args
from dataloaders.datasets.ddd17 import DDD17Segmentation
from retrain_model.build_model_retrain_SNN import AutoRetrain
import random

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def convert_str2index(this_str, is_b=False, is_wight=False, is_cell=False):
    if is_wight:
        this_str = this_str.split('.')[:-1] + ['conv1','weight']
    elif is_b:
        this_str = this_str.split('.')[:-1] + ['snn_optimal','b']
    elif is_cell:
        this_str = this_str.split('.')[:5]
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


def main(start_epoch, epochs):
    assert torch.cuda.is_available(), NotImplementedError('No cuda available ')
    if not osp.exists('data/'):
        os.mkdir('data/')
    if not osp.exists('log/'):
        os.mkdir('log/')
    args = obtain_evaluate_args()
    # fix seed
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    model_fname = 'logs/retrain/retrain_best_model/trained_model/{0}_{1}_epoch%d.pth'.format(args.dataset, args.exp)

    if args.dataset == 'ddd17':
        dataset_val = DDD17Segmentation(args, root=Path.db_root_dir(args.dataset), split='test')
        args.num_classes = 6         # for different dataset
    elif args.dataset == 'ddd17_images':
        dataset_val = DDD17_images_Segmentation(args, root=Path.db_root_dir(args.dataset), split='test')
        args.num_classes = 7         # for different dataset
    else:
        return NotImplementedError
    if args.backbone == 'autodeeplab':
        model = AutoRetrain(args)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    if not args.train:
        val_dataloader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)
        model = torch.nn.DataParallel(model).cuda()
        evaluator = Evaluator(dataset_val.NUM_CLASSES)
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
        model.eval()
        print("======================start evaluate=======================")
        for epoch in range(40, epochs, 10):   # event only best:40 epoch
        # for epoch in range(100, epochs, 10):   # ssam (e+f) best:100 epoch
            evaluator.reset()
            print("evaluate epoch {:}".format(epoch + start_epoch))
            checkpoint_name = model_fname % (epoch + start_epoch)
            print(checkpoint_name)
            checkpoint = torch.load(checkpoint_name)
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
            model.module.load_state_dict(state_dict)
            for i, sample in enumerate(val_dataloader):
                inputs, target ,seq= sample['image'], sample['label'], sample['seq']
                N, S, H, W = target.shape
                total_outputs = torch.zeros((N, dataset_val.NUM_CLASSES, H, W)).cuda()
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
                    target_new = tar_retrain.cpu().numpy()
                    pred_new = out_retrain.data.cpu().numpy()
                    pred_new = np.argmax(pred_new, axis=1)
                    evaluator.add_batch(target_new, pred_new)
            miou_new= evaluator.Mean_Intersection_over_Union()
            Acc = evaluator.Pixel_Accuracy()
            Acc_class = evaluator.Pixel_Accuracy_Class()
            FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, miou_new, FWIoU))
            a=b

if __name__ == "__main__":
    # epochs = 45
    epochs = 110
    state_epochs = 1
    main(epochs=epochs, start_epoch=state_epochs)
