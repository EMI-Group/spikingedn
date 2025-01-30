import os
from telnetlib import PRAGMA_HEARTBEAT
from xml.etree.ElementPath import prepare_descendant
import numpy as np
import torch
from tqdm import tqdm
from collections import OrderedDict
from mypath import Path
from dataloaders import make_data_loader
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from models.build_model_search import AutoSearch
from config_utils.snn_search_args import obtain_search_args
from utils.copy_state_dict import copy_state_dict
from utils.utils import AverageMeter, inter_and_union
import random
torch.backends.cudnn.benchmark = True
import apex

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def convert_str2index(this_str, is_b=False, is_wight=False, is_cell=False, is_search =False):
    if is_wight:
        this_str = this_str.split('.')[:-1] + ['conv1','weight']
    elif is_b:
        this_str = this_str.split('.')[:-1] + ['snn_optimal','b']
    elif is_cell:
        this_str = this_str.split('.')[:5]
    elif is_search:
        this_str = this_str.split('.')[:2]
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


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.use_amp = True if (APEX_AVAILABLE and args.use_amp) else False
        self.opt_level = args.opt_level
        self.num_layer = args.num_layer
        # timestep for RGB image
        self.temp_steps = args.timestep
        self.initial_channels = args.initial_channels
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last':True}
        self.train_loaderA, self.train_loaderB, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                raise NotImplementedError
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        # Define network
        model = AutoSearch(self.nclass, self.num_layer, self.temp_steps, self.criterion, self.args.filter_multiplier,
                             self.args.block_multiplier, self.args.step, self.initial_channels, self.args.sequence, self.args.burning_time, self.args.is_allsnn)
        
        optimizer = torch.optim.SGD(
                model.autodeeplab.weight_parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )

        self.model, self.optimizer = model, optimizer

        self.architect_optimizer = torch.optim.Adam(self.model.autodeeplab.arch_parameters(),
                                                    lr=args.arch_lr, betas=(0.9, 0.999),
                                                    weight_decay=args.arch_weight_decay)

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loaderA), min_lr=args.min_lr)
        # TODO: Figure out if len(self.train_loader) should be devided by two ? in other module as well
        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()

        # mixed precision
        if self.use_amp and args.cuda:
            keep_batchnorm_fp32 = True if (self.opt_level == 'O2' or self.opt_level == 'O3') else None

            # fix for current pytorch version with opt_level 'O1'
            if self.opt_level == 'O1' and torch.__version__ < '1.3':
                for module in self.model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        # Hack to fix BN fprop without affine transformation
                        if module.weight is None:
                            module.weight = torch.nn.Parameter(
                                torch.ones(module.running_var.shape, dtype=module.running_var.dtype,
                                           device=module.running_var.device), requires_grad=False)
                        if module.bias is None:
                            module.bias = torch.nn.Parameter(
                                torch.zeros(module.running_var.shape, dtype=module.running_var.dtype,
                                            device=module.running_var.device), requires_grad=False)

            # print(keep_batchnorm_fp32)
            self.model, [self.optimizer, self.architect_optimizer] = amp.initialize(
                self.model, [self.optimizer, self.architect_optimizer], opt_level=self.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale="dynamic")
            print('cuda finished')

        all_keys = [convert_str2index(name,is_cell=True) for name, value in model.named_parameters() if '_ops' in name] 
        all_keys = list(set(all_keys))
        self.mem_down_keys = list()
        self.mem_up_keys = list()
        self.mem_same_keys = list()
        for key in all_keys:
            key_2 = convert_str2index(key,is_search=True)
            try:
                if key.split('.')[2][:7] == "_ops_do":
                    eval('model.%s.mem_down'% key_2)
                    self.mem_down_keys.append(key_2)
                elif key.split('.')[2][:7] == "_ops_sa":
                    eval('model.%s.mem_same'% key_2)
                    self.mem_same_keys.append(key_2)
                elif key.split('.')[2][:7] == "_ops_up":
                    eval('model.%s.mem_up'% key_2)
                    self.mem_up_keys.append(key_2)
                else:
                    print("none")
            except:
                print(key)
                pass

         # for save grad
        self.model_all_keys = [name for name, value in self.model.named_parameters()]
        self.model_grad_dict = dict([(k,[]) for k in self.model_all_keys])
        self.exp = max([int(x.split('_')[-1]) for x in self.saver.runs]) + 1 if self.saver.runs else 0
        # print(" self.exp:", self.exp)
        self.grad_save_path = "/home/z50021440/Semantic_Segmentation/autodeeplab-new_master/search_grad" +'/deeplab_{0}_{1}_{2}'.format(args.backbone, args.dataset, self.exp)
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            # if the weights are wrapped in module object we have to clean it
            if args.clean_module:
                self.model.load_state_dict(checkpoint['state_dict'])
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' of dataparallel
                    new_state_dict[name] = v
                # self.model.load_state_dict(new_state_dict)
                copy_state_dict(self.model.state_dict(), new_state_dict)

            else:
                if torch.cuda.device_count() > 1 or args.load_parallel:
                    copy_state_dict(self.model.module.state_dict(), checkpoint['state_dict'])
                else:
                    copy_state_dict(self.model.state_dict(), checkpoint['state_dict'])


            if not args.ft:
                copy_state_dict(self.optimizer.state_dict(), checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
    
        # randomseed
        if self.args.randomseed:
            if self.args.num_randomseed > 0:
                self.randomseed_interval = int(self.args.epochs / int(self.args.num_randomseed-1))
            else:
                raise ValueError('Unknown num for random seed.')

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loaderA)        # iterator
        num_img_tr = len(self.train_loaderA)
        for i, sample in enumerate(tbar):
            
            image, target = sample['image'], sample['label']  # image [B,step, H,W,channels] or  [B,step, H,W,channels]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            self.reset_mem()

            output = self.model(image)
            out = output.reshape(-1,output.shape[2],output.shape[3],output.shape[4])
            tar = target[:,int(self.args.burning_time):].reshape(-1,target.shape[2],target.shape[3])
            loss = self.criterion(out, tar)
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            if epoch >= self.args.alpha_epoch:
                search = next(iter(self.train_loaderB))
                image_search, target_search = search['image'], search['label']  
                if self.args.cuda:
                    image_search, target_search = image_search.cuda (), target_search.cuda ()

                self.architect_optimizer.zero_grad()
                self.reset_mem()
                output_search = self.model(image_search)
                out_search = output_search.reshape(-1,output_search.shape[2],output_search.shape[3],output_search.shape[4])
                tar_search = target_search[:,int(self.args.burning_time):].reshape(-1,target_search.shape[2],target_search.shape[3])
                arch_loss = self.criterion(out_search, tar_search)

                if self.use_amp:
                    with amp.scale_loss(arch_loss, self.architect_optimizer) as arch_scaled_loss:
                        arch_scaled_loss.backward()
                else:
                    arch_loss.backward()
                self.architect_optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            save_grad = False
            if save_grad == True:
                # save grad
                for name, model_param in self.model.named_parameters():
                    try:
                        this_grad = torch.mean(torch.abs(model_param.grad)).cpu().item()
                        self.model_grad_dict[name].append(this_grad)
                    except:
                        pass
                np.save(self.grad_save_path +'model_grad_dict.npy',self.model_grad_dict)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                if self.args.dataset !='ddd17' and self.args.dataset !='ddd17_evsn':
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        inter_meter = AverageMeter()
        union_meter = AverageMeter()

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                self.reset_mem()
                output = self.model(image)
                out_search = output.reshape(-1,output.shape[2],output.shape[3],output.shape[4])
                tar_search = target[:,int(self.args.burning_time):].reshape(-1,target.shape[2],target.shape[3])
            loss = self.criterion(out_search, tar_search)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = out_search.data.cpu().numpy()
            target = tar_search.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            # # Add batch sample into evaluator
            # other method
            _, pred_e = torch.max(out_search, 1)
            pred_e = pred_e.detach().cpu().numpy().squeeze().astype(np.uint8)
            mask = tar_search.cpu().numpy().astype(np.uint8)
            inter, union = inter_and_union(pred_e, mask, 7)
            inter_meter.update(inter)
            union_meter.update(union)

        iou_e = inter_meter.sum / (union_meter.sum + 1e-10)
        miou_e = iou_e.mean() * 100
        print('epoch: {0} Mean IoU: {1:.2f}'.format(epoch, miou_e))
 
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        print("mIoU:",mIoU)

        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/mIoU_mean', miou_e, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)       # for all dataset
        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        if self.args.randomseed:
            if self.args.num_randomseed > 0:
                if epoch == 0 or ((epoch+1) % self.randomseed_interval==0):
                    if torch.cuda.device_count() > 1:
                        state_dict = self.model.module.state_dict()
                    else:
                        state_dict = self.model.state_dict()
                    self.saver.save_epoch_module({
                        'epoch': epoch + 1,
                        'state_dict': state_dict,
                        'optimizer': self.optimizer.state_dict(),
                        'pred': mIoU,
                    }, epoch)

    def reset_mem(self):
        for key in self.mem_down_keys:
            exec('self.model.%s.mem_down=None'%key)
        for key in self.mem_same_keys:
            exec('self.model.%s.mem_same=None'%key)
        for key in self.mem_up_keys:
            exec('self.model.%s.mem_up=None'%key)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = obtain_search_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
   # fix seed
    set_seed(args.seed)
    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 40,
            'pascal': 50,
            'kd':10
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    #args.lr = args.lr / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    # torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
    trainer.writer.close()

if __name__ == "__main__":
   main()

