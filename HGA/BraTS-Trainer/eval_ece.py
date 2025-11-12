#coding=utf-8
import argparse
import logging
import os
import random
import time
from collections import OrderedDict
import csv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
# import torch.optim

from data.data_utils import init_fn
from data.datasets_nii import (Brats_loadall_test_nii,
                               Brats_loadall_val_nii)
from data.transforms import *

from models import mmformer, ensemble_inc

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import criterions
from utils.predict import AverageMeter, test_ece_softmax
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from utils.parser import setup


parser = argparse.ArgumentParser()

parser.add_argument('--model', default='ensemble', type=str)
parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
# parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--dataname', default='BraTS/BRATS2020', type=str)
parser.add_argument('--datapath', default='BraTS/BRATS2020_Training_none_npy', type=str)
parser.add_argument('--savepath', default='outputs/eval_ece_ensemble_hga', type=str)
parser.add_argument('--resume', default='/home/sjj/PASSION/code/outputs/hga_idt_mr1379_hgaw_hps_bs2_epoch300_lr2e-3_temp4_seed1037/model_last.pth', type=str)
# parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--mask_type', default='idt', type=str)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=1037, type=int)
parser.add_argument('--needvalid', default=False, type=bool)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'testing')
args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

# currentdirPath = os.path.dirname(__file__)
# relativePath = '../datasets'
# datarootPath = os.path.abspath(os.path.join(currentdirPath,relativePath))
# #### Note: or directly set datarootPath as your data-saving path (absolute root):
# # datarootPath = 'your data-saving path root'
# datarootPath = '/home/sjj/MMMSeg'
# dataPath = os.path.abspath(os.path.join(datarootPath,args.datapath))

currentdirPath = os.path.dirname(__file__)
relativePath = '../datasets'
datarootPath = os.path.abspath(os.path.join(currentdirPath,relativePath))
#### Note: or directly set datarootPath as your data-saving path (absolute root):
# datarootPath = 'your data-saving path (root)'
datarootPath = '/home/sjj/MMMSeg'

# args = parser.parse_args()

args.datarootPath = datarootPath
args.datasetPath = os.path.abspath(os.path.join(args.datarootPath,args.datapath))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]

# masks_test = [[True, False, False, False]]
# mask_name = ['flair']
masks_torch = torch.from_numpy(np.array(masks_test))
mask_name = ['t2', 't1c', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int())

masks_valid = [[False, False, False, True], [True, False, False, False], [False, True, False, False], [False, False, True, False],
        [True, False, False, True], [False, True, False, True], [False, False, True, True], [True, True, False, False], [True, False, True, False], [False, True, True, False],
        [True, True, False, True], [True, False, True, True], [False, True, True, True], [True, True, True, False],
        [True, True, True, True]]
# t1,t1cet1,flairticet2,flairt1cet1t2
masks_valid_torch = torch.from_numpy(np.array(masks_valid))
masks_valid_array = np.array(masks_valid)
masks_all = [True, True, True, True]
masks_all_torch = torch.from_numpy(np.array(masks_all))

mask_name_valid = ['t2', 'flair', 't1c', 't1',
            'flairt2', 't1cet2', 't1t2', 'flairt1ce', 'flairt1', 't1cet1',
            'flairt1cet2', 'flairt1t2', 't1cet1t2', 'flairt1cet1',
            'flairt1cet1t2']
print (masks_valid_torch.int())

def main():
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['BraTS/BRATS2023', 'BraTS/BRATS2020', 'BraTS/BRATS2018']:
        num_cls = 4
    else:
        print ('dataset is error')
        exit(0)

    if args.model == 'mmformer':
        model = mmformer.Model(num_cls=num_cls)
    elif args.model == 'ensemble':
        model = ensemble_inc.Ensemble(in_channels=4, out_channels=num_cls, width_ratio=0.5)

    print (model)
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs, warmup=args.region_fusion_start_epoch, mode='warmuppoly')
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.AdamW(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ##########Setting data
        ####BRATS2020
    if args.dataname == 'BraTS/BRATS2020':
        test_file = os.path.join(args.datasetPath, 'test.txt')
    elif args.dataname == 'BraTS/BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        test_file = os.path.join(args.datasetPath, 'test1.txt')
    elif args.dataname == 'BraTS/BRATS2023':
        ####BRATS2021
        test_file = os.path.join(args.datasetPath, 'new.txt')

    logging.info(str(args))

    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datasetPath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    #########Evaluate
    ##########Evaluate last epoch
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('last epoch: {}'.format(checkpoint['epoch']+1))
        model.load_state_dict(checkpoint['state_dict'])
        test_ece_score = AverageMeter()
        with torch.no_grad():
            logging.info('###########test last epoch model###########')
            for i, mask in enumerate(masks_test):
                logging.info('{}'.format(mask_name[i]))
                ece_score = test_ece_softmax(
                                test_loader,
                                model,
                                dataname = args.dataname,
                                feature_mask = mask,
                                mask_name = mask_name[i],
                                )
                test_ece_score.update(ece_score)

            logging.info('Avg ECE scores: {}'.format(test_ece_score.avg))
            exit(0)

if __name__ == '__main__':
    main()
