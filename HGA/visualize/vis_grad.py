#coding=utf-8
import argparse
import logging
import os
import random
import time
from collections import OrderedDict
import csv
import copy
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
# import torch.optim
# from gmd import GMD
from optim.grad_vis import Grad_Vis
from data.data_utils import init_fn
from data.datasets_nii import (Brats_loadall_train_nii_pdt, Brats_loadall_test_nii, Brats_loadall_train_nii_idt_wimaskvalue,
                               Brats_loadall_val_nii, Brats_loadall_train_nii_idt, Brats_loadall_metaval_nii_idt)
from data.transforms import *
from models import ensemble_inc
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from options import args_parser
from utils import criterions
from utils.predict import AverageMeter, test_dice_hd95_softmax
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from utils.parser import setup, set_seed

## parse arguments
args = args_parser()
## training setup
setup(args, 'training')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
## checkpoints saving path
ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks_test))
mask_name = ['t2', 't1c', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int())

# masks_valid = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
#          [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
#          [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
#          [True, True, True, True]]
# masks_valid_torch = torch.from_numpy(np.array(masks_valid))
# masks_valid_array = np.array(masks_valid)
masks_all = [True, True, True, True]
masks_all_torch = torch.from_numpy(np.array(masks_all))

# mask_name_valid = ['t2', 't1c', 't1', 'flair',
#             't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
#             'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
#             'flairt1cet1t2']
# mask_name_single = ['flair', 't1c', 't1', 't2']
# print (masks_valid_torch.int())

def main():
    ##########setting seed
    set_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['BraTS/BRATS2023', 'BraTS/BRATS2020', 'BraTS/BRATS2018']:
        num_cls = 4
    else:
        print ('dataset is error')
        exit(0)

    if args.model == 'ensemble':
        model = ensemble_inc.Ensemble(in_channels=4, out_channels=num_cls, width_ratio=0.5)


    print (model)
    model = torch.nn.DataParallel(model).cuda()
    model.module.mask_type = args.mask_type
    model.module.use_passion = args.use_passion
    ##########Setting learning schedule and optimizer
    # lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]

    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    grad_vis = Grad_Vis(optimizer, reduction='mean', writer=writer)


    temp = args.temp

    valid_file = os.path.join(args.datasetPath, 'all.txt')
    valid_setv = Brats_loadall_val_nii(transforms=args.train_transforms, root=args.datasetPath, num_cls=num_cls, train_file=valid_file)

    logging.info(str(args))
    set_seed(args.seed)
    v_loader = MultiEpochsDataLoader(
        dataset=valid_setv,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)

    #### Whether use pretrained model

    checkpoint = torch.load(args.resume)
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # logging.info('pretrained_dict: {}'.format(pretrained_dict))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    logging.info('load ok')

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)


    iter_per_epoch = len(v_loader)
    v_iter = iter(v_loader)
    logging.info('#############Gradient Conflict Visualization############')
    b = time.time()
    csv_name = os.path.join(ckpts, '{}.csv'.format(args.model))
    file = open(csv_name, "a+")
    csv_writer = csv.writer(file)
    csv_writer.writerow(['FLa-D', 'T1c-D', 'T1-D','T2-D', 'FLa-KL', 'T1c-KL', 'T1-KL','T2-KL'])
    for i in range(iter_per_epoch):
        step = (i+1)
                ###Data load
        try:
            data = next(v_iter)
        except:
            v_iter = iter(v_loader)
            data = next(v_iter)
        x, target, mask, name = data
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        mask = mask.cuda(non_blocking=True)

        model.module.is_training = True
        model.train()

        out_bs, de_fe_bs = model(x, mask, target=target)
        losses = []
        losses_summary = []
        kl_losses = []
        fuse_loss = []
        fuse_losses = []

        for modc in range(4):
            if mask[0,modc]:
                losses.append(criterions.DiceCoef(out_bs[modc], target, num_cls=num_cls))
                # losses_summary.append(criterions.DiceCoef(out_bs[modc], target, num_cls=num_cls))
                fuse_losses.append(criterions.DiceCoef(out_bs[modc], target, num_cls=num_cls) + criterions.temp_kl_loss(out_bs[modc], out_bs[4].detach(), target, num_cls=num_cls, temp=temp))
                # cnt += 1
        fuse_loss.append(criterions.DiceCoef(out_bs[4], target, num_cls=num_cls) + criterions.softmax_loss(torch.nn.Softmax(dim=1)(out_bs[4]), target, num_cls=num_cls))
        fuse_losses.append(criterions.DiceCoef(out_bs[4], target, num_cls=num_cls) + criterions.softmax_loss(torch.nn.Softmax(dim=1)(out_bs[4]), target, num_cls=num_cls))
        
        # losses_summary.append(criterions.DiceCoef(out_bs[4], target, num_cls=num_cls))
        # cnt += 1
        # logging.info(losses_summary)
        # kl_loss_m = torch.stack(kl_losses, dim=0)
        # term_kl = kl_loss_m.sum()

        # proto0, dist0 = criterions.prototype_passion_loss_bs(de_fe_bs[0], de_fe_bs[4].detach(), target, num_cls=num_cls, temp=temp)
        # proto1, dist1 = criterions.prototype_passion_loss_bs(de_fe_bs[1], de_fe_bs[4].detach(), target, num_cls=num_cls, temp=temp)
        # proto2, dist2 = criterions.prototype_passion_loss_bs(de_fe_bs[2], de_fe_bs[4].detach(), target, num_cls=num_cls, temp=temp)
        # proto3, dist3 = criterions.prototype_passion_loss_bs(de_fe_bs[3], de_fe_bs[4].detach(), target, num_cls=num_cls, temp=temp)

        # proto_loss = torch.cat((proto0.unsqueeze(0), proto1.unsqueeze(0), proto2.unsqueeze(0), proto3.unsqueeze(0)), dim=0)
        # proto_loss_m = proto_loss * mask[0]

        # dist_ = torch.cat((dist0.unsqueeze(0), dist1.unsqueeze(0), dist2.unsqueeze(0), dist3.unsqueeze(0)), dim=0)
        # dist_m_bs = mask[0] * dist_
        # dist_m = dist_m_bs

        # dist_avg_bs = sum(dist_m_bs)/sum(mask[0])

        # rp_iter = mask[0]*(1 - dist_m_bs/(dist_avg_bs + 1e-8))
        # rp_mask = rp_iter > 0


        # fuse_losses.append((rp_mask * proto_loss_m).sum())
        fuse_losses = [sum(fuse_losses)]

        # kl_proto_losses = list(kl_proto_m.unbind(dim=0))
        # kl_proto_losses = list(kl_loss_m.unbind(dim=0))

        grad_vis.zero_grad()
        degree_ = grad_vis.gradvis_backward(losses, fuse_loss)
        degree_kl = grad_vis.gradvis_backward(losses, fuse_losses)
        logging.info('No KL Degree:[{:.2f},{:.2f},{:.2f},{:.2f}], KL Degree:[{:.2f},{:.2f},{:.2f},{:.2f}]'.format(degree_[0].item(),degree_[1].item(),degree_[2].item(),degree_[3].item(), degree_kl[0].item(),degree_kl[1].item(),degree_kl[2].item(),degree_kl[3].item()))
        # optimizer.step()
        csv_writer.writerow([degree_[0].item(),degree_[1].item(),degree_[2].item(),degree_[3].item(), degree_kl[0].item(),degree_kl[1].item(),degree_kl[2].item(),degree_kl[3].item()])
    file.close()


if __name__ == '__main__':
    main()
