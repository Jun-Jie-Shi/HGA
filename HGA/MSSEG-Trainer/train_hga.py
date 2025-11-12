#coding=utf-8
import argparse
import logging
import os
import random
import time
from collections import OrderedDict
import csv
import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
# from MMMSeg.LongTail.models import baseline
from data.data_utils import init_fn
from data.datasets_nii import (MSSEG_loadall_test_nii, MSSEG_loadall_train_nii_idt, MSSEG_loadall_metaval_nii_idt, MSSEG_loadall_val_nii)
from data.transforms import *
# from visualizer import get_local
# get_local.activate()
from models import ensemble_inc
from predict import AverageMeter, test_dice_hd95_softmax_msseg
# from predict import AverageMeter, test_softmax, test_softmax_visualize
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import criterions
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from utils.parser import setup, set_seed
from utils.sequence import MetaSampler
# from utils.visualize import visualize_heads

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='ensemble', type=str)
parser.add_argument('-batch_size', '--batch_size', default=4, type=int, help='Batch size')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--iter_per_epoch', default=34, type=int)
parser.add_argument('--dataname', default='/data/MSSEG/MSSEG2016', type=str)
parser.add_argument('--datapath', default='/home/sjj/MMMSeg/MSSEG/MSSEG2016_Training_none_npy', type=str)
parser.add_argument('--imbmrpath', default='/home/sjj/MMMSeg/MSSEG/msseg_split/MSSEG2016_imb_split_mr97531.csv', type=str, help='csv path')
parser.add_argument('--savepath', default='outputs/idt_mr97531_hganouni_fullfix_bs4_epoch300_lr2e-4_temp4_seed3407', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--mask_type', default='idt', type=str)
parser.add_argument('--seq_strategy', default='group_fixed_order', type=str, help='sequence strategy: fixed, random, group_fixed_order or group_sequential')
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=3407, type=int)
parser.add_argument('--needvalid', default=False, type=bool)
parser.add_argument('--temp', default=4.0, type=float)
parser.add_argument('--gpu', type=str, default='3', help='GPU to use')
# parser.add_argument('--csvpath', default='/home/sjj/MMMSeg/VQ/csv/', type=str)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.train_transforms = 'Compose([CenterCrop3D((224,224,1)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
# args.train_transforms = 'Compose([CenterCrop3D((256,256,1)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([CenterCrop3D((224,224,1)), NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

# csvpath = args.csvpath
# os.makedirs(csvpath, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks_test = [[False, False, False, False, True], [False, False, False, True, False], [False, False, True, False, False], [False, True, False, False, False], [True, False, False, False, False],
        [False, False, False, True, True], [False, False, True, False, True], [False, True, False, False, True], [True, False, False, False, True], [False, False, True, True, False], [False, True, False, True, False], [True, False, False, True, False], [False, True, True, False, False], [True, False, True, False, False], [True, True, False, False, False],
        [False, False, True, True, True], [False, True, False, True, True], [True, False, False, True, True], [False, True, True, False, True], [True, False, True, False, True], [True, True, False, False, True], [False, True, True, True, False], [True, False, True, True, False], [True, True, False, True, False], [True, True, True, False, False],
        [False, True, True, True, True], [True, False, True, True, True], [True, True, False, True, True], [True, True, True, False, True], [True, True, True, True, False],
        [True, True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks_test))
mask_name = ['GADO', 'DP', 'T2', 'T1', 'FLAIR',
            'DP-GADO', 'T2-GADO', 'T1-GADO', 'FLAIR-GADO', 'T2-DP', 'T1-DP', 'FLAIR-DP', 'T1-T2', 'FLAIR-T2', 'FLAIR-T1',
            'T2-DP-GADO', 'T1-DP-GADO', 'FLAIR-DP-GADO', 'T1-T2-GADO', 'FLAIR-T2-GADO', 'FLAIR-T1-GADO', 'T1-T2-DP', 'FLAIR-T2-DP', 'FLAIR-T1-DP', 'FLAIR-T1-T2',
            'T1-T2-DP-GADO', 'FLAIR-T2-DP-GADO', 'FLAIR-T1-DP-GADO', 'FLAIR-T1-T2-GADO', 'FLAIR-T1-T2-DP',
            'FLAIR-T1-T2-DP-GADO']
print (masks_torch.int())

masks_valid = [[False, False, False, False, True],
            [False, False, False, True, False],
            [False, False, True, False, False],
            [False, True, False, False, True],
            [True, False, False, False, False],
            [True, True, True, True, True]]
# masks_valid = [[False, False, True], [False, True, False], [True, False, False],
#          [False, True, True], [True, True, False], [True, False, True],
#          [True, True, True]]

# # t1,t1cet1,flairticet2,flairt1cet1t2
masks_valid_torch = torch.from_numpy(np.array(masks_valid))
masks_valid_array = np.array(masks_valid)
masks_all = [True, True, True, True, True]
masks_all_torch = torch.from_numpy(np.array(masks_all))
mask_name_valid = ['GADO',
                'DP',
                'T2',
                'T1',
                'FLAIR',
                'FLAIR-T1-T2-DP-GADO']

def main():
    ##########setting seed
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    set_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['/data/MSSEG/MSSEG2016']:
        num_cls = 2
    else:
        print ('dataset is error')
        exit(0)

    if args.model == 'ensemble':
        model = ensemble_inc.Ensemble(in_channels=5, out_channels=num_cls, width_ratio=0.5)
    # elif args.model == 'mmformer':
    #     model = mmformer.Model(num_cls=num_cls)

    # elif args.model == 'vqbaseline':
    #     model = baseline.Model(num_cls=num_cls)

    print (model)
    model = torch.nn.DataParallel(model).cuda()
    model.module.mask_type = args.mask_type
    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    temp = args.temp
    ##########Setting data
        ####BRATS2020
    if args.dataname == '/data/MSSEG/MSSEG2016':
        # train_file = '/home/sjj/MMMSeg/BraTS/BRATS2020_Training_none_npy/train.txt'
        train_file = args.imbmrpath
        test_file = '/home/sjj/MMMSeg/MSSEG/MSSEG2016_Training_none_npy/test.txt'
        valid_file = '/home/sjj/MMMSeg/MSSEG/MSSEG2016_Training_none_npy/val.txt'

    logging.info(str(args))
    set_seed(args.seed)
    # train_set = MSSEG_loadall_train_nii_idt(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, mask_type=args.mask_type, train_file=train_file)
    test_set = MSSEG_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    meta_set = [MSSEG_loadall_metaval_nii_idt(transforms=args.train_transforms, root=args.datapath, mask_value=i, num_cls=num_cls, train_file=train_file) for i in range(31)]
    valid_set = MSSEG_loadall_val_nii(transforms=args.test_transforms, root=args.datapath, num_cls=num_cls, train_file=valid_file)
    # train_loader = MultiEpochsDataLoader(
    #     dataset=train_set,
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     pin_memory=True,
    #     shuffle=True,
    #     worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    meta_loader = [MultiEpochsDataLoader(
        dataset=meta_set[i],
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn,
        drop_last=True) for i in range(31)]
    valid_loader = MultiEpochsDataLoader(
        dataset=valid_set,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        worker_init_fn=init_fn)

    if args.resume is not None:
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

    # if args.use_passion:
    meta_iters = [iter(meta_loader[i]) for i in range(31)]
    meta_list = list(range(31))
    # np.random.shuffle(meta_list)
    meta_list_iter = iter(meta_list)
    sampler = MetaSampler(
                n_sources=31,
                strategy=args.seq_strategy,
                groups=[[0,1,2,3,4], [30], [5,6,7,8,9,10,11,12,13,14], [30], [15,16,17,18,19,20,21,22,23,24], [30], [25,26,27,28,29], [30]]
            )
    meta_mask_value_pre = 31

    iter_per_epoch = args.iter_per_epoch

    # iter_per_epoch = len(train_loader)
    # train_iter = iter(train_loader)

    ### IDT Init-Imb-Weight Setting
    imb_mr_csv_data = pd.read_csv(train_file)
    modal_num = torch.tensor((0,0,0,0,0), requires_grad=False).cuda().float()
    for sample_mask in imb_mr_csv_data['mask']:
        modal_num += torch.tensor(eval(sample_mask), requires_grad=False).cuda().float()
    logging.info('Training Imperfect Datasets with Mod.FLAIR-{:d}, Mod.T1-{:d}, Mod.T2-{:d}, Mod.DP-{:d}, Mod.GADO-{:d}'\
    .format(int(modal_num[0].item()), int(modal_num[1].item()), int(modal_num[2].item()), int(modal_num[3].item()), int(modal_num[4].item())))

    modal_weight = torch.tensor((1,1,1,1,1), requires_grad=False).cuda().float()
    # modal_weight = (iter_per_epoch/modal_num).cuda().float()
    # valid_iter = iter(valid_loader)
    imb_beta = torch.tensor((1,1,1,1,1), requires_grad=False).cuda().float()
    eta = 0.01
    eta_ext = 1.5
    # phi_tilde = [p.clone() for p in model.parameters()]
    # 初始化：phi_tilde 保存在 CPU 上
    phi_tilde = [p.detach().clone().cpu() for p in model.parameters()]
    epsilon = 0.1
    for epoch in range(args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        epsilon = 0.1 + 0.9 * (epoch / args.num_epochs)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        epoch_fuse_losses = torch.zeros(1).cpu().float()
        epoch_sep_losses = torch.zeros(1).cpu().float()
        epoch_prm_losses = torch.zeros(1).cpu().float()
        epoch_kl_losses = torch.zeros(1).cpu().float()
        epoch_proto_losses = torch.zeros(1).cpu().float()
        # epoch_dist_losses = torch.zeros(1).cpu().float()
        epoch_losses = torch.zeros(1).cpu().float()
        epoch_sep_m = torch.zeros(5).cpu().float()
        epoch_kl_m = torch.zeros(5).cpu().float()
        epoch_proto_m = torch.zeros(5).cpu().float()
        epoch_dist_m = torch.zeros(5).cpu().float()

        b = time.time()
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            meta_mask_value = next(sampler)
            # if meta_mask_value_pre==30:

            #     with torch.no_grad():  # 禁用梯度
            #         for p, g in zip(model.parameters(), phi_tilde):
            #             # g 是 CPU 张量，需要移到 GPU
            #             g_gpu = g.to(p.device, non_blocking=True)  # 异步传输
            #             # 原地更新：p = g + epsilon * (p - g)
            #             p.data.copy_(g_gpu + epsilon * (p - g_gpu))
            #             # 更新 phi_tilde：把当前 p 拷贝回 CPU
            #             g.copy_(p.detach().cpu())  # 原地更新 CPU 副本
            # meta_mask_value_pre = meta_mask_value


            try:
                data = next(meta_iters[meta_mask_value])
            except:
                meta_iters[meta_mask_value] = iter(meta_loader[meta_mask_value])
                data = next(meta_iters[meta_mask_value])
            x, target, mask, name = data
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True
            model.train()

            # kl_loss_m = torch.zeros(4).cuda().float()
            # sep_loss_m = torch.zeros(4).cuda().float()
            # proto_loss_m = torch.zeros(4).cuda().float()
            # dist_m = torch.zeros(4).cuda().float()
            # prm_loss = torch.zeros(1).cuda().float()
            # fuse_loss = torch.zeros(1).cuda().float()
            rp_iter = torch.zeros(5).cuda().float()

            # fuse_pred, prm_loss_bs, sep_loss_m_bs, kl_loss_m_bs, proto_loss_m_bs, dist_m_bs = model(x, mask, target=target, temp=temp)
            out_bs, de_fe_bs = model(x, mask, target=target)
            losses = []
            losses_summary = []
            kl_losses = []
            # for out_bs_ava in out_bs:
            #     losses.append(criterions.DiceCoef(out_bs_ava, target, num_cls=num_cls))
            # cnt = 0
            for modc in range(5):
                if mask[0,modc]:
                    # criterions.DiceCoef(out_bs[modc], target, num_cls=num_cls) +
                    # losses.append(criterions.DiceCoef2(out_bs[modc], target, num_cls=num_cls) + criterions.softmax_weighted_loss(torch.nn.Softmax(dim=1)(out_bs[modc]), target, num_cls=num_cls))
                    losses_summary.append(criterions.DiceCoef2(out_bs[modc], target, num_cls=num_cls))
                    kl_losses.append(criterions.temp_weighted_kl_loss(out_bs[modc], out_bs[5].detach(), target, num_cls=num_cls, temp=temp))
                    # cnt += 1
                else:
                    losses_summary.append(torch.tensor(0).cuda().float())
                    kl_losses.append(torch.tensor(0).cuda().float())
            #  + criterions.softmax_weighted_loss(torch.nn.Softmax(dim=1)(out_bs[4]), target, num_cls=num_cls)
            losses.append(criterions.DiceCoef2(out_bs[5], target, num_cls=num_cls)+ criterions.softmax_weighted_loss(torch.nn.Softmax(dim=1)(out_bs[5]), target, num_cls=num_cls))
            losses_summary.append(criterions.DiceCoef2(out_bs[5], target, num_cls=num_cls))
            # cnt += 1
            # logging.info(losses_summary)
            epoch_summary = torch.stack(losses_summary, dim=0)
            loss = torch.mean(epoch_summary)
            kl_loss_m = torch.stack(kl_losses, dim=0)
            # losses /= cnt


            proto0, dist0 = criterions.prototype_passion_loss_bs(de_fe_bs[0], de_fe_bs[5].detach(), target, num_cls=num_cls, temp=temp)
            proto1, dist1 = criterions.prototype_passion_loss_bs(de_fe_bs[1], de_fe_bs[5].detach(), target, num_cls=num_cls, temp=temp)
            proto2, dist2 = criterions.prototype_passion_loss_bs(de_fe_bs[2], de_fe_bs[5].detach(), target, num_cls=num_cls, temp=temp)
            proto3, dist3 = criterions.prototype_passion_loss_bs(de_fe_bs[3], de_fe_bs[5].detach(), target, num_cls=num_cls, temp=temp)
            proto4, dist4 = criterions.prototype_passion_loss_bs(de_fe_bs[4], de_fe_bs[5].detach(), target, num_cls=num_cls, temp=temp)

            proto_loss = torch.cat((proto0.unsqueeze(0), proto1.unsqueeze(0), proto2.unsqueeze(0), proto3.unsqueeze(0), proto4.unsqueeze(0)), dim=0)
            proto_loss_m = proto_loss * mask[0]

            dist_ = torch.cat((dist0.unsqueeze(0), dist1.unsqueeze(0), dist2.unsqueeze(0), dist3.unsqueeze(0), dist4.unsqueeze(0)), dim=0)
            dist_m_bs = mask[0] * dist_
            dist_m = dist_m_bs

            # for bs in range(x.size(0)):
            dist_avg_bs = sum(dist_m_bs)/sum(mask[0])
            # rp_iter += mask[bs]*(dist_m_bs[bs]/(dist_avg_bs + 1e-8)-1)
            rp_iter += mask[0]*(1 - dist_m_bs/(dist_avg_bs + 1e-8))
            rp_mask = rp_iter > 0

            term_kl = (imb_beta * modal_weight * kl_loss_m).sum()
            term_proto = (rp_mask * modal_weight * proto_loss_m).sum()
            # term_seg_dice = criterions.DiceCoef(out_bs[4], target, num_cls=num_cls)
            # term_seg_dice = criterions.softmax_weighted_loss(torch.nn.Softmax(dim=1)(out_bs[4]), target, num_cls=num_cls)
            loss_reg = 0.5 * term_kl + 0.02 * term_proto + torch.sum(torch.stack(losses, dim=0))
            # + 0.02 * term_proto

            # pc_adam.zero_grad()
            # pc_adam.pc_backward(losses)
            # loss_reg.backward()
            # pc_adam.step()
            optimizer.zero_grad()
            loss_reg.backward()
            optimizer.step()

            epoch_losses += (loss/iter_per_epoch).detach().cpu()
            epoch_sep_m += (epoch_summary[0:5]/iter_per_epoch).detach().cpu()
            epoch_fuse_losses += (epoch_summary[5]/iter_per_epoch).detach().cpu()
            if args.mask_type == 'idt':
                epoch_kl_m += (kl_loss_m/iter_per_epoch).detach().cpu()
                # epoch_sep_m += (sep_loss_m/iter_per_epoch).detach().cpu()
                epoch_proto_m += (proto_loss_m/iter_per_epoch).detach().cpu()
                epoch_dist_m += (dist_m/iter_per_epoch).detach().cpu()

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'fuse_loss:{:.4f}, '.format(epoch_summary[5].item())
            # msg += 'sep_loss:{:.4f}, '.format(sep_loss.item())
            # msg += 'kl_loss:{:.4f}, proto_loss:{:.4f},'.format(kl_loss.item(), proto_loss.item())
            msg += 'seplist:[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}] '.format(epoch_summary[0].item(), epoch_summary[1].item(), epoch_summary[2].item(), epoch_summary[3].item(), epoch_summary[4].item())
            msg += 'kllist:[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}] '.format(kl_loss_m[0].item(), kl_loss_m[1].item(), kl_loss_m[2].item(), kl_loss_m[3].item(), kl_loss_m[4].item())
            msg += 'protolist:[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}] '.format(proto_loss_m[0].item(), proto_loss_m[1].item(), proto_loss_m[2].item(), proto_loss_m[3].item(), proto_loss_m[4].item())
            # msg += 'distlist:[{:.4f},{:.4f},{:.4f},{:.4f}] '.format(dist_m[0].item(), dist_m[1].item(), dist_m[2].item(), dist_m[3].item())
            # for bs_n in range(x.size(0)):
            #     msg += '{:>20}, '.format(name[bs_n])
            # msg += 'rp_iter[{:.2f},{:.2f},{:.2f},{:.2f}]'.format(rp_iter[0].item(), rp_iter[1].item(), rp_iter[2].item(), rp_iter[3].item())
            logging.info(msg)
        b_train = time.time()
        logging.info('train time per epoch: {}'.format(b_train - b))

        epoch_dist_avg = (sum(epoch_dist_m)/5.0).cpu().float()
        # rp_epoch = ((epoch_dist_avg - epoch_dist_m) / epoch_dist_avg)
        rp_epoch = ((epoch_dist_m - epoch_dist_avg) / epoch_dist_avg)
        if epoch < args.region_fusion_start_epoch:
            imb_beta = imb_beta.cuda()
        else:
            if epoch % 100 == 0:
                eta = eta * eta_ext
            imb_beta = imb_beta.cpu() - eta * rp_epoch
            imb_beta = torch.clamp(imb_beta, min=0.1, max=5.0)
            imb_beta = 5**(0.5) * imb_beta / (sum(imb_beta**2)**(0.5))
            imb_beta = imb_beta.cuda()


        logging.info('rp_epoch:[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}]'.format(rp_epoch[0].item(), rp_epoch[1].item(), rp_epoch[2].item(), rp_epoch[3].item(), rp_epoch[4].item()))
        logging.info('imb_beta:[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}]'.format(imb_beta[0].item(), imb_beta[1].item(), imb_beta[2].item(), imb_beta[3].item(), imb_beta[4].item()))


        writer.add_scalar('epoch_losses', epoch_losses.item(), global_step=(epoch+1))
        writer.add_scalar('epoch_fuse_losses', epoch_fuse_losses.item(), global_step=(epoch+1))
        # writer.add_scalar('epoch_prm_losses', epoch_prm_losses.item(), global_step=(epoch+1))
        # writer.add_scalar('epoch_sep_losses', epoch_sep_losses.item(), global_step=(epoch+1))
        # writer.add_scalar('epoch_kl_losses', epoch_kl_losses.item(), global_step=(epoch+1))
        # # writer.add_scalar('epoch_dist_losses', epoch_dist_losses.item(), global_step=(epoch+1))
        # writer.add_scalar('epoch_proto_losses', epoch_proto_losses.item(), global_step=(epoch+1))
        for m in range(5):
            writer.add_scalar('kl_m{}'.format(m), epoch_kl_m[m].item(), global_step=(epoch+1))
            writer.add_scalar('sep_m{}'.format(m), epoch_sep_m[m].item(), global_step=(epoch+1))
            writer.add_scalar('proto_m{}'.format(m), epoch_proto_m[m].item(), global_step=(epoch+1))
            writer.add_scalar('dist_m{}'.format(m), epoch_dist_m[m].item(), global_step=(epoch+1))
            writer.add_scalar('rp_m{}'.format(m), rp_epoch[m].item(), global_step=(epoch+1))


        #########model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            },
            file_name)

        if (epoch+1) % 50 == 0 or (epoch>=(args.num_epochs-5)):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)
    
#########Validate this epoch model
        valid_iter = iter(valid_loader)
        if ((epoch+1) % 10 == 0):
            with torch.no_grad():
                logging.info('#############validation############')
                score_modality = torch.zeros(7)
                for j, masks in enumerate(masks_valid_array):
                    logging.info('{}'.format(mask_name_valid[j]))
                    for i in range(len(valid_loader)):
                    # step = (i+1) + epoch*iter_per_epoch
                    ###Data load
                        try:
                            data = next(valid_iter)
                        except:
                            valid_iter = iter(valid_loader)
                            data = next(valid_iter)
                        x, target= data[:2]
                        x = x.cuda(non_blocking=True)
                        target = target.cuda(non_blocking=True)
                        batchsize=x.size(0)


                        mask = torch.unsqueeze(torch.from_numpy(masks), dim=0)
                        mask = mask.repeat(batchsize,1)
                        mask = mask.cuda(non_blocking=True)

                        model.module.is_training = False
                        # fuse_pred, sep_preds, prm_preds = model(x, mask)

                        fuse_pred = model(x, mask)

                        ###Loss compute
                        fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
                        # fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
                        fuse_loss = fuse_dice_loss
                        loss = fuse_loss

                        # loss = fuse_loss
                        # score -= loss
                        score_modality[j] += 1.0 - loss.item()
                        score_modality[6] += 1.0 - loss.item()
                val_score_avg = score_modality[6] / len(masks_valid_array)
                writer.add_scalar('valid-dice', val_score_avg.item(), global_step=(epoch+1))
                if (epoch+1) == 10:
                    best_score = val_score_avg
                    best_epoch = epoch
                elif val_score_avg > best_score:
                    best_score = val_score_avg
                    best_epoch = epoch
                    file_name = os.path.join(ckpts, 'model_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict(),
                        },
                        file_name)
                logging.info('val_score:{}, best_score:{}'.format(val_score_avg, best_score))
    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)


    ##########Test the best epoch model
    best_model_ck = os.path.join(ckpts, 'model_best.pth')
    checkpoint = torch.load(best_model_ck)
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # logging.info('pretrained_dict: {}'.format(pretrained_dict))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    logging.info('Load the best model ok!')


    # visualize_step = 0
    test_dice_score = AverageMeter()
    test_hd95_score = AverageMeter()
    csv_name = os.path.join(ckpts, '{}.csv'.format(args.model))
    with torch.no_grad():
        logging.info('###########test best epoch model###########')
        file = open(csv_name, "a+")
        csv_writer = csv.writer(file)
        csv_writer.writerow(['MS Dice', 'MS HD95'])
        file.close()
        for i, mask in enumerate(masks_test):
            logging.info('{}'.format(mask_name[i]))
            file = open(csv_name, "a+")
            csv_writer = csv.writer(file)
            csv_writer.writerow([mask_name[i]])
            file.close()
            dice_score, hd95_score = test_dice_hd95_softmax_msseg(
                            test_loader,
                            model,
                            dataname = args.dataname,
                            feature_mask = mask,
                            mask_name = mask_name[i],
                            csv_name = csv_name,
                            )
            test_dice_score.update(dice_score)
            test_hd95_score.update(hd95_score)

        logging.info('Avg Dice scores: {}'.format(test_dice_score.avg))
        logging.info('Avg HD95 scores: {}'.format(test_hd95_score.avg))
    ##########Test the best epoch model
    # best_model_ck = os.path.join(ckpts, 'model_best.pth')
    # checkpoint = torch.load(best_model_ck)
    # pretrained_dict = checkpoint['state_dict']
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # logging.info('pretrained_dict: {}'.format(pretrained_dict))
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # logging.info('Load the best model ok!')
    # writer_visualize = SummaryWriter(log_dir="visualize/result")
    # visualize_step = 0
    # test_dice_score = AverageMeter()
    # test_hd95_score = AverageMeter()
    # csv_name = os.path.join(ckpts, '{}.csv'.format(args.model))
    # with torch.no_grad():
    #     logging.info('###########test last epoch model###########')
    #     file = open(csv_name, "a+")
    #     csv_writer = csv.writer(file)
    #     csv_writer.writerow(['MS Dice', 'MS HD95'])
    #     file.close()
    #     for i, mask in enumerate(masks_test):
    #         logging.info('{}'.format(mask_name[i]))
    #         file = open(csv_name, "a+")
    #         csv_writer = csv.writer(file)
    #         csv_writer.writerow([mask_name[i]])
    #         file.close()
    #         dice_score, hd95_score = test_dice_hd95_softmax_msseg(
    #                         test_loader,
    #                         model,
    #                         dataname = args.dataname,
    #                         feature_mask = mask,
    #                         mask_name = mask_name[i],
    #                         csv_name = csv_name,
    #                         )
    #         test_dice_score.update(dice_score)
    #         test_hd95_score.update(hd95_score)

    #     logging.info('Avg Dice scores: {}'.format(test_dice_score.avg))
    #     logging.info('Avg HD95 scores: {}'.format(test_hd95_score.avg))


    # ##########Test the best epoch model
    # file_name = os.path.join(ckpts, 'model_best.pth')
    # checkpoint = torch.load(file_name)
    # logging.info('best epoch: {}'.format(checkpoint['epoch']+1))
    # model.load_state_dict(checkpoint['state_dict'])
    # test_best_score = AverageMeter()
    # with torch.no_grad():
    #     logging.info('###########test validate best model###########')
    #     for i, mask in enumerate(masks_test[::-1]):
    #         logging.info('{}'.format(mask_name[::-1][i]))
    #         dice_best_score = test_softmax(
    #                         test_loader,
    #                         model,
    #                         dataname = args.dataname,
    #                         feature_mask = mask,
    #                         mask_name = mask_name[::-1][i])
    #         test_best_score.update(dice_best_score)
    #     logging.info('Avg scores: {}'.format(test_best_score.avg))



if __name__ == '__main__':
    main()
