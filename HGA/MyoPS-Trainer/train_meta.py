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
from data.datasets_nii import (Myops_loadall_test_nii, Myops_loadall_train_nii_longtail, Myops_loadall_val_nii, Myops_loadall_metaval_nii_longtail)
from data.transforms import *
# from visualizer import get_local
# get_local.activate()
from models import mmformer_2d_meta
from predict import AverageMeter, test_dice_hd95_softmax_myops
# from predict import AverageMeter, test_softmax, test_softmax_visualize
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import criterions
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from utils.parser import setup, set_seed
# from utils.visualize import visualize_heads
# from gmd import GMD

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='mmformer_2d_passion', type=str)
parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--iter_per_epoch', default=84, type=int)
parser.add_argument('--dataname', default='/data/MyoPS/MyoPS2020', type=str)
parser.add_argument('--datapath', default='/data/MyoPS/MyoPS2020_Training_none_npy', type=str)
parser.add_argument('--imbmrpath', default='/data/MyoPS/myops_split/MyoPS2020_longtail_split_573.csv', type=str, help='csv path')
parser.add_argument('--savepath', default='outputs/idt_mr573_meta_imb_ra_reptile_linear_maml_bs1_epoch300_lr2e-4_temp4', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--mask_type', default='nosplit_longtail', type=str)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=1037, type=int)
parser.add_argument('--needvalid', default=False, type=bool)
parser.add_argument('--temp', default=4.0, type=float)
parser.add_argument('--gpu', type=str, default='3', help='GPU to use')
# parser.add_argument('--csvpath', default='/home/sjj/MMMSeg/VQ/csv/', type=str)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.train_transforms = 'Compose([CenterCrop3D((256,256,1)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
# args.train_transforms = 'Compose([CenterCrop3D((256,256,1)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([CenterCrop3D((256,256,1)), NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

# csvpath = args.csvpath
# os.makedirs(csvpath, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks_test = [[False, False, True], [False, True, False], [True, False, False],
         [False, True, True], [True, True, False], [True, False, True],
         [True, True, True]]
masks_torch = torch.from_numpy(np.array(masks_test))
mask_name = ['t2', 'lge', 'bsffp',
            'lge-t2', 'bsffp-lge', 'bsffp-t2',
            'bsffp-lge-t2']
print (masks_torch.int())

# masks_valid = [[False, False, True, False],
#             [False, True, True, False],
#             [True, True, False, True],
#             [True, True, True, True]]
masks_valid = [[False, False, True], [False, True, False], [True, False, False],
         [False, True, True], [True, True, False], [True, False, True],
         [True, True, True]]

# t1,t1cet1,flairticet2,flairt1cet1t2
masks_valid_torch = torch.from_numpy(np.array(masks_valid))
masks_valid_array = np.array(masks_valid)
masks_all = [True, True, True]
masks_all_torch = torch.from_numpy(np.array(masks_all))
# mask_name_valid = ['t1',
#                 't1cet1',
#                 'flairt1cet2',
#                 'flairt1cet1t2']
mask_name_valid = ['t2', 'lge', 'bsffp',
            'lge-t2', 'bsffp-lge', 'bsffp-t2',
            'bsffp-lge-t2']
mask_name_single = ['bsffp', 'lge', 't2']
print (masks_valid_torch.int())

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
    if args.dataname in ['/data/MyoPS/MyoPS2020']:
        num_cls = 6
    else:
        print ('dataset is error')
        exit(0)

    if args.model == 'mmformer_2d_passion':
        model = mmformer_2d_meta.Model(num_cls=num_cls)
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
    # pc_adam = GMD(optimizer, reduction='mean', writer=writer)
    temp = args.temp
    ##########Setting data
        ####BRATS2020
    if args.dataname == '/data/MyoPS/MyoPS2020':
        # train_file = '/home/sjj/MMMSeg/BraTS/BRATS2020_Training_none_npy/train.txt'
        train_file = args.imbmrpath
        test_file = '/data/MyoPS/MyoPS2020_Training_none_npy/test.txt'
        # valid_file = '/home/sjj/M2FTrans/BraTS/BRATS2020_Training_none_npy/val.txt'

    logging.info(str(args))
    set_seed(args.seed)
    train_set = Myops_loadall_train_nii_longtail(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, mask_type=args.mask_type, train_file=train_file)
    test_set = Myops_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    valid_set = Myops_loadall_val_nii(transforms=args.test_transforms, root=args.datapath, num_cls=num_cls, train_file=test_file)
    # meta_set = [Myops_loadall_metaval_nii_longtail(transforms=args.train_transforms, root=args.datapath, mask_value=i, num_cls=num_cls, train_file=train_file) for i in range(7)]
    meta_set = Myops_loadall_metaval_nii_longtail(transforms=args.train_transforms, root=args.datapath, mask_value=6, num_cls=num_cls, train_file=train_file)
    generator_metaval = torch.Generator()
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    valid_loader = MultiEpochsDataLoader(
        dataset=valid_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        worker_init_fn=init_fn)
    metaval_loader = MultiEpochsDataLoader(
        dataset=meta_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn,
        generator=generator_metaval)
    # meta_loader = [MultiEpochsDataLoader(
    #     dataset=meta_set[i],
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     pin_memory=True,
    #     shuffle=True,
    #     worker_init_fn=init_fn) for i in range(7)]

    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # logging.info('pretrained_dict: {}'.format(pretrained_dict))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logging.info('load ok')


    #########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = len(train_loader)
    # iter_per_epoch = args.iter_per_epoch
    train_iter = iter(train_loader)
    metaval_iter = iter(metaval_loader)
    metaval_len = len(metaval_loader)
    logging.info('Meta Validation Samples with Full-Modalities-{:d}'.format(int(metaval_len)))
    # meta_iters = [iter(meta_loader[i]) for i in range(7)]
    # meta_list = list(range(7))
    # np.random.shuffle(meta_list)
    # meta_list_iter = iter(meta_list)

    iter_per_epoch = args.iter_per_epoch
    imb_mr_csv_data = pd.read_csv(train_file)
    modal_num = torch.tensor((0,0,0), requires_grad=False).cuda().float()
    for sample_mask in imb_mr_csv_data['mask']:
        modal_num += torch.tensor(eval(sample_mask), requires_grad=False).cuda().float()
    logging.info('Training Imperfect Datasets with Mod.Bssfp-{:d}, Mod.LGE-{:d}, Mod.T2-{:d}'\
    .format(int(modal_num[0].item()), int(modal_num[1].item()), int(modal_num[2].item())))
    # modal_num = torch.tensor((64,46,26), requires_grad=False).cuda().float()
    # modal_weight = modal_num.sum() / (4.0 * modal_num).cuda().float()
    modal_weight = torch.tensor((1,1,1), requires_grad=False).cuda().float()
    modal_weight = (iter_per_epoch/modal_num).cuda().float()
    imb_beta = torch.tensor((1,1,1), requires_grad=False).cuda().float()
    phi_0 = [p.clone() for p in model.parameters()]
    # eta = 0.01
    # eta = 0.01
    # eta_ext = 1.5
    # phi_tilde = [p.clone() for p in model.parameters()]
    # epsilon = 0.1
    for epoch in range(args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        # epsilon = 0.1 + 0.9 * (epoch / args.num_epochs)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        epoch_fuse_losses = torch.zeros(1).cpu().float()
        epoch_sep_losses = torch.zeros(1).cpu().float()
        epoch_prm_losses = torch.zeros(1).cpu().float()
        epoch_kl_losses = torch.zeros(1).cpu().float()
        epoch_proto_losses = torch.zeros(1).cpu().float()
        # epoch_fkl_losses = torch.zeros(1).cpu().float()
        epoch_losses = torch.zeros(1).cpu().float()
        epoch_sep_m = torch.zeros(3).cpu().float()
        # epoch_kl_m = torch.zeros(3).cpu().float()
        # epoch_proto_m = torch.zeros(3).cpu().float()
        # epoch_fkl_m = torch.zeros(3).cpu().float()
        # epoch_imb_m = torch.zeros(4).cpu().float()
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask, name = data
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # batchsize = mask.size(0)
            # all modalities test
            # mask = masks_all_torch.repeat(batchsize, 1)
            # mask = mask[0].repeat(batchsize, 1)  ## to be test

            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True

            fuse_pred, (prm_cross_loss, prm_dice_loss), (sep_cross_loss, sep_dice_loss) = model(x, mask, target=target, temp=temp)

            ###Loss compute
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            prm_loss = prm_cross_loss + prm_dice_loss
            sep_loss_m = sep_cross_loss + sep_dice_loss
            sep_loss = (modal_weight * sep_loss_m).sum()
            loss = fuse_loss + sep_loss + prm_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            phi_tilde = [p.clone() for p in model.parameters()]

            try:
                data_val = next(metaval_iter)
            except:
                metaval_iter = iter(metaval_loader)
                data_val = next(metaval_iter)
            x_val, target_val, mask_val, patients_name_val = data_val
            target_val = target_val.cuda(non_blocking=True)
            mask_val = mask_val.cuda(non_blocking=True)

            fuse_pred_val, (prm_cross_loss_val, prm_dice_loss_val), (sep_cross_loss_val, sep_dice_loss_val) = model(x_val, mask_val, target=target_val, temp=temp)

            ###Loss compute
            fuse_cross_loss_val = criterions.softmax_weighted_loss(fuse_pred_val, target_val, num_cls=num_cls)
            fuse_dice_loss_val = criterions.dice_loss(fuse_pred_val, target_val, num_cls=num_cls)
            fuse_loss_val = fuse_cross_loss_val + fuse_dice_loss_val

            prm_loss_val = prm_cross_loss_val + prm_dice_loss_val
            sep_loss_m_val = sep_cross_loss_val + sep_dice_loss_val
            sep_loss_val = (modal_weight * sep_loss_m_val).sum()
            loss_val = fuse_loss_val + sep_loss_val + prm_loss_val

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            with torch.no_grad():
                for p, g, g0 in zip(model.parameters(), phi_tilde, phi_0):
                    p.data.copy_(g0 + p - g)
                phi_0 = [p.clone() for p in model.parameters()]

            msg_val = 'Meta-Validation Loss {:.4f}, '.format(loss_val.item())
            msg_val += 'Multi-Uni:[{:.4f},{:.4f},{:.4f},{:.4f}]'.format(fuse_loss_val.item(), sep_loss_m_val[0].item(), sep_loss_m_val[1].item(), sep_loss_m_val[2].item())
            for bs_n in range(x_val.size(0)):
                msg_val += ' {:>20},'.format(patients_name_val[bs_n])
            logging.info(msg_val)

            epoch_losses += (loss/iter_per_epoch).detach().cpu()
            epoch_fuse_losses += (fuse_loss/iter_per_epoch).detach().cpu()
            epoch_prm_losses += (prm_loss/iter_per_epoch).detach().cpu()
            epoch_sep_losses += (sep_loss/iter_per_epoch).detach().cpu()
            # epoch_kl_losses += (kl_loss/iter_per_epoch).detach().cpu()
            # epoch_proto_losses += (proto_loss/iter_per_epoch).detach().cpu()
            # epoch_fkl_losses += (fkl_loss/iter_per_epoch).detach().cpu()

            # epoch_kl_m += (kl_loss_m/modal_num).detach().cpu()
            epoch_sep_m += (sep_loss_m/modal_num).detach().cpu()
            # epoch_proto_m += (proto_loss_m/modal_num).detach().cpu()
            # epoch_fkl_m += (fkl_loss_m/modal_num).detach().cpu()
            # epoch_imb_m += (ra_imb_m/modal_num).detach().cpu()

            # epoch_kl_m += (kl_loss_m/iter_per_epoch).detach().cpu()
            # epoch_sep_m += (sep_loss_m/iter_per_epoch).detach().cpu()
            # epoch_proto_m += (proto_loss_m/iter_per_epoch).detach().cpu()
            # epoch_fkl_m += (fkl_loss_m/iter_per_epoch*2).detach().cpu()

            ###log
            # writer.add_scalar('loss', loss.item(), global_step=step)
            # writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            # writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            # writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            # writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            # writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            # writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)
            # writer.add_scalar('kl_loss', kl_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f}, '.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f}, '.format(prm_cross_loss.item(), prm_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f}, '.format(sep_cross_loss.sum().item(), sep_dice_loss.sum().item())
            # msg += 'kl:{:.4f}, '.format(kl_loss.item())
            # msg += 'kl_loss:{:.4f}, proto_loss:{:.4f},'.format(kl_loss.item(), proto_loss.item())
            msg += 'seplist:[{:.4f},{:.4f},{:.4f}] '.format(sep_loss_m[0].item(), sep_loss_m[1].item(), sep_loss_m[2].item())
            # msg += 'kllist:[{:.4f},{:.4f},{:.4f}] '.format(kl_loss_m[0].item(), kl_loss_m[1].item(), kl_loss_m[2].item())
            # msg += 'protolist:[{:.4f},{:.4f},{:.4f}] '.format(proto_loss_m[0].item(), proto_loss_m[1].item(), proto_loss_m[2].item())
            # msg += 'distlist:[{:.4f},{:.4f},{:.4f}] '.format(fkl_loss_m[0].item(), fkl_loss_m[1].item(), fkl_loss_m[2].item())
            msg += '{:>20}, '.format(name[0])
            # msg += 'rp_iter[{:.2f},{:.2f},{:.2f},{:.2f}]'.format(rp_iter[0].item(), rp_iter[1].item(), rp_iter[2].item(), rp_iter[3].item())
            # msg += 'imb[{:.2f},{:.2f},{:.2f},{:.2f}]'.format(ra_imb_m[0].item(), ra_imb_m[1].item(), ra_imb_m[2].item(), ra_imb_m[3].item())
            logging.info(msg)
        b_train = time.time()
        logging.info('train time per epoch: {}'.format(b_train - b))


        writer.add_scalar('epoch_losses', epoch_losses.item(), global_step=(epoch+1))
        writer.add_scalar('epoch_fuse_losses', epoch_fuse_losses.item(), global_step=(epoch+1))
        writer.add_scalar('epoch_prm_losses', epoch_prm_losses.item(), global_step=(epoch+1))
        writer.add_scalar('epoch_sep_losses', epoch_sep_losses.item(), global_step=(epoch+1))
        writer.add_scalar('epoch_kl_losses', epoch_kl_losses.item(), global_step=(epoch+1))
        # writer.add_scalar('epoch_fkl_losses', epoch_fkl_losses.item(), global_step=(epoch+1))
        writer.add_scalar('epoch_proto_losses', epoch_proto_losses.item(), global_step=(epoch+1))
        for m in range(3):
            # writer.add_scalar('kl_m{}'.format(m), epoch_kl_m[m].item(), global_step=(epoch+1))
            writer.add_scalar('sep_m{}'.format(m), epoch_sep_m[m].item(), global_step=(epoch+1))
            # writer.add_scalar('proto_m{}'.format(m), epoch_proto_m[m].item(), global_step=(epoch+1))
            # writer.add_scalar('dist_m{}'.format(m), epoch_fkl_m[m].item(), global_step=(epoch+1))
            # writer.add_scalar('rp_m{}'.format(m), ra[m].item(), global_step=(epoch+1))
            # writer.add_scalar('epoch_imb_m{}'.format(m), epoch_imb_m[m].item(), global_step=(epoch+1))

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
                score_modality = torch.zeros(8)
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
                        score_modality[7] += 1.0 - loss.item()
                val_score_avg = score_modality[7] / len(masks_valid_array)
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

        #         for z, _ in enumerate(masks_valid_array):
        #             writer.add_scalar('{}'.format(mask_name_valid[z]), score_modality[z].item(), global_step=epoch+1)
        #         writer.add_scalar('score_average', score_modality[15].item(), global_step=epoch+1)
        #         logging.info('epoch total score: {}'.format(score_modality[15].item()))
        #         logging.info('best score: {}'.format(best_score.item()))
        #         logging.info('best epoch: {}'.format(best_epoch + 1))
        #         logging.info('validate time per epoch: {}'.format(time.time() - b_train))

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    ##########Test the last epoch model
    test_dice_score = AverageMeter()
    test_hd95_score = AverageMeter()
    csv_name = os.path.join(ckpts, '{}.csv'.format(args.model))
    with torch.no_grad():
        logging.info('###########test last epoch model###########')
        file = open(csv_name, "a+")
        csv_writer = csv.writer(file)
        csv_writer.writerow(['LVB Dice', 'RVB Dice', 'MYO Dice', 'LVB HD95', 'RVB HD95', 'MYO HD95'])
        file.close()
        for i, mask in enumerate(masks_test[::-1]):
            logging.info('{}'.format(mask_name[::-1][i]))
            file = open(csv_name, "a+")
            csv_writer = csv.writer(file)
            csv_writer.writerow([mask_name[::-1][i]])
            file.close()
            dice_score, hd95_score = test_dice_hd95_softmax_myops(
                            test_loader,
                            model,
                            dataname = args.dataname,
                            feature_mask = mask,
                            mask_name = mask_name[::-1][i],
                            csv_name = csv_name,
                            )
            test_dice_score.update(dice_score)
            test_hd95_score.update(hd95_score)

        logging.info('Avg Dice scores: {}'.format(test_dice_score.avg))
        logging.info('Avg HD95 scores: {}'.format(test_hd95_score.avg))

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
    # writer_visualize = SummaryWriter(log_dir="visualize/result")
    # visualize_step = 0
    test_dice_score = AverageMeter()
    test_hd95_score = AverageMeter()
    csv_name = os.path.join(ckpts, '{}.csv'.format(args.model))
    with torch.no_grad():
        logging.info('###########test best epoch model###########')
        file = open(csv_name, "a+")
        csv_writer = csv.writer(file)
        csv_writer.writerow(['LVB Dice', 'RVB Dice', 'MYO Dice', 'LVB HD95', 'RVB HD95', 'MYO HD95'])
        file.close()
        for i, mask in enumerate(masks_test[::-1]):
            logging.info('{}'.format(mask_name[::-1][i]))
            file = open(csv_name, "a+")
            csv_writer = csv.writer(file)
            csv_writer.writerow([mask_name[::-1][i]])
            file.close()
            dice_score, hd95_score = test_dice_hd95_softmax_myops(
                            test_loader,
                            model,
                            dataname = args.dataname,
                            feature_mask = mask,
                            mask_name = mask_name[::-1][i],
                            csv_name = csv_name,
                            )
            test_dice_score.update(dice_score)
            test_hd95_score.update(hd95_score)

        logging.info('Avg Dice scores: {}'.format(test_dice_score.avg))
        logging.info('Avg HD95 scores: {}'.format(test_hd95_score.avg))


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
