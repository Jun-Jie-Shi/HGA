#coding=utf-8
import argparse
import logging
import os
import random
import time
from collections import OrderedDict
import csv
import gc
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import math
import higher
import copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters
# import torch.optim
from data.data_utils import init_fn
from data.datasets_nii import (Brats_loadall_train_nii_pdt, Brats_loadall_test_nii, Brats_loadall_metaval_nii_idt,
                               Brats_loadall_val_nii, Brats_loadall_train_nii_idt)
from data.transforms import *
from models import rfnet, mmformer, m2ftrans, mmformer_task, mmformer_nopassion
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from options import args_parser
from utils import criterions
from utils.predict import AverageMeter, test_dice_hd95_softmax
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from utils.parser import setup, set_seed, to_var

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
    if args.dataname in ['BraTS/BRATS2021', 'BraTS/BRATS2020', 'BraTS/BRATS2018']:
        num_cls = 4
    else:
        print ('dataset is error')
        exit(0)

    if args.model == 'mmformer':
        if args.use_multitask:
            model = mmformer_nopassion.Model(num_cls=num_cls)
        else:
            model = mmformer.Model(num_cls=num_cls)
    elif args.model == 'rfnet':
        model = rfnet.Model(num_cls=num_cls)
    elif args.model == 'm2ftrans':
        model = m2ftrans.Model(num_cls=num_cls)

    print (model)
    model = torch.nn.DataParallel(model).cuda()
    model.module.mask_type = args.mask_type
    model.module.use_passion = args.use_passion
    model.module.use_multitask = args.use_multitask
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    # log_meta_lr = torch.tensor([math.log(args.lr)], requires_grad=True, device=device)
    # log_meta_lr = torch.tensor([math.log(args.lr)], requires_grad=True, device=device)
    # log_meta_lr = to_var(torch.zeros(1))
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    # meta_train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    
    
    temp = args.temp
    ##########Setting data
        ####BRATS2020
    if args.dataname == 'BraTS/BRATS2020':
        train_file = os.path.join(args.datarootPath, args.imbmrpath)
        test_file = os.path.join(args.datasetPath, 'test.txt')
        # valid_file = os.path.join(args.datasetPath, 'val.txt')
    #### Other Datasets Setting (Like BraTS2020)
    elif args.dataname == 'BraTS/BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        train_file = os.path.join(args.datarootPath, 'BraTS/brats_split/Brats2018_imb_split_mr2468.csv')
        test_file = os.path.join(args.datasetPath, 'test1.txt')
    #   valid_file = os.path.join(args.datasetPath, 'val1.txt')
    # elif args.dataname == 'BraTS/BRATS2021':
    #     ####BRATS2021

    logging.info(str(args))
    set_seed(args.seed)
    if args.mask_type in ['pdt', 'idt', 'idt_drop']:
        train_set = Brats_loadall_train_nii_idt(transforms=args.train_transforms, root=args.datasetPath, num_cls=num_cls, mask_type=args.mask_type, train_file=train_file)
        # train_set = Brats_loadall_train_nii_idt(transforms=args.train_transforms, root=args.datasetPath, num_cls=num_cls, mask_type='idt', train_file=train_file)
    else:
        print ('training setting is error')
        exit(0)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datasetPath, test_file=test_file)
    valid_set = Brats_loadall_metaval_nii_idt(transforms=args.train_transforms, root=args.datasetPath, mask_value=14, num_cls=num_cls, train_file=train_file)
    # generator_train = torch.Generator()
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
    metaval_loader = MultiEpochsDataLoader(
        dataset=valid_set,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn,
        generator=generator_metaval)

    #### Whether use pretrained model
    if args.resume is not None and args.use_pretrain:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # logging.info('pretrained_dict: {}'.format(pretrained_dict))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logging.info('load ok')


    ##########Training
    masks_multi = torch.from_numpy(np.array([[True]])).detach()
    masks_alluni = torch.from_numpy(np.array([[True, True, True, True]])).detach()

    start = time.time()
    torch.set_grad_enabled(True)

    iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)

    metaval_iter = iter(metaval_loader)
    metaval_len = len(metaval_loader)
    logging.info('Meta Validation Samples with Full-Modalities-{:d}'.format(int(metaval_len)))

    ### IDT Init-Imb-Weight Setting
    imb_mr_csv_data = pd.read_csv(train_file)
    modal_num = torch.tensor((0,0,0,0), requires_grad=False).cuda().float()
    for sample_mask in imb_mr_csv_data['mask']:
        modal_num += torch.tensor(eval(sample_mask), requires_grad=False).cuda().float()
    logging.info('Training Imperfect Datasets with Mod.Flair-{:d}, Mod.T1c-{:d}, Mod.T1-{:d}, Mod.T2-{:d}'\
    .format(int(modal_num[0].item()), int(modal_num[1].item()), int(modal_num[2].item()), int(modal_num[3].item())))
    phi_0 = [p.clone() for p in model.parameters()]

    if args.use_multitask:
        if args.mask_type == 'pdt':
            logging.info('#############MultiTask-PDT-Training############')
        elif args.mask_type == 'idt':
            logging.info('#############MultiTask-IDT-Training############')
        for epoch in range(args.num_epochs):
            step_lr = lr_schedule(optimizer, epoch)
            writer.add_scalar('lr', step_lr, global_step=(epoch+1))
            epoch_fuse_losses = torch.zeros(1).cpu().float()
            # epoch_uni_losses = torch.zeros(1).cpu().float()
            epoch_losses = torch.zeros(1).cpu().float()
            epoch_uni_m = torch.zeros(4).cpu().float()

            b = time.time()
            for i in range(iter_per_epoch):
                step = (i+1) + epoch*iter_per_epoch
                ###Data load
                try:
                    data = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    data = next(train_iter)
                x, target, mask, patients_name = data
                x = x.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                mask = mask.cuda(non_blocking=True)

                model.module.is_training = True

                uni_loss_m = torch.zeros(4).cuda().float()
                # meta_model = copy.copy(model)

                masks_multi_bs = torch.tile(masks_multi, [x.size(0), 1]).cuda()
                if args.mask_type == 'idt':
                    masks_multi_uni = torch.cat((masks_multi_bs,mask), dim=1)
                else:
                    masks_alluni_bs = torch.tile(masks_alluni, [x.size(0), 1]).cuda()
                    masks_multi_uni = torch.cat((masks_multi_bs,masks_alluni_bs), dim=1)
                # masks_alluni_bs = torch.tile(masks_alluni, [x.size(0), 1]).cuda()
                # masks_multi_uni = torch.cat((masks_multi_bs,masks_alluni_bs), dim=1)

                fuse_pred, uni_loss_m_bs, prm_preds = model(x, mask, target=target, temp=temp)
                fuse_loss_bs = criterions.softmax_weighted_loss_bs(fuse_pred, target, num_cls=num_cls) + criterions.dice_loss_bs(fuse_pred, target, num_cls=num_cls)
                fuse_loss = torch.sum(fuse_loss_bs)
                uni_loss_m = torch.sum(uni_loss_m_bs, dim=0)
                # uni_loss = (uni_loss_m).sum()

                multi_uni_loss_bs = torch.cat((fuse_loss_bs,uni_loss_m_bs), dim=1)

                w_tilde = masks_multi_uni

                loss_bs = w_tilde * multi_uni_loss_bs

                prm_loss = torch.zeros(1).cuda().float()
                weight_prm = 1.0
                for prm_pred in prm_preds:
                    weight_prm /= 2.0
                    prm_loss += weight_prm * (criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls) + criterions.dice_loss(prm_pred, target, num_cls=num_cls))
                loss = loss_bs.sum() + prm_loss

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

                # model.module.is_training = True

                # uni_loss_m = torch.zeros(4).cuda().float()
                fuse_pred_val, uni_loss_m_bs_val, prm_preds_val = model(x_val, mask_val, target=target_val, temp=temp)
                fuse_loss_bs_val = criterions.softmax_weighted_loss_bs(fuse_pred_val, target_val, num_cls=num_cls) + criterions.dice_loss_bs(fuse_pred_val, target_val, num_cls=num_cls)
                fuse_loss_val = torch.sum(fuse_loss_bs_val)
                uni_loss_m_val = torch.sum(uni_loss_m_bs_val, dim=0)

                multi_uni_loss_bs_val = torch.cat((fuse_loss_bs_val,uni_loss_m_bs_val), dim=1)

                masks_multi_bs_val = torch.tile(masks_multi, [x_val.size(0), 1]).cuda()
                masks_multi_uni_val = torch.cat((masks_multi_bs_val,mask_val), dim=1)
                w_tilde_val = masks_multi_uni_val

                loss_bs_val = w_tilde_val * multi_uni_loss_bs_val

                prm_loss_val = torch.zeros(1).cuda().float()
                weight_prm = 1.0
                for prm_pred_val in prm_preds_val:
                    weight_prm /= 2.0
                    prm_loss_val += weight_prm * (criterions.softmax_weighted_loss(prm_pred_val, target_val, num_cls=num_cls) + criterions.dice_loss(prm_pred_val, target_val, num_cls=num_cls))
                loss_val = loss_bs_val.sum() + prm_loss_val

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                with torch.no_grad():
                    for p, g, g0 in zip(model.parameters(), phi_tilde, phi_0):
                        p.data.copy_(g0 + p - g)
                    phi_0 = [p.clone() for p in model.parameters()]

                msg_val = 'Meta-Validation Loss {:.4f}, '.format(loss_val.item())
                msg_val += 'Multi-Uni:[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}]'.format(fuse_loss_val.item(), uni_loss_m_val[0].item(), uni_loss_m_val[1].item(), uni_loss_m_val[2].item(), uni_loss_m_val[3].item())
                for bs_n in range(x_val.size(0)):
                    msg_val += ' {:>20},'.format(patients_name_val[bs_n])
                logging.info(msg_val)

                epoch_losses += (loss/iter_per_epoch).detach().cpu()
                epoch_fuse_losses += (fuse_loss/iter_per_epoch).detach().cpu()
                # epoch_uni_losses += (uni_loss/iter_per_epoch).detach().cpu()

                if args.mask_type == 'idt':
                    epoch_uni_m += (uni_loss_m/modal_num).detach().cpu()
                else:
                    epoch_uni_m += (uni_loss_m/iter_per_epoch).detach().cpu()

                msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
                # msg += 'meta_lr:{:.6f}, '.format(torch.exp(log_meta_lr).item())
                # msg += 'fuse_loss:{:.4f}, '.format(fuse_loss.item())
                # msg += 'uni_loss:{:.4f}, '.format(uni_loss.item())
                msg += 'Multi-Uni:[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}]'.format(fuse_loss.item(), uni_loss_m[0].item(), uni_loss_m[1].item(), uni_loss_m[2].item(), uni_loss_m[3].item())
                for bs_n in range(x.size(0)):
                    msg += ' {:>20},'.format(patients_name[bs_n])
                # msg += 'weight[{:.2f},{:.2f},{:.2f},{:.2f}] '.format(modal_weight[0].item(), modal_weight[1].item(), modal_weight[2].item(), modal_weight[3].item())
                logging.info(msg)
            b_train = time.time()
            logging.info('train time per epoch: {}'.format(b_train - b))

            # writer.add_scalar('meta_lr', torch.exp(log_meta_lr).item(), global_step=(epoch+1))

            writer.add_scalar('epoch_losses', epoch_losses.item(), global_step=(epoch+1))
            writer.add_scalar('epoch_fuse_losses', epoch_fuse_losses.item(), global_step=(epoch+1))
            # writer.add_scalar('epoch_uni_losses', epoch_uni_losses.item(), global_step=(epoch+1))
            for m in range(4):
                writer.add_scalar('uni_m{}'.format(m), epoch_uni_m[m].item(), global_step=(epoch+1))

            #########model save
            file_name = os.path.join(ckpts, 'model_last.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)

            if (epoch+1) % 100 == 0 or (epoch>=(args.num_epochs-5)):
                file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    },
                    file_name)


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
        csv_writer.writerow(['WT Dice', 'TC Dice', 'ET Dice','ETPro Dice', 'WT HD95', 'TC HD95', 'ET HD95' 'ETPro HD95'])
        file.close()
        for i, mask in enumerate(masks_test[::-1]):
            logging.info('{}'.format(mask_name[::-1][i]))
            file = open(csv_name, "a+")
            csv_writer = csv.writer(file)
            csv_writer.writerow([mask_name[::-1][i]])
            file.close()
            dice_score, hd95_score = test_dice_hd95_softmax(
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
