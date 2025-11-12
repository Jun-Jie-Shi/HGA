import logging
import os
import time
from unittest.mock import patch

import nibabel as nib
import numpy as np
import scipy.misc
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from medpy.metric import hd95
import csv
from torch.utils.tensorboard import SummaryWriter
# from utils.visualize import visualize_heads
# from visualizer import get_local

# get_local.activate()

cudnn.benchmark = True

path = os.path.dirname(__file__)

patch_size = 80

def compute_BraTS_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)
    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 1.0
            # follow ACN and SMU-Net
            # return 373.12866
            # follow nnUNet
    elif num_pred == 0 and num_ref != 0:
        return 1.0
        # follow ACN and SMU-Net
        # return 373.12866
        # follow in nnUNet
    else:
        return hd95(pred, ref, (1, 1, 1))

# def cal_hd95(output, target):
#      # whole tumor
#     mask_gt = (target != 0).astype(int)
#     mask_pred = (output != 0).astype(int)
#     hd95_whole = compute_BraTS_HD95(mask_gt, mask_pred)
#     del mask_gt, mask_pred

#     # tumor core
#     mask_gt = ((target == 1) | (target ==3)).astype(int)
#     mask_pred = ((output == 1) | (output ==3)).astype(int)
#     hd95_core = compute_BraTS_HD95(mask_gt, mask_pred)
#     del mask_gt, mask_pred

#     # enhancing
#     mask_gt = (target == 3).astype(int)
#     mask_pred = (output == 3).astype(int)
#     hd95_enh = compute_BraTS_HD95(mask_gt, mask_pred)
#     del mask_gt, mask_pred

#     mask_gt = (target == 3).astype(int)
#     if np.sum((output == 3).astype(int)) < 500:
#        mask_pred = (output == 3).astype(int) * 0
#     else:
#        mask_pred = (output == 3).astype(int)
#     hd95_enhpro = compute_BraTS_HD95(mask_gt, mask_pred)
#     del mask_gt, mask_pred

#     return (hd95_whole, hd95_core, hd95_enh, hd95_enhpro)

def cal_hd95_msseg(output, target):
    # LVB
    mask_gt = (target == 1).astype(int)
    mask_pred = (output == 1).astype(int)
    hd95_ms = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    return (hd95_ms,)

def softmax_output_dice_class2_msseg(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    ms_dice = intersect1 / denominator1

    dice_separate = torch.unsqueeze(ms_dice, 1)
    dice_evaluate = torch.unsqueeze(ms_dice, 1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test_dice_hd95_softmax_msseg(
        test_loader,
        model,
        dataname = '/data/MSSEG/MSSEG2016',
        feature_mask=None,
        mask_name=None,
        csv_name=None,
        ):

    # H, W, T = 256, 256, 1
    model.eval()
    vals_dice_evaluation = AverageMeter()
    vals_hd95_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    # one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().cuda()

    if dataname in ['/data/MSSEG/MSSEG2016']:
        num_cls = 2
        class_evaluation= ('ms',)
        class_separate = ('ms',)
    # elif dataname == '/home/sjj/MMMSeg/BraTS/BRATS2015':
    #     num_cls = 5
    #     class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
    #     class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'


    for i, data in enumerate(test_loader):
        target = data[1].cuda()
        x = data[0].cuda()
        names = data[-1]
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()
        model.module.is_training=False
        pred = model(x, mask)
        b = time.time()
        pred = torch.argmax(pred, dim=1)

        if dataname in ['/data/MSSEG/MSSEG2016']:
            scores_separate, scores_evaluation = softmax_output_dice_class2_msseg(pred, target)
            scores_hd95 = np.array(cal_hd95_msseg(pred[0].cpu().numpy(), target[0].cpu().numpy()))

        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_dice_evaluation.update(scores_evaluation[k])
            vals_hd95_evaluation.update(scores_hd95)
            # print(scores_evaluation)
            # print(scores_hd95)
            msg += 'DSC: '
            msg += ', '.join(['{}: {:.4f}'.format(c, s) for c, s in zip(class_evaluation, scores_evaluation[k])])
            msg += ', HD95: '
            msg += ', '.join(['{}: {:.4f}'.format(c, s) for c, s in zip(class_evaluation, scores_hd95)])
            file = open(csv_name, "a+")
            csv_writer = csv.writer(file)
            csv_writer.writerow([scores_evaluation[k][0], scores_hd95[0]])
            file.close()
            #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

            logging.info(msg)
    msg = 'Average scores:'
    msg += ' DSC: '
    msg += ', '.join(['{}: {:.4f}'.format(c, s) for c, s in zip(class_evaluation, vals_dice_evaluation.avg)])
    msg += ', HD95: '
    msg += ', '.join(['{}: {:.4f}'.format(c, s) for c, s in zip(class_evaluation, vals_hd95_evaluation.avg)])
    #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    print(msg)
    logging.info(msg)
    model.train()
    return vals_dice_evaluation.avg, vals_hd95_evaluation.avg

def test_ece_softmax_msseg(
        test_loader,
        model,
        dataname = '/data/MSSEG/MSSEG2016',
        feature_mask=None,
        mask_name=None,
        patch_size = 224,
        ):

    all_fg_probs = []
    all_fg_labels = []
    num_cls = 2
    class_names = ['MS']
    model.eval()

    # one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().cuda()
    for i, data in enumerate(test_loader):
        target = data[1].cuda()
        x = data[0].cuda()
        names = data[-1]
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()
        B, _, H, W, Z = x.size()
        model.module.is_training=False
        pred = model(x, mask)
    for i, data in enumerate(test_loader):
        target = data[1].cuda()
        x = data[0].cuda()
        names = data[-1]
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()
        B, _, H, W, Z = x.size()

        logging.info('Subject {}/{}'.format((i+1), len(test_loader)))

        for b in range(B):
            prob_b = pred[b]  # (4, H, W, D)
            label_b = target[b]     # (H, W, D)
            
            # Flatten spatial dimensions
            C, H, W, D = prob_b.shape
            prob_flat = prob_b.reshape(C, -1).transpose(1, 0)  # (H*W*D, 4)
            label_flat = label_b.flatten()  # (H*W*D,)
            
            # Only foreground pixels (label != 0)
            fg_mask = (label_flat != 0)
            if fg_mask.sum() == 0:
                continue  # Skip if no foreground
            
            fg_probs = prob_flat[fg_mask]    # (num_fg_b, 4)
            fg_labels = label_flat[fg_mask]  # (num_fg_b,)
            
            all_fg_probs.append(fg_probs.cpu().numpy())
            all_fg_labels.append(fg_labels.cpu().numpy())

        # all_probs.append(pred.cpu().numpy())
        # all_labels.append(target.cpu().numpy())

    # 转为 numpy
    all_fg_probs = np.concatenate(all_fg_probs, axis=0)
    all_fg_labels = np.concatenate(all_fg_labels, axis=0)

    total_fg = len(all_fg_labels)
    
    ece_foreground = []
    bin_boundaries = np.linspace(0, 1, 20 + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    for c_idx, class_label in enumerate([1,]):  # ET, TC, WT
        confidences = all_fg_probs[:, class_label]  # (TotalFG,)
        binary_labels = (all_fg_labels == class_label).astype(int)
        
        ece = 0.0
        for i in range(20):
            in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
            if in_bin.sum() > 0:
                acc = binary_labels[in_bin].mean()
                conf = confidences[in_bin].mean()
                count = in_bin.sum()
                ece += np.abs(acc - conf) * count
        
        ece /= total_fg  # Normalize by total foreground pixels
        ece_foreground.append(ece)

    macro_ece_fg = np.mean(ece_foreground)

    msg = 'Average scores:'
    msg += ' ECE: '
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_names, ece_foreground)])
    #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    print(msg)
    logging.info(msg)
    model.train()
    return np.array(ece_foreground)

