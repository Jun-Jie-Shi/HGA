import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['sigmoid_dice_loss','softmax_dice_loss','FocalLoss', 'dice_loss', 'temp_kl_loss', 'prototype_passion_loss', 'prototype_pmr_loss']

cross_entropy = F.cross_entropy

def DiceCoef(preds, targets, num_cls):
    eps = 1e-7
    class_num = num_cls
    sigmoid = nn.Softmax(dim=1)
    preds = sigmoid(preds)
    loss = torch.zeros(class_num, device=preds.device)
    for i in range(class_num):
        pred = preds[:,i,:,:,:]
        target = targets[:,i,:,:,:]
        intersection = (pred * target).sum()

        loss[i] += 1 - ((2.0 * intersection) / (pred.sum() + target.sum() + eps))
    return torch.mean(loss)

def DiceCoef2(preds, targets, num_cls):
    eps = 1e-7
    class_num = num_cls
    sigmoid = nn.Softmax(dim=1)
    preds = sigmoid(preds)
    loss = torch.zeros(class_num-1, device=preds.device)
    for i in range(class_num):
        if i==0:
            continue
        pred = preds[:,i,:,:,:]
        target = targets[:,i,:,:,:]
        intersection = (pred * target).sum()

        loss[i-1] += 1 - ((2.0 * intersection) / (pred.sum() + target.sum() + eps))
    return torch.mean(loss)

def DiceKDLoss(student_preds, teacher_preds, target, num_cls=5, temp=1.0, up_op=None):
    """
    Dice-based Knowledge Distillation Loss
    计算学生与教师概率图之间的软 Dice 损失

    Args:
        student_preds (Tensor): 学生模型输出的 logits 或概率图 shape [B, C, H, W, D] 或 [B, C, H, W]
        teacher_preds (Tensor): 教师模型输出的 soft 概率图 shape 同 student_preds
        num_cls (int): 类别数（含背景）

    Returns:
        loss (Tensor): 标量，平均的 Dice-KD 损失
    """
    eps=1e-7
    class_num = num_cls
    device = student_preds.device
    loss = torch.zeros(class_num, device=device)

    # 确保输入是概率图（如果是 logits，先加 softmax）
    # 假设输入已经是 softmax 后的概率图，否则取消注释下一行
    student_preds = torch.softmax(student_preds/temp, dim=1)
    teacher_preds = torch.softmax(teacher_preds/temp, dim=1)  # 如果 teacher 是 logits

    for i in range(class_num):
        # 提取第 i 个类别的预测概率图
        pred_s = student_preds[:, i, ...]  # 学生
        pred_t = teacher_preds[:, i, ...]  # 教师
        # 计算交集
        intersection = (pred_s * pred_t).sum()

        # Dice Loss for class i
        loss[i] += 1 - (2.0 * intersection) / (pred_s.sum() + pred_t.sum() + eps)
        # loss[i] = 1.0 - dice  # 转为损失

    # 返回所有类别的平均 Dice-KD 损失
    return torch.mean(loss)

def dice_loss(output, target, num_cls=5, eps=1e-7, up_op=None):
    target = target.float()
    if up_op:
        output = up_op(output)
    for i in range(num_cls):
        num = torch.sum(output[:,i,:,:,:] * target[:,i,:,:,:])
        l = torch.sum(output[:,i,:,:,:])
        r = torch.sum(target[:,i,:,:,:])
        if i == 0:
            # dice = 2.0 * num / (l+r+eps)
            dice = 0.0
        else:
            dice += 2.0 * num / (l+r+eps)
    return 1.0 - 1.0 * dice / (num_cls-1)

def dice_loss_bs(output, target, num_cls=5, eps=1e-7, up_op=None):
    target = target.float()
    if up_op:
        output = up_op(output)
    for i in range(num_cls):
        num = torch.sum(output[:,i,:,:,:] * target[:,i,:,:,:], dim=(1,2,3))
        l = torch.sum(output[:,i,:,:,:], dim=(1,2,3))
        r = torch.sum(target[:,i,:,:,:], dim=(1,2,3))
        if i == 0:
            dice = 2.0 * num / (l+r+eps)
        else:
            dice += 2.0 * num / (l+r+eps)
        dice_loss = (1.0 - 1.0 * dice / num_cls).unsqueeze(1)
    return dice_loss

def softmax_weighted_loss(output, target, num_cls=5, up_op=None):
    target = target.float()
    if up_op:
        output = up_op(output)
    B, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
        weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
            # cross_loss = -1.0 * weighted * targeti * torch.log(outputi).float()
        else:
            # cross_loss += -1.0 * weighted * targeti * torch.log(outputi).float()
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss

def softmax_weighted_loss_bs(output, target, num_cls=5, up_op=None):
    target = target.float()
    if up_op:
        output = up_op(output)
    B, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
        weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
            # cross_loss = -1.0 * weighted * targeti * torch.log(outputi).float()
        else:
            # cross_loss += -1.0 * weighted * targeti * torch.log(outputi).float()
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss, dim=(1,2,3)).unsqueeze(1)
    return cross_loss


def temp_kl_loss(logit_s, logit_t, target, num_cls=5, temp=1.0, up_op=None):
    pred_s = F.softmax(logit_s/temp, dim=1)
    pred_t = F.softmax(logit_t/temp, dim=1)
    if up_op:
        pred_s = up_op(pred_s)
        pred_t = up_op(pred_t)
    pred_s = torch.clamp(pred_s, min=0.005, max=1)
    pred_t = torch.clamp(pred_t, min=0.005, max=1)
    pred_s = torch.log(pred_s)
    kl_loss = temp * temp * torch.mul(pred_t, torch.log(pred_t)-pred_s)
    kl_loss = torch.mean(kl_loss)
    return kl_loss

def temp_weighted_kl_loss(logit_s, logit_t, target, num_cls=5, temp=1.0, up_op=None):
    target = target.float()
    pred_s = F.softmax(logit_s/temp, dim=1)
    pred_t = F.softmax(logit_t/temp, dim=1)
    if up_op:
        pred_s = up_op(pred_s)
        pred_t = up_op(pred_t)
    B, _, H, W, Z = pred_s.size()
    pred_s = torch.clamp(pred_s, min=0.005, max=1)
    pred_t = torch.clamp(pred_t, min=0.005, max=1)
    # pred_s = torch.log(pred_s)
    class_weights = []
    for i in range(num_cls):
        count_i = torch.sum(target[:, i, :, :, :], (1,2,3))  # (B,)
        total = torch.sum(target, (1,2,3,4))                # (B,)
        w = 1.0 - (count_i / total)  # (B,)
        class_weights.append(w.unsqueeze(1))  # (B, 1)
    class_weights = torch.cat(class_weights, dim=1).view(B, num_cls, 1, 1, 1)
    pixel_weights = torch.sum(target * class_weights, dim=1, keepdim=True)
    pred_s = torch.log(pred_s)
    kl_loss = temp * temp * torch.mul(pred_t, torch.log(pred_t)-pred_s)
    # kl_loss = temp * temp * torch.mul(pred_t, torch.log(pred_t)-pred_s)
    kl_loss = torch.mean(pixel_weights * kl_loss)
    return kl_loss

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def temp_kl_loss_bs(logit_s, logit_t, target, num_cls=5, temp=1.0, up_op=None):
    pred_s = F.softmax(logit_s/temp, dim=1)
    pred_t = F.softmax(logit_t/temp, dim=1)
    if up_op:
        pred_s = up_op(pred_s)
        pred_t = up_op(pred_t)
    pred_s = torch.clamp(pred_s, min=0.005, max=1)
    pred_t = torch.clamp(pred_t, min=0.005, max=1)
    pred_s = torch.log(pred_s)
    kl_loss = temp * temp * torch.mul(pred_t, torch.log(pred_t)-pred_s)
    kl_loss = torch.mean(kl_loss, dim=(1,2,3,4)).unsqueeze(1)
    return kl_loss


# def temp_kl_loss_bs(logit_s, logit_t, target, num_cls=5, temp=1.0, up_op=None, alpha=1.1):
#     pred_s = F.softmax(logit_s, dim=1)
#     pred_t = F.softmax(logit_t, dim=1)
#     if up_op:
#         pred_s = up_op(pred_s)
#         pred_t = up_op(pred_t)
#     pred_s = torch.clamp(pred_s, min=0.005, max=1)
#     pred_t = torch.clamp(pred_t, min=0.005, max=1)
#     integral_pq = (pred_s * pred_t).sum(dim=1)
#     integral_p_alpha = (pred_s ** alpha).sum(dim=1) ** (1 / alpha)
#     beta = alpha / (alpha - 1)
#     integral_q_beta = (pred_t ** beta).sum(dim=1) ** (1 / beta)
#     divergence = -torch.log(integral_pq / (integral_p_alpha * integral_q_beta))
#     holder_loss = torch.mean(divergence, dim=(1,2,3)).unsqueeze(1)

#     return holder_loss


def prototype_passion_loss(feature_s, feature_t, target, logit_s, logit_t, num_cls=5, temp=1.0, up_op=None):
    target = target.float()
    eps = 1e-5
    N = len(feature_s.size()) - 2
    s = []
    t = []
    st = []
    logit_ss = []
    logit_tt = []
    proto_fs = torch.zeros_like(feature_s).cuda().float()

    for i in range(num_cls):
        targeti = target[:, i, :, :, :]
        if (torch.sum(targeti,dim=(-3,-2,-1))>0).all():
            proto_s =  torch.sum(feature_s*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_t =  torch.sum(feature_t*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_fs += proto_s[:,:,None,None,None] * targeti[:,None]
            proto_map_s = F.cosine_similarity(feature_s,proto_s[:,:,None,None,None],dim=1,eps=eps)
            proto_map_t = F.cosine_similarity(feature_t,proto_t[:,:,None,None,None],dim=1,eps=eps)
            s.append(proto_map_s.unsqueeze(1))
            t.append(proto_map_t.unsqueeze(1))
            logit_ss.append(logit_s[:, i, :, :, :].unsqueeze(1))
            logit_tt.append(logit_t[:, i, :, :, :].unsqueeze(1))

    for i in range(num_cls):
        targeti = target[:, i, :, :, :]
        if (torch.sum(targeti,dim=(-3,-2,-1))>0).all():
            proto_t =  torch.sum(feature_t*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_map_st = F.cosine_similarity(proto_fs,proto_t[:,:,None,None,None],dim=1,eps=eps)
            st.append(proto_map_st.unsqueeze(1))
    sim_map_s = torch.cat(s,dim=1)
    sim_map_t = torch.cat(t,dim=1)
    proto_loss = torch.mean((sim_map_s-sim_map_t)**2)

    dist = torch.mean(torch.sqrt((sim_map_s-sim_map_t)**2))

    return proto_loss, dist

def prototype_passion_loss_bs(feature_s, feature_t, target, logit_s=None, logit_t=None, num_cls=5, temp=1.0, up_op=None):
    target = target.float()
    eps = 1e-5
    # N = len(feature_s.size()) - 2
    s = []
    t = []
    gt = []
    # st = []
    # logit_ss = []
    # logit_tt = []
    # proto_fs = torch.zeros_like(feature_s).cuda().float()

    for i in range(num_cls):
        targeti = target[:, i, :, :, :]
        if (torch.sum(targeti,dim=(-3,-2,-1))>0).all():
            proto_s =  torch.sum(feature_s*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            # proto_t =  torch.sum(feature_t*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            # proto_fs += proto_s[:,:,None,None,None] * targeti[:,None]
            # proto_map_s = F.cosine_similarity(feature_s,proto_s[:,:,None,None,None],dim=1,eps=eps)
            # proto_map_t = F.cosine_similarity(feature_t,proto_t[:,:,None,None,None],dim=1,eps=eps)
            proto_map_s = torch.sqrt(torch.sum((feature_s-proto_s[:,:,None,None,None])**2, dim=1))
            # proto_map_t = -torch.sqrt(torch.sum((feature_t-proto_t[:,:,None,None,None])**2, dim=1))
            s.append(proto_map_s.unsqueeze(1))
            # t.append(proto_map_t.unsqueeze(1))
            gt.append(targeti[:,None])
            # logit_ss.append(logit_s[:, i, :, :, :].unsqueeze(1))
            # logit_tt.append(logit_t[:, i, :, :, :].unsqueeze(1))

    dist_map_s = torch.cat(s,dim=1)
    sim_map_s = torch.nn.Softmax(dim=1)(-dist_map_s)
    # sim_map_t = torch.nn.Softmax(dim=1)(torch.cat(t,dim=1))
    gt = torch.cat(gt,dim=1)
    # sim_map_s_gt = torch.sum(sim_map_s*gt, dim=1)
    # dist_map_s_gt = torch.sum(dist_map_s*gt, dim=1)

    # dce_loss = torch.mean(-torch.log(torch.clamp(sim_map_s_gt, min=0.005, max=1)))
    # pl_loss = torch.mean(dist_map_s_gt)

    # proto_loss = dce_loss + 0.001 * pl_loss

    # dist = torch.mean(sim_map_s_gt)

    sim_foreground = sim_map_s[:, 1, ...]      # [B, H, W]
    gt_foreground = gt[:, 1, ...]              # [B, H, W]

    # 只对前景位置计算损失
    valid_mask = (gt_foreground > 0.5)


    sim_pos = sim_foreground[valid_mask]
    dce_loss = torch.mean(-torch.log(torch.clamp(sim_pos, min=0.005, max=1)))

    # Prototype loss: 前景类的距离
    dist_foreground = dist_map_s[:, 1, ...]
    pl_loss = torch.mean(dist_foreground[valid_mask])
    proto_loss = dce_loss + 0.001 * pl_loss

    dist = torch.mean(sim_pos)

    return proto_loss, dist
    # target = target.float()
    # eps = 1e-5
    # N = len(feature_s.size()) - 2
    # ss = []
    # gt = []

    # for i in range(num_cls):
    #     targeti = target[:, i, :, :, :]
    #     if (torch.sum(targeti,dim=(-3,-2,-1))>0).all():
    #         proto_s =  torch.sum(feature_s*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
    #         # proto_t =  torch.sum(feature_t*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
    #         proto_map_ss = -torch.sqrt(torch.sum((feature_s-proto_s[:,:,None,None,None])**2, dim=1))
    #         ss.append(proto_map_ss.unsqueeze(1))
    #         gt.append(targeti[:,None])

    # softmax_s = torch.nn.Softmax(dim=1)(torch.cat(ss,dim=1))
    # gt = torch.cat(gt,dim=1)

    # proto_distri_s = torch.sum(softmax_s*gt, dim=1)
    # proto_loss = torch.mean(-torch.log(torch.clamp(proto_distri_s, min=0.005, max=1)))
    # dist = torch.mean(proto_distri_s)

    # return proto_loss, dist


def prototype_pmr_loss(feature_s, feature_t, target, logit_s=None, logit_t=None, num_cls=5, temp=1.0, up_op=None):
    target = target.float()
    eps = 1e-5
    N = len(feature_s.size()) - 2
    ss = []
    gt = []

    for i in range(num_cls):
        targeti = target[:, i, :, :, :]
        if (torch.sum(targeti,dim=(-3,-2,-1))>0).all():
            proto_s =  torch.sum(feature_s*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            # proto_t =  torch.sum(feature_t*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_map_ss = -torch.sqrt(torch.sum((feature_s-proto_s[:,:,None,None,None])**2, dim=1))
            ss.append(proto_map_ss.unsqueeze(1))
            gt.append(targeti[:,None])

    softmax_s = torch.nn.Softmax(dim=1)(torch.cat(ss,dim=1))
    gt = torch.cat(gt,dim=1)

    proto_distri_s = torch.sum(softmax_s*gt, dim=1)
    proto_loss = torch.mean(-torch.log(torch.clamp(proto_distri_s, min=0.005, max=1)))
    kl_loss = torch.mean(proto_distri_s)

    return proto_loss, kl_loss

def softmax_loss(output, target, num_cls=5):
    target = target.float()
    _, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        if i == 0:
            cross_loss = -1.0 * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss

def FocalLoss(output, target, alpha=0.25, gamma=2.0):
    target[target == 4] = 3 # label [4] -> [3]
    # target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4,H,W,D]
    if output.dim() > 2:
        output = output.view(output.size(0), output.size(1), -1)  # N,C,H,W,D => N,C,H*W*D
        output = output.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
        output = output.contiguous().view(-1, output.size(2))  # N,H*W*D,C => N*H*W*D,C
    if target.dim() == 5:
        target = target.contiguous().view(target.size(0), target.size(1), -1)
        target = target.transpose(1, 2)
        target = target.contiguous().view(-1, target.size(2))
    if target.dim() == 4:
        target = target.view(-1) # N*H*W*D
    # compute the negative likelyhood
    logpt = -F.cross_entropy(output, target)
    pt = torch.exp(logpt)
    # compute the loss
    loss = -((1 - pt) ** gamma) * logpt
    # return loss.sum()
    return loss.mean()

def dice(output, target,eps =1e-5): # soft dice loss
    target = target.float()
    # num = 2*(output*target).sum() + eps
    num = 2*(output*target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

def sigmoid_dice_loss(output, target,alpha=1e-5):
    # output: [-1,3,H,W,T]
    # target: [-1,H,W,T] noted that it includes 0,1,2,4 here
    loss1 = dice(output[:,0,...],(target==1).float(),eps=alpha)
    loss2 = dice(output[:,1,...],(target==2).float(),eps=alpha)
    loss3 = dice(output[:,2,...],(target == 4).float(),eps=alpha)
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))
    return loss1+loss2+loss3


def softmax_dice_loss(output, target,eps=1e-5): #
    # output : [bsize,c,H,W,D]
    # target : [bsize,H,W,D]
    loss1 = dice(output[:,1,...],(target==1).float())
    loss2 = dice(output[:,2,...],(target==2).float())
    loss3 = dice(output[:,3,...],(target==4).float())
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))

    return loss1+loss2+loss3



