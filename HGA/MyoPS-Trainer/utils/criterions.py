import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['sigmoid_dice_loss','softmax_dice_loss','GeneralizedDiceLoss','FocalLoss', 'dice_loss']

cross_entropy = F.cross_entropy

def dice_loss(output, target, num_cls=5, eps=1e-7, up_op=None):
    target = target.float()
    if up_op:
        output = up_op(output)
    for i in range(num_cls):
        num = torch.sum(output[:,i,:,:,:] * target[:,i,:,:,:])
        l = torch.sum(output[:,i,:,:,:])
        r = torch.sum(target[:,i,:,:,:])
        if i == 0:
            dice = 2.0 * num / (l+r+eps)
        else:
            dice += 2.0 * num / (l+r+eps)
    return 1.0 - 1.0 * dice / num_cls

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

def temp_weighted_kl_loss(logit_s, logit_t, target, num_cls=5, temp=1.0, up_op=None):
    target = target.float()
    # pred_s = F.log_softmax(logit_s/temp, dim=-1)
    pred_s = F.softmax(logit_s/temp, dim=1) + 10**(-7)
    pred_t = F.softmax(logit_t/temp, dim=1) + 10**(-7)
    if up_op:
        pred_s = up_op(pred_s)
        pred_t = up_op(pred_t)
    pred_s = torch.log(pred_s)
    B, _, H, W, Z = pred_t.size()
    for i in range(num_cls):
        # outputi = output[:, i, :, :, :]
        pred_si = pred_s[:, i, :, :, :]
        pred_ti = pred_t[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
        weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
        if i == 0:
            kl_loss = temp * temp * weighted * torch.mul(pred_ti, torch.log(pred_ti)-pred_si)
        else:
            kl_loss += temp * temp * weighted * torch.mul(pred_ti, torch.log(pred_ti)-pred_si)
    kl_loss = torch.mean(kl_loss)
    return kl_loss

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
    # for i in range(num_cls):
    #     pred_si = pred_s[:, i, :, :, :]
    #     pred_ti = pred_t[:, i, :, :, :]
    #     if i == 0:
    #         kl_loss = temp * temp * torch.mul(pred_ti, torch.log(pred_ti)-pred_si)
    #     else:
    #         kl_loss += temp * temp * torch.mul(pred_ti, torch.log(pred_ti)-pred_si)
    # kl_loss = torch.mean(kl_loss)
    # return kl_loss

def prototype_loss(feature_s, feature_t, target, num_cls=5, temp=1.0, up_op=None):
    target = target.float()
    eps = 1e-5
    N = len(feature_s.size()) - 2
    s = []
    t = []
    for i in range(num_cls):
        targeti = target[:, i, :, :, :]
        if (torch.sum(targeti,dim=(-3,-2,-1))>0).all():
            proto_s =  torch.sum(feature_s*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_t =  torch.sum(feature_t*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_map_s = F.cosine_similarity(feature_s,proto_s[:,:,None,None,None],dim=1,eps=eps)
            proto_map_t = F.cosine_similarity(feature_t,proto_t[:,:,None,None,None],dim=1,eps=eps)
            s.append(proto_map_s.unsqueeze(1))
            t.append(proto_map_t.unsqueeze(1))
    sim_map_s = torch.cat(s,dim=1)
    sim_map_t = torch.cat(t,dim=1)
    proto_loss = torch.mean((sim_map_s-sim_map_t)**2)
    return proto_loss

def prototype_balance_loss(feature_s, feature_t, target, num_cls=5, temp=1.0, up_op=None):
    target = target.float()
    eps = 1e-5
    N = len(feature_s.size()) - 2
    s = []
    t = []
    # ra_imb = []
    # ss = []
    # tt = []
    # gt = []

    for i in range(num_cls):
        targeti = target[:, i, :, :, :]
        if (torch.sum(targeti,dim=(-3,-2,-1))>0).all():
            proto_s =  torch.sum(feature_s*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_t =  torch.sum(feature_t*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_map_s = F.cosine_similarity(feature_s,proto_s[:,:,None,None,None],dim=1,eps=eps)
            proto_map_t = F.cosine_similarity(feature_t,proto_t[:,:,None,None,None],dim=1,eps=eps)
            s.append(proto_map_s.unsqueeze(1))
            t.append(proto_map_t.unsqueeze(1))
            # proto_map_ss = -torch.sqrt(torch.sum((feature_s-proto_s[:,:,None,None,None])**2, dim=1))
            # proto_map_tt = -torch.sqrt(torch.sum((feature_t-proto_t[:,:,None,None,None])**2, dim=1))
            # proto_map_ss = torch.sqrt(torch.sum((feature_s*targeti[:,None]-proto_s[:,:,None,None,None]*targeti[:,None])**2, dim=1)).unsqueeze(1)
            # proto_map_ss = torch.sum(proto_map_ss*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            # proto_distance = torch.sqrt(torch.sum((proto_t-proto_s)**2, dim=1)).unsqueeze(1)
            # ra_imb_cls = proto_distance
            # ra_imb.append(ra_imb_cls)
            # tt.append(proto_map_tt.unsqueeze(1))
            # gt.append(targeti[:,None])

    sim_map_s = torch.cat(s,dim=1)
    sim_map_t = torch.cat(t,dim=1)
    # ra = torch.mean(torch.cat(ra_imb, dim=1))
    # softmax_t = torch.nn.Softmax(dim=1)(torch.cat(tt,dim=1))
    # gt = torch.cat(gt,dim=1)

    # proto_distri_s = torch.sum(softmax_s*gt, dim=1)
    # ra = torch.mean(proto_distri_s)
    # proto_distri_t = torch.sum(softmax_t*gt, dim=1)
    # imb = torch.mean(proto_distri_t)/torch.mean(proto_distri_s)
    # proto_self = torch.mean(-torch.log(torch.clamp(proto_distri_s, min=0.005, max=1)))

    proto_loss = torch.mean((sim_map_s-sim_map_t)**2)
    return proto_loss

def prototype_kl_loss(feature_s, feature_t, target, logit_s, logit_t, num_cls=5, temp=1.0, up_op=None):
    target = target.float()
    eps = 1e-5
    N = len(feature_s.size()) - 2
    s = []
    t = []
    st = []
    # logit_ss = []
    # logit_tt = []
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
            # logit_ss.append(logit_s[:, i, :, :, :].unsqueeze(1))
            # logit_tt.append(logit_t[:, i, :, :, :].unsqueeze(1))

    for i in range(num_cls):
        targeti = target[:, i, :, :, :]
        if (torch.sum(targeti,dim=(-3,-2,-1))>0).all():
            proto_t =  torch.sum(feature_t*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_map_st = F.cosine_similarity(proto_fs,proto_t[:,:,None,None,None],dim=1,eps=eps)
            st.append(proto_map_st.unsqueeze(1))
    sim_map_s = torch.cat(s,dim=1)
    sim_map_t = torch.cat(t,dim=1)
    proto_loss = torch.mean((sim_map_s-sim_map_t)**2)

    sim_map_st = torch.cat(st,dim=1)
    softmax_st = torch.nn.Softmax(dim=1)(sim_map_st)
    # kl_loss = torch.mean(((sim_map_s-sim_map_t)**2) * softmax_st)
    kl_loss = torch.mean(((sim_map_st-sim_map_t)**2))

    # logit_sss = torch.cat(logit_ss, dim=1)
    # logit_ttt = torch.cat(logit_tt, dim=1)

    # pred_s = F.softmax(logit_sss/temp, dim=1)
    # pred_t = F.softmax(logit_ttt/temp, dim=1)
    # if up_op:
    #     pred_s = up_op(pred_s)
    #     pred_t = up_op(pred_t)
    # pred_s = torch.clamp(pred_s, min=0.005, max=1)
    # pred_t = torch.clamp(pred_t, min=0.005, max=1)
    # pred_s = torch.log(pred_s)
    # pred_t = pred_t * softmax_st
    # kl_loss = - temp * temp * torch.mul(pred_t, pred_s)
    # kl_loss = torch.mean(kl_loss)

    return proto_loss, kl_loss

def prototype_passion_loss(feature_s, feature_t, target, logit_s, logit_t, num_cls=5, temp=1.0, up_op=None):
    target = target.float()
    eps = 1e-5
    s = []
    gt = []
    coss = []
    cost = []

    for i in range(num_cls):
        targeti = target[:, i, :, :, :]
        if (torch.sum(targeti,dim=(-3,-2,-1))>0).all():
            proto_s =  torch.sum(feature_s*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_map_s = torch.sqrt(torch.sum((feature_s-proto_s[:,:,None,None,None])**2, dim=1))
            s.append(proto_map_s.unsqueeze(1))
            gt.append(targeti[:,None])

    dist_map_s = torch.cat(s,dim=1)
    sim_map_s = torch.nn.Softmax(dim=1)(-dist_map_s)

    gt = torch.cat(gt,dim=1)
    sim_map_s_gt = torch.sum(sim_map_s*gt, dim=1)
    dist_map_s_gt = torch.sum(dist_map_s*gt, dim=1)

    dce_loss = torch.mean(-torch.log(torch.clamp(sim_map_s_gt, min=0.005, max=1)))
    pl_loss = torch.mean(dist_map_s_gt)
    # proto_loss = dce_loss + 0.001 * pl_loss

    proto_loss = dce_loss + 0.001 * pl_loss

    dist = torch.mean(sim_map_s_gt)

    return proto_loss, dist


def prototype_pmr_loss(feature_s, feature_t, target, logit_s, logit_t, num_cls=5, temp=1.0, up_op=None):
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


# Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
def GeneralizedDiceLoss(output,target,eps=1e-5,weight_type='square'): # Generalized dice loss
    """
        Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """

    # target = target.float()
    if target.dim() == 4:
        target[target == 4] = 3 # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4，H,W,D]

    output = flatten(output)[1:,...] # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:,...] # [class, N*H*W*D]

    target_sum = target.sum(-1) # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :',weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2*intersect[0] / (denominator[0] + eps)
    loss2 = 2*intersect[1] / (denominator[1] + eps)
    loss3 = 2*intersect[2] / (denominator[2] + eps)
    #logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(loss1.data, loss2.data, loss3.data))

    return 1 - 2. * intersect_sum / denominator_sum, [loss1.data, loss2.data, loss3.data]


def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:,1,:,:,:] = (x == 1)
        xx[:,2,:,:,:] = (x == 2)
        xx[:,3,:,:,:] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:,0,:,:,:] = (x == 1)
        xx[:,1,:,:,:] = (x == 2)
        xx[:,2,:,:,:] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)
