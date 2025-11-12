import copy
import pdb
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging


def purity_score_bs(feature_s, target, num_cls=2):
    target = target.float()
    eps = 1e-5
    s = []
    gt = []
    cls_num = 0
    for i in range(num_cls):
        targeti = target[:, i, :, :, :]
        if (torch.sum(targeti,dim=(-3,-2,-1))>0).all():
            proto_s =  torch.sum(feature_s*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_map_s = torch.sqrt(torch.sum((feature_s-proto_s[:,:,None,None,None])**2, dim=1))
            # proto_map_t = -torch.sqrt(torch.sum((feature_t-proto_t[:,:,None,None,None])**2, dim=1))
            s.append(proto_map_s.unsqueeze(1))
            # t.append(proto_map_t.unsqueeze(1))
            gt.append(targeti[:,None])
            cls_num += 1

    dist_map_s = torch.cat(s,dim=1)
    sim_map_s = torch.nn.Softmax(dim=1)(-dist_map_s)
    # sim_map_t = torch.nn.Softmax(dim=1)(torch.cat(t,dim=1))
    gt = torch.cat(gt,dim=1)

    pred_indices = torch.argmax(sim_map_s, dim=1)
    pred_one_hot = F.one_hot(pred_indices, num_classes=cls_num).permute(0, 4, 1, 2, 3)

    pred_gt_sum = torch.sum(pred_one_hot*gt, dim=(1,2,3,4))
    gt_sum = torch.sum(gt, dim=(1,2,3,4))
    purity_score = pred_gt_sum / gt_sum

    return purity_score

class DRe(nn.Module):
    
    def __init__(self, optimizer, n_tasks, writer=None, epsilon=1e-8):
    # def __init__(self, num_tasks, temperature=2.0, eps=1e-8):
        super(DRe, self).__init__()
        self.n_tasks = n_tasks
        self.epsilon = epsilon

        # 所有任务的权重（可监控）
        self.task_weights = nn.Parameter(torch.ones(n_tasks), requires_grad=False)
        self._optim = optimizer
        first_param = optimizer.param_groups[0]['params'][0]  # 第一个参数
        device = first_param.device
        # 每个任务独立维护历史损失
        self.rebalanced = True
        self.all_modal_account = 0
        self.phi_tilde = []
        for group in self._optim.param_groups:
            for p in group['params']:
                self.phi_tilde.append(p.detach().clone().cpu())


    def zero_grad(self):
        self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()


    def dre_backward(self, objectives, outputs, target, active_tasks=None):

        device = self.task_weights.device

        active_mask = torch.tensor(
                [obj.item() > self.epsilon for obj in objectives],
                dtype=torch.bool, device=device
            )
        active_indices = torch.where(active_mask)[0]
        if len(active_indices) == 5:
            self.all_modal_account += 1
            if self.all_modal_account % 20 == 0:
                self.rebalanced = False
        self._optim.zero_grad()
        total_loss = 0.0
        for i in range(self.n_tasks):
            if active_mask[i]:
                total_loss += objectives[i]
        total_loss.backward()
        return

    def re_learning(self, objectives, outputs, target, active_tasks=None):
        # --- 更新存在的任务 ---
        device = self.task_weights.device
        alpha_k = torch.zeros(self.n_tasks, device=device)
        # purity_d = torch.zeros(self.n_tasks, device=device)
        # purity_v = torch.zeros(self.n_tasks, device=device)
        g_k = torch.zeros(self.n_tasks, device=device)
        for i in range(self.n_tasks):
            purity_score = purity_score_bs(outputs[i], target)
            # purity_d[i] = purity_score[0]
            # purity_v[i] = purity_score[1]
            g_k[i] = torch.abs(purity_score[0] - purity_score[1])
            alpha_k[i] = torch.tanh(3.0 * g_k[i])
        # 手动更新 task_weights（只更新存在的任务）
        grads, shapes, has_grads, param = self._pack_grad(objectives)
        param_tilde = self._flatten_grad(self.phi_tilde, shapes)
        self.reinit_update(shapes, has_grads, param, param_tilde, alpha_k)
        logging.info('Re-Learning Alpha:[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}]'.format(alpha_k[0].item(),alpha_k[1].item(),alpha_k[2].item(),alpha_k[3].item(),alpha_k[4].item()))
        self.rebalanced = True
        return

    def reinit_update(self, shapes, has_grads, param, param_tilde, alpha_k):
        shared = torch.stack(has_grads).prod(dim=0).bool()
        for i, has_grad in enumerate(has_grads):
            param[has_grad.bool() & ~shared] -= alpha_k[i] * (param[has_grad.bool() & ~shared] - param_tilde[has_grad.bool() & ~shared])
        param = self._unflatten_grad(param, shapes[0])
        with torch.no_grad():
            for group in self._optim.param_groups:
                for p, g in zip(group['params'], param):
                    g_gpu = g.to(p.device, non_blocking=True)
                    p.data.copy_(g_gpu)
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for ii, obj in enumerate(objectives):
            self._optim.zero_grad(set_to_none=True)
            if obj.item() > self.epsilon:
                obj.backward(retain_graph=True)
            # obj.backward(retain_graph=True)
            grad, shape, has_grad, param = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        self._optim.zero_grad()
        param = self._flatten_grad(param, shape)
        return grads, shapes, has_grads, param

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        grad, shape, has_grad, param = [], [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    param.append(p.detach().clone().cpu())
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
                param.append(p.detach().clone().cpu())
        return grad, shape, has_grad, param

    def check_relearning(self):
        return self.rebalanced

    def reset_history(self):
        """可选：重置历史（如新 epoch 开始）"""
        self.prev_loss.zero_()
        self.prev_prev_loss.zero_()
        self.task_initialized = [False] * self.n_tasks