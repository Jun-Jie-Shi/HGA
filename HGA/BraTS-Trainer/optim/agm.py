import copy
import pdb
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

class AGM(nn.Module):
    
    def __init__(self, optimizer, n_tasks, writer=None, epsilon=1e-8):
    # def __init__(self, num_tasks, temperature=2.0, eps=1e-8):
        super(AGM, self).__init__()
        self.n_tasks = n_tasks
        self.epsilon = epsilon

        # 所有任务的权重（可监控）
        self.task_weights = nn.Parameter(torch.ones(n_tasks), requires_grad=False)
        self._optim = optimizer
        first_param = optimizer.param_groups[0]['params'][0]  # 第一个参数
        device = first_param.device
        # 每个任务独立维护历史损失
        self.train_score = torch.zeros(n_tasks, device=device, requires_grad=False)
        self.train_iter = torch.zeros(n_tasks, requires_grad=False)


    def zero_grad(self):
        self._optim.zero_grad(set_to_none=True)

    def step(self):

        return self._optim.step()

    def agm_backward(self, objectives, ce_scores, active_tasks=None):
    # def forward(self, current_losses, task_mask=None):
        device = self.task_weights.device

        if active_tasks is not None:
            active_mask = torch.tensor(active_tasks, dtype=torch.bool, device=device)
        else:
            active_mask = torch.tensor(
                [obj.item() > self.epsilon for obj in objectives],
                dtype=torch.bool, device=device
            )

        active_indices = torch.where(active_mask)[0]
        if len(active_indices) == 0:
            print("GradNorm Warning: No active tasks. Skipping.")
            self._optim.zero_grad()
            (sum(objectives) * 0).backward()
            return

        losses = [obj.detach() for obj in objectives]
        scores = [score.detach() for score in ce_scores]
        scores = torch.stack(scores, dim=0)

        # --- 更新存在的任务 ---
        weights = torch.zeros(self.n_tasks, device=device)
        for i in range(self.n_tasks):
            if losses[i].item() > self.epsilon:
                if self.train_score[i] == 0:
                    self.train_score[i] = scores[i]
    
        mean_score = torch.mean(scores[active_mask])
        modal_len = len(active_indices)
        ratio = torch.exp((mean_score-scores[active_mask])*modal_len/(modal_len-1))
        optimal_mean_score = torch.mean(self.train_score[active_mask])
        optimal_ratio = torch.exp((optimal_mean_score-self.train_score[active_mask])*modal_len/(modal_len-1))
        weights[active_mask] = torch.exp(1.0 * (torch.min(optimal_ratio.to(device)-ratio.to(device), torch.tensor(7).to(device))))

        for i in range(self.n_tasks):
            if losses[i].item() > self.epsilon:
                self.train_iter[i] += 1
                self.train_score[i] = self.train_score[i] * (self.train_iter[i] - 1) / self.train_iter[i] + scores[i] / self.train_iter[i]

        # 手动更新 task_weights（只更新存在的任务）
        with torch.no_grad():
            if len(active_indices) == 1:
                self.task_weights[active_mask] = 1.0
            else:
                self.task_weights[active_mask] = weights[active_mask]

        # --- 计算加权损失 ---
        # weighted_loss = (self.task_weights[active_mask] * objectives[active_mask]).sum()
        self._optim.zero_grad()
        msg_ = 'Modality Weight:['
        total_loss = 0.0
        for i in range(self.n_tasks):
            if losses[i].item() > self.epsilon:
                total_loss += self.task_weights[i] * objectives[i]
                msg_ += '{:.4f}, '.format(self.task_weights[i].item())
            else:
                msg_ += '{:.4f}, '.format(0.0)
        msg_ += ']'
        logging.info(msg_)
        total_loss.backward()  # This will be used by optimizer.step()

        return

    def get_weights(self):
        """返回当前所有任务权重"""
        return self.task_weights.detach().cpu().numpy()

    def reset_history(self):
        """可选：重置历史（如新 epoch 开始）"""
        self.prev_loss.zero_()
        self.prev_prev_loss.zero_()
        self.task_initialized = [False] * self.n_tasks