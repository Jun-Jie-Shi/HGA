import copy
import pdb
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

class DWA(nn.Module):
    """
    DWA that supports partial task availability.
    Only updates weights for tasks present in current batch.
    Maintains per-task loss history independently.
    """
    
    def __init__(self, optimizer, n_tasks, temperature=2.0, writer=None, epsilon=1e-8):
    # def __init__(self, num_tasks, temperature=2.0, eps=1e-8):
        super(DWA, self).__init__()
        self.n_tasks = n_tasks
        self.temperature = temperature
        self.epsilon = epsilon
        
        # 所有任务的权重（可监控）
        self.task_weights = nn.Parameter(torch.ones(n_tasks), requires_grad=False)
        self._optim = optimizer
        
        # 每个任务独立维护历史损失
        self.register_buffer('prev_loss', torch.zeros(n_tasks))        # L^{(t-1)}
        self.register_buffer('prev_prev_loss', torch.zeros(n_tasks))   # L^{(t-2)}
        
        # 标记哪些任务已经出现过（用于判断是否可计算比率）
        self.task_initialized = [False] * n_tasks  # t-2 是否已设置

    def zero_grad(self):
        self._optim.zero_grad(set_to_none=True)

    def step(self):

        return self._optim.step()

    def dwa_backward(self, objectives, active_tasks=None):
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


        # --- 更新存在的任务 ---
        weights = torch.zeros(self.n_tasks, device=device)
        
        for i in range(self.n_tasks):
            if losses[i].item() > self.epsilon:
                prev = self.prev_loss[i]
                prev_prev = self.prev_prev_loss[i]
                can_compute = self.task_initialized[i]
                if not can_compute:
                    # 无法计算比率：使用默认权重（如 1.0）
                    w_i = 1.0
                else:
                    # 计算相对下降率 r_i = L_i^{(t-1)} / L_i^{(t-2)}
                    r_i = prev / (prev_prev + self.epsilon)
                    w_i = torch.exp(r_i / self.temperature).item()  # 转为标量
            
                weights[i] = w_i
        
        # 只在存在的任务间归一化
        valid_weights = weights[active_mask]
        normalized_weights = valid_weights / (valid_weights.sum() + self.epsilon) * len(valid_weights)
        # 写回
        weights[active_mask] = normalized_weights

        # 手动更新 task_weights（只更新存在的任务）
        with torch.no_grad():
            if len(active_indices) == 1:
                self.task_weights[active_mask] = 1.0
            else:
                self.task_weights[active_mask] = weights[active_mask]

            for i in range(self.n_tasks):
                if losses[i].item() > self.epsilon:
                    # 向前滚动
                    self.prev_prev_loss[i] = self.prev_loss[i]
                    self.prev_loss[i] = losses[i]
                    # 标记已初始化（从第二轮存在开始）
                    if not self.task_initialized[i]:
                        # 第一次出现：prev 已设，但 prev_prev 还没设
                        # 第二次出现时才能计算比率
                        if self.prev_prev_loss[i] > 0:  # 初始为 0，第一次更新后 >0
                            self.task_initialized[i] = True
                    else:
                        # 已初始化，正常更新
                        pass

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