import copy
import pdb
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

class UncertaintyWeight:
    """
    Uncertainty Weighted Multi-Task Learning
    
    参考: Kendall, Alex, Yarin Gal, and Roberto Cipolla.
    "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." CVPR 2018.
    """
    def __init__(self, optimizer, n_tasks, writer=None, epsilon=1e-8):

        

        self.n_tasks = n_tasks
        self.epsilon = epsilon
        # 可学习参数：sigma_k
        first_param = optimizer.param_groups[0]['params'][0]  # 第一个参数
        device = first_param.device
        self.sigmas = nn.Parameter(torch.ones(n_tasks, device=device, requires_grad=True))
        # self.sigmas_optimizer = optim.Adam([self.sigmas], lr=optimizer.param_groups[0]['lr'] * 10)
        optimizer.add_param_group({'params': [self.sigmas], 'lr': optimizer.param_groups[0]['lr'] * 10})
        self._optim = optimizer
        # self.sigmas = nn.Parameter(torch.ones(n_tasks))  # 初始化 sigma = 1

        # 记录最后一次的权重和 sigma，用于监控
        self.last_weights = None
        self.last_sigmas = None

    def zero_grad(self):
        self._optim.zero_grad(set_to_none=True)

    def step(self):
        self._optim.step()

        # 记录用于监控
        with torch.no_grad():
            self.last_weights = (0.5 / (self.sigmas ** 2)).detach().cpu().numpy()
            self.last_sigmas = self.sigmas.detach().cpu().numpy()
            logging.info('Modality Weights:[{:.4f},{:.4f},{:.4f},{:.4f}]'.format(self.last_weights[0].item(),self.last_weights[1].item(),self.last_weights[2].item(),self.last_weights[3].item()))
        
        return
    
    def uncertainw_backward(self, objectives, active_tasks=None):
        assert len(objectives) == self.n_tasks, f"Expected {self.n_tasks} losses, got {len(objectives)}"
        device = objectives[0].device

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

        self._optim.zero_grad()
        total_loss = 0.0
        for i in range(self.n_tasks):
            if losses[i].item() > self.epsilon:
                total_loss += 0.5 / (self.sigmas[i] ** 2) * objectives[i] + torch.log(1 + self.sigmas[i] ** 2)
        total_loss.backward()  # This will be used by optimizer.step()
        return

    def get_weights(self):
        """返回当前各任务的权重（1/sigma^2）"""
        if self.last_weights is not None:
            return self.last_weights.copy()
        return None

    def get_sigmas(self):
        """返回当前各任务的不确定性 (sigma)"""
        if self.last_sigmas is not None:
            return self.last_sigmas.copy()
        return None
