import copy
import pdb
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

class GradNorm:
    def __init__(self, optimizer, n_tasks, alpha=1.5, writer=None, epsilon=1e-8):
        self._optim = optimizer
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.writer = writer
        self.epsilon = epsilon
        self.iter = 0

        # 可学习任务权重
        first_param = optimizer.param_groups[0]['params'][0]  # 第一个参数
        device = first_param.device
        self.task_weights = nn.Parameter(torch.ones(n_tasks, device=device, requires_grad=True))
        self.weight_optimizer = optim.Adam([self.task_weights], lr=optimizer.param_groups[0]['lr'] * 10)

        self.L0 = None  # 初始损失
        self.L1 = None  # 判断是否记录L0
        self._grad_shape_info = None  # 用于存储参数形状信息

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        self._optim.zero_grad(set_to_none=True)

    def step(self):
        self._optim.step()
        self.weight_optimizer.step()
        with torch.no_grad():
            self.task_weights.clamp_(min=self.epsilon)
        weights = (self.task_weights / self.task_weights.sum() * self.n_tasks).detach()
        # weights = self.task_weights
            # logging.info(self.L0)
            # logging.info(self.L1)
        self.task_weights = torch.nn.Parameter(weights, requires_grad=True)
        self.weight_optimizer = optim.Adam([self.task_weights], lr=self._optim.param_groups[0]['lr'] * 10)
        logging.info('Modality Weights:[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}]'.format(self.task_weights[0].item(),self.task_weights[1].item(),self.task_weights[2].item(),self.task_weights[3].item(),self.task_weights[4].item()))
        return
    
    def _retrieve_grad(self):
        """
        Retrieve gradient and has_grad flag for all parameters.
        Returns:
            grad: list of gradient tensors
            has_grad: list of 0/1 tensors (1 if grad exists)
        """
        grad, has_grad = [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    # No gradient: use zero tensor
                    g = torch.zeros_like(p)
                    h = torch.zeros_like(p)
                else:
                    g = p.grad.clone()
                    h = torch.ones_like(p)
                grad.append(g)
                has_grad.append(h)
        return grad, has_grad
    
    def _pack_grad(self, objectives):
        """
        PCGrad-style: compute per-task gradients and pack them.
        Returns:
            packed_grads: [n_tasks, D] tensor, each row is flattened gradient
            packed_has_grad: [n_tasks, D] tensor, 1 if parameter has gradient
        """
        packed_grads, packed_has_grad = [], []
        for obj in objectives:
            self._optim.zero_grad()
            if obj.item() > self.epsilon:
                obj.backward(retain_graph=True)
            grad, has_grad = self._retrieve_grad()
            # Flatten and concatenate
            flat_grad = torch.cat([g.flatten() for g in grad])
            flat_has_grad = torch.cat([h.flatten() for h in has_grad])
            packed_grads.append(flat_grad)
            packed_has_grad.append(flat_has_grad)
            self._optim.zero_grad()  # Clear for next task
        self._optim.zero_grad()  # Final clean
        # Stack into matrices: [n_tasks, total_numel]
        grads_matrix = torch.stack(packed_grads)        # [T, D]
        has_grad_matrix = torch.stack(packed_has_grad)  # [T, D]
        return grads_matrix, has_grad_matrix
    
    # def to_device(self, device):
    #     self.task_weights.to(device)
    
    def gradnorm_backward(self, objectives, active_tasks=None):
        """
        Perform GradNorm update.
        
        :param objectives: list of T scalar loss tensors
        :param active_tasks: list of bool, or None (auto-detect)
        """
        assert len(objectives) == self.n_tasks, f"Expected {self.n_tasks} losses, got {len(objectives)}"
        device = objectives[0].device

        # self.to_device(device)

        # --- 1. Determine active tasks ---
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

        # --- 2. Initialize L0 (initial loss for each task) ---
        if self.L0 is None:
            self.L0 = torch.ones(self.n_tasks, dtype=torch.float32, device=device)
            self.L1 = torch.zeros(self.n_tasks, dtype=torch.float32, device=device)
            for i in range(self.n_tasks):
                if losses[i].item() > self.epsilon:
                    self.L0[i] = losses[i].item()
                    self.L1[i] = 1
            # print(f"[GradNorm] Initialized L0 = {self.L0.cpu().numpy()}")

        # Ensure L0 is updated for newly active tasks
        if self.L1.sum() < 5:
            for i in range(self.n_tasks):
                if self.L1[i] == 0 and losses[i].item() > self.epsilon:
                    self.L0[i] = losses[i].item()
                    self.L1[i] = 1

        # --- 3. Extract per-task gradients ---
        grads_matrix, has_grad_matrix = self._pack_grad(objectives)  # [T, D], [T, D]

        # --- 4. Identify shared parameters: appear in all tasks' backward ---
        shared_mask = torch.prod(has_grad_matrix, dim=0).bool()  # [D]: 1 if in all tasks

        # Debug: log how many parameters are shared
        # num_shared = shared_mask.sum().item()
        # total_params = shared_mask.numel()
        # if self.writer and self.iter == 0:
        #     self.writer.add_text("gradnorm/shared_params", 
        #                        f"Shared: {num_shared}/{total_params} ({num_shared/total_params:.2%})")
        # --- 5. Compute gradient norm on shared parameters for each task ---
        grad_norms = torch.zeros(self.n_tasks, device=device)
        for i in range(self.n_tasks):
            if active_mask[i]:
                shared_grad = grads_matrix[i][shared_mask]  # Only shared part
                grad_norms[i] = (self.task_weights[i] * shared_grad).norm(2)

        # --- 6. Compute relative training rate ---
        loss_ratios = torch.zeros(self.n_tasks, device=device)
        for i in range(self.n_tasks):
            if active_mask[i]:
                loss_ratios[i] = losses[i].item() / (self.L0[i].item() + self.epsilon)

        r_avg = loss_ratios[active_mask].mean()

        # --- 7. Compute average gradient norm (on shared params) ---
        G_avg = grad_norms[active_mask].mean()

        # --- 8. Compute target gradient norms ---
        target_norms = G_avg * (loss_ratios[active_mask] / (r_avg + self.epsilon)) ** self.alpha
        target_norms = target_norms.detach() # Stop gradient
        current_norms = grad_norms[active_mask]

        # --- 9. GradNorm loss: MSE between current and target norms ---
        gradnorm_loss = (current_norms - target_norms).abs().sum()

        # --- 10. Update task weights ---
        self.weight_optimizer.zero_grad()
        gradnorm_loss.backward()
        # self.weight_optimizer.step()

        # --- 11. Final weighted loss backward (with updated weights) ---
        self._optim.zero_grad()
        total_loss = 0.0
        for i in range(self.n_tasks):
            if losses[i].item() > self.epsilon:
                total_loss += self.task_weights[i].detach() * objectives[i]
        total_loss.backward()  # This will be used by optimizer.step()

    def get_task_weights(self):
        """Return current task weights as numpy array"""
        return self.task_weights.detach().cpu().numpy().copy()
    
    def state_dict(self):
        """Save state for checkpointing"""
        return {
            'optimizer': self._optim.state_dict(),
            'weight_optimizer': self.weight_optimizer.state_dict(),
            'task_weights': self.task_weights.data.clone(),
            'L0': self.L0.clone() if self.L0 is not None else None,
            'iter': self.iter
        }

    def load_state_dict(self, state_dict):
        """Load from checkpoint"""
        self._optim.load_state_dict(state_dict['optimizer'])
        self.weight_optimizer.load_state_dict(state_dict['weight_optimizer'])
        self.task_weights.data.copy_(state_dict['task_weights'])
        if state_dict['L0'] is not None:
            self.L0 = state_dict['L0'].clone()
        self.iter = state_dict.get('iter', 0)