import copy
import pdb
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
from scipy.optimize import minimize


class BML:
    def __init__(self, optimizer, n_tasks, reduction='mean', writer=None, epsilon=1e-8):
        self._optim, self._reduction = optimizer, reduction
        self.epsilon = epsilon
        self.iter = 0
        self.n_tasks = n_tasks
        self.writer = writer
        first_param = optimizer.param_groups[0]['params'][0]  # 第一个参数
        device = first_param.device
        self.miu = torch.zeros(n_tasks, device=device, requires_grad=False)
        self.miu_count = torch.zeros(n_tasks, device=device, requires_grad=False)
        self.accelerate_mask = torch.tensor([True, True, True, True], dtype=torch.bool, device=device, requires_grad=False)
        self.rebalanced = True

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def bml_backward(self, objectives, ddp_model=None):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads, param = self._pack_grad(objectives, ddp_model)

        self.effective_update(grads, has_grads, param, objectives)
        if self.iter % 10 == 0:
            self.accelerate_mask = self.compute_dspeed()
            self.rebalanced = False
            need_balance_indices = torch.where(self.accelerate_mask)[0]
            logging.info('Modality {} need to be Rebalanced!'.format(need_balance_indices[0]))

        device = objectives[0].device


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
        
        if not self.rebalanced:
            rebalance_indices = torch.where(active_mask * self.accelerate_mask)[0]
            if len(rebalance_indices) != 0:
                active_mask = active_mask * self.accelerate_mask
                self.rebalanced = True
                logging.info('Modality {} is Rebalanced once!'.format(rebalance_indices[0]))


        self._optim.zero_grad()
        total_loss = 0.0
        for i in range(self.n_tasks):
            if active_mask[i]:
                total_loss += objectives[i]
        total_loss.backward()  # This will be used by optimizer.step()
        return


    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
        return
    
    def effective_update(self, grads, has_grads, param, objectives):
        shared = torch.stack(has_grads).prod(dim=0).bool()
        for i, (grad, has_grad, obj) in enumerate(zip(grads, has_grads, objectives)):
            if obj.item() > self.epsilon:
                self.miu[i] += grad[has_grad.bool() & ~shared].norm(2) / (param[has_grad.bool() & ~shared].norm(2) + self.epsilon)
                self.miu_count[i] += 1
        self.iter += 1
        return

    def compute_dspeed(self):
        miu_mean = self.miu / (self.miu_count + self.epsilon)
        values = torch.where(miu_mean > self.epsilon, miu_mean, torch.tensor(float('inf'), device=miu_mean.device))
        active_mask = (values == values.min()) if values.min() != float('inf') else torch.zeros_like(miu_mean, dtype=torch.bool, device=miu_mean.device)

        for i in range(self.n_tasks):
            self.miu[i] = 0.0
            self.miu_count[i] = 0

        return active_mask


    def _pack_grad(self, objectives, ddp):
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
                    param.append(p.data.clone())
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
                param.append(p.data.clone())
        return grad, shape, has_grad, param
