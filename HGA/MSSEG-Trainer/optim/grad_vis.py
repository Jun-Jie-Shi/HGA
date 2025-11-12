import copy
import pdb
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Grad_Vis:
    def __init__(self, optimizer, reduction='mean', writer=None):
        self._optim, self._reduction = optimizer, reduction
        self.iter = 0
        self.writer = writer

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

    def gradvis_backward(self, objectives, main_obj, ddp_model=None):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives, ddp_model)
        grad_main, shape_main, has_grad_main = self._pack_grad(main_obj, ddp_model)
        degree = self._project_conflicting(grads, grad_main[0], has_grads)

        return degree
    def _project_conflicting(self, grads, grad_main, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        degree = torch.ones(num_task, dtype=torch.float32, device=grads[0].device)
        g_j = grad_main[shared]
        for i, grad_i in enumerate(pc_grad):
            g_i = grad_i[shared]
            g_i_g_j = torch.dot(g_i, g_j)
            cos_theta = g_i_g_j / (torch.norm(g_i) * torch.norm(g_j) + 1e-8)
            angle_radian = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
            degree[i] = torch.rad2deg(angle_radian)
            # coef = torch.clamp(coef, min=-2.0, max=1) ## Test

        return degree

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
            # if ii == 0: continue
            # out_tensors = list(_find_tensors(obj))
            # ddp.reducer.prepare_for_backward(out_tensors)
            # if ii < len(objectives) - 1:
            #     obj.backward(retain_graph=True)
            # else:
            #     obj.backward(retain_graph=False)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

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
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


# class TestNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self._linear = nn.Linear(3, 4)

#     def forward(self, x):
#         return self._linear(x)


# class MultiHeadTestNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self._linear = nn.Linear(3, 2)
#         self._head1 = nn.Linear(2, 4)
#         self._head2 = nn.Linear(2, 4)

#     def forward(self, x):
#         feat = self._linear(x)
#         return self._head1(feat), self._head2(feat)


# if __name__ == '__main__':

#     # fully shared network test
#     torch.manual_seed(4)
#     x, y = torch.randn(2, 3), torch.randn(2, 4)
#     net = TestNet()
#     y_pred = net(x)
#     pc_adam = GMD(optim.Adam(net.parameters()))
#     pc_adam.zero_grad()
#     loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
#     loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

#     pc_adam.pc_backward([loss1, loss2])
#     for p in net.parameters():
#         print(p.grad)

#     print('-' * 80)
#     # seperated shared network test

#     torch.manual_seed(4)
#     x, y = torch.randn(2, 3), torch.randn(2, 4)
#     net = MultiHeadTestNet()
#     y_pred_1, y_pred_2 = net(x)
#     pc_adam = GMD(optim.Adam(net.parameters()))
#     pc_adam.zero_grad()
#     loss1_fn, loss2_fn = nn.MSELoss(), nn.MSELoss()
#     loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)

#     pc_adam.pc_backward([loss1, loss2])
#     for p in net.parameters():
#         print(p.grad)