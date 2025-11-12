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


class MGDA:
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

    def mgda_backward(self, objectives, ddp_model=None):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives, ddp_model)
        lambda_weights = self._solve_qp(grads)
        logging.info(lambda_weights)
        mgda_grad = self._combine_grads(grads, lambda_weights, has_grads)
        mgda_grad = self._unflatten_grad(mgda_grad, shapes[0])
        self._set_grad(mgda_grad)
        return

    def _solve_qp(self, grads):
        """
        求解：
        min_λ ||∑ λ_i g_i||^2
        s.t. ∑λ_i = 1, λ_i ≥ 0
        """
        device = grads[0].device
        N = len(grads)

        if N == 1:
            return torch.tensor([1.0], device=device)

        # 构建 Gram 矩阵 G[i,j] = g_i^T g_j
        G = torch.stack([torch.stack([torch.dot(gi, gj) for gj in grads]) for gi in grads])  # [N, N]
        G_np = G.detach().cpu().numpy()

        # 目标函数
        def objective(λ):
            return 0.5 * λ @ G_np @ λ

        def grad_objective(λ):
            return G_np @ λ

        # 约束：sum λ_i = 1
        constraints = {
            'type': 'eq',
            'fun': lambda λ: np.sum(λ) - 1.0,
            'jac': lambda λ: np.ones_like(λ)
        }

        # bounds: λ_i >= 0
        bounds = [(0, None) for _ in range(N)]

        # 初始值
        λ0 = np.ones(N) / N

        # 求解
        result = minimize(
            objective,
            λ0,
            method='SLSQP',
            jac=grad_objective,
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'maxiter': 100}
        )

        if not result.success:
            warnings.warn(f"MGDA QP solve failed: {result.message}. Using uniform weights.")
            λ_star = np.ones(N) / N
        else:
            λ_star = result.x

        return torch.tensor(λ_star, device=device, dtype=torch.float32)
    
    def _combine_grads(self, grads, lambda_weights, has_grads):
        """
        计算加权梯度，并处理 shared / non-shared 参数
        """
        # has_grads: [N, D] -> shared 参数 mask
        shared = torch.stack(has_grads).prod(dim=0).bool()  # [D,] 仅在所有任务都有梯度时为 True
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)

        # 加权组合
        weighted_grads = [w * g for w, g in zip(lambda_weights, grads)]
        total_grad = sum(weighted_grads)  # [D,]

        # 处理共享与非共享部分
        if self._reduction == 'mean':
            merged_grad[shared] = total_grad[shared]
            # 非共享部分：直接使用各自梯度（但已被加权）
            # 注意：MGDA 通常只对 shared 参数使用，这里简化处理
            # merged_grad[~shared] = total_grad[~shared]
            merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in grads]).sum(dim=0)

        else:
            raise ValueError(f"Invalid reduction: {self._reduction}")

        return merged_grad


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