"""
adapted from https://github.com/xunzheng/notears
"""
try:
  from .locally_connected import LocallyConnected
  from .lbfgsb_scipy import LBFGSBScipy
  from .trace_expm import trace_expm
except ImportError:
  from locally_connected import LocallyConnected
  from lbfgsb_scipy import LBFGSBScipy
  from trace_expm import trace_expm

import torch
import torch.nn as nn
import numpy as np
import math


class NotearsMLP(nn.Module):
    def __init__(self, dims, out_dims, bias=True):
        torch.set_default_dtype(torch.float64)
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.dims, self.pd = dims, dims[0]
        self.d = out_dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(self.pd, self.d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(self.pd, self.d * dims[1], bias=bias)
        # nn.init.zeros_(self.fc1_pos.weight)
        # nn.init.zeros_(self.fc1_pos.bias)
        # nn.init.zeros_(self.fc1_neg.weight)
        # nn.init.zeros_(self.fc1_neg.bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.d
        pd = self.pd
        bounds = []
        for j in range(pd):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.d, self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        A_i = torch.zeros([self.dims[0], self.d])
        for i in range(self.d):
            # print(fc1_weight[:self.dims[1]])
            A_i[:,  i] = torch.sum(fc1_weight[i*self.dims[1]:self.dims[1] * (i+1)]**2, axis=0)
        # print(fc1_weight.shape)
        # fc1_weight = fc1_weight.t().view(self.dims[0], self.dims[1], self.d)
        # A = torch.sum(fc1_weight ** 2, dim=1)
        # print(A.shape)
        # A = torch.sum(fc1_weight ** 2, dim=1)[:self.d]  # [i, j]
        A_i= A_i[:self.d]

        # print(A)
        # print(A_i)
        A = A_i
        h = trace_expm(A) - self.d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight        
        A_i = torch.zeros([self.dims[0], self.d])
        for i in range(self.d):
            A_i[:,  i] = torch.sum(fc1_weight[i*self.dims[1]:self.dims[1] * (i+1)]**2, axis=0)
        # print(fc1_weight.shape)
        # fc1_weight = fc1_weight.t().view(self.dims[0], self.dims[1], self.d)
        # A = torch.sum(fc1_weight ** 2, dim=1)
        # print(A.shape)
        # print(A)
        # print(A_i)
        A = A_i
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()  # [i, j]
        return W



def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def dual_ascent_step(model, X, Y, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = torch.optim.LBFGS(model.parameters())
    X_torch = torch.from_numpy(X)
    Y_torch = torch.from_numpy(Y)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(Y_torch)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            with torch.no_grad():
                print(loss, rho)
            return primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      Y: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, Y, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def main(mlp=True):
    from timeit import default_timer as timer
    torch.manual_seed(1)

    from generate_data import SyntheticDataset
    
    n, d, p = 1000, 5, 3
    dag_obj = SyntheticDataset(n, d, p, B_scale=1.0, graph_type='ER', degree=2, A_scale=1.0, noise_type='EV', mlp=mlp)

    A_true = dag_obj.A
    X = dag_obj.X
    Y = dag_obj.Y
    model = NotearsMLP(dims=[(p+1) * d, 30, 1], out_dims=d, bias=True)
    W_est = notears_nonlinear(model, X, Y, lambda1=0.01, lambda2=0.01)
    return W_est, A_true
    

    
   


if __name__ == '__main__':
    main()
