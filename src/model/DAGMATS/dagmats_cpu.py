"""
adapeted from https://github.com/kevinsbello/dagma
"""

try:
   from .locally_connected_dagma import LocallyConnected
except ImportError:
   from locally_connected_dagma import LocallyConnected

import torch
import torch.nn as nn
import numpy as np
from  torch import optim
import copy
from tqdm.auto import tqdm
import typing


__all__ = ["DagmaTS", "DagmaLinear"]


class DagmaTS(nn.Module): 
    """
    Class that models the structural equations for the causal graph using MLPs.
    """
    
    def __init__(self, n, d, p, dtype: torch.dtype = torch.double):
        r"""
        Parameters
        ----------
        dims : typing.List[int]
            Number of neurons in hidden layers of each MLP representing each structural equation.
        bias : bool, optional
            Flag whether to consider bias or not, by default ``True``
        dtype : torch.dtype, optional
            Float precision, by default ``torch.double``
        """
        torch.set_default_dtype(dtype)
        super(DagmaTS, self).__init__()
        self.n = n
        self.d = d
        self.p = p # autoregressive order
        self.d_prime = (self.p  + 1) * self.d
        self.I = torch.eye(self.d)
        
        self.A = nn.Parameter(torch.zeros([self.d_prime, self.d]))

        
    def forward(self, y: torch.Tensor) -> torch.Tensor:  # [n, d] -> [n, d]
        r"""
        Applies the current states of the structural equations to the dataset X

        Parameters
        ----------
        x : torch.Tensor
            Input dataset with shape :math:`(n,d)`.

        Returns
        -------
        torch.Tensor
            Result of applying the structural equations to the input data.
            Shape :math:`(n,d)`.
        """
        
        return y @ self.A

    def h_func(self, s: float = 1.0) -> torch.Tensor:
        r"""
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG

        Parameters
        ----------
        s : float, optional
            Controls the domain of M-matrices, by default 1.0

        Returns
        -------
        torch.Tensor
            A scalar value of the log-det acyclicity function :math:`h(\Theta)`.
        """
        W = self.A[:self.d]

        h = -torch.slogdet(s * self.I - W * W)[1] + self.d * np.log(s)
        return h

    def fc1_l1_reg(self) -> torch.Tensor:
        r"""
        Takes L1 norm of the weights in the first fully-connected layer

        Returns
        -------
        torch.Tensor
            A scalar value of the L1 norm of first FC layer. 
        """
        return torch.sum(torch.abs(self.A))




class DagmaLinear:
    """
    Class that implements the DAGMA algorithm
    """
    
    def __init__(self, model: nn.Module, verbose: bool = False, dtype: torch.dtype = torch.double):
        """
        Parameters
        ----------
        model : nn.Module
            Neural net that models the structural equations.
        verbose : bool, optional
            If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
            as defined in :py:meth:`~dagma.nonlinear.DagmaNonlinear.fit`. Defaults to ``False``.
        dtype : torch.dtype, optional
            float number precision, by default ``torch.double``.
        """
        self.vprint = print if verbose else lambda *a, **k: None
        self.model = model
        self.dtype = dtype
    
    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the logarithm of the MSE loss:
            .. math::
                \frac{d}{2} \log\left( \frac{1}{n} \sum_{i=1}^n (\mathrm{output}_i - \mathrm{target}_i)^2 \right)
        
        Parameters
        ----------
        output : torch.Tensor
            :math:`(n,d)` output of the model
        target : torch.Tensor
            :math:`(n,d)` input dataset

        Returns
        -------
        torch.Tensor
            A scalar value of the loss.
        """
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss

    def minimize(self, 
                 max_iter: float, 
                 lr: float, 
                 lambda1: float, 
                 lambda2: float, 
                 mu: float, 
                 s: float,
                 lr_decay: float = False, 
                 tol: float = 1e-6, 
                 pbar: typing.Optional[tqdm] = None,
        ) -> bool:
        r"""
        Solves the optimization problem: 
            .. math::
                \arg\min_{W(\Theta) \in \mathbb{W}^s} \mu \cdot Q(\Theta; \mathbf{X}) + h(W(\Theta)),
        where :math:`Q` is the score function, and :math:`W(\Theta)` is the induced weighted adjacency matrix
        from the model parameters. 
        This problem is solved via (sub)gradient descent using adam acceleration.

        Parameters
        ----------
        max_iter : float
            Maximum number of (sub)gradient iterations.
        lr : float
            Learning rate.
        lambda1 : float
            L1 penalty coefficient. Only applies to the parameters that induce the weighted adjacency matrix.
        lambda2 : float
            L2 penalty coefficient. Applies to all the model parameters.
        mu : float
            Weights the score function.
        s : float
            Controls the domain of M-matrices.
        lr_decay : float, optional
            If ``True``, an exponential decay scheduling is used. By default ``False``.
        tol : float, optional
            Tolerance to admit convergence. Defaults to 1e-6.
        pbar : tqdm, optional
            Controls bar progress. Defaults to ``tqdm()``.

        Returns
        -------
        bool
            ``True`` if the optimization succeded. This can be ``False`` when at any iteration, the model's adjacency matrix 
            got outside of the domain of M-matrices.
        """
        self.vprint(f'\nMinimize s={s} -- lr={lr}')
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)
        if lr_decay is True:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16
        for i in range(max_iter):
            optimizer.zero_grad()
            h_val = self.model.h_func(s)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            X_hat = self.model(self.Y)
            score = self.log_mse_loss(X_hat, self.X)
            l1_reg = lambda1 * self.model.fc1_l1_reg()
            obj = mu * (score + l1_reg) + h_val
            obj.backward()
            optimizer.step()
            if lr_decay and (i+1) % 1000 == 0: #every 1000 iters reduce lr
                scheduler.step()
            if i % self.checkpoint == 0 or i == max_iter-1:
                obj_new = obj.item()
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter-i)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return True

    def fit(self, 
            X: typing.Union[torch.Tensor, np.ndarray],
            Y,
            lambda1: float = .02, 
            lambda2: float = .005,
            T: int = 4, 
            mu_init: float = .1, 
            mu_factor: float = .1, 
            s: float = 1.0,
            warm_iter: int = 5e4, 
            max_iter: int = 8e4, 
            lr: float = .0002, 
            w_threshold: float = 0.3, 
            checkpoint: int = 1000,
        ) -> np.ndarray:
        r"""
        Runs the DAGMA algorithm and fits the model to the dataset.

        Parameters
        ----------
        X : typing.Union[torch.Tensor, np.ndarray]
            :math:`(n,d)` dataset.
        lambda1 : float, optional
            Coefficient of the L1 penalty, by default .02.
        lambda2 : float, optional
            Coefficient of the L2 penalty, by default .005.
        T : int, optional
            Number of DAGMA iterations, by default 4.
        mu_init : float, optional
            Initial value of :math:`\mu`, by default 0.1.
        mu_factor : float, optional
            Decay factor for :math:`\mu`, by default .1.
        s : float, optional
            Controls the domain of M-matrices, by default 1.0.
        warm_iter : int, optional
            Number of iterations for :py:meth:`~dagma.nonlinear.DagmaNonlinear.minimize` for :math:`t < T`, by default 5e4.
        max_iter : int, optional
            Number of iterations for :py:meth:`~dagma.nonlinear.DagmaNonlinear.minimize` for :math:`t = T`, by default 8e4.
        lr : float, optional
            Learning rate, by default .0002.
        w_threshold : float, optional
            Removes edges with weight value less than the given threshold, by default 0.3.
        checkpoint : int, optional
            If ``verbose`` is ``True``, then prints to stdout every ``checkpoint`` iterations, by default 1000.

        Returns
        -------
        np.ndarray
            Estimated DAG from data.
        
        
        .. important::

            If the output of :py:meth:`~dagma.nonlinear.DagmaNonlinear.fit` is not a DAG, then the user should try larger values of ``T`` (e.g., 6, 7, or 8) 
            before raising an issue in github.
        """
        torch.set_default_dtype(self.dtype)
        if type(X) == torch.Tensor:
            self.X = X.type(self.dtype)
        elif type(X) == np.ndarray:
            self.X = torch.from_numpy(X).type(self.dtype)
        else:
            ValueError("X should be numpy array or torch Tensor.")

        if type(Y) == torch.Tensor:
            self.Y = Y.type(self.dtype)
        elif type(Y) == np.ndarray:
            self.Y = torch.from_numpy(Y).type(self.dtype)
        
        self.checkpoint = checkpoint
        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.") 
        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
            for i in range(int(T)):
                self.vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
                success, s_cur = False, s[i]
                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)
                lr_decay = False
                while success is False:
                    success = self.minimize(inner_iter, lr, lambda1, lambda2, mu, s_cur, 
                                        lr_decay, pbar=pbar)
                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy())
                        lr *= 0.5 
                        lr_decay = True
                        if lr < 1e-10:
                            break # lr is too small
                        s_cur = 1
                mu *= mu_factor
        W_est = self.model.A.detach().numpy()
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est


def test():
    from timeit import default_timer as timer
    torch.manual_seed(1)

    from generate_data import SyntheticDataset
    
    n, d, p = 1000, 5, 3
    dag_obj = SyntheticDataset(n, d, p, B_scale=1.0, graph_type='ER', degree=2, A_scale=1.0, noise_type='EV', mlp=False)

    A_true = dag_obj.A
    X = dag_obj.X
    Y = dag_obj.Y

    eq_model = DagmaTS(n=n, p=p, d=d)
    model = DagmaLinear(eq_model, verbose=True)
    W_est = model.fit(X, Y, lambda1=0.01, lambda2=0.03, lr=0.02, w_threshold=0)
    return W_est, A_true
    
    
if __name__ == '__main__':
    test()
