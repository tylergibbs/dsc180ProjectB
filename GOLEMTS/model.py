"""
addapted from https://github.com/ignavierng/golem dec 2023
"""


import logging

import torch.nn as nn
import torch


class GolemTS(nn.Module):
    """Set up the objective function of GOLEM.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, n, d, p, Y, lambda_1, lambda_2,
                 seed=1, A_init=None):
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
            lambda_1 (float): Coefficient of L1 penalty.
            lambda_2 (float): Coefficient of DAG penalty.
            equal_variances (bool): Whether to assume equal noise variances
                for likelibood objective. Default: True.
            seed (int): Random seed. Default: 1.
            B_init (numpy.ndarray or None): [d, d] weighted matrix for
                initialization. Set to None to disable. Default: None.
        """

        super(GolemTS, self).__init__()

        self.n = n
        self.d = d
        self.p = p # autoregressive order
        self.seed = seed
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.A_init = A_init

        self.U = torch.vstack([torch.eye(d), torch.zeros((p * d, d))])

        # Placeholders and variables
        self.lr = 1e-3
        self.X = torch.zeros([self.n, self.d], dtype=torch.float32)
        self.Y = torch.zeros([self.n, (self.p + 1) * self.d])
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.A = nn.Parameter(torch.zeros([self.d * (self.p + 1), self.d], dtype=torch.float32))
        if self.A_init is not None:
            self.A = nn.Parameter(torch.tensor(self.A_init, dtype=torch.float32))
        else:
            self.A = nn.Parameter(torch.zeros([self.d * (self.p + 1), self.d], dtype=torch.float32))
        with torch.no_grad():
            self.A = self._preprocess(self.A)

        # Likelihood, penalty terms and score
        self.likelihood = self._compute_likelihood()
        self.L1_penalty = self._compute_L1_penalty()
        self.h = self._compute_h()
        self.score = self.likelihood + self.lambda_1 * self.L1_penalty + self.lambda_2 * self.h
        # Optimizer
        self.train_op = torch.optim.Adam(self.parameters(), lr=self.lr)
        self._logger.debug("Finished building PYTORCH graph.")
    
    def get_W(self):
        return self.A[:self.d].T

    def set_learning_rate(self, lr):
        self.lr = lr
        self.train_op = torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def run(self, Y):
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.likelihood = self._compute_likelihood()
        self.L1_penalty = self._compute_L1_penalty()
        self.h = self._compute_h()
        self.score = self.likelihood + self.lambda_1 * self.L1_penalty + self.lambda_2 * self.h
        return self.score, self.likelihood, self.h


    def _preprocess(self, B):
        """Set the diagonals of B to zero.

        Args:
            B (tf.Tensor): [d, d] weighted matrix.

        Returns:
            tf.Tensor: [d, d] weighted matrix.
        """
        return B.fill_diagonal_(0)

    def _compute_likelihood(self):
        """Compute (negative log) likelihood in the linear Gaussian case.

        Returns:
            tf.Tensor: Likelihood term (scalar-valued).
        """
        Binv = self.U - self.A
        B = torch.pinverse(Binv)
        # print(self.A)
        # print(Binv)
        # print(self.Y @ Binv)
        omega = torch.diag(torch.diag((self.Y @ Binv).T @ (self.Y @ Binv))) / self.n
        # print((self.Y @ Binv).T @ (self.Y @ Binv))
        return 0.5 * torch.logdet(B.T @ omega @ B)

    def _compute_L1_penalty(self):
        """Compute L1 penalty.

        Returns:
            tf.Tensor: L1 penalty term (scalar-valued).
        """
        return torch.norm(self.A, p=1)

    def _compute_h(self):
        """Compute DAG penalty.

        Returns:
            tf.Tensor: DAG penalty term (scalar-valued).
        """
        return torch.trace(torch.linalg.matrix_exp(self.get_W() * self.get_W())) - self.d





