"""
addapted from https://github.com/ignavierng/golem dec 2023
"""


import logging

from .model import Model
from ..utill.constraint import Constraint

import torch.nn as nn
import torch


class GOLEM(Model):
    """Set up the objective function of GOLEM.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, w, v, ET="EV", l1=1e-3, l2=1e-3, l3=1e-3, constraint = None,
                         W_init = None, seed = 1, verbose=False):

            n, d, p, Y, lambda_1, lambda_2,
                 seed=1, A_init=None, ev=False):
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
        p = w d=v n=not used    lambda_1 (float): Coefficient of L1 penalty.
            lambda_2 (float): Coefficient of DAG penalty.
            equal_variances (bool): Whether to assume equal noise variances
                for likelibood objective. Default: True.
            seed (int): Random seed. Default: 1.
            B_init (numpy.ndarray or None): [d, d] weighted matrix for
                initialization. Set to None to disable. Default: None.
        """

        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        if constraint is None:
           constraint = Constraint([], [], w, v, v)
        super().__init__(w, v, v, mu, constraint, W_init, None, seed, verbose)

    def loss(self, X):
        W = self.WA.W
        A = self.WA.A

        A = torch.vstack([W] + [A[i] for i in range(len(A))])       
        X = torch.vstack([X[i] for i in range(self.w + 1)])
 
        I = torch.eye((self.w + 1) * self.v)
        ep = 1e-5
        Binv = U - A
        B = torch.pinverse(Binv)

        omega = torch.diag(torch.diag((X @ Binv).T @ (X @ Binv))) / self.n
        if self.ET == "EV":
            likelihood = (0.5 * torch.log(torch.trace(omega) / self.d)) 
        elif self.ET == "NV":
            likelihood = 0.5 * torch.logdet(B.T @ omega @ B + ep * I)

        elif self.ET == "OG":
             #actualy just compute a giant squar matrix with the A an identity to match the result
             likelihood = 0.5 * self.v * torch.log(
                torch.square(
                    torch.norm(X - X @ W@ , p=2)
                )
            ) - torch.linalg.slogdet(torch.eye(self.d) - self.B)[1]

        else:
            raise ValueError("ET should be EV or NV") 

        sparcityW = torch.norm(W, 1)
        sparcityA = torch.norm(A, 1)
        acyclicity = torch.trace(torch.linalg.matrix_exp(self.get_W() * self.get_W())) - self.d

        info = {
             "likelyhood" : likelyhood,
             "sparcityW" : sparcityW,
             "sparcityA" : sparcityA,
             "acyclicity" : acyclicity,
        }
        return likelyhood + self.l1*sparcityW + self.l2*sparcityA + self.l3*acyclicity, info




