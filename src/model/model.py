import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
import scipy as sc
import torch

import torch.nn as nn

from ..utill.WAgraph import WAgraph


class Model(nn.Module):
    def __init__(self, w, v, vA, mu, constraint, W_init = None, A_init = None, seed = 1, verbose=False):
        super(Model, self).__init__()
        self.w = w
        self.v = v
        self.vA = vA
        self.mu = mu
        self.cons = constraint

        #TODO i dont think this does anything??
        self.seed = seed

        #TODO move to logging
        self.vprint = print if verbose else lambda *a, **k: None
 

        # Placeholders and variables
        # They are assigned to self and to WA so self.paraeters works
        #TODO make less clunky 
        if W_init is None:
           self.W = nn.Parameter(torch.zeros([v, v], dtype=torch.float32))
        else:
           assert(v == W_init.size()[0])
           assert(v == W_init.size()[1])
           self.W = nn.Parameter(torch.tensor(W_init), dtype=torch.float32)

        if A_init is None:
           self.A = nn.Parameter(torch.zeros([w, v, vA], dtype=torch.float32))
        else:
           assert((w, v, vA).equals(A_init.shape))
           self.A = nn.Parameter(torch.tensor(A_init), dtype=torch.float32)

        self.WA = WAgraph(self.W, self.A)

        self._preprocess()

        self.train_op = torch.optim.Adam(self.parameters())

    def _preprocess(self):
        with torch.no_grad():
           self.WA.zero_diag()
           self.WA.constrain(self.cons)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.train_op = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def run(self, X):
        X = torch.tensor(X, dtype=torch.float32) 
        score, info = self.loss(X)
        return score, info

    def update(self, X):
        X = torch.tensor(X, dtype=torch.float32) 
        grad = self.loss_grad(X)
        if grad is not None:
           #TODO(increase mu as cons dynamicaly .25 of score)
           pass
        else:
           self.train_op.zero_grad()

           score, info = self.run(X)

           score.backward()
           self.train_op.step()

        return score, info

    def loss(self, data):
        #TODO make correct error
        raise ValueError("not implemented")

    def loss_cons(self, data):
        return 0

    def loss_grad(self, data):
        return None

    def loss_cons_grad(self, data):
        return None

    def checkpoint(self, data, loss, gradW, gradA):
        pass

    def postProcess(self):
        raise ValueError("not implemented")
