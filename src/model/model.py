import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
import scipy as sc
import torch

import torch.nn as nn

class Model(nn):
    def __init__(self, w, v, vA, lr=1e-3, early_stop = 1e-5 
                verbose=False):
        super(Model, self).__init__()
        self.constraint = constraint
        self.lr=lr
        self.vprint = print if verbose else lambda *a, **k: None

        # Placeholders and variables
        self.lr = lr
        self.X = torch.zeros([self.n, self.d], dtype=torch.float32)
        self.W = nn.Parameter(torch.zeros([self.d, self.d], dtype=torch.float32))
        self.A = nn.Parameter(torch.zeros([self.d, self.d], dtype=torch.float32))

        self.score = loss(self.X)

        self.train_op = torch.optim.Adam(self.parameters(), lr=self.lr)


    def iter(C, data):
        pass


    def train(data, max_iter=100, checkpoint_iter=1000):
        def CtoWA(C):
            w = C[:varables*varables].reshape(v, v)
            a = C[varables*varables].reshape(w, v, v)
            return WAgraph(w, a).constrain(self.constraint).fill_diagonal_(0)

        cons = {'type': 'eq', 'fun': loss_cons}

        self.info(data)

        C = torch.tensor(np.rand(w*v*vA + v*v), requires_grad = True)

        last_loss = 1e16

        for iter in range(max_iter):
            C.zero_grad() 
            WA = CtoWA(C)
            loss = self.loss(data, WA)
            gradW, gradA = self.loss_grad(data, WA) 
            if grad is None:
               #TODO incorperate loss_cons
               loss.backward()
               gradW = W.grad
               gradA = A.grad
            else:
               W.grad = gradW
               A.grad = gradA

            self.train_op().step()

            b = model.B.detach().numpy()
            np.fill_diagonal(b,0)

            if iter%checkpoint_iter == 0:
               self.checkpoint(data, loss, gradW, gradA)

            if loss-last_loss < early_stop:
               self.vprint("EARLY STOP iter: {}".format(iter))
               self.checkpoint(data, loss, gradW, gradA)
               break 
        

        self.postProcess()

        return WAgraph(W, A)

    def loss(data):
        throw ValueError("not implemented")

    def loss_cons(data):
        return None

    def loss_grad(data):
        return None

    def info(data):
        self.vprint("not implemented")

    def checkpoint(data, loss, gradW, gradA):
        self.vprint("not implemented")

    def postProcess():
        self.vprint("not implemented")
