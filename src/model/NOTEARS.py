from .model import Model
from ..utill.constraint import Constraint
import torch
import numpy as np


class NOTEARS(Model):

      def __init__(self, w, v, l1=1e-3, l2=1e-3, mu=10, constraint = None,
                         W_init = None, seed = 1, verbose=False):
          self.l1 = l1
          self.l2 = l2
          if constraint is None:
             constraint = Constraint([], [], w, v, v)
          super().__init__(w, v, v, mu, constraint, W_init, None, seed, verbose)
      
      def loss(self, X):
          #TODO different error types this is l2
          W = self.WA.W
          A = self.WA.A

          X_NOW = X[0]
          X_W = torch.matmul(W, X_NOW)
          X_A = torch.stack([torch.tensordot(A, X[i:i+self.w], dims=((0,1),(0,1)))
                    for i in range(1, X.size()[0]-self.w+1)])

          likelyhood = torch.square(torch.norm(X_NOW - X_W - X_A, "fro"))/X.size()[2]
          sparcityW = torch.norm(W, 1) 
          sparcityA = torch.norm(A, 1)

          acyclicity = self.loss_cons(X)    

          info = {
             "likelyhood" : likelyhood,
             "sparcityW" : sparcityW,
             "sparcityA" : sparcityA,
             "acyclicity" : acyclicity,
          }

          return likelyhood + self.l1*sparcityW + self.l2*sparcityA + self.mu*acyclicity, info 

      def loss_cons(self, X):
          return torch.trace(torch.matrix_exp(torch.mul(self.WA.W,self.WA.W))) - self.WA.W.size()[0]

      def loss_grad(self, X):
          return None

      def loss_cons_grad(self, X):
          return None

      def checkpoint(self):
          pass 

      
