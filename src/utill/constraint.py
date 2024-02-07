import numpy as np


class Constraint:
      def __init__(self, W, A, w=None, v=None, vA=None):

          self.w = w
          self.v = v
          if  vA is None:
              self.vA = v
          else: 
              self.vA = vA
 
          if isinstance(W, type(np.ndarray)) and 2 == len(W.shape):
             assert(W.shape[0] == W.shape[1])
             assert(v is None or v == W.shape[0])
             self.v = W.shape[0]
             W = arrayToCons(W)
          
          elif isinstance(W, list) and (len(W) == 0 or 3 == len(W[0])):
             pass
          else:
             raise ValueError("W constraint must be 2d array or list of 3 tuples")

          if isinstance(A, np.ndarray) and 3 == len(A.shape):
             A = arrayToCons(A)
             assert(w is None or w == A.shape[0])
             assert(v is None or v == A.shape[1])
             assert(vA is None or vA == A.shape[2] or (A.shape[1] == A.shape[2]))
             self.w = A.shape[0]
             self.v = A.shape[1]
             self.vA = A.shape[2]

          if isinstance(A, np.ndarray) and 2 == len(A.shape):
             A = arrayToCons(np.array([A]))

             assert(w is None or w == 1)
             assert(v is None or v == A.shape[1])
             assert(vA is None or vA == A.shape[2] or (A.shape[1] == A.shape[2]))
             self.w = 1
             self.v = A.shape[1]
             self.vA = A.shape[2]


          elif isinstance(A, list) and (len(A) == 0 or  4 == len(A[0])):
             pass
          else:
             raise ValueError("A constraint must be 2d or 3d array or list of 3 tuples")


          self.W_cons = W
          self.A_cons = A

          assert(self.w is not None)
          assert(self.v is not None)
          if self.vA is None:
             self.vA == self.v

      def constrainWA(self, WA):
          return WA.constrain(self)

      def constrainedWA(self, WA):
          return WA == constrainWA(WA)

      def constrainW(self, W):
          for x, y, val in self.W_cons:
              W[x][y] = val

      def constrainA(self, A):
          for x, y, z, val in self.A_cons:
              A[x][y][z] = val

      def arrayToCons(self, array):
          ret = []
          for i in range(len(array)):
              for j in range(len(array[0])):
                  val = array[i][j]
                  if isinstance(val, list):
                     for k in range(len(array[0][0])):
                         if val is not np.nan:
                            val = array[i][j][k]
                            ret.append(i, j, k, val)
                  if val is not np.nan:
                     ret.append(i, j, val)
          return ret 

      def consWToArray(self, cons):
          ret = np.empty(v,v)
          ret[:] = np.nan
          return constrainW(ret)
         
      def consAToArray(self, cons):
          ret = np.empty(w, v, vA)
          ret[:] = np.nan
          return constrainA(ret)

      def validate(self, w, v, vA=None):
          assert(self.w == w) 
          assert(self.v == v) 
          assert(self.vA == vA or (vA is None and self.v == self.vA))
          return True 
