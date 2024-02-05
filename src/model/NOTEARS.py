


class NOTEARS(Model):

      def __init__(self, v, t, w, l1, l2, lr, verbose=False):
          self.l1 = l1
          self.l2 = l2
          super(self).__init__(self, 'l2', v, t, w, lr,
                Constraint([]), verbose)
      
      def loss(data, W, A, ls, loss_type):
          l1, l2 = ls
          X = data.getX()
          loss = np.square(
                np.linalg.norm(
                    np.matmul(np.eye(varables, varables) - W, X)
                    , "fro"
                )
            )/np.prod(X.shape)

      def loss_cons(...):
          return ...

      def checkpoint(data, loss, gradW, gradA):
          self.vprint() 

      
