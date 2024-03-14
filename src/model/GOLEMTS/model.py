"""
addapted from https://github.com/ignavierng/golem dec 2023
"""


import torch.nn as nn
import torch


class GolemTS(nn.Module):
    """Set up the objective function of GOLEM.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """

    def __init__(self, n, d, p, Y, lambda_1, lambda_2, lambda_3, device,
                 seed=1, A_init=None, ev=False, lr=1e-3):
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
        self.ev = ev
        self.d = d
        self.p = p # autoregressive order
        self.seed = seed
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.A_init = A_init

        self.d_prime = (p + 1) * d


        # Placeholders and variables
        self.lr = lr
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.X = self.Y[:, :self.d]

        self.A = torch.zeros([self.d * (self.p + 1), self.d], dtype=torch.float32)
        if self.A_init is not None:
            self.A = torch.tensor(self.A_init, dtype=torch.float32)
        else:
            self.A = torch.zeros([self.d * (self.p + 1), self.d], dtype=torch.float32)
        with torch.no_grad():
            self.A = self._preprocess(self.A)
        
        
        self.device = device

        # new VAR
        self.E = torch.zeros([self.d_prime, self.p * self.d])
        self.B = nn.Parameter(torch.hstack([self.A, self.E]).to(self.device))
        
        
        self.I = torch.eye(self.d_prime).to(self.device)

        
        self.Y = self.Y.to(self.device)
        self.X = self.X.to(self.device)
        # self.B = self.B.to(self.device)
        

        # Likelihood, penalty terms and score
        self.likelihood, self.ev_res = self._compute_likelihood()
        self.L1_penalty_E = self._compute_L1_penalty_E()
        self.L1_penalty_A = self._compute_L1_penalty_A()
        self.h = self._compute_h()
        self.score = self.likelihood + self.lambda_1 * self.L1_penalty_A + self.lambda_2 * self.h + self.lambda_3 * self.L1_penalty_E
        # Optimizer
        self.train_op = torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def get_W(self):
        return self.B[:self.d, :self.d]

    def set_learning_rate(self, lr):
        self.lr = lr
        self.train_op = torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def run(self, Y):
        self.Y = torch.tensor(Y.clone().detach(), dtype=torch.float32)
        self.likelihood, self.ev_res = self._compute_likelihood()
        self.L1_penalty_E = self._compute_L1_penalty_E()
        self.L1_penalty_A = self._compute_L1_penalty_A()
        self.h = self._compute_h()
        self.score = self.likelihood + self.lambda_1 * self.L1_penalty_A + self.lambda_2 * self.h + self.lambda_3 * self.L1_penalty_E
        return self.score, self.likelihood, self.ev_res


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
        ev_res = torch.square(
                    torch.norm(self.X - self.cut(self.Y @ self.B), p=2)
                )
        if self.ev:
            return 0.5 * self.d * torch.log(
                torch.square(
                    torch.norm(self.X - self.cut(self.Y @ self.B), p=2)
                )
              # ) - torch.linalg.slogdet(torch.eye(self.d) - self.get_W())[1], ev_res
               ) - torch.linalg.slogdet(self.I - self.B)[1], ev_res
        else:
            return 0.5 * torch.sum(
                torch.log(
                    torch.sum(
                        torch.square(self.X - self.cut(self.Y @ self.B)), axis=0
                    )
                )
            ) - torch.linalg.slogdet(self.I - self.B)[1], ev_res


    def cut(self, M):
        return M[:, :self.d]

    def get_A(self):
        return self.B[:, :self.d]
    

    def get_E(self):
        return self.B[:, self.d:]

    def _compute_L1_penalty_A(self):
        """Compute L1 penalty.

        Returns:
            tf.Tensor: L1 penalty term (scalar-valued).
        """
        return torch.norm(self.get_A(), p=1)
    
    def _compute_L1_penalty_E(self):
        """Compute L1 penalty.

        Returns:
            tf.Tensor: L1 penalty term (scalar-valued).
        """
        return torch.norm(self.get_E(), p=1)

    def _compute_h(self):
        """Compute DAG penalty.

        Returns:
            tf.Tensor: DAG penalty term (scalar-valued).
        """
        return torch.trace(torch.linalg.matrix_exp(self.get_W() * self.get_W())) - self.d





