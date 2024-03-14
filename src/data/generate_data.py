import networkx as nx

import logging

import igraph as ig
import networkx as nx
import numpy as np
from scipy.special import expit as sigmoid


def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))



class SyntheticDataset:
    """Generate synthetic data.

    Key instance variables:
        X (numpy.ndarray): [n, d] data matrix.
        B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
        B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.

    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, n, d, p, graph_type=None, degree=None, noise_type=None, B_scale=1, A_scale = 1.1, mlp=False, seed=1):
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
            graph_type ('ER' or 'SF'): Type of graph.
            degree (int): Degree of graph.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            B_scale (float): Scaling factor for range of B.
            seed (int): Random seed. Default: 1.
        """
        if isinstance(n, int):
           self.noise = noise_type
           self.n = n
           self.d = d
           self.p = p
           self.graph_type = graph_type
           self.degree = degree
           self.mlp = mlp
           self.noise_type = noise_type
           self.B_ranges = ((B_scale * -2.0, B_scale * -0.5),
                         (B_scale * 0.5, B_scale * 2.0))
           
           self.d_prime = (p+1) * d
           self.A_ranges = []
           for i in range(self.p):
               A_s = (1 / (A_scale**(i)))
               self.A_ranges.append(((A_s * -0.3, A_s * -0.5),
                         (A_s * 0.3, A_s * 0.5)))

           
           self.rs = np.random.RandomState(seed)    # Reproducibility
        
           self._setup()
           self._logger.debug("Finished setting up dataset class.")
        else:
           X = n
           Y = d
           self.n = X.shape[0]
           self.d = X.shape[1]
           self.degree = (Y!=0).sum().sum()/Y.shape[0]
           self.B = Y
           self.X = X

    def _setup(self):
        """Generate B_bin, B and X."""
        A_list = []
        for i in range(self.p):
            Abin = SyntheticDataset.simulate_A_bin(self.d, self.degree, self.p)
            if self.mlp:
                A_list.append(Abin)
            else:
                Aw = SyntheticDataset.simulate_weight(Abin, self.A_ranges[i])
                A_list.append(Aw)
                

        self.B_bin = SyntheticDataset.simulate_random_dag(self.d, self.degree,
                                                         self.graph_type, self.rs)
        if self.mlp:
            self.B = self.B_bin
        else:
            
            self.B = SyntheticDataset.simulate_weight(self.B_bin, self.B_ranges, self.rs)
        self.A = np.vstack([self.B.T, np.vstack(A_list)])
        if self.mlp:
            self.Z, self.Y = SyntheticDataset.simulate_nonlinear_sem(self.A, self.n, self.noise_type)
        else:
            self.Z, self.Y = SyntheticDataset.simulate_ts(self.A, self.n, self.noise_type)
        self.X = self.Y[:, :self.d]

        assert is_dag(self.B)


    @staticmethod
    def simulate_nonlinear_sem(A, n, noise_type):
        d = A.shape[1]
        p  = int(A.shape[0] / A.shape[1]) - 1
        
        Adag = np.hstack([A, np.zeros(((p+ 1) * d, p * d))])
        Y = np.zeros([n, (p+ 1) * d])
        G = nx.DiGraph(Adag)
        Z = SyntheticDataset.gen_Z(n, A.shape[0], noise_type)

        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == (p+1) * d
        for i in ordered_vertices:
            parents = list(G.predecessors(i))
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[len(parents), hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            Y[:, i] = sigmoid(Y[:, parents] @ W1) @ W2 + Z[:, i]
        return Z, Y


    @staticmethod
    def gen_Z(n, d_prime, noise_type):
        if noise_type == 'EV':
            Z = np.random.multivariate_normal(np.zeros(d_prime), np.random.uniform(1.0, 2.0) * np.eye(d_prime), size=(n))
        elif noise_type == 'NV':
            Z = np.random.multivariate_normal(np.zeros(d_prime), np.diag(np.random.uniform(1.0, 2.0, (d_prime))), size=(n))
        
        elif noise_type == 'EXP':
            Z = np.random.exponential(size=(n, d_prime))
        elif noise_type == 'GUMBEL':
            Z = np.random.gumbel(size=(n, d_prime))

        else:
            return -1
        return Z

    @staticmethod
    def simulate_ts(A, n, noise_type):


        # generate noize vector Z (d x d)
        d = A.shape[1]
        p  = int(A.shape[0] / A.shape[1]) - 1
        Z = SyntheticDataset.gen_Z(n, A.shape[0], noise_type)
        # create Adag
        Adag = np.hstack([A, np.zeros(((p+ 1) * d, p * d))])
        Y = np.zeros([n, (p+ 1) * d])
        G = nx.DiGraph(Adag)
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == (p+1) * d
        for i in ordered_vertices:
            parents = list(G.predecessors(i))
            Y[:, i] = Y[:, parents] @ Adag[parents, i] + Z[:, i]
        return Z, Y


        


        # U = np.vstack([np.eye(d), np.zeros((d*p, d))])
        # Y = Z @ np.linalg.pinv(U - A)
        return Z, Y 

    @staticmethod
    def simulate_A_bin(d, degree, p):
        prob = float(degree) / (p*d - 1)
        if prob >= 1.0:
            prob = 0.5
        return np.random.choice([0, 1], (d, d), p=[1-prob, prob])

    @staticmethod
    def simulate_er_dag(d, degree, rs=np.random.RandomState(1)):
        """Simulate ER DAG using NetworkX package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _get_acyclic_graph(B_und):
            return np.tril(B_und, k=-1)

        def _graph_to_adjmat(G):
            return nx.to_numpy_array(G)

        p = float(degree) / (d - 1)
        G_und = nx.generators.erdos_renyi_graph(n=d, p=p, seed=rs)
        B_und_bin = _graph_to_adjmat(G_und)    # Undirected
        B_bin = _get_acyclic_graph(B_und_bin)
        return B_bin

    @staticmethod
    def simulate_sf_dag(d, degree):
        """Simulate ER DAG using igraph package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        m = int(round(degree / 2))
        # igraph does not allow passing RandomState object
        G = ig.Graph.Barabasi(n=d, m=m, directed=True)
        B_bin = np.array(G.get_adjacency().data)
        return B_bin

    @staticmethod
    def simulate_random_dag(d, degree, graph_type, rs=np.random.RandomState(1)):
        """Simulate random DAG.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            graph_type ('ER' or 'SF'): Type of graph.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _random_permutation(B_bin):
            # np.random.permutation permutes first axis only
            P = rs.permutation(np.eye(B_bin.shape[0]))
            return P.T @ B_bin @ P

        if graph_type == 'ER':
            B_bin = SyntheticDataset.simulate_er_dag(d, degree, rs)
        elif graph_type == 'SF':
            B_bin = SyntheticDataset.simulate_sf_dag(d, degree)
        else:
            print(graph_type)
            raise ValueError("Unknown graph type.")
        return _random_permutation(B_bin)

    @staticmethod
    def simulate_weight(B_bin, B_ranges, rs=np.random.RandomState(1)):
        """Simulate the weights of B_bin.

        Args:
            B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
            B_ranges (tuple): Disjoint weight ranges.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] weighted adjacency matrix of DAG.
        """
        B = np.zeros(B_bin.shape)
        S = rs.randint(len(B_ranges), size=B.shape)  # Which range
        for i, (low, high) in enumerate(B_ranges):
            U = rs.uniform(low=low, high=high, size=B.shape)
            B += B_bin * (S == i) * U
        return B
    """
    @staticmethod
    def simulate_linear_sem(B, n, noise_type, rs=np.random.RandomState(1)):
        Simulate samples from linear SEM with specified type of noise.

        Args:
            B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
            n (int): Number of samples.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [n, d] data matrix.
        
        def _simulate_single_equation(X, B_i):
            Simulate samples from linear SEM for the i-th node.

            Args:
                X (numpy.ndarray): [n, number of parents] data matrix.
                B_i (numpy.ndarray): [d,] weighted vector for the i-th node.

            Returns:
                numpy.ndarray: [n,] data matrix.
            
            if noise_type == 'gaussian_ev':
                # Gaussian noise with equal variances
                N_i = rs.normal(scale=1.0, size=n)
            elif noise_type == 'gaussian_nv':
                # Gaussian noise with non-equal variances
                scale = rs.uniform(low=1.0, high=2.0)
                N_i = rs.normal(scale=scale, size=n)
            elif noise_type == 'exponential':
                # Exponential noise
                N_i = rs.exponential(scale=1.0, size=n)
            elif noise_type == 'gumbel':
                # Gumbel noise
                N_i = rs.gumbel(scale=1.0, size=n)
            else:
                raise ValueError("Unknown noise type.")
            return X @ B_i + N_i

        d = B.shape[0]
        X = np.zeros([n, d])
        G = nx.DiGraph(B)
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for i in ordered_vertices:
            parents = list(G.predecessors(i))
            X[:, i] = _simulate_single_equation(X[:, parents], B[parents, i])

        return X
"""


if __name__ == '__main__':
    n, d = 1000, 20
    graph_type, degree = 'ER', 4    # ER2 graph
    B_scale = 1.0
    noise_type = 'gaussian_ev'

    dataset = SyntheticDataset(n, d, graph_type, degree,
                               noise_type, B_scale, seed=1)
    print("dataset.X.shape: {}".format(dataset.X.shape))
    print("dataset.B.shape: {}".format(dataset.B.shape))
    print("dataset.B_bin.shape: {}".format(dataset.B.shape))
