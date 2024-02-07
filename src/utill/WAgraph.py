import numpy as np
import math
import copy
import torch 
import igraph as ig
import networkx as nx

class WAgraph:
      def __init__(self, w, v, vA=None, rs=1):
          self.rs = rs

          if isinstance(w, int):
             self.w = w
             self.W = None
             self.W_B = None
          elif(isinstance(w, np.ndarray)):
             self.v = w.shape[0]
             self.setW(w)
          elif(isinstance(w, torch.Tensor)):
             self.v = w.size()[0]
             self.setW(w)
          else: 
             raise ValueError("w was unreadable")

          if(isinstance(v, int) and (isinstance(vA, int) or vA is None)):
             self.v = v
             if vA is not None:
                self.vA= vA
             else:
                self.vA= v
             self.A = None
             self.A_B = None
          elif(isinstance(v, np.ndarray)):
             self.w = v.shape[0]
             self.v = v.shape[1]
             if vA is not None:
                self.vA= vA
             else:
                self.vA= v.shape[2]
             self.setA(v)
          elif(isinstance(v, torch.Tensor)):
             self.w = v.size()[0]
             self.v = v.size()[1]
             if vA is not None:
                self.vA= vA
             else:
                self.vA= v.size()[2]
             self.setA(v)
          else: 
             raise ValueError("v was unreadable")

          if(isinstance(w, np.ndarray)):
             assert(self.v == w.shape[0])
             assert(self.vA == w.shape[1] or self.vA is None)


      def getA(self):
          return self.A.detach().numpy()

      def getA_B(self):
          return self.A_B.detach().numpy()

      def getAFlat(self):
          return self.flatten(self.getA())
 
      def getW(self):
          return self.W.detach().numpy()
      
      def getW_B(self):
          return self.W_B.detach().numpy()

      def getWindow(self):
          return self.w

      def getV(self):
          return self.v

      def getVA(self):
          return self.vA

      def setA(self, grf):
        grf = self.castA(grf)

        #if self.A_B is None:#TODO verify are the same
        grf_B = (grf != 0).int()
        
        self.setA_B(grf_B)
        self.A = grf

      def setA_B(self, grf_B):
        grf_B = self.castA(grf_B)

        if ((grf_B == 0) | (grf_B == 1)).all():
           assert(grf_B.size()[0] == self.w)
           assert(grf_B.size()[1] == self.v)
           assert(grf_B.size()[2] == self.vA)
           self.A_B = grf_B
        else:
           raise ValueError("binary graph must be only 1s and 0s")

      def castA(self, grf):
        if isinstance(grf, list):
           grf = np.array(grf)
        if isinstance(grf, np.ndarray):
           grf = torch.tensor(grf)
        if isinstance(grf, torch.Tensor):
           if(len(grf.size()) == 2):
             dag = np.array([dag])
           elif(len(grf.size()) == 3):
             dag = grf
           else:
             raise ValueError("dag must be 2 or 3 dimentional")
        else:
           raise ValueError("dag unreadable")
        return grf


      def setW(self, dag):
        dag = self.castW(dag)

        #if self.W_B is None:
        dag_B = (dag.detach().numpy() != 0).astype(int)
        self.setW_B(dag_B)
        self.W = dag

      def setW_B(self, dag_B):
        dag_B = self.castW(dag_B)
  
        if ((dag_B == 0) | (dag_B == 1)).all():
           assert(dag_B.shape[0] == dag_B.shape[1])
           assert(dag_B.shape[0] == self.v)
           self.W_B = dag_B
        else:
           raise ValueError("binary graph must be only 1s and 0s")
    
      def castW(self, dag):
        if isinstance(dag, list):
           dag = np.array(dag)
        if isinstance(dag, np.ndarray):
           dag = torch.tensor(dag)
        if isinstance(dag, torch.Tensor):
           if(len(dag.size()) == 2):
             dag = dag
           else:
             raise ValueError("dag must be 2 dimentional")
        else:
           raise ValueError("dag unreadable") 
        return dag


      def generateW(self, distr=np.random.rand):
        assert(self.W_B is not None)

        dag = torch.mul(torch.tensor([[torch.tensor(distr()) 
              for j in range(len(self.W_B[0]))] for i in range(len(self.W_B))]), self.W_B)

        self.setW(dag)

      def generateW_B(self, edges, graph_type, edge_dist="fixed", method="SF"):
        edges = self.getEdgeNum(edges, edge_dist)

        if method == "ER":
           dag_B = np.zeros((self.v, self.v))
           p = float(edges) / (self.v - 1)
           G_und = nx.generators.erdos_renyi_graph(n=self.v, p=p)
           B_und_bin = nx.to_numpy_array(G_und)    # Undirected
           dag_B = np.tril(B_und_bin, k=-1)

           self.setW_B(dag_B)
        elif method == "SF":
           m = int(math.ceil(edges / 2))
           print(self.v, m)
           G = ig.Graph.Barabasi(n=self.v, m=m, directed=True)
           dag_B = np.array(G.get_adjacency().data)

           self.setW_B(dag_B)
        elif method == "BA":
           self.W_B = nx.barabasi_albert_graph(self.v, edges, seed=self.rs).to_numpy_array()

      def generateA(self, distr=np.random.rand, u=1):
        assert(self.A_B is not None)
   
        grf = torch.mul(torch.tensor([[[torch.tensor(distr()*(u**i)) 
              for k in range(len(self.A_B[0][0]))] for j in range(len(self.A_B[0]))] 
              for i in range(len(self.A_B))]), self.A_B)

        self.setA(grf)


      def generateA_B(self, edges, graph_type, node_dist="fixed", method="ER"):
        nodes = self.getEdgeNum(edges, node_dist)

        if method == "ER":
           grf_B = np.zeros((self.w, self.v, self.v))
           for i in range(nodes):
               pos = np.random.randint(0, self.w*self.v*self.v)
               z = pos//(self.v*self.vA)
               y = (pos%(self.v*self.vA))//self.v
               x = pos%self.v
               grf_B[z][y][x] = 1
           self.setA_B(grf_B)
        elif method == "block":
           raise ValueError("not implemented")
           #sizes = [] 
           #probs = [[]]
	   #self.A_B = np.rearrange(nx.stochastic_block_model(sizes, probs, seed=self.rs), self.w, self.v, self.v)     
 

      def getEdgeNum(self, edges, edge_dist):
        if isinstance(edge_dist, str):
           if edge_dist=="fixed":
              edge_dist = lambda x: x
           else:
              raise ValueError("node_dist not recognised")
        if not callable(edge_dist):
           raise ValueError("node_dist must be string or function")
        return edge_dist(edges)

      def zero_diag(self):
          self.W.fill_diagonal_(0)

      def constrain(self, cons):
          cons.constrainW(self.W)
          cons.constrainW(self.W_B)
          self.W_B = (self.W_B != 0).int() 
          cons.constrainA(self.A)
          cons.constrainA(self.A_B)
          self.A_B = (self.A_B != 0).int() 

      def flatten(self, array):
          return np.reshape(array, self.v*self.w, self.v)

      def validate(self, w, v, vA=None):
          assert(w == self.w)
          assert(v == self.v)
          assert(vA == self.vA or (vA is None and self.v == self.vA))
