import numpy as np
import math
import copy

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
          else: 
             raise ValueError("v was unreadable")

          if(isinstance(w, np.ndarray)):
             assert(self.v == w.shape[0])
             assert(self.vA == w.shape[1] or self.vA is None)


      def getA(self):
          return self.A

      def getAFlat(self):
          return self.flatten(self.A)
 
      def getW(self):
          return self.W

      def getWindow(self):
          return self.w

      def getV(self):
          return self.v

      def getVA(self):
          return self.vA

      def setA(self, grf):
        if isinstance(grf, list):
           grf = np.array(grf)
        if isinstance(grf, np.ndarray):
           if(len(grf.shape) == 2):
             dag = np.array([dag])
           elif(len(grf.shape) == 3):
             dag = np.array(grf)
           else:
             raise ValueError("dag must be 2 or 3 dimentional")
        else:
           raise ValueError("dag unreadable")
        #if self.A_B is None:#TODO verify are the same
        grf_B = (grf != 0).astype(int)
        self.setA_B(grf_B)
        self.A = grf

      def setA_B(self, grf_B):
        if isinstance(grf_B, list):
           grf_B = np.array(grf_B)
        if isinstance(grf_B, np.ndarray):
           if(len(grf_B.shape) == 2):
             grf_B = np.array([grf_B])
           elif(len(grf_B.shape) == 3):
             grf_B = np.array(grf_B)
           else:
             raise ValueError("binary graph must be 2 or 3 dimentional")
        else:
           raise ValueError("binary graph unreadable")

        if ((grf_B == 0) | (grf_B == 1)).all():
           assert(grf_B.shape[0] == self.w)
           assert(grf_B.shape[1] == self.v)
           assert(grf_B.shape[2] == self.vA)
           self.A_B = grf_B
        else:
           raise ValueError("binary graph must be only 1s and 0s")

      def setW(self, dag):
        if isinstance(dag, list):
           dag = np.array(dag)
        if isinstance(dag, np.ndarray):
           if(len(dag.shape) == 2):
             dag = np.array(dag)
           else:
             raise ValueError("dag must be 2 or 3 dimentional")
        else:
           raise ValueError("dag unreadable")
        #if self.W_B is None:
        dag_B = (dag != 0).astype(int)
        self.setW_B(dag_B)
        self.W = dag

      def setW_B(self, dag_B):
        if isinstance(dag_B, list):
           grf_B = np.array(dag_B)
        if isinstance(dag_B, np.ndarray):
           if(len(dag_B.shape) == 2):
             dag_B = np.array(dag_B)
           else:
             raise ValueError("binary graph must be 2 or 3 dimentional")
        else:
           raise ValueError("binary graph unreadable")
      
        if ((dag_B == 0) | (dag_B == 1)).all():
           assert(dag_B.shape[0] == dag_B.shape[1])
           assert(dag_B.shape[0] == self.v)
           self.W_B = dag_B
        else:
           raise ValueError("binary graph must be only 1s and 0s")
     

      def generateW(self, distr=np.random.rand):
        assert(self.W_B is not None)

        dag = np.vectorize(lambda x: distr())(self.W_B)*self.W_B

        self.setW(dag)

      def generateW_B(self, edges, graph_type, edge_dist="fixed", method="ER"):
        edges = self.getEdgeNum(edges, edge_dist)

        if method == "ER":
           dag_B = np.zeros((self.v, self.v))
           for i in range(edges):
               pos = np.random.randint(0, (self.v*self.v-self.v)//2)
               y = math.floor(.5*(np.sqrt(8*pos+9)))
               x = pos-(y*(y-1)//2)
               dag_B[x][y] = 1
           np.random.shuffle(dag_B)
           np.random.shuffle(np.transpose(dag_B))
           self.W_B = dag_B
        elif method == "BA":
           self.W_B = nx.barabasi_albert_graph(self.v, edges, seed=self.rs).to_numpy_array()

      def generateA(self, distr=np.random.rand, u=1):
        assert(self.A_B is not None)

        grf = np.array([np.vectorize(lambda x: distr()*(u**i))(self.A_B[i]) for i in range(len(self.A_B))])*self.A_B

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
           self.A_B = grf_B
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

      def constrain(self, cons):
          ret = WAgraph(copy.deepcopy(self.W), copy.deepcopy(self.A))
          cons.constrainW(ret.W)
          cons.constrainW(ret.W_B)
          ret.W_B = (ret.W_B != 0).astype(int) 
          cons.constrainA(ret.A)
          cons.constrainA(ret.A_B)
          ret.A_B = (ret.A_B != 0).astype(int) 
          return ret

      def flatten(self, array):
          return np.reshape(array, self.v*self.w, self.v)

      def validate(self, w, v, vA=None):
          assert(w == self.w)
          assert(v == self.v)
          assert(vA == self.vA or (vA is None and self.v == self.vA))
