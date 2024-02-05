import igraph as ig
import networkx as nx
import numpy as np
import semopy

class Dataset:
    """"""

    def __init__(self, n, v, vA, t, w, WA, cons, seed=1):
        """Initialize self.

 Args:
            n (int): Number of samples.
            v (int): Number of variables
            WA(WAgraph): graph of W and A
            seed (int): Random seed. Default: 1.
"""
        self.n = n
        self.v = v
        self.vA = vA
        assert(t > w)
        self.t = t
        self.w = w
        self.rs = np.random.RandomState(seed)
        #assert isinstance(WA, WAgraph)
        cons.validate(w, v, vA)
        self.cons=cons
        WA.validate(w, v, vA)
        self.WA = cons.constrainWA(WA)

    def getData(self):
        return self.data

    def getDataWindow(self):
        return self.data[:self.w]

    def getWA(self):
        return self.WA

    def setData(self, data):
        if isinstance(data, list):
           data = np.array(data)
        if(len(data.shape) == 2):
           data = np.array([data])
        elif(len(data.shape) == 3):
             data = data
        else:
             raise ValueError("data must be 2 or 3 dimentional")
        self.data = data
        self.validate()



    def generateData(self, noise):
        noiseFunc = self.toNoiseFunc(noise)

        data = noiseFunc(np.arange(self.n*self.v*self.w).reshape(self.w, self.v, self.n))


        def step(data):
            dataA = np.tensordot(self.WA.A, data[:self.w], axes=((0,1),(0,1)))
            dataR = noiseFunc(data[0])
            dataW = np.matmul(self.WA.W, dataR)

            newData = np.array([dataW + dataR + dataA])

            return np.append(newData, data, axis = 0)

        for i in range(self.w):
            data = step(data)
        data = data[:self.w]

        for i in range(self.t-self.w):
            data = step(data)

        self.setData(data)


    def validate(self):
        t, v, n = self.data.shape
        assert(self.t == t)
        assert(self.v == v)
        assert(self.n == n)

    def toNoiseFunc(self, noise):
        if isinstance(noise, str):
           pass
        else:
           return np.vectorize(lambda x: noise(self.rs))
