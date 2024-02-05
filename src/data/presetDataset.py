import igraph as ig
import networkx as nx
import numpy as np
from timeseriesDataset import TimeseriesDataset
from timeseriesDataset import TimeseriesDataset

def getPresetDataset(name, **args):
    try:
       return eval(str(name, " ", args)))
    except:
       throw ValueError(name + " is not a defined preset dataset")
