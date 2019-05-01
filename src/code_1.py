"""
課題1のコード
test_code_1.pyにテストコード
"""
import numpy as np
from common import ReLU

class GNN:
  def __init__(self, D, T):
    self.W = np.ones(D)
    self.T = T
  
  def aggregate(self, G, x):
    for t in range(self.T):
      copy_x = np.copy(x)
      for i in range(x.shape[0]):
        linked_xs = x[G[i] == 1, :]
        a_i = np.sum(linked_xs, axis=0)
        x_i_new = ReLU(np.dot(a_i, self.W))
        copy_x[i] = x_i_new
      x = copy_x
    return x
  
  def readout(self, agg_x):
    h = np.sum(x, axis=0)
    return h

if __name__ == "__main__":
  D = (8, 8)
  T = 2
  G = np.array([[0, 1, 0, 0],
                [1, 0, 1, 1],
                [0, 1, 0, 1],
                [0, 1, 1, 0]])
  x = np.zeros((G.shape[0], D[0]))
  x[:, 0] = 1
  T = 1

  gnn = GNN(D, T)
  agg_x = gnn.aggregate(G, x)
  h = gnn.readout(agg_x)

  print("aggregated X : ", agg_x)
  print("ReadOut h : ", h)