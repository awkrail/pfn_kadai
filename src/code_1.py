"""
課題1のコード
test_code_1.pyにテストコード
"""
import numpy as np

class GNN:
  def __init__(self, G, T):
    self.G = G
    self.rows, self.cols = self.G.shape
    
    # 重み行列
    self.W = np.ones((self.rows, self.cols))
    # one-hot-vectorで各特徴ベクトルを表現
    self.x = np.eye(self.rows)
    self.T = T
  
  def aggregate(self):
    # 集約1, 集約2を計算する
    for t in range(self.T):
      copy_x = np.copy(self.x)
      for i in range(self.x.shape[0]):
        linked_xs = self.x[self.G[i] == 1, :] # 隣接行列から接続している特徴ベクトルを抜き出す
        a_i = np.sum(linked_xs, axis=0)
        x_i_new = self.ReLU(a_i)
        copy_x[i] = x_i_new
      self.x = copy_x
    return self.x
  
  def readout(self):
    # READOUTを計算する
    h = np.sum(self.x, axis=0)
    return h
  
  @staticmethod
  def ReLU(x):
    return np.maximum(0, x)

if __name__ == "__main__":
  G = np.array([[0, 1, 0, 0],
                [1, 0, 1, 1],
                [0, 1, 0, 1],
                [0, 1, 1, 0]])
  T = 1
  gnn = GNN(G, T)
  agg_x = gnn.aggregate()
  h = gnn.readout()