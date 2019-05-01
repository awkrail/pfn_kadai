"""
課題2のコード
test_code_2.pyにテストコード
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
    
    # UPDATE
    self.A = np.ones((self.rows, 1))
    self.b = np.ones(1)
  
  ####### updated #######
  def binary_cross_entropy(self, h, y):
    # 損失関数を計算
    s = np.dot(h, self.A) + self.b
    L = y * np.log(1 + np.exp(-s)) + (1-y) * np.log(1 + np.exp(s))
    return L
  
  def numerical_gradient(self):
    pass
  
  def update_params(self):
    pass
  #######################

  def aggregate(self):
    # 集約1, 集約2を計算
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
    # READOUTを計算
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

  # 課題2, 前半
  # binary cross entropyの計算
  y = 1
  gnn.binary_cross_entropy(h, y)

  # 課題2, 後半
  # パラメータW, A, bの再計算
  # TODO : G, 10頂点程度に修正
  epoch = 10
  T = 2
  gnn = GNN(G, T)
  y = 1

  for i in range(epoch):
    agg_x = gnn.aggregate()
    h = gnn.readout()
    loss = gnn.binary_cross_entropy(h, y)
    gnn.update_params()
    print("epoch : ", i, " loss : ", loss)
    
    if loss <= 0.01:
      print("done!")
      break