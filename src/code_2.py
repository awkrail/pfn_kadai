"""
課題2のコード
"""
import numpy as np
from common import binary_cross_entropy, ReLU, numerical_gradient

class GNN:
  def __init__(self, D, T):
    self.params = {
      "W" : np.random.normal(0, 0.4, D),
      "A" : np.random.normal(0, 0.4, D[0]),
      "b" : np.zeros(1)
    }
    self.D = D
    self.T = T
  
  def forward(self, G, x):
    # 1. aggregate
    agg_x = self.aggregate(G, x)
    # 2. readout
    h = self.readout(agg_x)
    # 3. linear transform
    s = np.dot(h, self.params["A"]) + self.params["b"]
    return s

  def loss(self, G, x, t):
    y = self.forward(G, x)
    return binary_cross_entropy(y, t)
  
  def numerical_gradient(self, G, x, t):
    loss_f = lambda f : self.loss(G, x, t)
    # 損失関数を計算
    grad_W = numerical_gradient(loss_f, self.params["W"])
    grad_A = numerical_gradient(loss_f, self.params["A"])
    grad_b = numerical_gradient(loss_f, self.params["b"])

    grads = {
      "W" : grad_W,
      "A" : grad_A,
      "b" : grad_b
    }
    return grads
  
  def aggregate(self, G, x):
    for t in range(self.T):
      agg_x = np.zeros_like(x)
      for i in range(x.shape[0]):
        linked_xs = x[G[i] == 1, :]
        a_i = np.sum(linked_xs, axis=0)
        x_i_new = ReLU(np.dot(a_i, self.params["W"]))
        agg_x[i] = x_i_new
      x = agg_x
    return x
  
  def readout(self, agg_x):
    h = np.sum(agg_x, axis=0)
    return h

if __name__ == "__main__":
  D = (8, 8)
  T = 2
  G = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]])
  x = np.zeros((G.shape[0], D[0]))
  x[:, 0] = 1 # 各ノードの特徴ベクトルは先頭要素だけ1, 他は0としたベクトルで表現
  t = 0
  lr = 0.001
    
  # 課題2(前半) : binary cross entropyを実装
  gnn = GNN(D, T)
  h = gnn.forward(G, x)
  y = gnn.loss(G, x, t)
  print("binary cross entropy : ", y[0])

  # 課題2(後半) : binary cross entropyを数値微分
  gnn = GNN(D, T)
  for epoch in range(1000):
    grads = gnn.numerical_gradient(G, x, t)
    for key in grads.keys():
      gnn.params[key] -= lr * grads[key]
    loss = gnn.loss(G, x, t)
    print("epoch : ", epoch, "loss : ", loss[0])

    # 初期値の設定によってはNaNになることもある
    if np.isnan(loss) or np.isinf(loss):
      print("Oops! Would you run this script again? The initial weights seems to be bad.")
      exit(1)

    if loss <= 0.01:
      final_epoch = epoch
      final_loss = loss
      break
  
  print("final epoch : ", final_epoch)
  print("final loss : ", final_loss[0])