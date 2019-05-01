import numpy as np
import pickle
from utils import *
from code_2 import GNN
from common import binary_cross_entropy, numerical_gradient

np.random.seed(42)

class BatchGNN(GNN):
  def batch_loss(self, Gs, ts):
    # ここにバッチ処理を書く予定
    losses = []
    for G, t in zip(Gs, ts):
      x = np.zeros((G.shape[0], self.D[0]))
      x[:, 0] = 1
      y = self.forward(G, x)
      loss = binary_cross_entropy(y, t)
      losses.append(loss)
    return np.mean(np.array(losses))
  
  def batch_gradient(self, Gs, ts):
    loss_f = lambda f : self.batch_loss(Gs, ts)
    
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


class SGD:
  def __init__(self, lr=0.001):
    self.lr = lr
  
  def step(self, params, grads):
    for key in grads.keys():
      params[key] -= lr * grads[key]


class MomentumSGD:
  def __init__(self, lr=0.001, eta=0.9):
    self.lr = lr
    self.eta = eta
    self.w = None
  
  def step(self, params, grads):
    if self.w is None:
      self.w = {}
      for key, value in params.items():
        self.w[key] = np.zeros_like(value)
  
    for key in grads.keys():
      params[key] = params[key] - self.lr * grads[key] + self.eta * self.w[key]
      self.w[key] = self.eta * self.w[key] - self.lr * grads[key]


if __name__ == "__main__":
  graphs, labels = load_graph_from_train_data()
  graphs = np.array(graphs)
  labels = np.array(labels, dtype=np.int32)
  
  # hyper parameters
  N = len(graphs) # データ数
  D = (8, 8) # 行列の次元数
  lr = 0.001 # 学習率
  eta = 0.9 # モーメント
  epoch = 10 # エポック数
  B = 50 # バッチサイズ
  T = 2 # 集約数

  # model, optimizer
  gnn = BatchGNN(D, T)
  optimizer = SGD(lr)
  losses = []

  for ep in range(epoch):
    shuffle_idx = np.random.permutation(N)
    ep_losses = []
    for idx in range(0, N, B):
      Gs = graphs[shuffle_idx[idx:(idx+B) if idx+B < N else N]]
      ts = labels[shuffle_idx[idx:(idx+B) if idx+B < N else N]]
      grads = gnn.batch_gradient(Gs, ts)
      optimizer.step(gnn.params, grads)
      ep_losses.append(gnn.batch_loss(Gs, ts))
    
    mean_loss = np.array(ep_losses).mean()
    losses.append(mean_loss)
    print("epoch : ", ep, " Loss : ", mean_loss)

  """
  あとで描画する時用に保存
  """
  with open("results/SGD.pkl", "wb") as f:
    pickle.dump(losses, f)