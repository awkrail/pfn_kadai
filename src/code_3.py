import numpy as np
import pickle
from utils import *
from code_2 import GNN
from common import binary_cross_entropy, numerical_gradient, sigmoid
import random

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


def calculate_accuracy(model, val_Gs, val_ts):
  correct_count = 0
  all_count = len(val_Gs)
  val_loss = []

  for val_G, val_t in zip(val_Gs, val_ts):
    x = np.zeros((val_G.shape[0], model.D[0]))
    x[:, 0] = 1
    y = model.forward(val_G, x)
    hat_y = sigmoid(y)
    predict = 1 if hat_y[0] > 0.5 else 0
    if predict == val_t:
      correct_count += 1
    val_loss.append(model.loss(val_G, x, val_t))
  val_loss = np.mean(np.array(val_loss))
  return correct_count/all_count, val_loss

def train_val_split(Xs, ys):
  Xy_tuples = [(X, y) for X, y in zip(Xs, ys)]
  random.shuffle(Xy_tuples)
  threshold = int(0.8 * len(Xy_tuples))
  train_tuples, test_tuples = Xy_tuples[:threshold], Xy_tuples[threshold:]
  train_graphs, train_labels = np.array([Xy[0] for Xy in train_tuples]), np.array([Xy[1] for Xy in train_tuples])
  val_graphs, val_labels = np.array([Xy[0] for Xy in test_tuples]), np.array([Xy[1] for Xy in test_tuples])
  return train_graphs, train_labels, val_graphs, val_labels

if __name__ == "__main__":
  graphs, labels = load_graph_from_train_data()

  # split train/val
  train_graphs, train_labels, val_graphs, val_labels = train_val_split(graphs, labels)

  # hyper parameters
  N = len(train_graphs) # データ数
  D = (8, 8) # 行列の次元数
  lr = 0.0001 # 学習率
  eta = 0.9 # モーメント
  epoch = 100 # エポック数
  B = 30 # バッチサイズ
  T = 3 # 集約数

  # model, optimizer
  gnn = BatchGNN(D, T)
  # optimizer = SGD(lr)
  optimizer = MomentumSGD(lr, eta)
  loss_transition = []
  accuracy_transition = []

  for ep in range(epoch):
    shuffle_idx = np.random.permutation(N)
    ep_losses = []
    for idx in range(0, N, B):
      Gs = train_graphs[shuffle_idx[idx:(idx+B) if idx+B < N else N]]
      ts = train_labels[shuffle_idx[idx:(idx+B) if idx+B < N else N]]
      grads = gnn.batch_gradient(Gs, ts)

      for key in grads.keys():
        gnn.params[key] -= lr * grads[key]
    
      # optimizer.step(gnn.params, grads)
      ep_losses.append(gnn.batch_loss(Gs, ts))
    
    mean_loss = np.array(ep_losses).mean()
    loss_transition.append(mean_loss)

    # valデータでの正解率の計算
    accuracy, val_loss = calculate_accuracy(gnn, val_graphs, val_labels)
    accuracy_transition.append(accuracy)
    print("epoch : ", ep, " Loss : ", mean_loss, "Accuracy : ", accuracy, "val_loss : ", val_loss)
  
  """
  あとで描画する時用に保存
  """
  results = {
    "params" : gnn.params,
    "loss" : loss_transition,
    "accuracy" : accuracy_transition
  }

  with open("results/MomentumSGD.pkl", "wb") as f:
    pickle.dump(results, f)