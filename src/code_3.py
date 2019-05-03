import numpy as np
import pickle
from utils import *
from code_2 import GNN
from common import binary_cross_entropy, numerical_gradient, sigmoid
import random
import argparse


class BatchGNN(GNN):
  def batch_loss(self, Gs, ts):
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
  """
  課題3(前半)
  SGDの実装
  """
  def __init__(self, lr=0.001):
    self.lr = lr
  
  def step(self, params, grads):
    for key in grads.keys():
      params[key] -= lr * grads[key]


class MomentumSGD:
  """
  課題3(前半)
  MomentumSGDの実装
  """
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


def calculate_accuracy_loss(model, Gs, ts):
  correct_count = 0
  all_count = len(Gs)
  loss = []

  for G, t in zip(Gs, ts):
    x = np.zeros((G.shape[0], model.D[0]))
    x[:, 0] = 1
    y = model.forward(G, x)
    hat_y = sigmoid(y)
    predict = 1 if hat_y[0] > 0.5 else 0
    if predict == t:
      correct_count += 1
    loss.append(model.loss(G, x, t))

  loss = np.mean(np.array(loss))
  return correct_count/all_count, loss


def train_val_split(Xs, ys):
  Xy_tuples = [(X, y) for X, y in zip(Xs, ys)]
  random.shuffle(Xy_tuples)
  threshold = int(0.8 * len(Xy_tuples))
  train_tuples, test_tuples = Xy_tuples[:threshold], Xy_tuples[threshold:]
  train_graphs, train_labels = np.array([Xy[0] for Xy in train_tuples]), np.array([Xy[1] for Xy in train_tuples])
  val_graphs, val_labels = np.array([Xy[0] for Xy in test_tuples]), np.array([Xy[1] for Xy in test_tuples])
  return train_graphs, train_labels, val_graphs, val_labels

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--optim", "-o", type=str, default="SGD", help="Select optimizer. SGD[-o SGD], MomentumSGD [-o mSGD]")
  args = parser.parse_args()

  # split train/val
  graphs, labels = load_graph_from_train_data()
  train_graphs, train_labels, val_graphs, val_labels = train_val_split(graphs, labels)

  # hyper parameters
  N = len(train_graphs) # データ数
  D = (8, 8) # 行列の次元数
  lr = 0.001 # 学習率
  eta = 0.9 # モーメント
  epoch = 50 # エポック数
  B = 30 # バッチサイズ
  T = 2 # 集約数

  # model, optimizer
  gnn = BatchGNN(D, T)
  if args.optim == "SGD":
    optimizer = SGD(lr)
  elif args.optim == "mSGD":
    optimizer = MomentumSGD(lr, eta)
  else:
    print("Optimizer Error. Please select optimizer from SGD or momentumSGD[mSGD]")
    exit(1)

  # レポート用
  results = []

  for ep in range(epoch):
    shuffle_idx = np.random.permutation(N)
    for idx in range(0, N, B):
      Gs = train_graphs[shuffle_idx[idx:(idx+B) if idx+B < N else N]]
      ts = train_labels[shuffle_idx[idx:(idx+B) if idx+B < N else N]]
      grads = gnn.batch_gradient(Gs, ts)
      optimizer.step(gnn.params, grads)

    # 訓練データでのloss, 正解率の計算
    train_acc, train_loss = calculate_accuracy_loss(gnn, train_graphs, train_labels)
    # valデータでのloss, 正解率の計算
    val_acc, val_loss = calculate_accuracy_loss(gnn, val_graphs, val_labels)
    
    results.append({ "train_acc" : train_acc, "train_loss" : train_loss, "val_acc" : val_acc, "val_loss" : val_loss, "params" : gnn.params })
    print("epoch : ", ep, "train_loss : ", train_loss, " train_acc : ", train_acc, " val_loss : ", val_loss, "val_acc : ", val_acc)
  
  """
  あとで描画する時用に保存
  """
  if args.optim == "SGD":
    filename = "results/1NN/SGD.pkl"
  else:
    filename = "results/1NN/MomentumSGD.pkl"

  with open(filename, "wb") as f:
    pickle.dump(results, f)