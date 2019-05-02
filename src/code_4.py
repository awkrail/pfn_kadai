"""
課題4のコード
与えられた課題から, 私は以下の2つを選択し, 実装を行なった. レポートのTable.1にその結果を示す．
code_4.pyでは, コマンドライン引数に以下を選択することができる.

- 最適化手法(--optim, -o) : SGD, momentumSGD, Adam
- 層の深さ(--nn, -n) : 1 or 2
"""
import argparse
import pickle
from code_3 import SGD, MomentumSGD, BatchGNN
from code_3 import train_val_split, calculate_accuracy_loss
from utils import *

# Adam
class Adam:
  """
  https://arxiv.org/pdf/1412.6980v8.pdf
  """
  def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999):
    self.lr = lr
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.t = 0
    self.m = None
    self.v = None
  
  def step(self, params, grads):
    if self.m is None:
      self.m = {}
      self.v = {}
      for key, value in params.items():
        self.m[key] = np.zeros_like(value)
        self.v[key] = np.zeros_like(value)
    
    self.t += 1

    for key in params.keys():
      self.m[key] = self.beta_1 * self.m[key] + (1 - self.beta_1) * grads[key]
      self.v[key] = self.beta_2 * self.v[key] + (1 - self.beta_2) * (grads[key]**2)
      hat_m = self.m[key] / (1 - (self.beta_1 ** self.t))
      hat_v = self.v[key] / (1 - (self.beta_2 ** self.t))
      params[key] -= self.lr * hat_m / (np.sqrt(hat_v) + 1e-7)

# 2NN Network
"""
class BatchGNN_2NN(BatchGNN):
  def __init__(self, D, T):
    super()
"""

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--optim", "-o", type=str, default="SGD", help="Select optimizer. SGD[-o SGD], MomentumSGD [-o mSGD], or Adam [-o adam]")
  parser.add_argument("--nn", "-n", type=int, default=1, help="Select the depth of neural network, 1 or 2")
  args = parser.parse_args()

  nn_depth = args.nn
  optim_method = args.optim

  # split train/val
  graphs, labels = load_graph_from_train_data()
  train_graphs, train_labels, val_graphs, val_labels = train_val_split(graphs, labels)

  # hyper parameters
  N = len(train_graphs) # データ数
  D = (8, 8) # 行列の次元数
  lr = 0.001 # 学習率
  eta = 0.9 # モーメント
  beta_1=0.9 # Adam用(\beta_1)
  beta_2=0.999 # Adam用(\beta_2)
  epoch = 50 # エポック数
  B = 30 # バッチサイズ
  T = 2 # 集約数

  # model
  if nn_depth == 1:
    gnn = BatchGNN(D, T)
  # TODO : 2層verを追加する

  if optim_method == "SGD":
    optimizer = SGD(lr)
  elif optim_method == "mSGD":
    optimizer = MomentumSGD(lr, eta)
  elif optim_method == "adam":
    optimizer = Adam(lr, beta_1, beta_2)
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
  if optim_method == "SGD":
    filename = "results/SGD.pkl"
  elif optim_method == "mSGD":
    filename = "results/MomentumSGD.pkl"
  elif optim_method == "adam":
    filename = "results/Adam.pkl"

  with open(filename, "wb") as f:
    pickle.dump(results, f)