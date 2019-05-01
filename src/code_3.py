import numpy as np
from utils import *
from code_2 import GNN

np.random.seed(42)

class BatchGNN(GNN):
  def batch_gradient(self, x_batch, t_batch):
    # ここにバッチ処理を書く予定
    pass

class SGD:
  def __init__(self):
    pass

class MomentumSGD:
  def __init__(self):
    pass

if __name__ == "__main__":
  graphs, labels = load_graph_from_train_data()
  train_tuples = [(graph, label) for graph, label in zip(graphs, labels)]
  
  # hyper parameters
  D = (8, 8) # 行列の次元数
  lr = 0.001 # 学習率
  eta = 0.9 # モーメント
  epoch = 50 # エポック数
  B = 50 # バッチサイズ

  for ep in range(epoch):
    pass


  import ipdb; ipdb.set_trace()