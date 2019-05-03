"""
report.pdfでも書いたように, Adam+2NNが検証用データでの正解率70%と最も高かった.
このファイルでは, datasets/testのデータに対して予測を行なって保存をする
"""
import numpy as np
import os
from code_4 import BatchGNN_2NN
from common import sigmoid

def load_test_data():
  test_files = [test_file for test_file in os.listdir("datasets/test/")]
  test_files = sorted(test_files, key=lambda x : int(x.split("_")[0])) # 0_graph => 1_graph..., と数字の順番になるように
  test_Gs = []

  for test_file in test_files:
    with open("datasets/test/" + test_file, "r") as f:
      lines = f.readlines()
    node_num = int(lines[0])
    graph = []
    for line in lines[1:]:
      line = line.strip()
      row = [int(x) for x in line.split(" ")]
      graph.append(row)
    assert(len(graph) == node_num)
    test_Gs.append(np.array(graph))
  return test_Gs

if __name__ == "__main__":
  test_graphs = load_test_data()

  D = (8, 8)
  T = 2
  best_epoch = 35

  gnn = BatchGNN_2NN(D, T)
  gnn.load_params(best_epoch, "results/2NN/Adam.pkl")
  predict_results = []

  for test_G in test_graphs:
    # 1. 特徴ベクトルを作る
    x = np.zeros((test_G.shape[0], D[0]))
    x[:, 0] = 1
    y = gnn.forward(test_G, x)
    hat_y = sigmoid(y)
    predict = 1 if hat_y[0] > 0.5 else 0
    predict_results.append(predict)
  
  with open("../prediction.txt", "w") as f:
    for predict_result in predict_results:
      f.write(str(predict_result))
      f.write("\n")