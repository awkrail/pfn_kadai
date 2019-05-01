"""
ファイルの読み込み系のソースコード
"""
import os

def read_graph_from_file(data_dir):
  with open(data_dir, "r") as f:
    lines = f.readlines()
  node_num = int(lines[0])
  graph = []
  for line in lines[1:]:
    line = line.strip()
    row = [int(x) for x in line.split(" ")]
    graph.append(row)
  assert(len(graph) == node_num)
  return np.array(graph)

def read_label_from_file(data_dir):
  with open(data_dir, "r") as f:
    lines = f.readlines()
  return int(lines[0].strip())

def load_graph_from_train_data():
  graphs = []
  labels = []
  all_files = list(set([file.split("_")[0] for file in os.listdir("datasets/train")]))
  
  for data_file in all_files:
    graph_file = "datasets/train/" + data_file + "_graph.txt"
    label_file = "datasets/train/" + data_file + "_label.txt"
    graph = read_graph_from_file(graph_file)
    label = read_label_from_file(label_file)
    graphs.append(graph)
    labels.append(label)

  return graphs, labels