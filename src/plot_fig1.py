import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font="IPAexGothic", font_scale=1.2)

def plot_loss_transition(sgd_results, msgd_results):
  sgd_train_loss = np.array([sgd_result["train_loss"] for sgd_result in sgd_results])
  sgd_val_loss = np.array([sgd_result["val_loss"] for sgd_result in sgd_results])
  msgd_train_loss = np.array([msgd_result["train_loss"] for msgd_result in msgd_results])
  msgd_val_loss = np.array([msgd_result["val_loss"] for msgd_result in msgd_results])
  
  loss_change = np.concatenate([sgd_train_loss, sgd_val_loss, msgd_train_loss, msgd_val_loss])
  N = sgd_train_loss.shape[0]
  x = np.arange(N)
  x = np.concatenate([x, x, x, x])

  x = x.tolist()
  loss_change = loss_change.tolist()
  data_type = ["SGD train"] * N + ["SGD val"] * N + ["mSGD train"] * N + ["mSGD val"] * N
  
  acc = np.array([x, loss_change, data_type])
  acc_df = pd.DataFrame(acc.T, columns=["エポック数", "平均損失", "データ(train/val)"])
  acc_df["平均損失"] = pd.to_numeric(acc_df["平均損失"])
  acc_df["エポック数"] = pd.to_numeric(acc_df["エポック数"])
  plt.clf()
  plt.figure(figsize=(12, 6))
  plt.title("エポックごとの平均損失の推移")
  plt.legend(loc="upper right", prop={'size': 10}, bbox_to_anchor=(1.3, 1))
  sns.pointplot(x="エポック数", y="平均損失", hue="データ(train/val)", data=acc_df,
                markers=["^", "o", "v", ","], linestyles=[":", "-.", "--", "-"])
  plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
  plt.savefig("results/fig_loss.png")

def plot_acc_transition(sgd_results, msgd_results):
  sgd_train_loss = np.array([sgd_result["train_acc"] for sgd_result in sgd_results])
  sgd_val_loss = np.array([sgd_result["val_acc"] for sgd_result in sgd_results])
  msgd_train_loss = np.array([msgd_result["train_acc"] for msgd_result in msgd_results])
  msgd_val_loss = np.array([msgd_result["val_acc"] for msgd_result in msgd_results])
  
  loss_change = np.concatenate([sgd_train_loss, sgd_val_loss, msgd_train_loss, msgd_val_loss])
  N = sgd_train_loss.shape[0]
  x = np.arange(N)
  x = np.concatenate([x, x, x, x])

  x = x.tolist()
  loss_change = loss_change.tolist()
  data_type = ["SGD train"] * N + ["SGD val"] * N + ["mSGD train"] * N + ["mSGD val"] * N
  
  acc = np.array([x, loss_change, data_type])
  acc_df = pd.DataFrame(acc.T, columns=["エポック数", "平均正解率", "データ(train/val)"])
  acc_df["平均正解率"] = pd.to_numeric(acc_df["平均正解率"])
  acc_df["エポック数"] = pd.to_numeric(acc_df["エポック数"])
  plt.clf()
  plt.figure(figsize=(12, 6))
  # plt.title("エポックごとの平均正解率の推移")
  plt.legend(loc="upper right", prop={'size': 10}, bbox_to_anchor=(1.3, 1))
  sns.pointplot(x="エポック数", y="平均正解率", hue="データ(train/val)", data=acc_df,
                markers=["^", "o", "v", ","], linestyles=[":", "-.", "--", "-"])
  plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
  plt.savefig("results/fig_acc.png")

if __name__ == "__main__":
  with open("results/2NN/SGD.pkl", "rb") as f:
    sgd_results = pickle.load(f)
  
  with open("results/2NN/MomentumSGD.pkl", "rb") as f:
    m_sgd_results = pickle.load(f)
  
  with open("results/2NN/Adam.pkl", "rb") as f:
    adam_results = pickle.load(f)
  
  # 平均損失, 平均正解率のグラフをplotする
  # plot_loss_transition(sgd_results, m_sgd_results)
  # plot_acc_transition(sgd_results, m_sgd_results)

  # 最も検証用データで性能がよかったのは?
  sorted_m_sgd_results = sorted([(idx+1, m_sgd_result["val_acc"]) for idx, m_sgd_result in enumerate(m_sgd_results)], key=lambda x : x[1], reverse=True)
  sorted_sgd_results = sorted([(idx+1, sgd_result["val_acc"]) for idx, sgd_result in enumerate(sgd_results)], key=lambda x : x[1], reverse=True)
  sorted_adam_results = sorted([(idx+1, adam_result["val_acc"]) for idx, adam_result in enumerate(adam_results)], key=lambda x : x[1], reverse=True)
  import ipdb; ipdb.set_trace()