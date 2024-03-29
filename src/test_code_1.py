"""
課題1のテストコード
"""
import numpy as np
import unittest
from code_1 import GNN

class TestGNN(unittest.TestCase):
  """
  test class of code_1.py
  """

  def test_aggregate(self):
    """
    test for aggregate()
    G = [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
    X = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    注意 : Xは行で一つの特徴ベクトルとする.
    つまり, ノードの頂点の番号の要素を1, それ以外を0とする特徴ベクトルで表現

    aggregateの結果
    T = 1:
    expected_X = 
          [[0, 1, 0],
           [1, 0, 1],
           [0, 1, 0]]
    T = 2:
    expected_X = 
          [[1, 0, 1],
           [0, 2, 0],
           [1, 0, 1]]
    """
    D = (3, 3)
    G = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]])
    T = 1
    x = np.eye(3)
    expected_agg_x = np.array([[1, 1, 1],
                               [2, 2, 2],
                               [1, 1, 1]])
    gnn = GNN(D, T)
    agg_x = gnn.aggregate(G, x)
    np.testing.assert_almost_equal(agg_x, expected_agg_x)

    T = 2
    expected_agg_x = np.array([[6, 6, 6],
                               [6, 6, 6],
                               [6, 6, 6]])
    gnn = GNN(D, T)
    agg_x = gnn.aggregate(G, x)
    np.testing.assert_almost_equal(agg_x, expected_agg_x)
    
  def test_readout(self):
    """
    test for readout()
    上のケースで計算すると, T=1, T=2でそれぞれ
    T = 1:
      h = [1, 2, 1]
    T = 2:
      h = [2, 2, 2]
    と計算される
    """
    D = (3, 3)
    G = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]])
    T = 1
    x = np.eye(3)
    expected_h = np.array([4, 4, 4])
    gnn = GNN(D, T)
    agg_x = gnn.aggregate(G, x)
    h = gnn.readout(agg_x)
    np.testing.assert_almost_equal(h, expected_h)

    T = 2
    expected_h = np.array([18, 18, 18])
    gnn = GNN(D, T)
    agg_x = gnn.aggregate(G, x)
    h = gnn.readout(agg_x)
    np.testing.assert_almost_equal(h, expected_h)

if __name__ == "__main__":
  unittest.main()