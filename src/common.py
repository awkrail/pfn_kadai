import numpy as np

def binary_cross_entropy(y, t):
  return t * np.log(1 + np.exp(-y)) + (1 - t) * np.log(1 + np.exp(y))

def ReLU(x):
  return np.maximum(0, x)

def numerical_gradient_1d(f, x):
  eps = 1e-3
  grad = np.zeros_like(x)
  f_x = f(x)
  for idx in range(x.size):
    tmp_val = x[idx]
    x[idx] = float(tmp_val) + eps
    f_dx = f(x)
    grad[idx] = (f_dx - f_x) / eps
    x[idx] = tmp_val

  return grad

def numerical_gradient(f, X):
  if X.ndim == 1:
    return numerical_gradient_1d(f, X)
  else:
    grad = np.zeros_like(X)

    for idx, x in enumerate(X):
      grad[idx] = numerical_gradient_1d(f, x)
    return grad