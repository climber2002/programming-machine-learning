import numpy as np

def predict(X, w):
  return X * w

def loss(X, Y, w):
  return np.average((predict(X, w) - Y) ** 2)


