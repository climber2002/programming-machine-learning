import numpy as np

def predict(X, w, b):
  return X * w + b

def loss(X, Y, w, b):
  return np.average((predict(X, w, b) - Y) ** 2)

def gradient(X, Y, w, b):
  w_gradient = np.average(2 * X * (predict(X, w, b) - Y))
  b_gradient = np.average(2 * (predict(X, w, b) - Y))
  return (w_gradient, b_gradient)

def train(X, Y, iterations, lr):
  w = b = 0
  for i in range(iterations):
    print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, b)))
    w_gradient, b_gradient = gradient(X, Y, w, b)
    print("w_gradient: %.10f , b_gradient: %.10f" % (w_gradient, b_gradient))
    w -= w_gradient * lr
    b -= b_gradient * lr
  return w, b

if __name__ == "__main__":
  X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
  w, b = train(X, Y, iterations=20000, lr=0.001)
  print("\nw=%.10f, b=%.10f" % (w, b))
  print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))
