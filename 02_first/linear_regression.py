import numpy as np

def predict(X, w):
  return X * w

def loss(X, Y, w):
  return np.average((predict(X, w) - Y) ** 2)

def train(X, Y, iterations, lr):
  w = 0
  for i in range(iterations):
    current_loss = loss(X, Y, w)
    print("Iteration %4d => Loss: %.6f" % (i, current_loss))
    if loss(X, Y, w + lr) < current_loss:
      w += lr
    elif loss(X, Y, w - lr) < current_loss:
      w -= lr
    else:
      return w
  raise Exception("Couldn't converge within %d iterations" % iterations)

if __name__ == "__main__":
  # Import data
  X, Y = np.loadtxt("pizza.txt", skiprows = 1, unpack = True)

  # Train system
  w = train(X, Y, iterations = 10000, lr = 0.01)
  print("\nw=%.3f" % w)
  print("Prediction: x=%d => 7=%.2f" % (20, predict(20, w)))

   # Plot chart
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set()
  plt.plot(X, Y, "bo")
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  plt.xlabel("Reservations", fontsize=30)
  plt.ylabel("Pizzas", fontsize=30)
  x_edge, y_edge = 50, 50
  plt.axis([0, x_edge, 0, y_edge])
  plt.plot([0, x_edge], [0, predict(x_edge, w)], linewidth=1.0, color="g")
  plt.ion()
  plt.show()
  input("Press <Enter>...")