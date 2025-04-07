import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import *
import time

# Max number of SGD iterations (each over all N points)
max_iter = 50  # You can increase if needed
tol = 0.01

# Step size
eta = 0.003

# Load data
data = loadmat('data.mat')
X1, X2 = data['X1'], data['X2']

# Prepare X and target t
X = np.vstack((X1, X2))
X = np.hstack((X, np.ones((X.shape[0], 1))))
t = np.vstack((np.zeros((X1.shape[0], 1)), np.ones((X2.shape[0], 1))))

N = X.shape[0]

# Initialize weights
w = np.array([1., 0., 0.]).reshape(3, 1)

# Store error over iterations
e_all = []

# Set up slope-intercept figure
plt.figure(2)
plt.rcParams['font.size'] = 8
plt.title('Separator in slope-intercept space')
plt.xlabel('slope')
plt.ylabel('intercept')
plt.axis([-5, 5, -10, 0])

for iter in range(max_iter):
    e_iter = 0
    for i in range(N):
        xi = X[i].reshape(1, 3)
        ti = t[i]
        yi = sigmoid(w.T @ xi.T)
        e_i = -ti * np.log(yi + eps) - (1 - ti) * np.log(1 - yi + eps)
        e_iter += e_i
        grad_e = (yi - ti) * xi
        w_old = w.copy()
        w = w - eta * grad_e.T

        # Plot slope-intercept path ONLY in last 3 iterations
        if iter >= max_iter - 3 and i == 0:
            plt.figure(2)
            plotMB(plt, w, w_old)

    e_all.append(e_iter.item())

    # Plot separator in data space ONLY in last 3 iterations
    if iter >= max_iter - 3:
        plt.figure(1)
        plt.clf()
        plt.rcParams['font.size'] = 20
        plt.plot(X1[:, 0], X1[:, 1], 'g.')
        plt.plot(X2[:, 0], X2[:, 1], 'b.')
        drawSep(plt, w)
        plt.title(f'Separator in data space (Iter {iter})')
        plt.axis([-5, 15, -10, 10])
        plt.draw()
        plt.pause(1e-17)
        plt.savefig(f"sgd_separator_iter_{iter}.png")

    print(f"Iter {iter}, Negative Log-Likelihood: {e_iter.item():.4f}, w={w.T}")

    if iter > 0 and abs(e_all[-1] - e_all[-2]) < tol:
        break

# Final Error Plot
plt.figure(3, figsize= (8,6))
plt.rcParams['font.size'] = 20
plt.plot(e_all, 'b-')
plt.xlabel('Iteration')
plt.ylabel('neg. log likelihood')
plt.title('SGD: Minimize Negative Log-Likelihood')
plt.savefig("sgd_error_plot.png", bbox_inches='tight')
plt.show()
