import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import *
import time

# Parameters
max_iter = 500
tol = 0.01
eta = 0.001  # Start small for stability
lambdas = [0.1, 1, 10, 100]

# Load data
data = loadmat('data.mat')
X1, X2 = data['X1'], data['X2']
X = np.vstack((X1, X2))
X = np.hstack((X, np.ones((X.shape[0], 1))))
t = np.vstack((np.zeros((X1.shape[0], 1)), np.ones((X2.shape[0], 1))))

for lam in lambdas:
    print(f"\nRunning with λ = {lam}")
    w = np.array([1., 0., 0.]).reshape(3, 1)
    e_all = []

    for iter in range(max_iter):
        y = sigmoid(w.T @ X.T).T
        nll = -np.sum(t * np.log(y + eps) + (1 - t) * np.log(1 - y + eps))
        reg_term = 0.5 * lam * np.sum(w**2)
        e_total = nll + reg_term
        e_all.append(e_total)

        grad_e = np.sum((y - t) * X, 0, keepdims=True).T  # 3x1
        grad_reg = lam * w  # L2 regularization gradient
        grad_total = grad_e + grad_reg

        w_old = w.copy()
        w = w - eta * grad_total

        if iter > 0 and abs(e_all[-1] - e_all[-2]) < tol:
            break

    norm_w = np.linalg.norm(w)
    print(f"Final Negative Log-Likelihood: {nll:.4f}")
    print(f"||w||: {norm_w:.4f}")

    # Optional: Plot final separator for each λ
    plt.figure()
    plt.plot(X1[:, 0], X1[:, 1], 'g.')
    plt.plot(X2[:, 0], X2[:, 1], 'b.')
    drawSep(plt, w)
    plt.title(f'Separator with λ = {lam}')
    plt.axis([-5, 15, -10, 10])
    plt.savefig(f"separator_lambda_{lam}.png", bbox_inches='tight')
    plt.close()

    # Plot error
    plt.figure()
    plt.plot(e_all, 'b-')
    plt.title(f'Neg. Log-Likelihood + Reg (λ={lam})')
    plt.xlabel('Iteration')
    plt.ylabel('Total Cost')
    plt.savefig(f"error_lambda_{lam}.png", bbox_inches='tight')
    plt.close()
