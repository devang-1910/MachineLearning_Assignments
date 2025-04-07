import numpy as np
import matplotlib.pyplot as plt
from utils import loadData, normalizeData


# Gaussian kernel function
def gaussian_kernel(u, h):
    return (1 / (np.sqrt(2 * np.pi) * h)) * np.exp(- (u ** 2) / (2 * h ** 2))


# Kernel regression implementation
def kernel_regression(x_train, t_train, x_eval, h):
    # Compute pairwise distances (broadcasting works since both are 1D)
    dists = x_eval[:, None] - x_train[None, :]
    weights = gaussian_kernel(dists, h)

    # Numerator and denominator for each evaluation point
    numerator = weights @ t_train
    denominator = np.sum(weights, axis=1) + np.finfo(float).eps  # avoid div by zero
    return numerator / denominator


# Load and normalize data
t, X = loadData()
X_n = normalizeData(X)
t = normalizeData(t)

# Use only 3rd feature (index 2), shape = (n,)
x = X_n[:, 2]

# Split train and test
x_train = x[:100]
t_train = t[:100]
x_test = x[100:]
t_test = t[100:]

# Evaluation points for smooth plot
x_eval = np.linspace(np.min(x), np.max(x), 300)

# Bandwidth values to experiment with
h_values = [0.01, 0.1, 1, 2, 3, 4]

# Create plots for selected h values
for h in h_values:
    t_pred = kernel_regression(x_train, t_train, x_eval, h)

    plt.figure(figsize=(8, 5))
    plt.plot(x_eval, t_pred, label=f"Regression (h={h})", linewidth=2)
    plt.scatter(x_train, t_train, c='red', label='Training Data', s=20)
    plt.scatter(x_test, t_test, c='green', label='Test Data', s=20, alpha=0.6)
    plt.xlabel('Normalized Feature (3rd)')
    plt.ylabel('Normalized Target (MPG)')
    plt.title(f'Kernel Regression with Gaussian Kernel (h={h})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
