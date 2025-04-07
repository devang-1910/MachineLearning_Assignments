import numpy as np
import matplotlib.pyplot as plt
from utils import loadData, normalizeData

# Epanechnikov kernel function
def epanechnikov_kernel(u, h):
    u_scaled = u / h
    mask = np.abs(u_scaled) <= 1
    result = np.zeros_like(u)
    result[mask] = 0.75 * (1 - u_scaled[mask]**2)
    return result

# Kernel regression
def kernel_regression(x_train, t_train, x_eval, h):
    dists = x_eval[:, None] - x_train[None, :]
    weights = epanechnikov_kernel(dists, h)
    numerator = weights @ t_train
    denominator = np.sum(weights, axis=1) + np.finfo(float).eps
    return numerator / denominator

# Load and normalize data
t, X = loadData()
X_n = normalizeData(X)
t = normalizeData(t)

# Use only the 3rd feature
x = X_n[:, 2]

# Train/test split
x_train = x[:100]
t_train = t[:100]
x_test = x[100:]
t_test = t[100:]

# Evaluation points for plotting
x_eval = np.linspace(np.min(x), np.max(x), 300)

# Bandwidth values
h_values = [0.01, 0.1, 1, 2, 3, 4]

# Plot regression for each h
for h in h_values:
    t_pred = kernel_regression(x_train, t_train, x_eval, h)

    plt.figure(figsize=(8, 5))
    plt.plot(x_eval, t_pred, label=f"Regression (h={h})", linewidth=2)
    plt.scatter(x_train, t_train, c='red', label='Training Data', s=20)
    plt.scatter(x_test, t_test, c='green', label='Test Data', s=20, alpha=0.6)
    plt.xlabel('Normalized Feature (3rd)')
    plt.ylabel('Normalized Target (MPG)')
    plt.title(f'Epanechnikov Kernel Regression (h={h})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
