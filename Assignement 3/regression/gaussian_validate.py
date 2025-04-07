import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from utils import loadData, normalizeData

# Gaussian kernel function
def gaussian_kernel(u, h):
    return (1 / (np.sqrt(2 * np.pi) * h)) * np.exp(- (u ** 2) / (2 * h ** 2))

# Kernel regression
def kernel_regression(x_train, t_train, x_eval, h):
    dists = x_eval[:, None] - x_train[None, :]
    weights = gaussian_kernel(dists, h)
    numerator = weights @ t_train
    denominator = np.sum(weights, axis=1) + np.finfo(float).eps
    return numerator / denominator

# Load and normalize data
t, X = loadData()
X_n = normalizeData(X)
t = normalizeData(t)

# Use only the 3rd feature
x = X_n[:, 2]
n = len(x)

# Bandwidth values to evaluate
h_values = [0.01, 0.1, 0.25, 1, 2, 3, 4]
validation_errors = []

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for h in h_values:
    fold_errors = []
    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        t_train, t_val = t[train_index], t[val_index]

        t_pred = kernel_regression(x_train, t_train, x_val, h)
        error = np.mean((t_pred - t_val) ** 2)
        fold_errors.append(error)

    avg_error = np.mean(fold_errors)
    validation_errors.append(avg_error)
    print(f"h = {h}, Avg Validation MSE = {avg_error:.4f}")

# Plot validation error vs. h (log scale)
plt.figure(figsize=(8, 5))
plt.semilogx(h_values, validation_errors, marker='o', linewidth=2)
plt.xlabel('Bandwidth h (log scale)')
plt.ylabel('Average Validation MSE')
plt.title('10-Fold Cross-Validation Error vs. Bandwidth h')
plt.grid(True)
plt.tight_layout()
plt.show()
