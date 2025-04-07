import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from utils import loadData, normalizeData


# Epanechnikov kernel function
def epanechnikov_kernel(u, h):
    u_scaled = u / h
    result = np.zeros_like(u)
    mask = np.abs(u_scaled) <= 1
    result[mask] = 0.75 * (1 - u_scaled[mask] ** 2)
    return result


# Kernel regression using Epanechnikov
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
x = X_n[:, 2]  # use only the 3rd feature

# Cross-validation settings
h_values = [0.01, 0.1, 0.25, 1, 2, 3, 4]
validation_errors = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Run CV for each h
for h in h_values:
    fold_errors = []
    for train_idx, val_idx in kf.split(x):
        x_train, x_val = x[train_idx], x[val_idx]
        t_train, t_val = t[train_idx], t[val_idx]

        t_pred = kernel_regression(x_train, t_train, x_val, h)
        mse = np.mean((t_pred - t_val) ** 2)
        fold_errors.append(mse)

    avg_mse = np.mean(fold_errors)
    validation_errors.append(avg_mse)
    print(f"h = {h}, Avg Validation MSE = {avg_mse:.4f}")

# Plot the results
plt.figure(figsize=(8, 5))
plt.semilogx(h_values, validation_errors, marker='o', linewidth=2)
plt.xlabel('Bandwidth h (log scale)')
plt.ylabel('Average Validation MSE')
plt.title('10-Fold CV Error vs. h (Epanechnikov Kernel)')
plt.grid(True)
plt.tight_layout()
plt.show()
