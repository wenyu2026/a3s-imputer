"""Example: A3S / LA3S on synthetic alloy data."""
import numpy as np
from a3s_imputer import a3s, la3s, auto_a3s, check_compositional_validity, compositional_mae

# Generate synthetic steel-like compositions (Fe dominant, trace elements)
np.random.seed(42)
n_samples = 100
n_elements = 5  # Fe, C, Mn, Cr, Ni

# Fe is dominant (~70%), others are minor
X = np.random.dirichlet(np.ones(n_elements) * 0.5, size=n_samples) * 30.0
Fe = 100.0 - X.sum(axis=1, keepdims=True)
Fe = np.clip(Fe, 0.0, 100.0)
X = np.concatenate([X, Fe], axis=1)
row_sums = X.sum(axis=1, keepdims=True)
X = X / row_sums * 100.0

print("Original data shape:", X.shape)
print("First 3 samples:")
print(X[:3])

# Introduce 10% missing values
missing_rate = 0.10
n_missing = int(X.size * missing_rate)
missing_indices = np.random.choice(X.size, n_missing, replace=False)
X_missing = X.copy()
X_missing.flat[missing_indices] = np.nan

print(f"\nMissing rate: {missing_rate:.0%}")
print("Missing positions (first 5):")
print(np.argwhere(np.isnan(X_missing))[:5])

# Impute with A3S
X_a3s = a3s(X_missing, comp_idx=list(range(6)), total=100.0)
valid_a3s = check_compositional_validity(X_a3s, comp_idx=list(range(6)), total=100.0)
print("\n--- A3S Results ---")
print(f"Sum deviation: {valid_a3s['sum_dev_mean']:.2e}")
print(f"Valid: {valid_a3s['valid']}")

# Impute with LA3S
X_la3s = la3s(X_missing, comp_idx=list(range(6)), total=100.0, n_neighbors=5)
valid_la3s = check_compositional_validity(X_la3s, comp_idx=list(range(6)), total=100.0)
print("\n--- LA3S Results ---")
print(f"Sum deviation: {valid_la3s['sum_dev_mean']:.2e}")
print(f"Valid: {valid_la3s['valid']}")

# Compare accuracy (only on missing positions)
mask = np.isnan(X_missing)
if mask.any():
    mae_a3s = compositional_mae(X, X_a3s, mask)
    mae_la3s = compositional_mae(X, X_la3s, mask)
    print(f"\n--- Accuracy (MAE on missing positions) ---")
    print(f"A3S MAE:  {mae_a3s['overall_mae']:.4f}")
    print(f"LA3S MAE: {mae_la3s['overall_mae']:.4f}")

print("\n--- First 3 imputed samples (A3S) ---")
print(X_a3s[:3].round(2))
print("\n--- First 3 imputed samples (LA3S) ---")
print(X_la3s[:3].round(2))
