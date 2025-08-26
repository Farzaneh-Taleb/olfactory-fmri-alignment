import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from .config import SEED

def custom_ridge_regression(X, y, alpha=None):
    """
    Custom ridge regression with optional alpha parameter.
    
    Args:
        X: Feature matrix
        y: Target values
        alpha: Regularization parameter. If None, uses cross-validation.
        
    Returns:
        Fitted estimator
    """
    if alpha is None:
        linreg = RidgeCV(alphas=np.logspace(2, 7, 16), cv=5)
    else:
        linreg = Ridge(alpha=alpha)
    
    estimator = linreg.fit(X, y)
    return estimator



def train_and_eval_prekfold(
    X_train, y_train, X_test, y_test, n_components=None,
    *, n_permutations: int = 1000, use_abs_corr: bool = False
):
    """
    Train ridge models per pre-defined fold, evaluate with SciPy Pearson r, and
    estimate permutation p-values.

    Returns:
        predicteds, y_tests, correlations, mse_errors, p_value_correlation, p_value_mse, targets
    """
    rng = np.random.default_rng(SEED)
    print(X_train.shape,y_train.shape)

    # preds_list, y_list, target_ids = [], [], []

    
    # Optional PCA (fit on train only)
    if n_components is not None and n_components < X_train.shape[1]:
        pca = PCA(n_components=n_components, random_state=SEED)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Fit & predict (expects custom_ridge_regression in scope)
    model = custom_ridge_regression(X_train, y_train, None)
    predicted = model.predict(X_test)
    

    # preds_list.append(pred)
    # y_tests.append(y_test)

    n_te, n_vox = predicted.shape
    targets = np.repeat(np.arange(n_vox), n_te)
    # # Stack folds
    # predicteds = np.vstack(preds_list)     # (N_total, V)
    # y_tests    = np.vstack(y_list)         # (N_total, V)
    # targets    = np.concatenate(target_ids)  # (N_total * V,)

    V = y_test.shape[1]
    N = y_test.shape[0]
    # --- Metrics with SciPy ---
    # MSE per target (vectorized)
    mse_errors = ((predicted - y_test) ** 2).mean(axis=0)

    # Pearson r per target using scipy.stats.pearsonr
    correlations = np.empty(V, dtype=float)
    for j in range(V):
        r, _ = pearsonr(predicted[:, j], y_test[:, j])
        correlations[j] = r

    # --- Permutation tests (row-wise shuffle of y across samples) ---
    r_obs = np.abs(correlations) if use_abs_corr else correlations
    p_corr_counts = np.zeros(V, dtype=float)
    p_mse_counts  = np.zeros(V, dtype=float)

    for _ in range(n_permutations):
        perm = rng.permutation(N)
        y_perm = y_test[perm, :]

        # Permuted MSE (vectorized)
        mse_perm = ((predicted - y_perm) ** 2).mean(axis=0)

        # Permuted Pearson r via SciPy
        r_perm = np.empty(V, dtype=float)
        for j in range(V):
            rp, _ = pearsonr(predicted[:, j], y_perm[:, j])
            r_perm[j] = abs(rp) if use_abs_corr else rp

        # Count extreme permutations
        p_corr_counts += (r_perm >= r_obs)
        p_mse_counts  += (mse_perm <= mse_errors)

    denom = max(1, n_permutations)
    p_value_correlation = p_corr_counts / denom
    p_value_mse         = p_mse_counts  / denom

    return predicted, y_test, correlations, mse_errors, p_value_correlation, p_value_mse, targets


import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

# Assumes you have: SEED and custom_ridge_regression in scope


def compute_targetwise_metrics(predicted: np.ndarray, y_true: np.ndarray):
    """
    Compute per-voxel Pearson r (SciPy) and MSE.

    Args:
        predicted: (N, V) predictions
        y_true:    (N, V) ground truth

    Returns:
        correlations: (V,) Pearson r per voxel
        mse_errors:   (V,) MSE per voxel
    """
    V = y_true.shape[1]

    # MSE per voxel (vectorized)
    mse_errors = ((predicted - y_true) ** 2).mean(axis=0)

    # Pearson r per voxel (SciPy)
    correlations = np.empty(V, dtype=float)
    for j in range(V):
        r, _ = pearsonr(predicted[:, j], y_true[:, j])
        correlations[j] = r

    return correlations, mse_errors


def permutation_test_metrics(
    predicted: np.ndarray,
    y_true: np.ndarray,
    correlations: np.ndarray,
    mse_errors: np.ndarray,
    *,
    n_permutations: int = 1000,
    use_abs_corr: bool = False,
    seed: int | None = None,
):
    """
    Permutation test for voxelwise correlation and MSE by shuffling rows of y_true.

    Args:
        predicted:       (N, V) predictions
        y_true:          (N, V) ground truth
        correlations:    (V,) observed Pearson r per voxel
        mse_errors:      (V,) observed MSE per voxel
        n_permutations:  number of label permutations
        use_abs_corr:    whether to use |r| for the correlation test
        seed:            RNG seed for reproducibility

    Returns:
        p_value_correlation: (V,) permutation p-values for r (or |r|)
        p_value_mse:         (V,) permutation p-values for MSE
    """
    rng = np.random.default_rng(seed)
    N, V = y_true.shape

    r_obs = np.abs(correlations) if use_abs_corr else correlations
    p_corr_counts = np.zeros(V, dtype=float)
    p_mse_counts  = np.zeros(V, dtype=float)

    for _ in range(n_permutations):
        perm = rng.permutation(N)
        y_perm = y_true[perm, :]

        # Permuted MSE
        mse_perm = ((predicted - y_perm) ** 2).mean(axis=0)

        # Permuted Pearson r
        r_perm = np.empty(V, dtype=float)
        for j in range(V):
            rp, _ = pearsonr(predicted[:, j], y_perm[:, j])
            r_perm[j] = abs(rp) if use_abs_corr else rp

        # Count extreme permutations
        p_corr_counts += (r_perm >= r_obs)
        p_mse_counts  += (mse_perm <= mse_errors)

    denom = max(1, n_permutations)
    p_value_correlation = p_corr_counts / denom
    p_value_mse         = p_mse_counts  / denom
    return p_value_correlation, p_value_mse


def train_and_eval_prekfold(
    X_train, y_train, X_test, y_test, n_components=None,
    *, n_permutations: int = 1000, use_abs_corr: bool = False
):
    """
    Train ridge models per pre-defined fold, evaluate with SciPy Pearson r, and
    estimate permutation p-values.

    Returns:
        predicted, y_test, correlations, mse_errors, p_value_correlation, p_value_mse, targets
    """
    rng = np.random.default_rng(SEED)
    print(X_train.shape, y_train.shape)

    # Optional PCA (fit on train only)
    if n_components is not None and n_components < X_train.shape[1]:
        pca = PCA(n_components=n_components, random_state=SEED)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Fit & predict (expects custom_ridge_regression in scope)
    model = custom_ridge_regression(X_train, y_train, None)
    predicted = model.predict(X_test)

    # Targets indexing (per-voxel label for each row)
    n_te, n_vox = predicted.shape
    targets = np.repeat(np.arange(n_vox), n_te)

    # Metrics
    correlations, mse_errors = compute_targetwise_metrics(predicted, y_test)

    # Permutation tests (separate function)
    p_value_correlation, p_value_mse = permutation_test_metrics(
        predicted,
        y_test,
        correlations,
        mse_errors,
        n_permutations=n_permutations,
        use_abs_corr=use_abs_corr,
        seed=SEED,
    )

    return predicted, y_test, correlations, mse_errors, p_value_correlation, p_value_mse, targets


def pipeline(Xs_train, ys_train, Xs_test, ys_test, n_components=None):
    """
    Run the pipeline for regression analysis.

    Parameters:
        Xs_train: List of training feature matrices
        ys_train: List of training target matrices  
        Xs_test: List of test feature matrices
        ys_test: List of test target matrices
        voxels_retained: Retained voxel indices
        n_components: Number of components for dimensionality reduction

    Returns:
        DataFrame: Metrics containing correlations, MSE, targets, and p-values
    """
    predicteds, y_tests, correlations, mse_errors, p_value_correlation, p_value_mse, targets = \
    train_and_eval_prekfold(Xs_train, ys_train, Xs_test, ys_test, n_components=n_components)

    V = y_tests.shape[1]
    target_ids = np.arange(V)  # per-voxel identifier

    metrics_df = pd.DataFrame({
    "target": target_ids,
    "correlation": correlations,           # 1-D, length V
    "mse": mse_errors,                     # 1-D, length V
    "p_value_correlation": p_value_correlation,  # 1-D, length V
    "p_value_mse": p_value_mse,                  # 1-D, length V
})

    return metrics_df


def compute_correlation(Xs_train, ys_train, Xs_test, ys_test, n_components=None):
    """
    Compute correlations for regression analysis.

    Parameters:
        Xs_train: List of training feature matrices
        ys_train: List of training target matrices  
        Xs_test: List of test feature matrices
        ys_test: List of test target matrices
        voxels_retained: Retained voxel indices
        n_components: Number of components for dimensionality reduction

    Returns:
        DataFrame: Metrics from pipeline analysis
    """
    results = pipeline(Xs_train, ys_train, Xs_test, ys_test, n_components=n_components)
    return results