import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from .config import SEED
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold
# def custom_ridge_regression(X, y, alpha=None):
#     """
#     Custom ridge regression with optional alpha parameter.
    
#     Args:
#         X: Feature matrix
#         y: Target values
#         alpha: Regularization parameter. If None, uses cross-validation.
        
#     Returns:
#         Fitted estimator
#     """
#     if alpha is None:
#         linreg = RidgeCV(alphas=np.logspace(2, 7, 16), cv=5)
#     else:
#         linreg = Ridge(alpha=alpha)
    
#     estimator = linreg.fit(X, y)
#     return estimator


def _mean_pearson_corr(Y_true, Y_pred):
    Y_true = np.asarray(Y_true); Y_pred = np.asarray(Y_pred)
    if Y_true.ndim == 1: 
        Y_true = Y_true[:, None]; Y_pred = Y_pred[:, None]
    r_vals = []
    for j in range(Y_true.shape[1]):
        if np.std(Y_true[:, j]) == 0 or np.std(Y_pred[:, j]) == 0:
            r_vals.append(0.0)
        else:
            r, _ = pearsonr(Y_true[:, j], Y_pred[:, j])
            r_vals.append(r)
    return float(np.nanmean(r_vals))


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

  
    corr_scorer = make_scorer(_mean_pearson_corr, greater_is_better=True)

    
    linreg = RidgeCV(alphas=np.logspace(-4, 8, 25), scoring=corr_scorer)
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)

    oof_pred = np.zeros_like(y, dtype=float)
    oof_mask = np.zeros(len(X), dtype=bool)
    alphas = []

    for tr, te in splitter:
        model = linreg.fit(X[tr], y[tr])
        oof_pred[te] = model.predict(X[te])
        oof_mask[te] = True
        alphas.append(model.alpha_)

    r_target = []
    for j in range(y.shape[1]):
        if np.std(y[oof_mask, j]) == 0 or np.std(oof_pred[oof_mask, j]) == 0:
            r_target.append(0.0)
        else:
            r, _ = pearsonr(y[oof_mask, j], oof_pred[oof_mask, j])
            r_target.append(r)
    # results = pd.DataFrame({"target": np.arange(y.shape[1]), "r": r_target})
    # mean_r = float(np.mean(r_target))
    alpha_median = float(np.median(alphas))

    final_model = Ridge(alpha=alpha_median)
    final_model = final_model.fit(X, y)

    
    estimator = final_model.fit(X, y)
    return estimator



# def train_and_eval_prekfold(
#     X_train, y_train, X_test, y_test, n_components=None,
#     *, n_permutations: int = 1000, use_abs_corr: bool = False
# ):
#     """
#     Train ridge models per pre-defined fold, evaluate with SciPy Pearson r, and
#     estimate permutation p-values.

#     Returns:
#         predicteds, y_tests, correlations, mse_errors, p_value_correlation, p_value_mse, targets
#     """
#     rng = np.random.default_rng(SEED)
#     print(X_train.shape,y_train.shape)

#     # preds_list, y_list, target_ids = [], [], []

    
#     # Optional PCA (fit on train only)
#     if n_components is not None and n_components < X_train.shape[1]:
#         pca = PCA(n_components=n_components, random_state=SEED)
#         X_train = pca.fit_transform(X_train)
#         X_test = pca.transform(X_test)

#     # Fit & predict (expects custom_ridge_regression in scope)
#     model = custom_ridge_regression(X_train, y_train, None)
#     predicted = model.predict(X_test)
    

#     # preds_list.append(pred)
#     # y_tests.append(y_test)

#     n_te, n_vox = predicted.shape
#     targets = np.repeat(np.arange(n_vox), n_te)
#     # # Stack folds
#     # predicteds = np.vstack(preds_list)     # (N_total, V)
#     # y_tests    = np.vstack(y_list)         # (N_total, V)
#     # targets    = np.concatenate(target_ids)  # (N_total * V,)

#     V = y_test.shape[1]
#     N = y_test.shape[0]
#     # --- Metrics with SciPy ---
#     # MSE per target (vectorized)
#     mse_errors = ((predicted - y_test) ** 2).mean(axis=0)

#     # Pearson r per target using scipy.stats.pearsonr
#     correlations = np.empty(V, dtype=float)
#     for j in range(V):
#         r, _ = pearsonr(predicted[:, j], y_test[:, j])
#         correlations[j] = r

#     # --- Permutation tests (row-wise shuffle of y across samples) ---
#     r_obs = np.abs(correlations) if use_abs_corr else correlations
#     p_corr_counts = np.zeros(V, dtype=float)
#     p_mse_counts  = np.zeros(V, dtype=float)

#     for _ in range(n_permutations):
#         perm = rng.permutation(N)
#         y_perm = y_test[perm, :]

#         # Permuted MSE (vectorized)
#         mse_perm = ((predicted - y_perm) ** 2).mean(axis=0)

#         # Permuted Pearson r via SciPy
#         r_perm = np.empty(V, dtype=float)
#         for j in range(V):
#             rp, _ = pearsonr(predicted[:, j], y_perm[:, j])
#             r_perm[j] = abs(rp) if use_abs_corr else rp

#         # Count extreme permutations
#         p_corr_counts += (r_perm >= r_obs)
#         p_mse_counts  += (mse_perm <= mse_errors)

#     denom = max(1, n_permutations)
#     p_value_correlation = p_corr_counts / denom
#     p_value_mse         = p_mse_counts  / denom

#     return predicted, y_test, correlations, mse_errors, p_value_correlation, p_value_mse, targets


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
    X_train, y_train, X_test, n_components=None, z_score=False
):
    """
    Train ridge models per pre-defined fold, evaluate with SciPy Pearson r, and
    estimate permutation p-values.

    Returns:
        predicted, y_test, correlations, mse_errors, p_value_correlation, p_value_mse, targets
    """
    # rng = np.random.default_rng(SEED)
    print(X_train.shape, y_train.shape)

    
    
    
    # Optional PCA (fit on train only)

    if z_score:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    if n_components is not None and n_components < X_train.shape[1]:
        pca = PCA(n_components=n_components, random_state=SEED)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    

    # Fit & predict (expects custom_ridge_regression in scope)
    model = custom_ridge_regression(X_train, y_train, None)
    predicted = model.predict(X_test)
    print("predited",predicted.shape)

    # Targets indexing (per-voxel label for each row)
    # n_te, n_vox = predicted.shape
    # targets = np.repeat(np.arange(n_vox), n_te)


    # Metrics

    return predicted


def pipeline(Xs_train, ys_train, Xs_test, n_components=None,z_score=False):
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
    predicteds = train_and_eval_prekfold(Xs_train, ys_train, Xs_test, n_components=n_components,z_score=z_score)
    return predicteds
    


def compute_correlation(Xs_train, ys_train, Xs_test, ys_test, n_components=None,z_score=False):
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
    y_list = []
    preds_list=[]
    for X_train,y_train,X_test,y_test in zip(Xs_train, ys_train, Xs_test, ys_test):
        predicted = pipeline(X_train, y_train, X_test, n_components=n_components,z_score=z_score)
        preds_list.append(predicted)
        y_list.append(y_test)
        print()

    P = np.vstack(preds_list)   # (N_total, V)
    Y = np.vstack(y_list)       # (N_total, V)
    correlations, mse_errors = compute_targetwise_metrics(P, Y)

    # Permutation tests (separate function)
    p_value_correlation, p_value_mse = permutation_test_metrics(
        P,
        Y,
        correlations,
        mse_errors,
        n_permutations=1000,
        use_abs_corr=False,
        seed=SEED,
    )

    V = Y.shape[1]
    target_ids = np.arange(V)  # per-voxel identifier

    metrics_df = pd.DataFrame({
    "target_id": target_ids,
    "correlation": correlations,           # 1-D, length V
    "mse": mse_errors,                     # 1-D, length V
    "p_value_correlation": p_value_correlation,  # 1-D, length V
    "p_value_mse": p_value_mse,                  # 1-D, length V
})

    return metrics_df
