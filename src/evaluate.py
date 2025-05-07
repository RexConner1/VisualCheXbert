import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score

def compute_metrics(y_true, y_pred):
    """
    y_true, y_pred: numpy arrays of shape (N, C), binary
    Returns per-class F1, macro-F1, weighted-F1, average Kappa.
    """
    f1_per = f1_score(y_true, y_pred, average=None)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    kappas = [cohen_kappa_score(y_true[:,i], y_pred[:,i]) for i in range(y_true.shape[1])]
    return f1_per, macro_f1, weighted_f1, np.mean(kappas)

def bootstrap_ci(y_true, y_pred_a, y_pred_b, n_bootstrap=1000, alpha=0.05):
    """
    Returns (lower, upper) CI for macro-F1 difference between a and b.
    """
    rng = np.random.RandomState(42)
    diffs = []
    N = y_true.shape[0]
    for _ in range(n_bootstrap):
        idx = rng.choice(N, N, replace=True)
        _, mfa, _, _ = compute_metrics(y_true[idx], y_pred_a[idx])
        _, mfb, _, _ = compute_metrics(y_true[idx], y_pred_b[idx])
        diffs.append(mfa - mfb)
    lower = np.percentile(diffs, 100 * alpha/2)
    upper = np.percentile(diffs, 100 * (1-alpha/2))
    return lower, upper
