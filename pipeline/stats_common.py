"""Bootstrap AUC confidence intervals and paired AUC differences."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc_ci(y, p, n=2000, seed=0):
    y, p = np.asarray(y), np.asarray(p); ok = np.isfinite(p); y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return (np.nan,) * 3 + (len(y), int(y.sum()))
    a = roc_auc_score(y, p); rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            bs.append(roc_auc_score(y[i], p[i]))
    return (a, *np.percentile(bs, [2.5, 97.5]), len(y), int(y.sum()))


def fmt(r):
    a, lo, hi, n, k = r
    return "   n/a" if not np.isfinite(a) else f"{a:.3f} [{lo:.3f}, {hi:.3f}]  ({k}+/{n-k}-)"


def paired_delta(y, p1, p2, n=2000, seed=0):
    y, p1, p2 = map(np.asarray, (y, p1, p2)); rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    d = np.array(d)
    return (roc_auc_score(y, p1) - roc_auc_score(y, p2), *np.percentile(d, [2.5, 97.5]),
            max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / n))
