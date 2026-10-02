import numpy as np

FEATURE_SETS = {
    'S1': ['DQ_relative_logvar'],
    'S2': ['DQ_relative_logvar', 'C1', 'C2', 'Switch_SOC'],
    'S3': ['DQ_relative_logvar', 'C1', 'C2', 'Switch_SOC',
           'capacity_retention_100_pct', 'IR_10', 'Tavg_median_early'],
    'S2_plus_ratio': ['DQ_relative_logvar', 'C1', 'C2', 'Switch_SOC', 'C1_to_C2_ratio'],
    'S2_original_DQ': ['DQ_logvar', 'C1', 'C2', 'Switch_SOC'],
}

def derive_early_features(q10, q100, delta, voltage, c1, c2):
    """No lifetime labels or post-cycle-100 observations are used here."""
    valid = np.isfinite(q10) and q10 > 0
    rel = np.asarray(delta, dtype=float) / q10 if valid and delta is not None else None
    if rel is not None:
        order = np.argsort(voltage)
        v = np.asarray(voltage)[order]
        loss = np.trapezoid(np.maximum(-rel[order], 0), v) / (v[-1]-v[0])
        logvar = np.log10(max(float(np.var(rel)), 1e-16))
    else:
        loss = logvar = np.nan
    return {
        'DQ_relative_logvar': logvar, 'DQ_relative_loss_area': loss,
        'capacity_retention_100_pct': 100*q100/q10 if valid else np.nan,
        'C1_to_C2_ratio': c1/c2 if np.isfinite(c2) and c2 > 0 else np.nan,
    }

def check_redundant_features(df):
    cols = ['DQ_logvar', 'DQ_relative_logvar', 'DQ_mean',
            'DQ_relative_loss_area', 'Qd_change_10_100', 'capacity_retention_100_pct']
    return df.loc[df.split.eq('train'), cols].corr(method='spearman')
