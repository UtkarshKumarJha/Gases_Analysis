import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def refined_index_of_agreement(o, p):
    mask = ~np.isnan(o) & ~np.isnan(p)
    o = o[mask]
    p = p[mask]
    
    o_bar = np.mean(o)
    numerator = np.sum(np.abs(p - o))
    denominator = 2 * np.sum(np.abs(o - o_bar))
    
    if numerator <= denominator:
        return 1 - (numerator / denominator)
    else:
        return (denominator / numerator) - 1

def calculate_metrics(y_true, y_pred, y_baseline=None):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    ria = refined_index_of_agreement(y_true, y_pred)
    
    metrics = {'RMSE': rmse, 'R2': r2, 'RIA': ria}
    
    if y_baseline is not None:
        base_rmse = np.sqrt(mean_squared_error(y_true, y_baseline))
        metrics['Baseline_RMSE'] = base_rmse
        metrics['Improvement'] = base_rmse - rmse
        
    return metrics