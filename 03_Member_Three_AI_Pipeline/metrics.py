"""Dependency-free binary-classification metrics for baseline experiment reports."""
def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError('Both non-empty arrays must have the same length')
    tp = sum(a == b == 1 for a, b in zip(y_true, y_pred)); tn = sum(a == b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred)); fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    return {'sensitivity': recall, 'specificity': tn / (tn + fp) if tn + fp else 0.0, 'precision': precision, 'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
