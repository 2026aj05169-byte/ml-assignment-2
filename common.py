"""Shared schema checks and binary metrics. Target 1 always means malignant."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Support Vector Machine': 'svm.joblib',
}
METRICS = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']


def validate_data(frame, features):
    """Reject malformed uploads; return columns in the model's training order."""
    if frame.empty:
        raise ValueError('CSV has no data rows.')
    if len(frame) > 10000:
        raise ValueError('Upload at most 10,000 test rows.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not allowed.')
    missing = sorted(set(features + ['target']) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(features + ['target']))
    if missing or extra:
        raise ValueError(f'Column mismatch. Missing: {missing}. Unexpected: {extra}. Use the sample CSV headers.')
    try:
        numeric = frame[features + ['target']].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('Features and target must be numeric.') from exc
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError('Missing, infinite, and NaN values are not allowed.')
    if not numeric['target'].isin([0, 1]).all():
        raise ValueError('Target must be 0 (benign) or 1 (malignant).')
    return numeric[features], numeric['target'].astype(int)


def evaluate(model, X, y):
    predictions = model.predict(X)
    # Select the probability column by label, rather than assuming its position.
    positive_index = list(model.classes_).index(1)
    scores = model.predict_proba(X)[:, positive_index]
    values = {
        'Accuracy': accuracy_score(y, predictions),
        'AUC': roc_auc_score(y, scores) if y.nunique() == 2 else float('nan'),
        'Precision': precision_score(y, predictions, zero_division=0),
        'Recall': recall_score(y, predictions, zero_division=0),
        'F1': f1_score(y, predictions, zero_division=0),
        'MCC': matthews_corrcoef(y, predictions),
    }
    return values, predictions, scores
