"""Run in BITS Virtual Lab to reproduce every saved artifact without downloads."""
import json
import platform
import hashlib
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from common import ROOT, MODEL_FILES, evaluate


def main():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    # sklearn uses malignant=0. Explicitly remap so malignant is the positive class.
    y = (dataset.target == 0).astype(int).rename('target')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    # Hyperparameters are fixed before evaluation; the holdout is never used to tune.
    models = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=42)),
        'Decision Tree': make_pipeline(DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42)),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        'Naive Bayes': make_pipeline(StandardScaler(), GaussianNB()),
        'Support Vector Machine': make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', probability=True, random_state=42)),
    }
    (ROOT / 'model').mkdir(exist_ok=True)
    (ROOT / 'reports').mkdir(exist_ok=True)
    X_train.assign(target=y_train).to_csv(ROOT / 'train_data.csv', index=False)
    X_test.assign(target=y_test).to_csv(ROOT / 'test_data.csv', index=False)
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, ROOT / 'model' / MODEL_FILES[name])
        metrics, _, _ = evaluate(model, X_test, y_test)
        rows.append({'ML Model Name': name, **metrics})
    results = pd.DataFrame(rows)
    results.to_csv(ROOT / 'reports' / 'metrics.csv', index=False)
    metadata = {
        'dataset': 'UCI Breast Cancer Wisconsin (Diagnostic)',
        'source': 'https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic',
        'features': list(X.columns), 'target': {'0': 'benign', '1': 'malignant'},
        'total_rows': len(X), 'train_rows': len(X_train), 'test_rows': len(X_test),
        'split_seed': 42, 'test_fraction': 0.20,
        'train_class_counts': {str(k): int(v) for k, v in y_train.value_counts().items()},
        'test_class_counts': {str(k): int(v) for k, v in y_test.value_counts().items()},
        'train_indices': X_train.index.tolist(), 'test_indices': X_test.index.tolist(),
        'versions': {'python': platform.python_version(), 'scikit-learn': sklearn.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__},
        'test_csv_sha256': hashlib.sha256((ROOT / 'test_data.csv').read_bytes()).hexdigest(),
    }
    (ROOT / 'model' / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(results.to_string(index=False, float_format=lambda value: f'{value:.4f}'))
    print(f'\nSaved five pipelines. Train: {len(X_train)}, test: {len(X_test)}. Positive class: malignant.')


if __name__ == '__main__':
    main()
