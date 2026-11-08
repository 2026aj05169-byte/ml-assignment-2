"""Independent artifact, schema, metric, and Streamlit smoke checks."""
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from streamlit.testing.v1 import AppTest
from common import ROOT, MODEL_FILES, METRICS, validate_data, evaluate


def main():
    meta = json.loads((ROOT / 'model/metadata.json').read_text())
    test = pd.read_csv(ROOT / 'test_data.csv')
    train = pd.read_csv(ROOT / 'train_data.csv')
    assert len(train) == 455 and len(test) == 114
    assert not set(meta['train_indices']) & set(meta['test_indices'])
    assert len(set(meta['train_indices'] + meta['test_indices'])) == 569
    assert hashlib.sha256((ROOT / 'test_data.csv').read_bytes()).hexdigest() == meta['test_csv_sha256']
    X, y = validate_data(test, meta['features'])
    stored = pd.read_csv(ROOT / 'reports/metrics.csv').set_index('ML Model Name')
    for name, filename in MODEL_FILES.items():
        model = joblib.load(ROOT / 'model' / filename)
        values, predicted, probability = evaluate(model, X, y)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        # Independently check positive-class interpretation and six saved metrics.
        assert np.isclose(values['Accuracy'], (tp + tn) / len(y))
        assert np.isclose(values['Precision'], tp / (tp + fp))
        assert np.isclose(values['Recall'], tp / (tp + fn))
        assert np.isclose(values['F1'], 2 * tp / (2 * tp + fp + fn))
        denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        assert np.isclose(values['MCC'], (tp * tn - fp * fn) / denominator)
        assert np.isclose(values['AUC'], roc_auc_score(y, probability))
        np.testing.assert_allclose([values[m] for m in METRICS], stored.loc[name, METRICS].to_numpy(dtype=float))
        reordered, _ = validate_data(test[list(reversed(test.columns))], meta['features'])
        np.testing.assert_array_equal(predicted, model.predict(reordered))
        _, single_y = validate_data(test[test.target == 0], meta['features'])
        one_class = evaluate(model, X.loc[single_y.index], single_y)[0]
        assert np.isnan(one_class['AUC'])
        print(f'PASS {name}: TN={tn}, FP={fp}, FN={fn}, TP={tp}')
    bad_frames = [test.drop(columns='target'), test.assign(target=3), test.assign(unexpected=1),
                  test.assign(**{meta['features'][0]: np.nan}),
                  test.assign(**{meta['features'][0]: 'invalid'}), test.iloc[:0]]
    for bad in bad_frames:
        try:
            validate_data(bad, meta['features'])
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input accepted')
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    assert not app.exception, str(app.exception)
    for name in MODEL_FILES:
        app.selectbox[0].select(name).run()
        assert not app.exception, str(app.exception)
        assert len(app.metric) == 9
        assert len(app.dataframe) >= 3
    print('PASS schema checks, one-class AUC, feature ordering, and all five Streamlit selections')


if __name__ == '__main__':
    main()
