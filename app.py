"""Inference-only Streamlit app: uploaded data is never used for fitting."""
import json
import csv
import io
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
from common import ROOT, MODEL_FILES, METRICS, validate_data, evaluate

st.set_page_config(page_title='CellScope | Classifier comparison', page_icon='🔬', layout='wide')
st.title('CellScope')
st.caption('ML Assignment 2 · Five classifiers · One held-out dataset')
st.write('Compare how five trained models classify cell measurements as benign or malignant. Educational demonstration only.')

@st.cache_resource
def load_artifacts():
    metadata = json.loads((ROOT / 'model' / 'metadata.json').read_text())
    models = {name: joblib.load(ROOT / 'model' / filename) for name, filename in MODEL_FILES.items()}
    return metadata, models

try:
    metadata, models = load_artifacts()
except Exception as exc:
    st.error(f'Could not load model artifacts: {exc}. Run python train.py and commit the model folder.')
    st.stop()

with st.sidebar:
    st.header('Evaluation controls')
    name = st.selectbox('Select model', list(MODEL_FILES))
    upload = st.file_uploader('Upload labeled test CSV', type=['csv'])
    st.download_button('Download sample test CSV', (ROOT / 'test_data.csv').read_bytes(), 'test_data.csv', 'text/csv')
    st.caption('30 numeric features plus target. 0 = benign; 1 = malignant. Upload test data only.')
    st.write(f"Training rows: {metadata['train_rows']} | Holdout rows: {metadata['test_rows']}")
    st.caption('Models are loaded from disk. The app never retrains.')

try:
    if upload is not None:
        if upload.size > 5 * 1024 * 1024:
            raise ValueError('CSV must be 5 MB or smaller.')
        raw = upload.getvalue().decode('utf-8-sig')
        header = next(csv.reader(io.StringIO(raw)), [])
        if len(header) != len(set(header)):
            raise ValueError('Duplicate CSV headers are not allowed.')
        frame = pd.read_csv(io.StringIO(raw))
    else:
        frame = pd.read_csv(ROOT / 'test_data.csv')
    X, y = validate_data(frame, metadata['features'])
except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.stop()

st.info('Using uploaded test data.' if upload is not None else 'Using the bundled 114-row holdout. Upload a CSV to evaluate another labeled test set.')
if y.nunique() < 2:
    st.warning('AUC requires both classes and is shown as N/A. Other metrics use zero_division=0; MCC may be 0 for degenerate cases.')
a, b, c = st.columns(3)
a.metric('Evaluated rows', len(X))
b.metric('Benign (0)', int((y == 0).sum()))
c.metric('Malignant (1)', int((y == 1).sum()))
rows = []
outputs = {}
for model_name, model in models.items():
    values, predicted, probability = evaluate(model, X, y)
    rows.append({'ML Model Name': model_name, **values})
    outputs[model_name] = (values, predicted, probability)
comparison = pd.DataFrame(rows)
values, predicted, probability = outputs[name]
st.subheader(name)
for column, metric in zip(st.columns(6), METRICS):
    column.metric(metric, 'N/A' if pd.isna(values[metric]) else f'{values[metric]:.4f}')
st.caption('Precision, recall, and F1 use malignant (1) as positive. AUC uses malignant probabilities; predictions use each fitted classifier’s default decision rule.')
left, right = st.columns(2)
with left:
    st.write('Confusion matrix: rows = actual, columns = predicted')
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    st.dataframe(pd.DataFrame(matrix, index=['Actual benign', 'Actual malignant'], columns=['Predicted benign', 'Predicted malignant']), width='stretch')
with right:
    st.write('Classification report')
    report = classification_report(y, predicted, labels=[0, 1], target_names=['Benign', 'Malignant'], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), width='stretch')
st.subheader('All five models on the same test rows')
st.dataframe(comparison.set_index('ML Model Name').round(4), width='stretch')
st.download_button('Download comparison', comparison.to_csv(index=False), 'comparison.csv', 'text/csv')
with st.expander('Inspect predictions and test measurements'):
    predictions = X.copy()
    predictions['actual_target'] = y
    predictions['predicted_target'] = predicted
    predictions['malignant_probability'] = probability
    st.dataframe(predictions, width='stretch')
    st.download_button('Download selected-model predictions', predictions.to_csv(index=False), 'predictions.csv', 'text/csv')
with st.expander('Experiment design'):
    st.write('UCI Wisconsin Diagnostic dataset: 569 rows, 30 numeric inputs. Stratified 80/20 split, random_state=42. Scaling is fitted only on training rows inside saved pipelines. Hyperparameters were fixed before holdout evaluation. No tuning or training happens in this app.')
    st.write('A single holdout is an educational comparison and does not establish clinical validity. A winner on this split may differ on other splits.')
    st.json(metadata['versions'])
