# CellScope - ML Assignment 2

## a. Problem statement
Classify a cell sample as benign or malignant using measured cell-nucleus characteristics. Compare Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, and Support Vector Machine on the same held-out test set. The Streamlit interface loads saved models and evaluates labeled test uploads without retraining. This is an educational experiment, not a clinical diagnostic tool.

## b. Dataset description
Dataset: UCI Breast Cancer Wisconsin (Diagnostic), accessed through scikit-learn's bundled copy. It is a public UCI dataset, not synthetic data.

Source: https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic

Citation: Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5DW2B. UCI lists the dataset under CC BY 4.0.

| Property | Value |
| --- | --- |
| Instances | 569 |
| Input features | 30 numeric measurements; exceeds minimum 12 |
| Original class counts | 357 benign, 212 malignant |
| Target in this project | 0 = benign, 1 = malignant |
| Missing values | None in the bundled dataset |
| Train / test | 455 / 114 rows, stratified 80/20 |
| Train classes | 285 benign, 170 malignant |
| Test classes | 72 benign, 42 malignant |
| Split | random_state=42 |

Ten characteristics are measured: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. Each has a mean, standard error, and worst-value summary, giving 30 input features. IDs are not input features. Exact feature headers appear in model/metadata.json and the supplied CSVs.

Important label conversion: scikit-learn encodes malignant as 0; train.py deliberately remaps malignant to 1 so recall refers to malignant samples throughout this project.

## c. Github Repository Link
REPLACE_WITH_GITHUB_REPOSITORY_URL

Live Streamlit App Link: REPLACE_WITH_STREAMLIT_APP_URL

### Repository contents
- app.py: interactive Streamlit inference and evaluation interface.
- common.py: shared schema validation and metric calculations.
- train.py: reproducible dataset split, training, saving, and holdout evaluation.
- requirements.txt: pinned app and training dependencies.
- requirements-report.txt: optional PDF-generation dependency.
- train_data.csv and test_data.csv: separate, reproducible training and test records.
- model/: all five fitted pipelines and metadata.json.
- reports/metrics.csv: full precision holdout results.
- verify.py: metric, artifact, schema, and Streamlit checks.
- build_submission.py: creates the ordered submission PDF from real links and a real BITS Lab screenshot.
- DEPLOYMENT_GUIDE.md: local, BITS Lab, GitHub, cloud, and submission steps.
- LEARNING_NOTES.md: code walkthrough and questions to understand before submission.

## d. Models used
Scaling is learned only from training data inside each pipeline. Logistic Regression, KNN, Gaussian Naive Bayes, and SVM use StandardScaler. The Decision Tree does not require scaling. All five models use the same split. No parameter search or selection is performed using the test data. Hyperparameters below were fixed in advance.

| Model | Main settings |
| --- | --- |
| Logistic Regression | C=1.0, max_iter=3000, random_state=42 |
| Decision Tree | max_depth=5, min_samples_leaf=3, random_state=42 |
| KNN | n_neighbors=5, uniform weighting, Euclidean distance |
| Naive Bayes | GaussianNB, default var_smoothing=1e-9 |
| Support Vector Machine | RBF kernel, C=1.0, gamma=scale, probability=True, random_state=42 |

SVM probability=True performs internal probability calibration using training data. AUC is computed from continuous positive-class probabilities, not class predictions. Predictions use each classifier's own default decision rule.

### Evaluation metrics
Accuracy = correct predictions / all predictions.
Precision = TP / (TP + FP). Recall = TP / (TP + FN).
F1 = 2 * precision * recall / (precision + recall).
MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)).
AUC measures ranking discrimination across thresholds. All binary metrics use malignant as the positive class. Undefined precision/recall/F1 use zero_division=0. AUC is N/A for a one-class upload. MCC follows scikit-learn's convention of 0 for degenerate cases.

### Comparison table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9035 | 0.9471 | 0.9429 | 0.7857 | 0.8571 | 0.7926 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

Values are rounded to four decimals only for display. reports/metrics.csv contains the unrounded values. The app recomputes all six metrics on the currently selected test data.

### Observations
| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Accuracy 0.9649 and highest AUC 0.9960. It missed 3 malignant samples and flagged 1 benign sample. Scaled features support a strong linear baseline; this result alone does not prove the classes are linearly separable. |
| Decision Tree | Accuracy 0.9035, recall 0.7857, and MCC 0.7926 are lowest in this experiment. It missed 9 malignant samples and flagged 2 benign samples. A depth-limited axis-aligned tree was less effective on this split; a single test result cannot establish overfitting. |
| KNN | Accuracy 0.9561 and precision 0.9744. It missed 4 malignant samples and flagged 1 benign sample. Scaling makes distances comparable; results may change with k or another training split. |
| Naive Bayes | Accuracy 0.9211 and recall 0.8571. It missed 6 malignant samples and flagged 3 benign samples. AUC 0.9891 shows stronger ranking than default-threshold classification. Correlated nucleus measurements may limit its conditional-independence assumption. |
| Support Vector Machine | Highest accuracy 0.9737, F1 0.9630, and MCC 0.9442. It flagged no benign samples and missed 3 malignant samples. Precision is 1.0000 on these 114 test rows, not a guarantee of perfect performance. |
| Overall winner for this dataset? | SVM wins this particular holdout comparison by accuracy, F1, and MCC. Logistic Regression has the highest AUC and ties SVM on malignant recall. Winner depends on metric and split; no clinical superiority is claimed. |

### Confusion counts
| Model | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| Logistic Regression | 71 | 1 | 3 | 39 |
| Decision Tree | 70 | 2 | 9 | 33 |
| KNN | 71 | 1 | 4 | 38 |
| Naive Bayes | 69 | 3 | 6 | 36 |
| Support Vector Machine | 72 | 0 | 3 | 39 |

### Reproduce and run
Use Python 3.12 and run from this project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python verify.py
python -m streamlit run app.py
```

On Windows, replace the activation command with .venv\Scripts\activate. Saved models are included, so train.py is optional for a quick local demonstration but must be run in BITS Virtual Lab for the assignment execution. Model serialization is version-sensitive: retain pinned dependencies and Python 3.12 on the cloud. If any training choices are changed, regenerate the models, metrics, CSVs, and update all README numbers and observations together.

### App behavior and test CSV contract
The app initially evaluates the bundled test_data.csv. Upload a labeled CSV with exactly the same 30 feature names and target column. Columns can be reordered, but extra or missing columns, duplicate headers, nonnumeric values, missing values, infinities, or targets outside 0/1 are rejected. Limit: 5 MB and 10,000 rows. Raw measurements are required; do not manually scale them. Pipelines apply their fitted scaling.

The dropdown selects a detailed model view with six metrics, confusion matrix, classification report, and downloadable predictions. A comparison table always evaluates all five models on the same uploaded rows. The app never fits any model. Never upload private patient data or untrusted joblib/pickle files. Only the bundled, trusted model files are deserialized.

### Deployment and submission
Use Streamlit Community Cloud with app.py, branch main, Python 3.12, and requirements.txt at repository root. Follow DEPLOYMENT_GUIDE.md. Replace both link markers above with your actual public URLs before submission. The submitted PDF must contain, in order: GitHub link, live app link, BITS Virtual Lab execution screenshot, and this complete README.

### Limitations and learning declaration
This small dataset and single holdout provide an educational comparison. There is no external validation, repeated-split study, or clinical assessment. Test metrics must not be used to tune hyperparameters. Future work could tune on training-only cross-validation and reserve a fresh external test set.

This starter implementation was developed with AI assistance. The assignment permits AI for learning support and prohibits direct copy-paste submissions. Review and adapt the implementation, explain the choices in your own words, follow course disclosure requirements, and perform your own BITS Virtual Lab execution. Do not manufacture screenshots or commit history.
