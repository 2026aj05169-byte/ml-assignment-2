# Understand the implementation before adapting it

## Read the files in this order
1. train.py: loads a public dataset, remaps the target, makes one stratified split, trains five pipelines, saves them, and calculates holdout metrics.
2. common.py: validates the test schema and calculates all six metrics consistently.
3. app.py: loads trusted saved models, accepts only labeled test CSVs, evaluates every model, and displays the selected model's details.
4. verify.py: independently checks confusion-count formulas, saved results, row separation, input failures, and Streamlit model selection.

## Why these decisions matter
| Decision | Reason |
| --- | --- |
| Malignant = 1 | Positive-class recall counts correctly detected malignant samples. sklearn's original labels are inverted explicitly. |
| Stratified 80/20 split | Preserves approximately the same class proportions while keeping test rows outside training. |
| Scale inside pipelines | Fits mean and variance on training data only; uses the same fitted transformation during inference. |
| Scale LR, KNN, and SVM | Feature magnitudes influence regularization, distances, and RBF similarity. |
| GaussianNB | Continuous numeric measurements match Gaussian NB better than count-based MultinomialNB. Scaling is harmless for this modeling choice and preserves the common pipeline interface. |
| Limited tree depth | Constrains complexity in advance. It is not a test-based tuned optimum. |
| Probabilities for AUC | AUC evaluates rankings across thresholds, so hard 0/1 predictions discard useful information. |
| MCC | Summarizes all four confusion counts and provides another view with unequal class sizes. |
| No fitting in app.py | Uploaded data is test data, never training data. Saved preprocessing travels with each model. |

## One worked metric example
SVM: TN=72, FP=0, FN=3, TP=39; total=114.
Accuracy = (72 + 39) / 114 = 0.9737.
Precision = 39 / (39 + 0) = 1.0000.
Recall = 39 / (39 + 3) = 0.9286.
F1 = 2 * 39 / (2 * 39 + 0 + 3) = 0.9630.
MCC = (39 * 72 - 0 * 3) / sqrt(39 * 42 * 72 * 75) = 0.9442.
AUC = 0.9947 from the ranked malignant probabilities; it cannot be reconstructed from one confusion matrix alone.

## Be ready to explain
- Why can Logistic Regression have better AUC while SVM has better accuracy?
- Why is SVM precision 1.0 not evidence of a perfect cancer detector?
- What leakage would occur if scaling were fitted on the full dataset before splitting?
- Why should repeated experiments not choose their seed based on the test score?
- Why do correlated measurements challenge the Naive Bayes assumption?
- What is a false negative here, and which metric reflects missed malignant cases?
- What happens when the CSV contains only benign samples?

## Responsible adaptation
Do not submit code you cannot explain. Change the UI and discussion in ways you understand, document meaningful changes in genuine commits, and follow your institution's AI-assistance disclosure requirements. If you alter training choices, regenerate and reconcile every reported number. A future cross-validation experiment must use training folds only and keep a separate untouched final test set.
