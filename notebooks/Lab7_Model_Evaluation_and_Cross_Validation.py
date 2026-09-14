
# # Lab 7 — Model Evaluation and Cross-Validation

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

target = "is_late_delivery"

input_path = Path("data/processed/olist_orders_feature_engineered.csv")
df = pd.read_csv(input_path)
print("Shape:", df.shape)
df.head()


df[target].value_counts()

df[target].value_counts(normalize=True)


y = df[target]
X = df.drop(columns=[target])

# Drop identifier columns that are not real numeric features
if "order_id" in X.columns:
    X = X.drop(columns=["order_id"])

X = X.select_dtypes(include=np.number)
X = X.fillna(X.median())

print("X shape:", X.shape)
print("y shape:", y.shape)


# ## Part C — Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


# ## Part D — Train a Baseline Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# ## Part E — Confusion Matrix


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)


# ## Part F — Accuracy


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# ## Part G — Precision
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred, zero_division=0)
print("Precision:", precision)

# ## Part H — Recall
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred, zero_division=0)
print("Recall:", recall)
# ## Part I — F1-Score
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred, zero_division=0)
print("F1:", f1)
# ## Part J — Classification Report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, zero_division=0))

# ## Part K — ROC-AUC

y_prob = model.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", roc_auc)

# ## Part L (setup) — Classification Thresholds
threshold = 0.40
y_pred_40 = (y_prob >= threshold).astype(int)
recall_40 = recall_score(y_test, y_pred_40, zero_division=0)
print("Recall at threshold 0.40:", recall_40)
# ## Part L — Why One Train-Test Split Is Not Enough
# A single split's F1 score depends partly on which rows happened to land in
# train vs. test. We estimate that variability with cross-validation below.
# ## Part M / N — 5-Fold Cross-Validation
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="f1",
)
print("Fold scores:", cv_scores)
print("Mean F1:", cv_scores.mean())
print("Std F1:", cv_scores.std())
# ## Part O — Compare Multiple Metrics with Cross-Validaation
from sklearn.model_selection import cross_validate

scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
)

cv_summary = {}
for metric in scoring:
    values = cv_results["test_" + metric]
    cv_summary[metric] = (values.mean(), values.std())
    print(f"{metric:10s} mean={values.mean():.3f} std={values.std():.3f}")
# ## Part P — Compare Baseline and Engineered Features
# Model A uses only the "raw"/baseline columns that existed before Lab 5's

baseline_features = [
    "price",
    "freight_value",
    "product_weight_g",
    "product_volume_cm3",
    "seller_to_customer_distance_km",
    "payment_installments",
    "order_item_count",
    "review_score",
]
baseline_features = [c for c in baseline_features if c in X_train.columns]

engineered_extra = [
    "estimated_delivery_days",
    "carrier_avg_delay_days",
    "seller_avg_processing_days",
    "distance_x_installments",
    "is_holiday_season_order",
    "purchase_to_approval_hours",
]
engineered_extra = [c for c in engineered_extra if c in X_train.columns]
engineered_features = baseline_features + engineered_extra

model_a = LogisticRegression(max_iter=3000, class_weight="balanced")
model_b = LogisticRegression(max_iter=3000, class_weight="balanced")

cv_a = cross_val_score(
    model_a, X_train[baseline_features], y_train, cv=cv, scoring="f1"
)
cv_b = cross_val_score(
    model_b, X_train[engineered_features], y_train, cv=cv, scoring="f1"
)

print("Baseline features:", baseline_features)
print("Engineered features add:", engineered_extra)
print(f"Model A (baseline)   Mean F1={cv_a.mean():.3f} Std F1={cv_a.std():.3f}")
print(f"Model B (engineered) Mean F1={cv_b.mean():.3f} Std F1={cv_b.std():.3f}")

# ## Part Q — A More Correct Pipeline-Based Evaluation

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Re-derive X without the earlier median-fill so the pipeline's own imputer
# does that work inside each fold instead.
X_raw = df.drop(columns=[target])
if "order_id" in X_raw.columns:
    X_raw = X_raw.drop(columns=["order_id"])
X_raw = X_raw.select_dtypes(include=np.number)

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ]
)

pipeline_cv_scores = cross_val_score(
    pipeline,
    X_train_raw,
    y_train_raw,
    cv=cv,
    scoring="f1",
)
print("Pipeline CV scores:", pipeline_cv_scores)
print("Pipeline Mean F1:", pipeline_cv_scores.mean())
print("Pipeline Std F1:", pipeline_cv_scores.std())

# ## Part 26 — Train Final Model and Evaluate Once on the Test Set
pipeline.fit(X_train_raw, y_train_raw)
final_pred = pipeline.predict(X_test_raw)

print(classification_report(y_test_raw, final_pred, zero_division=0))

final_accuracy = accuracy_score(y_test_raw, final_pred)
final_precision = precision_score(y_test_raw, final_pred, zero_division=0)
final_recall = recall_score(y_test_raw, final_pred, zero_division=0)
final_f1 = f1_score(y_test_raw, final_pred, zero_division=0)
final_prob = pipeline.predict_proba(X_test_raw)[:, 1]
final_roc_auc = roc_auc_score(y_test_raw, final_prob)

print("Final Accuracy:", final_accuracy)
print("Final Precision:", final_precision)
print("Final Recall:", final_recall)
print("Final F1:", final_f1)
print("Final ROC-AUC:", final_roc_auc)

# ## Challenge Exercise — Threshold Sweep

thresholds_to_test = [0.30, 0.50, 0.70]
threshold_rows = []
for t in thresholds_to_test:
    y_pred_t = (y_prob >= t).astype(int)
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t, zero_division=0)
    f = f1_score(y_test, y_pred_t, zero_division=0)
    threshold_rows.append((t, p, r, f))
    print(f"Threshold={t:.2f}  Precision={p:.3f}  Recall={r:.3f}  F1={f:.3f}")

# ## Recommended Evaluation Report — Values to Copy Into the Report Table
print("=== Copy these into the Evaluation Report table ===")
print(f"Accuracy:   {accuracy:.3f}")
print(f"Precision:  {precision:.3f}")
print(f"Recall:     {recall:.3f}")
print(f"F1-score:   {f1:.3f}")
print(f"ROC-AUC:    {roc_auc:.3f}")
print(f"CV Mean F1: {cv_scores.mean():.3f}")
print(f"CV Std F1:  {cv_scores.std():.3f}")
