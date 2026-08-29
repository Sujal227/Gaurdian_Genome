"""
GUARDIAN GENOME — PHASE 3: V1 vs FULL LONGITUDINAL RANDOM FOREST
================================================================

Purpose
-------
Train two patient-level models using the SAME patient-level split:

1. V1 model:
   Uses the Visit-1 feature set only.
   It does NOT use previous-visit history.

2. Full model:
   Uses the complete longitudinal feature set, including:
       - Previous_Drug
       - Previous_Outcome_encoded
       - Previous_ADR_flag
       - Is_Return_Visit
       - Dose_Escalated_flag
       - pharmacogenomic × condition interactions

The test patients are identical for both models.

IMPORTANT:
- Hyperparameter tuning is performed ONLY on the training patients.
- GroupKFold is used during tuning so a patient cannot cross folds.
- The final test set is untouched until final evaluation.
- Model selection uses CV macro-F1.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# 25 searches for each model. This is deliberately moderate so it is
# practical to run on a normal laptop.
N_ITER_SEARCH = 20

print("=" * 75)
print("GUARDIAN GENOME — PHASE 3")
print("V1 vs FULL LONGITUDINAL RANDOM FOREST")
print("=" * 75)

# =====================================================================
# STEP 1 — LOAD DATA
# =====================================================================
print("\n" + "=" * 75)
print("STEP 1: Loading data and feature lists")
print("=" * 75)

df = pd.read_csv("cleaned_full.csv", encoding="latin-1")

with open("feature_list_v1.json", "r", encoding="utf-8") as f:
    FEATURES_V1 = json.load(f)

with open("feature_list_full.json", "r", encoding="utf-8") as f:
    FEATURES_FULL = json.load(f)

with open("drug_label_map.json", "r", encoding="utf-8") as f:
    drug_map = json.load(f)

print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Patients: {df['Patient_ID'].nunique()}")
print(f"V1 features: {len(FEATURES_V1)}")
print(f"Full features: {len(FEATURES_FULL)}")
print(
    "Additional Full features:",
    [f for f in FEATURES_FULL if f not in FEATURES_V1]
)

# Keep only valid target rows.
df = df[df["Drug_Label"] >= 0].copy()

# Verify features.
missing_v1 = [f for f in FEATURES_V1 if f not in df.columns]
missing_full = [f for f in FEATURES_FULL if f not in df.columns]

if missing_v1:
    raise ValueError(
        "Missing V1 features:\n" + "\n".join(missing_v1)
    )

if missing_full:
    raise ValueError(
        "Missing Full features:\n" + "\n".join(missing_full)
    )

# =====================================================================
# STEP 2 — SAME PATIENT-LEVEL SPLIT FOR BOTH MODELS
# =====================================================================
print("\n" + "=" * 75)
print("STEP 2: Creating SAME patient-level train/test split")
print("=" * 75)

groups = df["Patient_ID"].copy()
y = df["Drug_Label"].astype(int).copy()

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    gss.split(df, y, groups=groups)
)

df_train = df.iloc[train_idx].copy()
df_test = df.iloc[test_idx].copy()

train_patients = set(df_train["Patient_ID"].unique())
test_patients = set(df_test["Patient_ID"].unique())

overlap = train_patients.intersection(test_patients)

print(f"Training rows: {len(df_train)}")
print(f"Test rows:     {len(df_test)}")
print(f"Training patients: {len(train_patients)}")
print(f"Test patients:     {len(test_patients)}")
print(f"Patient overlap:   {len(overlap)}")

if overlap:
    raise RuntimeError(
        "PATIENT LEAKAGE DETECTED. Train/test patients overlap."
    )

# =====================================================================
# STEP 3 — PREPARE FEATURES
# =====================================================================
print("\n" + "=" * 75)
print("STEP 3: Preparing V1 and Full feature matrices")
print("=" * 75)


def prepare_features(frame, feature_list):
    X = frame[feature_list].copy()

    # Convert booleans to integers.
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    non_numeric = X.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            "Non-numeric features found:\n"
            + "\n".join(non_numeric)
        )

    return X


X_v1_train = prepare_features(df_train, FEATURES_V1)
X_v1_test = prepare_features(df_test, FEATURES_V1)

X_full_train = prepare_features(df_train, FEATURES_FULL)
X_full_test = prepare_features(df_test, FEATURES_FULL)

y_train = df_train["Drug_Label"].astype(int)
y_test = df_test["Drug_Label"].astype(int)

# =====================================================================
# STEP 4 — IMPUTATION
# =====================================================================
print("\n" + "=" * 75)
print("STEP 4: Training-only median imputation")
print("=" * 75)


def impute_train_test(X_train, X_test):
    imputer = SimpleImputer(strategy="median")

    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_imp, X_test_imp, imputer


X_v1_train, X_v1_test, imputer_v1 = impute_train_test(
    X_v1_train,
    X_v1_test,
)

X_full_train, X_full_test, imputer_full = impute_train_test(
    X_full_train,
    X_full_test,
)

print(
    "V1 remaining NaNs:",
    X_v1_train.isna().sum().sum(),
    X_v1_test.isna().sum().sum(),
)

print(
    "Full remaining NaNs:",
    X_full_train.isna().sum().sum(),
    X_full_test.isna().sum().sum(),
)

# =====================================================================
# STEP 5 — RANDOM FOREST SEARCH FUNCTION
# =====================================================================
print("\n" + "=" * 75)
print("STEP 5: Defining grouped cross-validation search")
print("=" * 75)

param_distributions = {
    "n_estimators": [200, 300, 400, 500, 700],
    "max_depth": [None, 12, 18, 24, 30, 40],
    "min_samples_split": [2, 4, 6, 10],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", "log2", 0.5, 0.75],
    "class_weight": [
        None,
        "balanced",
        "balanced_subsample",
    ],
    "criterion": ["gini", "entropy"],
}


def tune_random_forest(X_train, y_train, groups_train, model_name):
    print("\n" + "-" * 75)
    print(f"Tuning {model_name}")
    print("-" * 75)

    n_groups = groups_train.nunique()
    n_splits = min(CV_FOLDS, n_groups)

    if n_splits < 2:
        raise ValueError(
            f"Not enough patient groups for {model_name} CV."
        )

    cv = GroupKFold(n_splits=n_splits)

    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=False,
    )

    search.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    print(f"\n{model_name} best CV macro-F1:")
    print(f"  {search.best_score_:.4f}")

    print(f"\n{model_name} best parameters:")
    for key, value in search.best_params_.items():
        print(f"  {key}: {value}")

    return search.best_estimator_, search


groups_train = df_train["Patient_ID"]

# =====================================================================
# STEP 6 — TUNE V1 MODEL
# =====================================================================
v1_model, v1_search = tune_random_forest(
    X_v1_train,
    y_train,
    groups_train,
    "V1 MODEL",
)

# =====================================================================
# STEP 7 — TUNE FULL MODEL
# =====================================================================
full_model, full_search = tune_random_forest(
    X_full_train,
    y_train,
    groups_train,
    "FULL LONGITUDINAL MODEL",
)

# =====================================================================
# STEP 8 — FINAL TEST EVALUATION
# =====================================================================
print("\n" + "=" * 75)
print("STEP 8: FINAL TEST EVALUATION")
print("=" * 75)


def evaluate_model(model, X_test, y_test, name):
    pred = model.predict(X_test)

    result = {
        "model": name,
        "accuracy": float(
            accuracy_score(y_test, pred)
        ),
        "macro_precision": float(
            precision_score(
                y_test,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "predictions": pred,
    }

    print(f"\n{name}")
    print("-" * 50)
    print(
        f"Accuracy:        {result['accuracy']:.4f} "
        f"({result['accuracy'] * 100:.2f}%)"
    )
    print(
        f"Macro Precision: {result['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall:    {result['macro_recall']:.4f}"
    )
    print(
        f"Macro F1:        {result['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1:     {result['weighted_f1']:.4f}"
    )

    return result


v1_result = evaluate_model(
    v1_model,
    X_v1_test,
    y_test,
    "V1 MODEL — NO PREVIOUS-VISIT HISTORY",
)

full_result = evaluate_model(
    full_model,
    X_full_test,
    y_test,
    "FULL MODEL — WITH LONGITUDINAL HISTORY",
)

# =====================================================================
# STEP 9 — RETURN-VISIT ANALYSIS
# =====================================================================
print("\n" + "=" * 75)
print("STEP 9: RETURN-VISIT ANALYSIS")
print("=" * 75)

return_mask = df_test["Visit_Number"].astype(int) > 1
first_mask = df_test["Visit_Number"].astype(int) == 1

print(
    f"Test first-visit rows: {first_mask.sum()}"
)
print(
    f"Test return-visit rows: {return_mask.sum()}"
)


def subset_metrics(
    mask,
    label,
):
    if mask.sum() == 0:
        print(f"\n{label}: no rows")
        return None

    y_sub = y_test.loc[mask]

    pred_v1 = v1_result["predictions"][mask.values]
    pred_full = full_result["predictions"][mask.values]

    v1_acc = accuracy_score(y_sub, pred_v1)
    full_acc = accuracy_score(y_sub, pred_full)

    v1_f1 = f1_score(
        y_sub,
        pred_v1,
        average="macro",
        zero_division=0,
    )

    full_f1 = f1_score(
        y_sub,
        pred_full,
        average="macro",
        zero_division=0,
    )

    result = {
        "n": int(mask.sum()),
        "v1_accuracy": float(v1_acc),
        "full_accuracy": float(full_acc),
        "accuracy_delta": float(full_acc - v1_acc),
        "v1_macro_f1": float(v1_f1),
        "full_macro_f1": float(full_f1),
        "macro_f1_delta": float(full_f1 - v1_f1),
    }

    print(f"\n{label} (n={result['n']})")
    print(
        f"  V1:   Accuracy={v1_acc:.4f} | "
        f"Macro-F1={v1_f1:.4f}"
    )
    print(
        f"  Full: Accuracy={full_acc:.4f} | "
        f"Macro-F1={full_f1:.4f}"
    )
    print(
        f"  Accuracy delta: {full_acc - v1_acc:+.4f}"
    )
    print(
        f"  Macro-F1 delta: {full_f1 - v1_f1:+.4f}"
    )

    return result


first_visit_result = subset_metrics(
    first_mask,
    "FIRST VISITS",
)

return_visit_result = subset_metrics(
    return_mask,
    "RETURN VISITS (V2 + V3)",
)

# Per-visit analysis.
visit_results = {}

for visit_number in sorted(
    df_test["Visit_Number"].astype(int).unique()
):
    mask = (
        df_test["Visit_Number"].astype(int)
        == visit_number
    )

    visit_results[str(visit_number)] = subset_metrics(
        mask,
        f"VISIT {visit_number}",
    )

# =====================================================================
# STEP 10 — PER-CLASS REPORTS
# =====================================================================
print("\n" + "=" * 75)
print("STEP 10: Per-drug performance")
print("=" * 75)


def save_classification_report(
    y_true,
    predictions,
    model_name,
    filename,
):
    labels = sorted(
        np.unique(
            np.concatenate(
                [y_true.values, predictions]
            )
        )
    )

    target_names = [
        drug_map.get(
            str(label),
            f"Class_{label}",
        )
        for label in labels
    ]

    report = classification_report(
        y_true,
        predictions,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).T
    report_df.to_csv(filename)

    print(f"\n{model_name}:")
    print(
        report_df[
            ["precision", "recall", "f1-score", "support"]
        ].round(4).to_string()
    )

    return report_df


v1_report = save_classification_report(
    y_test,
    v1_result["predictions"],
    "V1 MODEL",
    "rf_v1_classification_report.csv",
)

full_report = save_classification_report(
    y_test,
    full_result["predictions"],
    "FULL MODEL",
    "rf_full_classification_report.csv",
)

# =====================================================================
# STEP 11 — SAVE MODELS AND PREDICTIONS
# =====================================================================
print("\n" + "=" * 75)
print("STEP 11: Saving models and results")
print("=" * 75)

joblib.dump(
    v1_model,
    "rf_v1_tuned.joblib",
)

joblib.dump(
    full_model,
    "rf_full_tuned.joblib",
)

joblib.dump(
    imputer_v1,
    "rf_v1_imputer.joblib",
)

joblib.dump(
    imputer_full,
    "rf_full_imputer.joblib",
)

# One prediction file containing both model predictions.
predictions_df = df_test[
    ["Patient_ID", "Visit_Number", "Drug_Label"]
].copy()

predictions_df["Actual_Drug"] = [
    drug_map.get(str(int(x)), f"Class_{x}")
    for x in predictions_df["Drug_Label"]
]

predictions_df["V1_Predicted_Label"] = (
    v1_result["predictions"]
)

predictions_df["V1_Predicted_Drug"] = [
    drug_map.get(
        str(int(x)),
        f"Class_{x}",
    )
    for x in v1_result["predictions"]
]

predictions_df["Full_Predicted_Label"] = (
    full_result["predictions"]
)

predictions_df["Full_Predicted_Drug"] = [
    drug_map.get(
        str(int(x)),
        f"Class_{x}",
    )
    for x in full_result["predictions"]
]

predictions_df.to_csv(
    "rf_v1_vs_full_predictions.csv",
    index=False,
)

# =====================================================================
# STEP 12 — FEATURE IMPORTANCE
# =====================================================================
v1_importance = pd.DataFrame(
    {
        "Feature": FEATURES_V1,
        "Importance": v1_model.feature_importances_,
    }
).sort_values(
    "Importance",
    ascending=False,
)

full_importance = pd.DataFrame(
    {
        "Feature": FEATURES_FULL,
        "Importance": full_model.feature_importances_,
    }
).sort_values(
    "Importance",
    ascending=False,
)

v1_importance.to_csv(
    "rf_v1_feature_importance.csv",
    index=False,
)

full_importance.to_csv(
    "rf_full_feature_importance.csv",
    index=False,
)

print("\nTop Full-model features:")
print(
    full_importance.head(20).to_string(
        index=False
    )
)

# =====================================================================
# STEP 13 — CONFUSION MATRICES
# =====================================================================
def save_confusion_matrix(
    y_true,
    predictions,
    filename,
):
    labels = sorted(
        np.unique(
            np.concatenate(
                [y_true.values, predictions]
            )
        )
    )

    names = [
        drug_map.get(
            str(label),
            f"Class_{label}",
        )
        for label in labels
    ]

    cm = pd.DataFrame(
        __import__("sklearn").metrics.confusion_matrix(
            y_true,
            predictions,
            labels=labels,
        ),
        index=names,
        columns=names,
    )

    cm.to_csv(filename)


save_confusion_matrix(
    y_test,
    v1_result["predictions"],
    "rf_v1_confusion_matrix.csv",
)

save_confusion_matrix(
    y_test,
    full_result["predictions"],
    "rf_full_confusion_matrix.csv",
)

# =====================================================================
# STEP 14 — COMPARISON SUMMARY
# =====================================================================
comparison = pd.DataFrame(
    [
        {
            "Model": "V1 — No history",
            "Accuracy": v1_result["accuracy"],
            "Macro_Precision": v1_result["macro_precision"],
            "Macro_Recall": v1_result["macro_recall"],
            "Macro_F1": v1_result["macro_f1"],
            "Weighted_F1": v1_result["weighted_f1"],
            "CV_Macro_F1": float(
                v1_search.best_score_
            ),
        },
        {
            "Model": "Full — With history",
            "Accuracy": full_result["accuracy"],
            "Macro_Precision": full_result["macro_precision"],
            "Macro_Recall": full_result["macro_recall"],
            "Macro_F1": full_result["macro_f1"],
            "Weighted_F1": full_result["weighted_f1"],
            "CV_Macro_F1": float(
                full_search.best_score_
            ),
        },
    ]
)

comparison.to_csv(
    "rf_v1_vs_full_comparison.csv",
    index=False,
)

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(
    comparison.round(4).to_string(
        index=False
    )
)

print("\nLongitudinal contribution:")
print(
    f"Accuracy improvement: "
    f"{full_result['accuracy'] - v1_result['accuracy']:+.4f}"
)

print(
    f"Macro-F1 improvement: "
    f"{full_result['macro_f1'] - v1_result['macro_f1']:+.4f}"
)

if return_visit_result:
    print("\nReturn-visit contribution:")
    print(
        f"Accuracy improvement: "
        f"{return_visit_result['accuracy_delta']:+.4f}"
    )
    print(
        f"Macro-F1 improvement: "
        f"{return_visit_result['macro_f1_delta']:+.4f}"
    )

# =====================================================================
# STEP 15 — SAVE COMPLETE JSON SUMMARY
# =====================================================================
summary = {
    "experiment": "Patient-level V1 vs Full Longitudinal Random Forest",
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "train_rows": int(len(df_train)),
    "test_rows": int(len(df_test)),
    "train_patients": int(len(train_patients)),
    "test_patients": int(len(test_patients)),
    "patient_overlap": int(len(overlap)),
    "v1_features": int(len(FEATURES_V1)),
    "full_features": int(len(FEATURES_FULL)),
    "v1": {
        "cv_macro_f1": float(
            v1_search.best_score_
        ),
        "accuracy": v1_result["accuracy"],
        "macro_precision": v1_result["macro_precision"],
        "macro_recall": v1_result["macro_recall"],
        "macro_f1": v1_result["macro_f1"],
        "weighted_f1": v1_result["weighted_f1"],
        "best_parameters": v1_search.best_params_,
    },
    "full": {
        "cv_macro_f1": float(
            full_search.best_score_
        ),
        "accuracy": full_result["accuracy"],
        "macro_precision": full_result["macro_precision"],
        "macro_recall": full_result["macro_recall"],
        "macro_f1": full_result["macro_f1"],
        "weighted_f1": full_result["weighted_f1"],
        "best_parameters": full_search.best_params_,
    },
    "longitudinal_contribution": {
        "overall_accuracy_delta": float(
            full_result["accuracy"]
            - v1_result["accuracy"]
        ),
        "overall_macro_f1_delta": float(
            full_result["macro_f1"]
            - v1_result["macro_f1"]
        ),
        "first_visits": first_visit_result,
        "return_visits": return_visit_result,
        "by_visit": visit_results,
    },
}

with open(
    "rf_v1_vs_full_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

# =====================================================================
# FINAL
# =====================================================================
print("\n" + "=" * 75)
print("PHASE 3 COMPLETE")
print("=" * 75)

print(
    f"V1 FINAL ACCURACY:   "
    f"{v1_result['accuracy'] * 100:.2f}%"
)

print(
    f"V1 FINAL MACRO-F1:   "
    f"{v1_result['macro_f1'] * 100:.2f}%"
)

print(
    f"FULL FINAL ACCURACY: "
    f"{full_result['accuracy'] * 100:.2f}%"
)

print(
    f"FULL FINAL MACRO-F1: "
    f"{full_result['macro_f1'] * 100:.2f}%"
)

print("\nSaved:")
print("  rf_v1_tuned.joblib")
print("  rf_full_tuned.joblib")
print("  rf_v1_imputer.joblib")
print("  rf_full_imputer.joblib")
print("  rf_v1_vs_full_predictions.csv")
print("  rf_v1_vs_full_comparison.csv")
print("  rf_v1_vs_full_summary.json")
print("  rf_v1_classification_report.csv")
print("  rf_full_classification_report.csv")
print("  rf_v1_feature_importance.csv")
print("  rf_full_feature_importance.csv")
print("  rf_v1_confusion_matrix.csv")
print("  rf_full_confusion_matrix.csv")

print("\nDo NOT run the old Phase 6 yet.")
print("Run this Phase 3 first and send the terminal output.")
print("=" * 75)
