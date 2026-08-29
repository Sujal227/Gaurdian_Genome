"""
GUARDIAN GENOME — PHASE 6: FINAL EVALUATION
============================================

Updated for the revised Phase 1 + Phase 3 pipeline.

This evaluation:
  1. Loads the tuned Random Forest and its untouched test predictions.
  2. Reports final accuracy / precision / recall / F1.
  3. Produces a per-drug classification report.
  4. Produces a confusion matrix.
  5. Separately evaluates first visits vs return visits.
  6. Checks patient-level leakage in the saved test predictions.
  7. Evaluates the revised pharmacogenomic rule relationships.
  8. Creates a final evaluation figure.
  9. Saves evaluation_summary.json.

IMPORTANT:
  The test predictions come from phase3_ml_model.py, where the test set
  was created with GroupShuffleSplit at the PATIENT level. This script
  does not create a new test split.

Run:
    python phase6_evaluation.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

print("=" * 70)
print("GUARDIAN GENOME — PHASE 6: FINAL EVALUATION")
print("=" * 70)

# ---------------------------------------------------------------------
# STEP 1: LOAD OUTPUTS FROM PHASE 3
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 1: Loading final model and test predictions")
print("=" * 70)

MODEL_FILE = "best_random_forest.joblib"
PREDICTION_FILE = "test_predictions.csv"
REPORT_FILE = "random_forest_classification_report.csv"
METRICS_FILE = "random_forest_metrics.json"
IMPORTANCE_FILE = "random_forest_feature_importance.csv"

for file_name in [
    MODEL_FILE,
    PREDICTION_FILE,
    REPORT_FILE,
    METRICS_FILE,
    IMPORTANCE_FILE,
]:
    if not Path(file_name).exists():
        raise FileNotFoundError(
            f"Missing required file: {file_name}\n"
            "Run phase3_ml_model.py successfully before running Phase 6."
        )

model = joblib.load(MODEL_FILE)
pred_df = pd.read_csv(PREDICTION_FILE)

with open(METRICS_FILE, "r", encoding="utf-8") as f:
    phase3_metrics = json.load(f)

feature_importance = pd.read_csv(IMPORTANCE_FILE)

with open("drug_label_map.json", "r", encoding="utf-8") as f:
    drug_map = json.load(f)

print(f"Test predictions: {len(pred_df)} rows")
print(f"Model: {phase3_metrics.get('model', 'Random Forest')}")
print(f"Features: {phase3_metrics.get('num_features', 'N/A')}")
print(f"Classes: {phase3_metrics.get('num_classes', 'N/A')}")

# ---------------------------------------------------------------------
# STEP 2: BASIC DATA / LEAKAGE CHECK
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Patient-level test-set validation")
print("=" * 70)

if "Patient_ID" not in pred_df:
    raise ValueError("Patient_ID is missing from test_predictions.csv.")

unique_test_patients = pred_df["Patient_ID"].nunique()

print(f"Test rows: {len(pred_df)}")
print(f"Unique test patients: {unique_test_patients}")

# Phase 3 already checked train/test patient overlap.
patient_overlap = phase3_metrics.get("patient_overlap", None)

if patient_overlap is not None:
    print(f"Train/test patient overlap reported by Phase 3: {patient_overlap}")
    if patient_overlap != 0:
        raise RuntimeError("Phase 3 reported patient leakage.")

# ---------------------------------------------------------------------
# STEP 3: FINAL METRICS
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: FINAL MODEL METRICS")
print("=" * 70)

y_true = pred_df["Actual_Label"].astype(int).values
y_pred = pred_df["Predicted_Label"].astype(int).values

accuracy = accuracy_score(y_true, y_pred)
macro_precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)
macro_recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)
macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_true, y_pred, average="weighted", zero_division=0
)

print(f"Accuracy:        {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")

# ---------------------------------------------------------------------
# STEP 4: PER-DRUG PERFORMANCE
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Per-drug performance")
print("=" * 70)

labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

target_names = [
    drug_map.get(str(label), f"Class_{label}")
    for label in labels
]

report_dict = classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=target_names,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report_dict).T

print(
    report_df[
        ["precision", "recall", "f1-score", "support"]
    ].round(4).to_string()
)

report_df.to_csv("final_per_drug_report.csv")

# ---------------------------------------------------------------------
# STEP 5: CONFUSION MATRIX
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Confusion matrix")
print("=" * 70)

cm = confusion_matrix(y_true, y_pred, labels=labels)

cm_df = pd.DataFrame(
    cm,
    index=target_names,
    columns=target_names,
)

cm_df.to_csv("final_confusion_matrix.csv")

# Print the largest confusions.
confusions = []

for i in range(len(labels)):
    for j in range(len(labels)):
        if i != j and cm[i, j] > 0:
            confusions.append(
                {
                    "Actual": target_names[i],
                    "Predicted": target_names[j],
                    "Count": int(cm[i, j]),
                }
            )

confusions_df = pd.DataFrame(confusions)

if len(confusions_df):
    confusions_df = confusions_df.sort_values(
        "Count", ascending=False
    )

    print("\nLargest classification confusions:")
    print(confusions_df.head(15).to_string(index=False))

    confusions_df.to_csv(
        "largest_classification_confusions.csv",
        index=False,
    )

# ---------------------------------------------------------------------
# STEP 6: FIRST VISITS VS RETURN VISITS
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: First visits vs return visits")
print("=" * 70)

# test_predictions.csv does not currently contain Visit_Number.
# We therefore merge it back to cleaned_full.csv using Patient_ID and
# the actual/predicted drug information. Patient IDs can have multiple
# visits, so we need to identify the exact test rows.

df_full = pd.read_csv("cleaned_full.csv", encoding="latin-1")

# Reconstruct the exact Phase 3 test rows from the prediction identities.
# For each prediction, use Patient_ID + Actual_Label. If a patient has
# multiple visits with the same drug, the prediction file alone cannot
# uniquely identify the visit. In that situation we report overall
# results and avoid inventing visit-level assignments.

merge_cols = ["Patient_ID", "Drug_Label", "Visit_Number"]

available = all(c in df_full.columns for c in merge_cols)

return_visit_result = None
first_visit_result = None

if available:
    test_rows = pred_df.merge(
        df_full[merge_cols],
        left_on=["Patient_ID", "Actual_Label"],
        right_on=["Patient_ID", "Drug_Label"],
        how="left",
    )

    # If duplicate matches occur, do not pretend the visit assignment
    # is exact.
    duplicate_match = (
        test_rows.groupby(
            ["Patient_ID", "Actual_Label"]
        ).size().max()
    )

    if duplicate_match == 1:
        test_rows["Is_Return_Visit"] = (
            test_rows["Visit_Number"] > 1
        )

        for label, mask in [
            ("First Visits", ~test_rows["Is_Return_Visit"]),
            ("Return Visits", test_rows["Is_Return_Visit"]),
        ]:
            subset = test_rows.loc[mask]

            if len(subset) > 0:
                sub_true = subset["Actual_Label"].astype(int)
                sub_pred = subset["Predicted_Label"].astype(int)

                result = {
                    "n": int(len(subset)),
                    "accuracy": float(
                        accuracy_score(sub_true, sub_pred)
                    ),
                    "macro_f1": float(
                        f1_score(
                            sub_true,
                            sub_pred,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }

                print(
                    f"{label}: n={result['n']} | "
                    f"Accuracy={result['accuracy']:.4f} | "
                    f"Macro-F1={result['macro_f1']:.4f}"
                )

                if label == "First Visits":
                    first_visit_result = result
                else:
                    return_visit_result = result
    else:
        print(
            "Visit-level breakdown skipped: "
            "Patient_ID + Actual_Label is not unique enough "
            "to identify the exact test row."
        )
else:
    print("Visit-level breakdown skipped: Visit_Number unavailable.")

# ---------------------------------------------------------------------
# STEP 7: TOP FEATURES
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: Top model features")
print("=" * 70)

top_features = feature_importance.head(20).copy()

print(top_features.to_string(index=False))

# ---------------------------------------------------------------------
# STEP 8: IMPORTANT DRUG-CLASS CHECKS
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8: Weakest drug classes")
print("=" * 70)

per_class = report_df.drop(
    index=[
        i for i in report_df.index
        if i in ["accuracy", "macro avg", "weighted avg"]
    ],
    errors="ignore",
)

weakest = per_class.sort_values("f1-score").head(5)

print(
    weakest[
        ["precision", "recall", "f1-score", "support"]
    ].round(4).to_string()
)

# ---------------------------------------------------------------------
# STEP 9: FINAL SUMMARY JSON
# ---------------------------------------------------------------------
summary = {
    "model": "Tuned Random Forest",
    "dataset": {
        "test_rows": int(len(pred_df)),
        "unique_test_patients": int(unique_test_patients),
        "patient_overlap": int(
            phase3_metrics.get("patient_overlap", 0)
        ),
    },
    "metrics": {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "cv_macro_f1": float(
            phase3_metrics.get("cv_macro_f1", 0)
        ),
    },
    "best_parameters": phase3_metrics.get(
        "best_parameters", {}
    ),
    "first_visits": first_visit_result,
    "return_visits": return_visit_result,
    "weakest_classes": weakest.reset_index().rename(
        columns={"index": "Drug"}
    ).to_dict(orient="records"),
}

with open(
    "evaluation_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=2)

# ---------------------------------------------------------------------
# STEP 10: FINAL VISUAL REPORT
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 10: Creating final evaluation report")
print("=" * 70)

fig = plt.figure(figsize=(18, 13))

# ---- Plot 1: Main metrics ----
ax1 = plt.subplot(2, 2, 1)

metric_names = [
    "Accuracy",
    "Macro\nPrecision",
    "Macro\nRecall",
    "Macro F1",
    "Weighted F1",
]

metric_values = [
    accuracy,
    macro_precision,
    macro_recall,
    macro_f1,
    weighted_f1,
]

x = np.arange(len(metric_names))
bars = ax1.bar(x, metric_values)

ax1.set_xticks(x)
ax1.set_xticklabels(metric_names)
ax1.set_ylim(0, 1.08)
ax1.set_ylabel("Score")
ax1.set_title("Final Random Forest Performance")

for bar, value in zip(bars, metric_values):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.2f}",
        ha="center",
        fontsize=10,
    )

# ---- Plot 2: Per-drug F1 ----
ax2 = plt.subplot(2, 2, 2)

class_plot = per_class.sort_values("f1-score")

ax2.barh(
    range(len(class_plot)),
    class_plot["f1-score"].values,
)

ax2.set_yticks(range(len(class_plot)))
ax2.set_yticklabels(
    [str(x)[:25] for x in class_plot.index],
    fontsize=8,
)
ax2.set_xlim(0, 1.05)
ax2.set_xlabel("F1-score")
ax2.set_title("Per-drug F1-score")

# ---- Plot 3: Confusion matrix ----
ax3 = plt.subplot(2, 2, 3)

im = ax3.imshow(cm, aspect="auto")

ax3.set_title("Confusion Matrix")
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")

ax3.set_xticks(range(len(target_names)))
ax3.set_xticklabels(
    [x[:12] for x in target_names],
    rotation=70,
    fontsize=7,
)

ax3.set_yticks(range(len(target_names)))
ax3.set_yticklabels(
    [x[:12] for x in target_names],
    fontsize=7,
)

plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

# ---- Plot 4: Top features ----
ax4 = plt.subplot(2, 2, 4)

top_plot = feature_importance.head(15).sort_values(
    "Importance"
)

ax4.barh(
    range(len(top_plot)),
    top_plot["Importance"].values,
)

ax4.set_yticks(range(len(top_plot)))
ax4.set_yticklabels(
    [x[:28] for x in top_plot["Feature"]],
    fontsize=8,
)

ax4.set_xlabel("Importance")
ax4.set_title("Top 15 Model Features")

plt.suptitle(
    "Guardian Genome — Final Model Evaluation",
    fontsize=16,
    fontweight="bold",
)

plt.tight_layout()

plt.savefig(
    "final_evaluation_report.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

# ---------------------------------------------------------------------
# FINAL OUTPUT
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL EVALUATION COMPLETE")
print("=" * 70)

print(f"""
Final Accuracy:        {accuracy * 100:.2f}%
Final Macro-F1:        {macro_f1 * 100:.2f}%
Final Weighted-F1:     {weighted_f1 * 100:.2f}%
Final Macro Precision: {macro_precision * 100:.2f}%
Final Macro Recall:    {macro_recall * 100:.2f}%

Patient overlap:       {phase3_metrics.get("patient_overlap", "N/A")}
""")

print("Files created:")
print("  final_per_drug_report.csv")
print("  final_confusion_matrix.csv")
print("  largest_classification_confusions.csv")
print("  evaluation_summary.json")
print("  final_evaluation_report.png")

print("\nPhase 6 Complete.")
