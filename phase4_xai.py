"""
GUARDIAN GENOME — PHASE 4: XAI Explainability (Longitudinal)
=============================================================
What this phase does:
  1. Global SHAP — which features matter most in V1 model vs Full model
  2. Longitudinal SHAP comparison — do Previous_Outcome/ADR features rank high
     in the Full model? This validates that the model learned from history.
  3. Local SHAP — per-patient force plots for all 3 visit scenarios
  4. NL explanation generator — includes visit context in explanation
     e.g. "Previous drug was ineffective → dose escalated based on..."

Run:  python phase4_xai.py
Outputs: shap_global_v1.png, shap_global_full.png,
         shap_longitudinal_comparison.png,
         shap_local_visits.png, explanation_samples.json
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: LOAD
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading models and data")
print("=" * 60)

model_v1   = joblib.load("rf_v1.pkl")
model_full = joblib.load("rf_full.pkl")

with open("feature_list_v1.json")   as f: FEATURES_V1   = json.load(f)
with open("feature_list_full.json") as f: FEATURES_FULL = json.load(f)
with open("drug_label_map.json")    as f: drug_map       = json.load(f)

drug_classes = [drug_map[str(i)] for i in range(len(drug_map))]

df_v1   = pd.read_csv("cleaned_visit1.csv",  encoding="latin-1")
df_full = pd.read_csv("cleaned_full.csv",    encoding="latin-1")

df_v1[FEATURES_V1]     = df_v1[FEATURES_V1].fillna(0)
df_full[FEATURES_FULL] = df_full[FEATURES_FULL].fillna(0)

X_v1   = df_v1[FEATURES_V1].values
X_full = df_full[FEATURES_FULL].values

print(f"V1 model:   {type(model_v1).__name__}, {len(FEATURES_V1)} features")
print(f"Full model: {type(model_full).__name__}, {len(FEATURES_FULL)} features")

LONG_FEATS = ["Previous_Outcome_encoded","Previous_ADR_flag",
              "Dose_Escalated_flag","Is_Return_Visit"]
GENOMIC_FEATS = ["CYP2C9_Phenotype_encoded","CYP2D6_Phenotype_encoded",
                 "VKORC1_Genotype_encoded","VKORC1_Sensitivity_encoded",
                 "SLCO1B1_Function_encoded","CYP2C19_Phenotype_encoded",
                 "TPMT_Activity_encoded","DPYD_Function_encoded",
                 "HLA_B5701_encoded","CYP2C9_AlleleScore"]

# ─────────────────────────────────────────────
# STEP 2: GLOBAL SHAP — BOTH MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Computing SHAP values (this takes ~2 min)")
print("=" * 60)

n_explain = min(250, len(X_v1), len(X_full))
X_v1_explain   = X_v1[:n_explain]
X_full_explain = X_full[:n_explain]

# SHAP compatibility:
# - RandomForest: TreeExplainer is fast and exact for tree models.
# - HistGradientBoosting: some SHAP versions do not support this estimator
#   through TreeExplainer. Fall back to a model-agnostic SHAP Explainer.
def build_shap_explainer(model, background):
    try:
        explainer = shap.TreeExplainer(model)
        return explainer, "TreeExplainer"
    except Exception as exc:
        print(f"  TreeExplainer unavailable for {type(model).__name__}: {exc}")
        print("  Falling back to model-agnostic SHAP Explainer.")
        # Keep the background small so Phase 4 remains practical on a laptop.
        bg = background[:min(50, len(background))]
        explainer = shap.Explainer(model.predict_proba, bg, algorithm="permutation")
        return explainer, "Permutation"

def get_shap_values(explainer, X):
    result = explainer(X)
    values = result.values

    # Modern SHAP multiclass output is usually:
    #   (samples, features, classes)
    # Some versions return:
    #   (samples, features, classes) through Explanation.values
    # Normalize binary/single-output cases to the same convention.
    arr = np.asarray(values)

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    return arr

explainer_v1, explainer_v1_type = build_shap_explainer(model_v1, X_v1_explain)
explainer_full, explainer_full_type = build_shap_explainer(model_full, X_full_explain)

shap_v1 = get_shap_values(explainer_v1, X_v1_explain)
shap_full = get_shap_values(explainer_full, X_full_explain)

print(f"  V1 SHAP method:   {explainer_v1_type}")
print(f"  Full SHAP method: {explainer_full_type}")
print(f"  V1 SHAP shape:    {shap_v1.shape}")
print(f"  Full SHAP shape:  {shap_full.shape}")

# Mean |SHAP| per feature across samples and classes.
def mean_abs_shap(shap_vals, n_classes):
    arr = np.asarray(shap_vals)
    return np.abs(arr).mean(axis=(0, 2))

shap_imp_v1 = pd.Series(
    mean_abs_shap(shap_v1, len(drug_classes)),
    index=FEATURES_V1
)

shap_imp_full = pd.Series(
    mean_abs_shap(shap_full, len(drug_classes)),
    index=FEATURES_FULL
)


print("\nTop 10 features (V1 model):")
print(shap_imp_v1.sort_values(ascending=False).head(10).to_string())
print("\nTop 10 features (Full model):")
print(shap_imp_full.sort_values(ascending=False).head(10).to_string())

# ─────────────────────────────────────────────
# STEP 3: LONGITUDINAL FEATURE IMPORTANCE COMPARISON
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Longitudinal feature SHAP rank analysis")
print("=" * 60)

# Check what rank the longitudinal features have in the Full model
shap_full_sorted = shap_imp_full.sort_values(ascending=False)
total_feats = len(FEATURES_FULL)
for lf in LONG_FEATS:
    if lf in shap_full_sorted.index:
        rank = list(shap_full_sorted.index).index(lf) + 1
        importance = shap_full_sorted[lf]
        print(f"  {lf}: rank {rank}/{total_feats}, mean|SHAP|={importance:.4f}")

# ─────────────────────────────────────────────
# STEP 4: GLOBAL SHAP FIGURES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Generating SHAP plots")
print("=" * 60)

# ── Figure 1: V1 model top features ───────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(10, 7))
top20_v1 = shap_imp_v1.sort_values(ascending=False).head(20)
clrs = ["#2ecc71" if f in GENOMIC_FEATS else "#9b59b6" if "Primary_Condition" in f
        else "#3498db" for f in top20_v1.index]
ax.barh(top20_v1.index[::-1], top20_v1.values[::-1], color=clrs[::-1])
ax.set_title("Global SHAP — V1 Model (Visit 1, Genetics Only)\n🟢Genomic  🔵Clinical  🟣Condition",
             fontweight="bold")
ax.set_xlabel("Mean |SHAP value|")
ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig("shap_global_v1.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_global_v1.png")

# ── Figure 2: Full model top features with longitudinal highlighted ────────────
fig2, ax = plt.subplots(figsize=(10, 7))
top20_full = shap_imp_full.sort_values(ascending=False).head(20)
clrs_full = []
for f in top20_full.index:
    if f in LONG_FEATS:       clrs_full.append("#e74c3c")
    elif f in GENOMIC_FEATS:  clrs_full.append("#2ecc71")
    elif "Primary_Condition" in f: clrs_full.append("#9b59b6")
    else:                     clrs_full.append("#3498db")
ax.barh(top20_full.index[::-1], top20_full.values[::-1], color=clrs_full[::-1])
ax.set_title("Global SHAP — Full Model (All Visits, +History)\n"
             "🟢Genomic  🔴Longitudinal  🔵Clinical  🟣Condition",
             fontweight="bold")
ax.set_xlabel("Mean |SHAP value|")
ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig("shap_global_full.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_global_full.png")

# ── Figure 3: Longitudinal feature SHAP across visit numbers ──────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle("Longitudinal Features SHAP — Do They Matter More on Return Visits?",
              fontsize=13, fontweight="bold")

for ax, (visit_num, label) in zip(axes3, [(1, "Visit 1"), (2, "Visit 2 & 3")]):
    if visit_num == 1:
        mask = df_full["Visit_Number"] == 1
    else:
        mask = df_full["Visit_Number"] > 1
    sub_X = df_full[mask][FEATURES_FULL].fillna(0).values[:min(100, mask.sum())]
    if len(sub_X) < 2:
        continue
    sub_shap = get_shap_values(explainer_full, sub_X)
    sub_imp = mean_abs_shap(sub_shap, len(drug_classes))
    sub_series = pd.Series(sub_imp, index=FEATURES_FULL)

    top12 = sub_series.sort_values(ascending=False).head(12)
    clrs_s = ["#e74c3c" if f in LONG_FEATS else "#2ecc71" if f in GENOMIC_FEATS
              else "#9b59b6" if "Primary_Condition" in f else "#3498db"
              for f in top12.index]
    ax.barh(top12.index[::-1], top12.values[::-1], color=clrs_s[::-1])
    ax.set_title(f"Full Model — {label}\n(🔴 longitudinal feats should rank higher on V2/V3)",
                 fontweight="bold", fontsize=9)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig("shap_longitudinal_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_longitudinal_comparison.png")

# ─────────────────────────────────────────────
# STEP 5: LOCAL SHAP — PATIENT JOURNEYS (3 VISITS)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Local SHAP — patient visit journeys")
print("=" * 60)

# Find a patient with 3 visits in the dataset for the showcase
orig_df = pd.read_csv("longitudinal_dataset.csv", encoding="latin-1")
visit_counts = orig_df.groupby("Patient_ID").size()
patients_3v  = visit_counts[visit_counts == 3].index.tolist()
patients_2v  = visit_counts[visit_counts == 2].index.tolist()
showcase_patients = patients_3v[:2] + patients_2v[:1]

fig4, axes4 = plt.subplots(
    max(1, len(showcase_patients)), 3,
    figsize=(18, 5 * max(1, len(showcase_patients)))
)
axes4 = np.atleast_2d(axes4)
fig4.suptitle("Local SHAP — Patient Visit Journeys (Drug Recommendation per Visit)",
              fontsize=13, fontweight="bold")

for pat_idx, pid in enumerate(showcase_patients):
    patient_visits = df_full[
        df_full.get("Patient_ID", pd.Series([None]*len(df_full))).values == pid
        if "Patient_ID" in df_full.columns else
        pd.Series([False]*len(df_full))
    ] if "Patient_ID" in df_full.columns else df_full[df_full.index == -1]

    # Fallback: just pick 3 rows from different visits
    v1_rows = df_full[df_full["Visit_Number"]==1].head(1)
    v2_rows = df_full[(df_full["Visit_Number"]==2) & (df_full["Previous_Outcome_encoded"]==1)].head(1)
    v3_rows = df_full[df_full["Visit_Number"] == 3].head(1)
    visit_rows_list = [(1, v1_rows), (2, v2_rows), (3, v3_rows)]

    for col_idx, (vnum, vrow) in enumerate(visit_rows_list):
        if len(vrow) == 0:
            continue
        ax = axes4[pat_idx][col_idx]
        patient_X = vrow[FEATURES_FULL].fillna(0).values
        pred_label = model_full.predict(patient_X)[0]
        drug_name = drug_map.get(str(pred_label), "Unknown")

        # Get SHAP
        sv = get_shap_values(explainer_full, patient_X)
        # sv shape is (n_samples, n_features, n_classes) on this version of SHAP
        # need to extract the slice for pred_label class
        sv_arr = np.array(sv)
        if sv_arr.ndim == 3:
        # shape: (n_samples, n_features, n_classes) → take class pred_label
            class_shap = sv_arr[0, :, pred_label]
        else:
        # shape: list of (n_samples, n_features) → take index pred_label
            class_shap = sv_arr[pred_label][0]

        top_n = 8
        top_idx = np.argsort(np.abs(class_shap))[::-1][:top_n]
        top_names = [FEATURES_FULL[i][:25] for i in top_idx]
        top_vals  = class_shap[top_idx]
        colors_bar = ["#e74c3c" if v < 0 else "#2ecc71" for v in top_vals]

        ax.barh(top_names[::-1], top_vals[::-1], color=colors_bar[::-1])
        ax.axvline(0, color="black", linewidth=0.8)

        # Outcome annotation
        prev_out = vrow["Previous_Outcome"].values[0] if "Previous_Outcome" in vrow.columns else "N/A"
        outcome_note = f"Prev outcome: {prev_out}" if vnum > 1 else "First visit"
        ax.set_title(f"Visit {vnum} → {drug_name}\n({outcome_note})",
                     fontweight="bold", fontsize=9)
        ax.set_xlabel("SHAP value", fontsize=8)
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig("shap_local_visits.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_local_visits.png")

# ─────────────────────────────────────────────
# STEP 6: NL EXPLANATION ENGINE (LONGITUDINAL-AWARE)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Natural language explanation engine")
print("=" * 60)

GENE_NL = {
    "CYP2C9_Phenotype_encoded": {
        0: ("CYP2C9 Poor Metabolizer", "significantly reduced drug metabolism"),
        1: ("CYP2C9 Intermediate Metabolizer", "moderately reduced metabolism"),
        2: ("CYP2C9 Normal Metabolizer", "normal drug metabolism"),
    },
    "CYP2D6_Phenotype_encoded": {
        0: ("CYP2D6 Poor Metabolizer", "very slow metabolism — drug accumulation risk"),
        1: ("CYP2D6 Intermediate Metabolizer", "reduced metabolism"),
        2: ("CYP2D6 Normal Metabolizer", "normal metabolism"),
        3: ("CYP2D6 Rapid Metabolizer", "faster than normal metabolism"),
        4: ("CYP2D6 Ultrarapid Metabolizer", "extremely rapid metabolism"),
    },
    "VKORC1_Sensitivity_encoded": {
        0: ("VKORC1 GG (Low Sensitivity)", "standard Warfarin dose appropriate"),
        1: ("VKORC1 GA (Moderate Sensitivity)", "moderate bleeding risk"),
        2: ("VKORC1 AA (High Sensitivity)", "high bleeding risk — reduced dose required"),
    },
    "SLCO1B1_Function_encoded": {
        0: ("SLCO1B1 Non/Poor Function", "high myopathy risk — avoid high-dose statins"),
        1: ("SLCO1B1 Decreased", "reduced statin transport — consider lower dose"),
        2: ("SLCO1B1 Normal", "normal statin transport"),
    },
    "TPMT_Activity_encoded": {
        0: ("TPMT Deficient", "CONTRAINDICATED: Azathioprine myelotoxicity risk"),
        1: ("TPMT Low", "Azathioprine dose reduction required"),
        2: ("TPMT Intermediate", "moderate dose adjustment"),
        3: ("TPMT Normal", "standard dosing appropriate"),
    },
    "DPYD_Function_encoded": {
        0: ("DPYD Non-functional", "CONTRAINDICATED: 5-FU severe toxicity"),
        1: ("DPYD Decreased", "50% fluoropyrimidine dose reduction required"),
        2: ("DPYD Normal", "normal 5-FU metabolism"),
    },
    "HLA_B5701_encoded": {
        0: ("HLA-B*5701 Negative", "Abacavir hypersensitivity risk is low"),
        1: ("HLA-B*5701 Positive", "CONTRAINDICATED: Abacavir — hypersensitivity"),
    },
}

VISIT_CONTEXT_NL = {
    "FIRST_VISIT": "This is the patient's first visit — recommendation is based solely on genomic profile.",
    "CONTINUE":    "Previous treatment was {prev_outcome} — continuing current regimen.",
    "ESCALATE_DOSE": "Previous treatment showed no response — dose escalated to improve efficacy.",
    "SWITCH_DRUG": "Previous drug caused an adverse reaction — switching to alternative drug class.",
}

DOSE_NL = {
    "Standard Dose": "standard dosing is appropriate",
    "Reduced Dose": "dose reduction is recommended",
    "Reduced Dose (50%)": "50% dose reduction is required",
    "Standard-Low Dose": "lower-end standard dose is recommended",
    "Low Dose": "low dosing is recommended",
    "Increased Dose": "dose escalation is required for therapeutic effect",
}

def generate_nl_explanation(encoded_dict, predicted_drug, dose_category,
                             top_shap_features, confidence, clinical_flag,
                             visit_number, prev_outcome, prev_drug):
    """Generate a visit-aware NL explanation."""
    lines = []

    # Visit context (NEW)
    visit_num = int(visit_number) if visit_number else 1
    if visit_num == 1:
        visit_line = VISIT_CONTEXT_NL["FIRST_VISIT"]
    elif str(prev_outcome) == "Adverse Reaction":
        visit_line = VISIT_CONTEXT_NL["SWITCH_DRUG"]
    elif str(prev_outcome) == "No Change":
        visit_line = VISIT_CONTEXT_NL["ESCALATE_DOSE"]
    else:
        visit_line = VISIT_CONTEXT_NL["CONTINUE"].format(prev_outcome=prev_outcome)

    lines.append(f"[Visit {visit_num}] {visit_line}")

    if visit_num > 1 and prev_drug and prev_drug != "None":
        lines.append(f"Previous drug: {prev_drug} → Outcome: {prev_outcome}")

    # Drug recommendation
    dose_text = DOSE_NL.get(dose_category, "standard dosing is appropriate")
    lines.append(f"Recommended: <strong>{predicted_drug}</strong> — {dose_text} "
                 f"(confidence: {confidence:.0%})")

    # Gene reasoning
    gene_reasons = []
    for feat, shap_val in top_shap_features[:3]:
        if feat in GENE_NL:
            enc = int(encoded_dict.get(feat, 0))
            if enc in GENE_NL[feat]:
                gene_name, effect = GENE_NL[feat][enc]
                gene_reasons.append(f"{gene_name}: {effect}.")
    if gene_reasons:
        lines.append("Genomic rationale: " + " ".join(gene_reasons))

    # Longitudinal contribution (NEW)
    long_contributors = [(f, v) for f, v in top_shap_features[:5]
                         if f in ["Previous_Outcome_encoded","Previous_ADR_flag",
                                  "Dose_Escalated_flag","Is_Return_Visit"]]
    if long_contributors:
        lf_name, lf_val = long_contributors[0]
        direction = "supported" if lf_val > 0 else "flagged a concern for"
        lines.append(f"Visit history {direction} this recommendation "
                     f"(feature: {lf_name}, SHAP={lf_val:+.3f}).")

    # Clinical flag
    clinical_flag = str(clinical_flag) if clinical_flag and str(clinical_flag) != "nan" else "None"
    if clinical_flag and clinical_flag not in ["None", "nan"]:
        clinical_flag = str(clinical_flag) if clinical_flag and str(clinical_flag) != "nan" else "None"
        icon = "🚨" if "CONTRAINDICATED" in clinical_flag else "⚠️"
        lines.append(f"{icon} Alert: {clinical_flag}")

    # Confidence
    if confidence >= 0.90:
        lines.append("High confidence — strong evidence supports this recommendation.")
    elif confidence >= 0.75:
        lines.append("Moderate confidence — clinical review advised.")
    else:
        lines.append("Lower confidence — additional judgment required.")

    return "\n".join(lines)


# Generate 10 sample explanations across different visit scenarios
print("Generating sample explanations...")

sample_explanations = []
scenarios = [
    ("First visit, normal genes", df_full[df_full["Visit_Number"]==1].head(3)),
    ("Visit 2, prev stable",      df_full[(df_full["Visit_Number"]==2) & (df_full["Previous_Outcome_encoded"]==2)].head(2)),
    ("Visit 2, prev no change",   df_full[(df_full["Visit_Number"]==2) & (df_full["Previous_Outcome_encoded"]==1)].head(2)),
    ("Visit 2, prev ADR",         df_full[(df_full["Visit_Number"]==2) & (df_full["Previous_ADR_flag"]==1)].head(2)),
    ("Visit 3", df_full[df_full["Visit_Number"] == 3].head(1)),
]

for scenario_name, rows_subset in scenarios:
    for _, row in rows_subset.iterrows():
        patient_X = row[FEATURES_FULL].fillna(0).values.reshape(1, -1)
        pred_label = model_full.predict(patient_X)[0]
        pred_drug  = drug_map.get(str(pred_label), "Unknown")
        proba      = model_full.predict_proba(patient_X)[0]
        confidence = float(proba[pred_label])

        sv = get_shap_values(explainer_full, patient_X)
        sv_arr = np.array(sv)
        class_shap = sv_arr[0, :, pred_label] if sv_arr.ndim == 3 else sv_arr[pred_label][0]
        top_shap = sorted(zip(FEATURES_FULL, class_shap), key=lambda x: abs(x[1]), reverse=True)[:5]

        encoded_dict = {f: float(row.get(f, 0)) for f in FEATURES_FULL}

        # Previous drug/outcome are already present in the longitudinal cleaned row.
        # No future/current-visit data is pulled into the explanation.

        explanation = generate_nl_explanation(
            encoded_dict, pred_drug,
            row.get("Dose_Category", "Standard Dose"),
            top_shap, confidence,
            row.get("Clinical_Flag", "None"),
            row.get("Visit_Number", 1),
            row.get("Previous_Outcome", "None"),
            row.get("Previous_Drug", "None"),
        )

        sample_explanations.append({
            "scenario": scenario_name,
            "visit_number": int(row.get("Visit_Number", 1)),
            "previous_outcome": str(row.get("Previous_Outcome", "None")),
            "previous_drug": str(row.get("Previous_Drug", "None")),
            "true_drug": str(row.get("Recommended_Drug", "?")),
            "predicted_drug": pred_drug,
            "correct": pred_drug == str(row.get("Recommended_Drug", "?")),
            "confidence": round(confidence, 3),
            "top_shap": [(f, round(float(v), 4)) for f, v in top_shap],
            "explanation": explanation,
        })

with open("explanation_samples.json", "w") as f:
    json.dump(sample_explanations, f, indent=2)
print("Saved: explanation_samples.json")

# Print sample
for exp in sample_explanations[:3]:
    print(f"\n[{exp['scenario']}]")
    print(f"  True: {exp['true_drug']} | Predicted: {exp['predicted_drug']} ({'✅' if exp['correct'] else '❌'})")
    print(f"  Explanation:\n    " + exp["explanation"].replace("\n", "\n    "))

print("\n✅ Phase 4 Complete")
print("   shap_global_v1.png, shap_global_full.png")
print("   shap_longitudinal_comparison.png, shap_local_visits.png")
print("   explanation_samples.json")