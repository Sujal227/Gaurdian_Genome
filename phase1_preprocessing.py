"""
GUARDIAN GENOME — PHASE 1: Preprocessing (Longitudinal Dataset)
================================================================
What this phase does:
  1. Loads longitudinal_dataset.csv (1 patient = 1–3 linked rows)
  2. Audits missing values and data quality flags
  3. Drops non-informative columns (Creatinine 97% missing, static genomic coords)
  4. Imputes missing values with column-appropriate strategies
  5. Encodes all categorical features (ordinal + OHE)
  6. Engineers 4 new longitudinal features from visit history
  7. Splits into Visit-1-only subset and full longitudinal subset
  8. Saves cleaned files + EDA report

NEW vs original project:
  - Previous_Drug and Previous_Outcome are now KEPT and engineered into features
  - Previous_Drug is one-hot encoded as a historical treatment feature
  - Previous_Outcome_encoded: ordinal score of previous visit outcome (−1 to 3)
  - Previous_ADR_flag: was the previous drug an adverse reaction? (binary)
  - Dose_Escalated_flag: did dose go up this visit vs last? (binary)
  - Is_Return_Visit: binary flag for visit 2 or 3
  - Visit_Number is a key model input (not just metadata)

Run:  python phase1_preprocessing.py
Outputs: cleaned_full.csv, cleaned_visit1.csv, eda_report.png,
         drug_label_map.json, feature_list_v1.json, feature_list_full.json
"""

import pandas as pd     #helps with reading of csv files
import numpy as np      # provides functions for numbers and arrays like 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: LOAD
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading longitudinal dataset")
print("=" * 60)

df = pd.read_csv("longitudinal_dataset.csv", encoding="latin-1")
df = df.replace("", np.nan)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Unique patients: {df['Patient_ID'].nunique()}")
print(f"Visit distribution:\n{df['Visit_Number'].value_counts().sort_index().to_string()}")

# ─────────────────────────────────────────────
# STEP 2: MISSING VALUE AUDIT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Missing value audit")
print("=" * 60)

missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
print(missing_pct.to_string())

# ─────────────────────────────────────────────
# STEP 3: DROP NON-INFORMATIVE COLUMNS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Dropping columns")
print("=" * 60)

DROP_COLS = {
    "Creatinine_mg_dL":  "97% missing — unusable",
    "Chromosome":        "constant value (always 10)",
    "Genomic_Position":  "redundant with rsID",
    "Ref_Allele":        "low variance, derivable from rsID",
    "Alt_Allele":        "low variance, derivable from rsID",
    # Keep Patient_ID in the cleaned dataset for patient-level train/test splitting.
    # It is NOT included in FEATURES_V1 / FEATURES_FULL.
}
for col, reason in DROP_COLS.items():
    print(f"  Dropping {col}: {reason}")
df.drop(columns=list(DROP_COLS.keys()), errors="ignore", inplace=True)

# ─────────────────────────────────────────────
# STEP 4: FLAG DATA QUALITY ROWS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Data quality flagging")
print("=" * 60)

error_mask = df["Clinical_Flag"].str.contains("DATA ENTRY ERROR", na=False)
df["data_quality_flag"] = error_mask.astype(int)
print(f"Data entry error rows flagged: {error_mask.sum()} ({error_mask.mean()*100:.1f}%)")

blank_cond = df["Primary_Condition"].isna()
df.loc[blank_cond, "Primary_Condition"] = "Unknown"
print(f"Blank Primary_Condition → 'Unknown': {blank_cond.sum()}")

# ─────────────────────────────────────────────
# STEP 5: IMPUTE MISSING VALUES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Imputation")
print("=" * 60)

GENOMIC_COLS = ["CYP2D6_Phenotype", "SLCO1B1_Function",
                "CYP2C19_Phenotype", "TPMT_Activity", "DPYD_Function"]
for col in GENOMIC_COLS:
    before = df[col].isna().sum()
    mode_map = df.groupby("Primary_Condition")[col].agg(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    )
    df[col] = df.apply(
        lambda row: mode_map.get(row["Primary_Condition"], np.nan)
        if pd.isna(row[col]) else row[col], axis=1
    )
    df[col] = df[col].fillna(df[col].mode()[0])
    print(f"  {col}: {before} → {df[col].isna().sum()} missing")

df["HLA_B5701"] = df["HLA_B5701"].fillna("Negative")

LAB_COLS = ["GFR_mL_min", "ALT_U_L", "BMI", "Weight_kg", "Height_cm", "Quality_Score_PHRED"]
for col in LAB_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    before = df[col].isna().sum()
    eth_med = df.groupby("Ethnicity")[col].transform("median")
    df[col] = df[col].fillna(eth_med).fillna(df[col].median())
    print(f"  {col}: {before} → {df[col].isna().sum()} missing")

df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(df["Age"].median())
df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])
df["Ethnicity"] = df["Ethnicity"].fillna(df["Ethnicity"].mode()[0])
for col in ["Dose_Min_mg", "Dose_Max_mg", "Visit_Number"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

print(f"\nRemaining nulls: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# STEP 6: LONGITUDINAL FEATURE ENGINEERING  ← KEY NEW STEP
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Longitudinal feature engineering (NEW for Path 2)")
print("=" * 60)

# Previous_Drug is legitimate pre-visit information:
# it refers to the treatment received at the immediately preceding visit.
# Visit 1 has no treatment history, so represent that state explicitly as "None".
df["Previous_Drug"] = df["Previous_Drug"].fillna("None").astype(str)
df.loc[df["Previous_Drug"].str.strip() == "", "Previous_Drug"] = "None"
print("  Previous_Drug categories:", df["Previous_Drug"].nunique())
print("  Previous_Drug history rows:", (df["Previous_Drug"] != "None").sum())

OUTCOME_ORDINAL = {
    "None": 0, "First Visit": 0, "Lost to Follow-up": 0,
    "No Change": 1,
    "Adverse Reaction": -1, "Switched Drug": -1,
    "Stable": 2, "Improved": 3,
}

# Feature 1: Previous_Outcome_encoded
# Ordinal score of what happened at the LAST visit (stored in Previous_Outcome column)
# Visit 1 patients have Previous_Outcome = "None" → encodes as 0
# Visit 2 patients have Previous_Outcome = result of visit 1 drug
# Visit 3 patients have Previous_Outcome = result of visit 2 drug
df["Previous_Outcome_encoded"] = df["Previous_Outcome"].map(OUTCOME_ORDINAL).fillna(0).astype(int)
print(f"  Previous_Outcome_encoded: {df['Previous_Outcome_encoded'].value_counts().to_dict()}")
print(f"  Meaning: -1=ADR/Switch, 0=First Visit/No Change, 2=Stable, 3=Improved")

# Feature 2: Previous_ADR_flag
# Binary: 1 if the last drug caused an Adverse Reaction
df["Previous_ADR_flag"] = (df["Previous_Outcome"] == "Adverse Reaction").astype(int)
print(f"  Previous_ADR_flag: {df['Previous_ADR_flag'].sum()} patients with prior ADR")



# Feature 4: Is_Return_Visit
df["Is_Return_Visit"] = (df["Visit_Number"] > 1).astype(int)
print(f"  Is_Return_Visit: {df['Is_Return_Visit'].sum()} return-visit rows")

# Feature 3: Dose_Escalated_flag
# Compares this visit's dose midpoint with the previous visit's midpoint.
# For Visit 1 there is no prior dose, so the flag is 0.
df = df.sort_values(["Patient_ID", "Visit_Number"]).copy()
dose_mid = (pd.to_numeric(df["Dose_Min_mg"], errors="coerce") +
            pd.to_numeric(df["Dose_Max_mg"], errors="coerce")) / 2
prev_dose_mid = dose_mid.groupby(df["Patient_ID"]).shift(1)
df["Dose_Escalated_flag"] = (
    (df["Visit_Number"] > 1) &
    prev_dose_mid.notna() &
    dose_mid.notna() &
    (dose_mid > prev_dose_mid)
).astype(int)
print(f"  Dose_Escalated_flag: {df['Dose_Escalated_flag'].sum()} escalated rows")

# Clinically meaningful interaction features.
# These make the revised pharmacogenomic treatment rules explicit to the model.
df["CYP2C9_Diabetes_interaction"] = (
    df["CYP2C9_Phenotype"].astype(str).eq("Poor Metabolizer") &
    df["Primary_Condition"].astype(str).eq("Type 2 Diabetes")
).astype(int)

df["SLCO1B1_Lipid_interaction"] = (
    df["SLCO1B1_Function"].astype(str).isin(
        ["Poor Function", "Non-functional", "Decreased Function"]
    ) &
    df["Primary_Condition"].astype(str).eq("Hyperlipidemia")
).astype(int)

df["CYP2C19_Depression_interaction"] = (
    df["CYP2C19_Phenotype"].astype(str).eq("Poor Metabolizer") &
    df["Primary_Condition"].astype(str).eq("Depression")
).astype(int)

df["TPMT_RA_interaction"] = (
    df["TPMT_Activity"].astype(str).isin(["Deficient Activity", "Low Activity"]) &
    df["Primary_Condition"].astype(str).eq("Rheumatoid Arthritis")
).astype(int)

df["DPYD_CRC_interaction"] = (
    df["DPYD_Function"].astype(str).isin(["Non-functional", "Decreased Function"]) &
    df["Primary_Condition"].astype(str).eq("Colorectal Cancer")
).astype(int)

df["HLA_HIV_interaction"] = (
    df["HLA_B5701"].astype(str).eq("Positive") &
    df["Primary_Condition"].astype(str).eq("HIV/AIDS")
).astype(int)

print("  Added pharmacogenomic × condition interaction features.")

# ─────────────────────────────────────────────
# STEP 7: ENCODING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Feature encoding")
print("=" * 60)

METABOLIZER_ORDER = {
    "Poor Metabolizer": 0, "Intermediate Metabolizer": 1,
    "Normal Metabolizer": 2, "Rapid Metabolizer": 3, "Ultrarapid Metabolizer": 4,
}
ORDINAL_MAPS = {
    "CYP2C9_Phenotype":     METABOLIZER_ORDER,
    "CYP2D6_Phenotype":     METABOLIZER_ORDER,
    "CYP2C19_Phenotype":    METABOLIZER_ORDER,
    "VKORC1_Sensitivity":   {"High Sensitivity": 2, "Moderate Sensitivity": 1, "Low Sensitivity": 0},
    "SLCO1B1_Function":     {"Non-functional": 0, "Poor Function": 0, "Decreased Function": 1, "Normal Function": 2},
    "DPYD_Function":        {"Non-functional": 0, "Decreased Function": 1, "Normal Function": 2},
    "TPMT_Activity":        {"Deficient Activity": 0, "Low Activity": 1, "Intermediate Activity": 2, "Normal Activity": 3},
    "Drug_Interaction_Risk":{"Low": 0, "Moderate": 1, "High": 2},
    "VKORC1_Genotype":      {"GG": 0, "GA": 1, "AA": 2},
    "Zygosity":             {"Homozygous": 0, "Heterozygous": 1},
    "HLA_B5701":            {"Negative": 0, "Positive": 1},
}
for col, mapping in ORDINAL_MAPS.items():
    df[col + "_encoded"] = df[col].map(mapping).fillna(0).astype(int)

df["Sex_encoded"] = (df["Sex"] == "Male").astype(int)
df["CYP2C9_AlleleScore"] = df["CYP2C9_Alleles"].apply(
    lambda a: sum({"*1":1.0,"*2":0.5,"*3":0.0}.get(p.strip(),0.5)
                  for p in str(a).split("/")) if pd.notna(a) else 2.0
)
df = pd.get_dummies(
    df,
    columns=["Primary_Condition", "Ethnicity", "Previous_Drug"],
    drop_first=False
)
print("  All encodings applied.")

# ─────────────────────────────────────────────
# STEP 8: TARGET ENCODING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Target variable encoding")
print("=" * 60)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
valid_mask = df["Recommended_Drug"].notna()
df.loc[valid_mask, "Drug_Label"] = le.fit_transform(df.loc[valid_mask, "Recommended_Drug"])
df["Drug_Label"] = df["Drug_Label"].fillna(-1).astype(int)
df["Dose_Label"] = df["Dose_Category"].map({
    "": -1, "Low Dose": 0, "Reduced Dose (50%)": 1, "Reduced Dose": 2,
    "Standard-Low Dose": 3, "Standard Dose": 4, "Increased Dose": 5,
}).fillna(-1).astype(int)

drug_map = {int(i): name for i, name in enumerate(le.classes_)}
with open("drug_label_map.json", "w") as f:
    json.dump(drug_map, f, indent=2)
print(f"  {len(le.classes_)} drug classes saved → drug_label_map.json")

# ─────────────────────────────────────────────
# STEP 9: FEATURE SETS + SAVE SPLITS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Feature sets and saving")
print("=" * 60)

GENOMIC_FEATURES = [
    "CYP2C9_Phenotype_encoded", "CYP2D6_Phenotype_encoded",
    "VKORC1_Genotype_encoded", "VKORC1_Sensitivity_encoded",
    "SLCO1B1_Function_encoded", "CYP2C19_Phenotype_encoded",
    "TPMT_Activity_encoded", "DPYD_Function_encoded",
    "HLA_B5701_encoded", "Zygosity_encoded", "CYP2C9_AlleleScore",
]
CLINICAL_FEATURES = [
    "Age", "Sex_encoded", "Weight_kg", "BMI", "GFR_mL_min", "ALT_U_L",
    "Drug_Interaction_Risk_encoded",
    "Comorbid_Hypertension", "Comorbid_T2Diabetes", "Comorbid_Hyperlipidemia",
    "Comorbid_AtrialFibrillation", "Comorbid_ChronicKidneyDis", "Comorbid_Obesity",
    "Visit_Number",
]
LONGITUDINAL_FEATURES = [
    "Previous_Outcome_encoded",
    "Previous_ADR_flag",
    "Is_Return_Visit",
    "Dose_Escalated_flag",
]
MED_FEATURES = [c for c in df.columns if c.startswith("Med_")]
OHE_FEATURES = [
    c for c in df.columns
    if c.startswith("Primary_Condition_")
    or c.startswith("Ethnicity_")
    or c.startswith("Previous_Drug_")
]

INTERACTION_FEATURES = [
    "CYP2C9_Diabetes_interaction",
    "SLCO1B1_Lipid_interaction",
    "CYP2C19_Depression_interaction",
    "TPMT_RA_interaction",
    "DPYD_CRC_interaction",
    "HLA_HIV_interaction",
]

FEATURES_V1   = [f for f in GENOMIC_FEATURES + CLINICAL_FEATURES + INTERACTION_FEATURES + MED_FEATURES + OHE_FEATURES if f in df.columns]
FEATURES_FULL = [f for f in GENOMIC_FEATURES + CLINICAL_FEATURES + INTERACTION_FEATURES + LONGITUDINAL_FEATURES + MED_FEATURES + OHE_FEATURES if f in df.columns]

with open("feature_list_v1.json", "w") as f:    json.dump(FEATURES_V1, f, indent=2)
with open("feature_list_full.json", "w") as f:  json.dump(FEATURES_FULL, f, indent=2)

df_full = df[df["Drug_Label"] >= 0].copy()
df_v1   = df_full[df_full["Visit_Number"] == 1].copy()
df_full.to_csv("cleaned_full.csv", index=False)
df_v1.to_csv("cleaned_visit1.csv", index=False)
print(f"  feature_list_v1.json   ({len(FEATURES_V1)} features  — Visit 1 model)")
print(f"  feature_list_full.json ({len(FEATURES_FULL)} features — Full longitudinal model, including Previous_Drug)")
print(f"  cleaned_full.csv       ({len(df_full)} rows)")
print(f"  cleaned_visit1.csv     ({len(df_v1)} rows — Visit 1 only)")

# ─────────────────────────────────────────────
# STEP 10: EDA REPORT
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18))
fig.suptitle("Guardian Genome — EDA Report (Longitudinal Dataset)", fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])
drug_counts = df_full["Recommended_Drug"].value_counts()
ax1.barh(drug_counts.index[::-1], drug_counts.values[::-1], color=plt.cm.tab20.colors[:len(drug_counts)])
ax1.set_title("Recommended Drug Distribution", fontweight="bold")
for i, v in enumerate(drug_counts.values[::-1]):
    ax1.text(v+1, i, str(v), va="center", fontsize=8)

ax2 = fig.add_subplot(gs[0, 2])
oc = df_full["Treatment_Outcome"].value_counts()
ax2.pie(oc.values, labels=oc.index, autopct="%1.1f%%",
        colors=["#2ecc71","#27ae60","#e67e22","#e74c3c","#9b59b6","#95a5a6","#3498db"][:len(oc)],
        textprops={"fontsize":8})
ax2.set_title("Treatment Outcomes", fontweight="bold")

ax3 = fig.add_subplot(gs[1, 0])
visit_dose = pd.crosstab(df_full["Visit_Number"], df_full["Dose_Category"])
visit_dose_pct = visit_dose.div(visit_dose.sum(axis=1), axis=0)*100
visit_dose_pct.plot(kind="bar", stacked=True, ax=ax3, colormap="RdYlGn", width=0.7)
ax3.set_title("Dose Category % by Visit\n(dose escalation pattern)", fontweight="bold")
ax3.set_xlabel("Visit Number")
ax3.legend(fontsize=7)
ax3.tick_params(axis="x", rotation=0)

ax4 = fig.add_subplot(gs[1, 1])
enc_dist = df_full["Previous_Outcome_encoded"].value_counts().sort_index()
clrs = {-1:"#e74c3c", 0:"#95a5a6", 1:"#e67e22", 2:"#3498db", 3:"#2ecc71"}
ax4.bar([str(k) for k in enc_dist.index], enc_dist.values,
        color=[clrs.get(k,"#95a5a6") for k in enc_dist.index])
ax4.set_title("Previous_Outcome_encoded\n(new longitudinal feature)", fontweight="bold")
ax4.set_xlabel("−1=ADR, 0=No History/No Change, 2=Stable, 3=Improved")

ax5 = fig.add_subplot(gs[1, 2])
war = df_full[df_full["Recommended_Drug"]=="Warfarin"]
ct = pd.crosstab(war["VKORC1_Genotype"], war["Dose_Category"])
ct.plot(kind="bar", ax=ax5, colormap="coolwarm", width=0.8)
ax5.set_title("Warfarin Dose by VKORC1\n(clinical validation)", fontweight="bold")
ax5.tick_params(axis="x", rotation=0)
ax5.legend(fontsize=7)

ax6 = fig.add_subplot(gs[2, 0])
esc = (
    df_full.assign(
        Dose_Escalated_EDA=(
            (df_full["Dose_Category"] == "Increased Dose") &
            (df_full["Visit_Number"] >= 2)
        ).astype(int)
    )
    .groupby("Visit_Number")["Dose_Escalated_EDA"]
    .mean() * 100
)
ax6.bar(esc.index, esc.values, color=["#2ecc71","#f39c12","#e74c3c"])
ax6.set_title("% Patients with Dose Escalation\nper Visit", fontweight="bold")
ax6.set_xlabel("Visit Number")
ax6.set_ylabel("% Escalated")
ax6.set_xticks([1,2,3])

ax7 = fig.add_subplot(gs[2, 1])
miss_data = pd.DataFrame({
    "col": ["CYP2D6","SLCO1B1","TPMT","DPYD","BMI","GFR","ALT","Creatinine"],
    "pct": [df["CYP2D6_Phenotype_encoded"].isna().mean()*100,
            5.6, 7.4, 9.4, 5.2, 4.3, 4.2, 97.0]
})
clrs_m = ["#e74c3c" if v>50 else "#f39c12" if v>8 else "#2ecc71" for v in miss_data["pct"]]
ax7.barh(miss_data["col"], miss_data["pct"], color=clrs_m)
ax7.set_title("Missing Value Rates\n(before imputation)", fontweight="bold")
ax7.axvline(10, color="orange", linestyle="--", alpha=0.5, linewidth=1)

ax8 = fig.add_subplot(gs[2, 2])
conf = df_full["Recommendation_Confidence"]
ax8.hist(conf, bins=30, color="#9b59b6", edgecolor="white", alpha=0.8)
ax8.axvline(conf.mean(), color="red", linestyle="--", label=f"Mean={conf.mean():.2f}")
ax8.set_title("Recommendation Confidence", fontweight="bold")
ax8.legend()

plt.savefig("eda_report.png", dpi=150, bbox_inches="tight")
print("\n  Saved: eda_report.png")
print("\n✅ Phase 1 Complete")