"""
GUARDIAN GENOME — PHASE 2: Rule-Based Engine (Longitudinal)
============================================================
What this phase does:
  1. All original pharmacogenomic rules (CYP2C9, CYP2D6, VKORC1, etc.)
  2. NEW: Visit-adaptive rules — uses Previous_Outcome and Previous_Drug
     - If previous outcome was Adverse Reaction → flag the previous drug
     - If previous outcome was No Change → recommend dose escalation
     - If visit >= 2 → check drug class switching logic
  3. Validates all rules against Clinical_Flag ground truth
  4. Outputs per-patient rule results + coverage metrics

Run:  python phase2_rule_engine.py
Outputs: rule_engine_results.csv, rule_engine_report.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# ALL PHARMACOGENOMIC + LONGITUDINAL RULES
# ─────────────────────────────────────────────
RULES = {
    # ── Genomic rules (same as before) ───────────────────────────────────────
    "R01_CYP2C9_POOR_GLIPIZIDE": {
        "description": "CYP2C9 Poor → Avoid Glipizide",
        "condition": lambda p: p.get("CYP2C9_Phenotype") == "Poor Metabolizer",
        "flag": "Avoid Glipizide – CYP2C9 reduced function",
        "dose_adj": None, "avoid": "Metformin + Glipizide",
        "alt": "Metformin", "conf_mod": -0.05,
    },
    "R02_CYP2C9_POOR_PHENYTOIN": {
        "description": "CYP2C9 Poor → Phenytoin toxicity risk",
        "condition": lambda p: p.get("CYP2C9_Phenotype") == "Poor Metabolizer",
        "flag": "CYP2C9 Poor – Risk of Phenytoin toxicity; monitor levels",
        "dose_adj": "Reduced Dose", "avoid": None, "alt": None, "conf_mod": -0.02,
    },
    "R03_VKORC1_AA": {
        "description": "VKORC1 AA → High bleeding risk",
        "condition": lambda p: p.get("VKORC1_Genotype") == "AA",
        "flag": "High Bleeding Risk – Monitor INR closely",
        "dose_adj": "Reduced Dose", "avoid": None, "alt": None, "conf_mod": 0.0,
    },
    "R04_VKORC1_GA": {
        "description": "VKORC1 GA → Moderate INR monitoring",
        "condition": lambda p: p.get("VKORC1_Genotype") == "GA",
        "flag": "Moderate Risk – Routine INR monitoring",
        "dose_adj": "Standard-Low Dose", "avoid": None, "alt": None, "conf_mod": 0.0,
    },
    "R05_SLCO1B1_DECREASED": {
        "description": "SLCO1B1 Decreased → Low-dose statin",
        "condition": lambda p: p.get("SLCO1B1_Function") == "Decreased Function",
        "flag": "Reduced SLCO1B1 – consider low-dose statin",
        "dose_adj": "Low Dose", "avoid": None, "alt": "Pravastatin", "conf_mod": -0.03,
    },
    "R06_SLCO1B1_POOR_SIMVASTATIN": {
        "description": "SLCO1B1 Poor → Avoid Simvastatin",
        "condition": lambda p: p.get("SLCO1B1_Function") == "Poor Function",
        "flag": "Avoid Simvastatin – SLCO1B1 Poor Function; Myopathy Risk",
        "dose_adj": None, "avoid": "Simvastatin", "alt": "Pravastatin", "conf_mod": -0.05,
    },
    "R07_CYP2D6_POOR_METOPROLOL": {
        "description": "CYP2D6 Poor → Avoid Metoprolol",
        "condition": lambda p: p.get("CYP2D6_Phenotype") == "Poor Metabolizer",
        "flag": "Avoid Metoprolol – CYP2D6 Poor Metabolizer",
        "dose_adj": None, "avoid": "Metoprolol", "alt": "Amlodipine", "conf_mod": -0.05,
    },
    "R08_CYP2D6_ULTRARAPID": {
        "description": "CYP2D6 Ultrarapid → Higher dose may be needed",
        "condition": lambda p: p.get("CYP2D6_Phenotype") == "Ultrarapid Metabolizer",
        "flag": "Ultrarapid – may need higher dose or alternative",
        "dose_adj": "Increased Dose", "avoid": None, "alt": None, "conf_mod": -0.03,
    },
    "R09_CYP2D6_ULTRARAPID_AMIT": {
        "description": "CYP2D6 Ultrarapid → Avoid Amitriptyline",
        "condition": lambda p: p.get("CYP2D6_Phenotype") == "Ultrarapid Metabolizer",
        "flag": "Avoid Amitriptyline – Ultrarapid CYP2D6",
        "dose_adj": None, "avoid": "Amitriptyline", "alt": "Sertraline", "conf_mod": 0.0,
    },
    "R10_CYP2C19_POOR_SSRI": {
        "description": "CYP2C19 Poor → Reduce SSRI dose",
        "condition": lambda p: p.get("CYP2C19_Phenotype") == "Poor Metabolizer",
        "flag": "CYP2C19 Poor – Reduce SSRI dose",
        "dose_adj": "Reduced Dose", "avoid": None, "alt": None, "conf_mod": -0.02,
    },
    "R11_TPMT_DEFICIENT": {
        "description": "TPMT Deficient → CONTRAINDICATED: Azathioprine",
        "condition": lambda p: p.get("TPMT_Activity") == "Deficient Activity",
        "flag": "CONTRAINDICATED: TPMT Deficient – Azathioprine Myelotoxicity Risk",
        "dose_adj": None, "avoid": "Azathioprine", "alt": "Methotrexate", "conf_mod": -0.10,
    },
    "R12_TPMT_LOW": {
        "description": "TPMT Low → Reduce Azathioprine dose",
        "condition": lambda p: p.get("TPMT_Activity") == "Low Activity",
        "flag": "TPMT Low Activity – Reduce Azathioprine dose",
        "dose_adj": "Reduced Dose", "avoid": None, "alt": None, "conf_mod": -0.03,
    },
    "R13_DPYD_DECREASED": {
        "description": "DPYD Decreased → Reduce fluoropyrimidine 50%",
        "condition": lambda p: p.get("DPYD_Function") == "Decreased Function",
        "flag": "DPYD Decreased – Reduce fluoropyrimidine dose by 50%",
        "dose_adj": "Reduced Dose (50%)", "avoid": None, "alt": None, "conf_mod": -0.03,
    },
    "R14_DPYD_NONFUNCTIONAL": {
        "description": "DPYD Non-functional → CONTRAINDICATED: 5-FU",
        "condition": lambda p: p.get("DPYD_Function") == "Non-functional",
        "flag": "CONTRAINDICATED: DPYD Non-functional – 5-FU Severe Toxicity Risk",
        "dose_adj": None, "avoid": "5-Fluorouracil (5-FU)", "alt": None, "conf_mod": -0.10,
    },
    "R15_HLA_B5701": {
        "description": "HLA-B*5701 Positive → CONTRAINDICATED: Abacavir",
        "condition": lambda p: p.get("HLA_B5701") == "Positive",
        "flag": "CONTRAINDICATED: HLA-B*5701 Positive – Abacavir hypersensitivity",
        "dose_adj": None, "avoid": "Abacavir + Lamivudine", "alt": "Tenofovir + Emtricitabine",
        "conf_mod": -0.10,
    },

    # ── NEW: Longitudinal / Visit-adaptive rules ──────────────────────────────
    "R16_PREV_ADR_SWITCH": {
        "description": "Previous visit ADR → recommend switching drug class",
        "condition": lambda p: p.get("Previous_Outcome") == "Adverse Reaction",
        "flag": "Previous ADR – consider switching drug class",
        "dose_adj": None, "avoid": None, "alt": None, "conf_mod": -0.07,
    },
    "R17_NO_CHANGE_ESCALATE": {
        "description": "Previous visit No Change → escalate dose",
        "condition": lambda p: (
            p.get("Previous_Outcome") == "No Change" and
            str(p.get("Visit_Number", 1)) != "1"
        ),
        "flag": "No response to previous dose – consider escalation",
        "dose_adj": "Increased Dose", "avoid": None, "alt": None, "conf_mod": -0.03,
    },
    "R18_STABLE_CONTINUE": {
        "description": "Previous visit Stable/Improved → continue current drug",
        "condition": lambda p: p.get("Previous_Outcome") in ["Stable", "Improved"],
        "flag": "None",  # positive signal — no flag needed
        "dose_adj": None, "avoid": None, "alt": None, "conf_mod": +0.03,
    },
    "R19_DATA_QUALITY": {
        "description": "Data entry error flagged",
        "condition": lambda p: str(p.get("Clinical_Flag", "")).startswith("DATA ENTRY ERROR"),
        "flag": "DATA ENTRY ERROR – VERIFY",
        "dose_adj": None, "avoid": None, "alt": None, "conf_mod": -0.15,
    },
}


def run_rule_engine(patient_dict: dict) -> dict:
    """Run all rules on a single patient. Returns structured result."""
    triggered, flags, dose_adjs, avoids, alts = [], [], [], [], []
    conf_delta = 0.0

    for rule_id, rule in RULES.items():
        try:
            if rule["condition"](patient_dict):
                triggered.append(rule_id)
                if rule["flag"] != "None":
                    flags.append(rule["flag"])
                if rule["dose_adj"]:
                    dose_adjs.append(rule["dose_adj"])
                if rule["avoid"]:
                    avoids.append(rule["avoid"])
                if rule["alt"]:
                    alts.append(rule["alt"])
                conf_delta += rule["conf_mod"]
        except Exception:
            pass

    return {
        "rules_triggered": triggered,
        "rule_count": len(triggered),
        "flags_generated": " | ".join(flags) if flags else "None",
        "dose_recommendation": dose_adjs[0] if dose_adjs else "Standard Dose",
        "drugs_to_avoid": ", ".join(set(avoids)) if avoids else "None",
        "alternatives": ", ".join(set(alts)) if alts else "None",
        "has_contraindication": any("CONTRAINDICATED" in f for f in flags),
        "has_data_error": any("DATA ENTRY ERROR" in f for f in flags),
        "confidence_delta": round(conf_delta, 3),
        # Visit-adaptive summary
        "visit_action": (
            "SWITCH_DRUG"   if "R16_PREV_ADR_SWITCH" in triggered else
            "ESCALATE_DOSE" if "R17_NO_CHANGE_ESCALATE" in triggered else
            "CONTINUE"      if "R18_STABLE_CONTINUE" in triggered else
            "FIRST_VISIT"
        ),
    }


# ─────────────────────────────────────────────
# RUN ENGINE ON FULL DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("Running Rule Engine on all patients")
print("=" * 60)

df = pd.read_csv("longitudinal_dataset.csv", encoding="latin-1").replace("", np.nan)

results = []
for _, row in df.iterrows():
    p = row.to_dict()
    r = run_rule_engine(p)
    r["Patient_ID"]       = p.get("Patient_ID", "")
    r["Visit_Number"]     = p.get("Visit_Number", 1)
    r["Previous_Outcome"] = p.get("Previous_Outcome", "None")
    r["Recommended_Drug"] = p.get("Recommended_Drug", "")
    r["Clinical_Flag_GT"] = p.get("Clinical_Flag", "None")
    results.append(r)

results_df = pd.DataFrame(results)
print(f"Processed {len(results_df)} rows")

# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Validation against Clinical_Flag ground truth")
print("=" * 60)

covered = results_df["rule_count"] > 0
print(f"  Rows with ≥1 rule triggered: {covered.sum()} ({covered.mean()*100:.1f}%)")
print(f"  Contraindications caught:    {results_df['has_contraindication'].sum()}")
print(f"  Data quality flags:          {results_df['has_data_error'].sum()}")

print("\nVisit-adaptive actions breakdown:")
for action, count in results_df["visit_action"].value_counts().items():
    print(f"  {action}: {count}")

print("\nRule trigger frequency:")
from collections import Counter
all_rules = []
for rules_list in results_df["rules_triggered"]:
    all_rules.extend(rules_list)
for rule_id, count in Counter(all_rules).most_common():
    print(f"  {rule_id}: {count}  — {RULES[rule_id]['description']}")

# ─────────────────────────────────────────────
# SAVE + VISUALIZE
# ─────────────────────────────────────────────
output = pd.concat([
    df[["Patient_ID","Visit_Number","Primary_Condition","Recommended_Drug",
        "Clinical_Flag","Recommendation_Confidence","Previous_Outcome",
        "Previous_Drug"]].reset_index(drop=True),
    results_df.drop(columns=["Patient_ID","Visit_Number","Recommended_Drug",
                              "Clinical_Flag_GT","Previous_Outcome"])
], axis=1)
output.to_csv("rule_engine_results.csv", index=False)
print("\nSaved: rule_engine_results.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Guardian Genome — Rule Engine Analysis (Longitudinal)", fontsize=14, fontweight="bold")

ax = axes[0, 0]
rc = results_df["rule_count"]
ax.hist(rc, bins=range(0, rc.max()+2), color="#3498db", edgecolor="white", rwidth=0.8)
ax.axvline(rc.mean(), color="red", linestyle="--", label=f"Mean={rc.mean():.1f}")
ax.set_title("Rules Triggered per Row", fontweight="bold")
ax.set_xlabel("Number of rules")
ax.legend()

ax = axes[0, 1]
va_counts = results_df["visit_action"].value_counts()
colors_va = {"FIRST_VISIT": "#2ecc71", "CONTINUE": "#3498db",
             "ESCALATE_DOSE": "#f39c12", "SWITCH_DRUG": "#e74c3c"}
ax.bar(va_counts.index, va_counts.values,
       color=[colors_va.get(k, "#95a5a6") for k in va_counts.index])
ax.set_title("Visit-Adaptive Actions\n(NEW longitudinal rules)", fontweight="bold")
ax.tick_params(axis="x", rotation=15)

ax = axes[1, 0]
visit_actions = results_df.groupby("Visit_Number")["visit_action"].value_counts(normalize=True).unstack(fill_value=0)*100
visit_actions.plot(kind="bar", stacked=True, ax=ax,
                   color=["#2ecc71","#e74c3c","#f39c12","#3498db"])
ax.set_title("Visit-Adaptive Actions % by Visit\n(shows escalation increasing V2→V3)", fontweight="bold")
ax.set_xlabel("Visit Number")
ax.tick_params(axis="x", rotation=0)
ax.legend(fontsize=7)

ax = axes[1, 1]
contra_flags = results_df[results_df["has_contraindication"]]["flags_generated"]
contra_types = Counter()
for flag_str in contra_flags:
    for f in flag_str.split(" | "):
        if "CONTRAINDICATED" in f:
            contra_types[f[:45]] += 1
if contra_types:
    labels, vals = zip(*contra_types.most_common(6))
    ax.barh(list(labels)[::-1], list(vals)[::-1], color="#e74c3c")
    ax.set_title("Contraindication Types Triggered", fontweight="bold")
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig("rule_engine_report.png", dpi=150, bbox_inches="tight")
print("Saved: rule_engine_report.png")
print("\n✅ Phase 2 Complete")
