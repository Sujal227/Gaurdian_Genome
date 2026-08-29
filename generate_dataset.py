"""
GUARDIAN GENOME — Longitudinal Synthetic Dataset Generator
===========================================================
Generates a realistic patient dataset where:
  - Each patient has 1, 2, or 3 linked visit rows (same Patient_ID)
  - Visit 1  → genetics-only recommendation (no history)
  - Visit 2  → genetics + outcome of Visit 1 informs dose/drug
  - Visit 3  → if Visit 2 was "No Change" or "Adverse Reaction",
               dose escalates OR drug switches

Design decisions grounded in the original dataset:
  - Same column schema as original (58 cols + 2 new longitudinal cols)
  - Missing value rates match original (2–9% per column, 96% Creatinine)
  - Numeric ranges match original (Age 18-90, GFR 5-130, etc.)
  - Drug-condition mappings preserved exactly
  - Gene-phenotype frequencies match original distributions
  - Comorbidity rates match original
  - ADR types and rates match original

Output: longitudinal_dataset.csv  (~1800 rows, ~600 unique patients)
"""

import random
import csv
import math
from collections import defaultdict

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — all grounded in original dataset analysis
# ─────────────────────────────────────────────────────────────────────────────

CONDITIONS = [
    "Hypertension", "Type 2 Diabetes", "Hyperlipidemia",
    "Atrial Fibrillation", "Depression", "Rheumatoid Arthritis",
    "HIV/AIDS", "Epilepsy", "Colorectal Cancer",
]
CONDITION_WEIGHTS = [320, 306, 261, 168, 174, 137, 135, 75, 204]

ETHNICITIES = ["South Asian", "Caucasian", "African", "East Asian", "Hispanic", "Middle Eastern"]
ETHNICITY_WEIGHTS = [594, 499, 352, 248, 126, 59]

SEXES = ["Male", "Female"]
SEX_WEIGHTS = [1033, 845]

ADR_TYPES = ["None", "Myopathy", "Hepatotoxicity", "Agranulocytosis", "Rash",
             "Nephrotoxicity", "Bleeding", "QT Prolongation", "Anaphylaxis", "Nausea/Vomiting"]
ADR_WEIGHTS = [1456, 76, 65, 48, 45, 43, 42, 40, 34, 29]

CURRENT_MEDS_POOL = [
    "None", "Metformin", "Aspirin", "Atorvastatin", "Furosemide",
    "Ibuprofen", "Lisinopril", "Omeprazole", "Pantoprazole",
    "Spironolactone", "Amlodipine", "Levothyroxine",
    "Furosemide; Atorvastatin", "Metformin; Lisinopril",
    "Omeprazole; Atorvastatin", "Aspirin; Pantoprazole",
]

RSID_MAP = {
    "*1/*1": ("rs1799853", "96702047", "C", "C"),
    "*1/*2": ("rs1799853", "96702047", "C", "T"),
    "*1/*3": ("rs1057910", "96741053", "A", "C"),
    "*2/*2": ("rs1799853", "96702047", "T", "T"),
    "*2/*3": ("rs1057910", "96741053", "A", "C"),
    "*3/*3": ("rs1057910", "96741053", "C", "C"),
}

# CYP2C9 allele → phenotype (exact from dataset)
CYP2C9_ALLELE_PHENO = {
    "*1/*1": "Normal Metabolizer",
    "*1/*2": "Intermediate Metabolizer",
    "*1/*3": "Intermediate Metabolizer",
    "*2/*2": "Poor Metabolizer",
    "*2/*3": "Poor Metabolizer",
    "*3/*3": "Poor Metabolizer",
}
CYP2C9_ALLELE_WEIGHTS = [941, 338, 235, 186, 125, 53]

CYP2D6_PHENOTYPES = ["Normal Metabolizer", "Intermediate Metabolizer",
                     "Poor Metabolizer", "Ultrarapid Metabolizer", "Rapid Metabolizer"]
CYP2D6_WEIGHTS = [800, 500, 200, 200, 178]

VKORC1_GENOS = ["GG", "GA", "AA"]
VKORC1_SENSITIVITY = {"GG": "Low Sensitivity", "GA": "Moderate Sensitivity", "AA": "High Sensitivity"}
VKORC1_WEIGHTS = [600, 800, 478]

SLCO1B1_FUNCS = ["Normal Function", "Decreased Function", "Poor Function"]
SLCO1B1_WEIGHTS = [1400, 350, 128]

CYP2C19_PHENOTYPES = ["Normal Metabolizer", "Intermediate Metabolizer",
                      "Poor Metabolizer", "Rapid Metabolizer", "Ultrarapid Metabolizer"]
CYP2C19_WEIGHTS = [800, 500, 200, 200, 178]

TPMT_ACTIVITIES = ["Normal Activity", "Intermediate Activity", "Low Activity", "Deficient Activity"]
TPMT_WEIGHTS = [1500, 200, 100, 78]

DPYD_FUNCS = ["Normal Function", "Decreased Function", "Non-functional"]
DPYD_WEIGHTS = [1600, 200, 78]

HLA_B5701 = ["Negative", "Positive"]
HLA_WEIGHTS = [1782, 96]

ZYGOSITY_OPTIONS = ["Homozygous", "Heterozygous"]

DRUG_INTERACTION_RISK = ["Low", "Moderate", "High"]
DRUG_RISK_WEIGHTS = [1300, 400, 178]

MED_COLS = ["Med_Amlodipine", "Med_Aspirin", "Med_Atorvastatin", "Med_Furosemide",
            "Med_Ibuprofen", "Med_Levothyroxine", "Med_Lisinopril", "Med_Metformin",
            "Med_Omeprazole", "Med_Pantoprazole", "Med_Spironolactone"]

# ─────────────────────────────────────────────────────────────────────────────
# DRUG RECOMMENDATION LOGIC
# Maps condition + gene profile → drug, dose, unit, flag, confidence
# ─────────────────────────────────────────────────────────────────────────────

def get_drug_recommendation(condition, cyp2c9, cyp2d6, vkorc1_geno, vkorc1_sens,
                             slco1b1, cyp2c19, tpmt, dpyd, hla,
                             prev_drug=None, prev_outcome=None, visit=1,
                             gfr=75, alt=25, age=50, weight=70):
    """
    Core pharmacogenomic recommendation engine.
    Returns (drug, dose_category, dose_min, dose_max, unit, flag, confidence)
    Visit 2+ logic: considers previous drug and outcome for dose adjustment.
    """
    flag = "None"
    confidence = round(random.uniform(0.82, 0.98), 3)

    # ── DOSE ESCALATION LOGIC (Visit 2 and 3) ────────────────────────────────
    # If returning visit with known outcome, adjust before genomic recommendation
    escalate = False
    switch_drug = False
    if visit >= 2 and prev_drug and prev_outcome:
        if prev_outcome == "No Change":
            escalate = True        # same drug, higher dose
        elif prev_outcome == "Adverse Reaction":
            switch_drug = True     # different drug in same class
            confidence = round(confidence - 0.07, 3)
        elif prev_outcome in ["Stable", "Improved"]:
            pass                   # continue same drug, same dose
        elif prev_outcome == "Switched Drug":
            switch_drug = True

    # ── CONDITION-SPECIFIC DRUG SELECTION ────────────────────────────────────

    # HYPERTENSION
    if condition == "Hypertension":
        if cyp2d6 == "Poor Metabolizer":
            drug = "Amlodipine"
            flag = "Avoid Metoprolol – CYP2D6 Poor Metabolizer"
            confidence -= 0.05
        elif switch_drug and prev_drug == "Metoprolol":
            drug = "Amlodipine"
        elif switch_drug and prev_drug == "Amlodipine":
            drug = "Metoprolol"
        else:
            drug = "Metoprolol"
            if cyp2d6 == "Ultrarapid Metabolizer":
                flag = "Ultrarapid – may need higher dose or alternative"
                confidence -= 0.03

        if drug == "Metoprolol":
            base_min, base_max = 25.0, 200.0
            unit = "mg/day"
            if escalate:
                dose_cat = "Increased Dose"
                dose_min = round(base_min * random.uniform(1.2, 1.5), 2)
                dose_max = round(base_max * random.uniform(1.2, 1.4), 2)
            elif visit == 1:
                dose_cat = "Standard Dose"
                dose_min = round(random.uniform(23.0, 32.0), 2)
                dose_max = round(random.uniform(190.0, 210.0), 2)
            else:
                dose_cat = "Standard Dose"
                dose_min = round(random.uniform(23.0, 32.0), 2)
                dose_max = round(random.uniform(190.0, 210.0), 2)
        else:  # Amlodipine
            base_min, base_max = 5.0, 10.0
            unit = "mg/day"
            if escalate:
                dose_cat = "Increased Dose"
                dose_min = round(random.uniform(7.5, 9.0), 2)
                dose_max = round(random.uniform(12.0, 15.0), 2)
            else:
                dose_cat = "Standard Dose"
                dose_min = round(random.uniform(4.5, 6.0), 2)
                dose_max = round(random.uniform(9.0, 11.0), 2)

    # TYPE 2 DIABETES
    elif condition == "Type 2 Diabetes":
        if cyp2c9 == "Poor Metabolizer":
            drug = "Metformin"
            flag = "Avoid Glipizide – CYP2C9 reduced function"
            confidence -= 0.05
        elif switch_drug and prev_drug == "Metformin":
            drug = "Metformin + Glipizide"
        elif switch_drug and prev_drug == "Metformin + Glipizide":
            drug = "Insulin (short-acting)"
        else:
            # Deterministic rule: normal/intermediate CYP2C9 -> combination therapy
            # This removes target noise while keeping the recommendation feature-driven.
            drug = "Metformin + Glipizide"
        unit = "mg/day"
        if drug == "Metformin":
            if escalate:
                dose_cat = "Increased Dose"
                dose_min = round(random.uniform(800, 1000), 2)
                dose_max = round(random.uniform(1800, 2200), 2)
            else:
                dose_cat = "Standard Dose"
                dose_min = round(random.uniform(490, 540), 2)
                dose_max = round(random.uniform(1950, 2100), 2)
        elif drug == "Metformin + Glipizide":
            if escalate:
                dose_cat = "Increased Dose"
                dose_min = round(random.uniform(800, 1000), 2)
                dose_max = round(random.uniform(1800, 2200), 2)
            else:
                dose_cat = "Standard Dose"
                dose_min = round(random.uniform(490, 540), 2)
                dose_max = round(random.uniform(1950, 2100), 2)
        else:  # Insulin
            dose_cat = "Standard Dose"
            dose_min = round(random.uniform(0.5, 2.0), 2)
            dose_max = round(random.uniform(3.0, 6.0), 2)
            unit = "units/kg/day"

    # HYPERLIPIDEMIA
    elif condition == "Hyperlipidemia":
        if slco1b1 == "Poor Function":
            drug = "Pravastatin"
            flag = "Avoid Simvastatin – SLCO1B1 Poor Function; Myopathy Risk"
            confidence -= 0.05
        elif slco1b1 == "Decreased Function":
            drug = random.choice(["Pravastatin", "Rosuvastatin"])
            flag = "Reduced SLCO1B1 – consider low-dose statin"
            confidence -= 0.03
        elif switch_drug and prev_drug == "Simvastatin":
            drug = "Rosuvastatin"
        elif switch_drug and prev_drug == "Rosuvastatin":
            drug = "Pravastatin"
        else:
            # Deterministic rule based on SLCO1B1 function.
            drug = "Simvastatin"
        unit = "mg/day"
        if escalate:
            dose_cat = "Increased Dose"
            dose_min = round(random.uniform(15.0, 25.0), 2)
            dose_max = round(random.uniform(55.0, 80.0), 2)
        elif slco1b1 in ["Decreased Function", "Poor Function"]:
            dose_cat = "Low Dose"
            dose_min = round(random.uniform(2.0, 4.0), 2)
            dose_max = round(random.uniform(10.0, 15.0), 2)
        else:
            dose_cat = "Standard Dose"
            dose_min = round(random.uniform(9.0, 11.0), 2)
            dose_max = round(random.uniform(38.0, 42.0), 2)

    # ATRIAL FIBRILLATION
    elif condition == "Atrial Fibrillation":
        drug = "Warfarin"
        unit = "mg/day"
        if hla == "Positive":   # just in case, shouldn't happen but safety
            flag = "HLA-B*5701 positive — monitor"
        if vkorc1_geno == "AA":
            dose_cat = "Reduced Dose"
            flag = "High Bleeding Risk – Monitor INR closely"
            confidence -= 0.02
            if escalate:
                dose_min = round(random.uniform(0.8, 1.2), 2)
                dose_max = round(random.uniform(4.5, 6.0), 2)
            else:
                dose_min = round(random.uniform(0.4, 0.7), 2)
                dose_max = round(random.uniform(3.5, 4.5), 2)
        elif vkorc1_geno == "GA":
            dose_cat = "Standard-Low Dose"
            flag = "Moderate Risk – Routine INR monitoring"
            if escalate:
                dose_min = round(random.uniform(0.6, 0.9), 2)
                dose_max = round(random.uniform(6.0, 8.0), 2)
            else:
                dose_min = round(random.uniform(0.35, 0.5), 2)
                dose_max = round(random.uniform(5.5, 6.5), 2)
        else:  # GG
            dose_cat = "Standard Dose"
            if escalate:
                dose_min = round(random.uniform(1.5, 3.0), 2)
                dose_max = round(random.uniform(12.0, 20.0), 2)
            else:
                dose_min = round(random.uniform(0.7, 1.2), 2)
                dose_max = round(random.uniform(8.0, 12.0), 2)

    # DEPRESSION
    elif condition == "Depression":
        if cyp2d6 == "Ultrarapid Metabolizer":
            drug = "Sertraline"
            flag = "Avoid Amitriptyline – Ultrarapid CYP2D6"
            confidence -= 0.03
        elif cyp2c19 == "Poor Metabolizer":
            drug = "Escitalopram"
            flag = "CYP2C19 Poor – Reduce SSRI dose"
            confidence -= 0.02
        elif switch_drug and prev_drug == "Escitalopram":
            drug = "Sertraline"
        elif switch_drug and prev_drug == "Sertraline":
            drug = "Escitalopram"
        else:
            # Deterministic rule: patients without poor CYP2C19 metabolism
            # receive Sertraline in the synthetic recommendation policy.
            drug = "Sertraline"
        unit = "mg/day"
        if cyp2c19 == "Poor Metabolizer":
            if escalate:
                dose_cat = "Reduced Dose"
                dose_min = round(random.uniform(4.0, 6.0), 2)
                dose_max = round(random.uniform(14.0, 18.0), 2)
            else:
                dose_cat = "Reduced Dose"
                dose_min = round(random.uniform(2.0, 3.5), 2)
                dose_max = round(random.uniform(10.0, 13.0), 2)
        else:
            if escalate:
                dose_cat = "Increased Dose"
                dose_min = round(random.uniform(10.0, 15.0), 2)
                dose_max = round(random.uniform(28.0, 40.0), 2)
            else:
                dose_cat = "Standard Dose"
                dose_min = round(random.uniform(4.5, 6.0), 2)
                dose_max = round(random.uniform(18.0, 22.0), 2)

    # RHEUMATOID ARTHRITIS
    elif condition == "Rheumatoid Arthritis":
        if tpmt == "Deficient Activity":
            drug = "Methotrexate"
            flag = "CONTRAINDICATED: TPMT Deficient – Azathioprine Myelotoxicity Risk"
            confidence -= 0.10
        elif tpmt == "Low Activity":
            drug = "Azathioprine"
            flag = "TPMT Low Activity – Reduce Azathioprine dose"
            confidence -= 0.03
        elif switch_drug and prev_drug == "Azathioprine":
            drug = "Methotrexate"
        elif switch_drug and prev_drug == "Methotrexate":
            drug = "Azathioprine"
        else:
            # Normal TPMT activity -> Azathioprine in the synthetic policy.
            drug = "Azathioprine"
        if drug == "Azathioprine":
            unit = "mg/kg/day"
            if tpmt == "Low Activity":
                dose_cat = "Reduced Dose (50%)"
                dose_min = round(random.uniform(0.4, 0.6), 2)
                dose_max = round(random.uniform(1.2, 1.6), 2)
            elif escalate:
                dose_cat = "Increased Dose"
                dose_min = round(random.uniform(1.2, 1.5), 2)
                dose_max = round(random.uniform(2.8, 3.5), 2)
            else:
                dose_cat = "Standard Dose"
                dose_min = round(random.uniform(0.9, 1.1), 2)
                dose_max = round(random.uniform(2.3, 2.7), 2)
        else:  # Methotrexate
            unit = "mg/week"
            dose_cat = "Standard Dose"
            dose_min = round(random.uniform(7.0, 8.0), 2)
            dose_max = round(random.uniform(24.0, 26.0), 2)

    # HIV/AIDS
    elif condition == "HIV/AIDS":
        if hla == "Positive":
            drug = "Tenofovir + Emtricitabine"
            flag = "CONTRAINDICATED: HLA-B*5701 Positive – Abacavir hypersensitivity"
            confidence -= 0.10
        elif switch_drug and prev_drug == "Abacavir + Lamivudine":
            drug = "Tenofovir + Emtricitabine"
        elif switch_drug and prev_drug == "Tenofovir + Emtricitabine":
            drug = "Abacavir + Lamivudine"
        else:
            # HLA-B*5701 negative -> Abacavir + Lamivudine in this synthetic policy.
            drug = "Abacavir + Lamivudine"
        unit = "mg/day"
        dose_cat = "Standard Dose"
        if drug == "Abacavir + Lamivudine":
            dose_min = round(random.uniform(555, 645), 2)
            dose_max = round(dose_min + random.uniform(0.0, 0.2), 2)
        else:
            dose_min = round(random.uniform(294, 307), 2)
            dose_max = round(dose_min + random.uniform(0.0, 10.0), 2)

    # EPILEPSY
    elif condition == "Epilepsy":
        drug = "Phenytoin"
        unit = "mg/day"
        if cyp2c9 == "Poor Metabolizer":
            flag = "CYP2C9 Poor – Risk of Phenytoin toxicity; monitor levels"
            confidence -= 0.03
        if escalate:
            dose_cat = "Increased Dose"
            dose_min = round(random.uniform(130, 160), 2)
            dose_max = round(random.uniform(480, 560), 2)
        else:
            dose_cat = "Standard Dose"
            dose_min = round(random.uniform(95, 107), 2)
            dose_max = round(random.uniform(380, 425), 2)

    # COLORECTAL CANCER
    elif condition == "Colorectal Cancer":
        if dpyd == "Non-functional":
            drug = "Irinotecan"
            flag = "CONTRAINDICATED: DPYD Non-functional – 5-FU Severe Toxicity Risk"
            confidence -= 0.10
        elif dpyd == "Decreased Function":
            drug = "Capecitabine"
            flag = "DPYD Decreased – Reduce fluoropyrimidine dose by 50%"
            confidence -= 0.03
        elif switch_drug and prev_drug == "5-Fluorouracil (5-FU)":
            drug = "Capecitabine"
        elif switch_drug and prev_drug == "Capecitabine":
            drug = "Irinotecan"
        else:
            # Normal DPYD function -> 5-FU in the synthetic recommendation policy.
            drug = "5-Fluorouracil (5-FU)"

        if drug in ["5-Fluorouracil (5-FU)", "Capecitabine"]:
            unit = "mg/m²/day"
            if dpyd == "Decreased Function":
                dose_cat = "Reduced Dose (50%)"
                if escalate:
                    dose_min = round(random.uniform(230, 260), 2)
                    dose_max = round(random.uniform(560, 700), 2)
                else:
                    dose_min = round(random.uniform(190, 230), 2)
                    dose_max = round(random.uniform(450, 560), 2)
            else:
                if escalate:
                    dose_cat = "Increased Dose"
                    dose_min = round(random.uniform(450, 500), 2)
                    dose_max = round(random.uniform(1100, 1200), 2)
                else:
                    dose_cat = "Standard Dose"
                    dose_min = round(random.uniform(373, 430), 2)
                    dose_max = round(random.uniform(936, 1070), 2)
        else:  # Irinotecan
            unit = "mg/m²"
            dose_cat = "Standard Dose"
            dose_min = round(random.uniform(120, 127), 2)
            dose_max = round(random.uniform(340, 366), 2)

    else:
        drug = "No Medication Required"
        dose_cat = ""
        dose_min = 0.0
        dose_max = 0.0
        unit = ""

    confidence = max(0.10, min(1.0, round(confidence, 3)))
    return drug, dose_cat, dose_min, dose_max, unit, flag, confidence


# ─────────────────────────────────────────────────────────────────────────────
# OUTCOME SIMULATOR
# Given drug + dose + genes, simulate what happens at next visit
# ─────────────────────────────────────────────────────────────────────────────

OUTCOME_PROBS = {
    # (condition, gene_risk_level) → {outcome: weight}
    "low_risk": {
        "Stable": 40, "Improved": 35, "No Change": 15,
        "Adverse Reaction": 5, "Switched Drug": 3, "Lost to Follow-up": 2
    },
    "moderate_risk": {
        "Stable": 30, "Improved": 25, "No Change": 25,
        "Adverse Reaction": 10, "Switched Drug": 7, "Lost to Follow-up": 3
    },
    "high_risk": {
        "Stable": 20, "Improved": 15, "No Change": 30,
        "Adverse Reaction": 20, "Switched Drug": 12, "Lost to Follow-up": 3
    },
}

def simulate_outcome(drug, dose_cat, cyp2c9, cyp2d6, vkorc1_geno,
                     slco1b1, tpmt, dpyd, hla, flag, prev_adr):
    """Simulate realistic treatment outcome for next visit."""
    risk = "low_risk"
    if ("CONTRAINDICATED" in flag or "Avoid" in flag or
            tpmt == "Deficient Activity" or dpyd == "Non-functional" or hla == "Positive"):
        risk = "high_risk"
    elif ("Reduce" in flag or "Risk" in flag or "monitor" in flag.lower() or
          cyp2c9 == "Poor Metabolizer" or cyp2d6 in ["Poor Metabolizer", "Ultrarapid Metabolizer"] or
          slco1b1 == "Decreased Function"):
        risk = "moderate_risk"
    if prev_adr and prev_adr != "None":
        risk = "moderate_risk" if risk == "low_risk" else "high_risk"
    if dose_cat == "Increased Dose":
        # Escalated dose slightly increases adverse reaction risk
        probs = dict(OUTCOME_PROBS[risk])
        probs["Adverse Reaction"] = probs.get("Adverse Reaction", 5) + 5
        probs["No Change"] = max(0, probs.get("No Change", 15) - 5)
    else:
        probs = OUTCOME_PROBS[risk]
    outcomes = list(probs.keys())
    weights = list(probs.values())
    return random.choices(outcomes, weights=weights)[0]


# ─────────────────────────────────────────────────────────────────────────────
# MISSING VALUE INJECTOR
# Matches original dataset missing rates per column
# ─────────────────────────────────────────────────────────────────────────────

MISSING_RATES = {
    "Weight_kg": 0.026, "Height_cm": 0.025, "BMI": 0.050,
    "GFR_mL_min": 0.044, "ALT_U_L": 0.038, "Quality_Score_PHRED": 0.045,
    "CYP2D6_Phenotype": 0.063, "SLCO1B1_Function": 0.060,
    "CYP2C19_Phenotype": 0.048, "TPMT_Activity": 0.080,
    "DPYD_Function": 0.090, "HLA_B5701": 0.051,
    "Creatinine_mg_dL": 0.966,   # intentionally high — will be dropped in preprocessing
    "Primary_Condition": 0.030,   # reduced slightly from 5.2% for longitudinal coherence
}

def maybe_missing(value, col):
    rate = MISSING_RATES.get(col, 0.0)
    return "" if random.random() < rate else value


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_patient_visits(patient_num):
    """Generate 1–3 linked visit rows for a single patient."""
    pid = f"PGX-{patient_num:04d}"

    # ── Static demographics (same across all visits) ──────────────────────────
    age = random.randint(18, 90)
    sex = random.choices(SEXES, weights=SEX_WEIGHTS)[0]
    ethnicity = random.choices(ETHNICITIES, weights=ETHNICITY_WEIGHTS)[0]
    weight = round(random.uniform(39.7, 115.6), 1)
    height = round(random.uniform(145.0, 196.0), 1)
    bmi = round(weight / ((height / 100) ** 2), 1)

    # ── Genomic profile (fixed for the patient, same across all visits) ───────
    cyp2c9_allele = random.choices(list(CYP2C9_ALLELE_PHENO.keys()),
                                    weights=CYP2C9_ALLELE_WEIGHTS)[0]
    cyp2c9_pheno = CYP2C9_ALLELE_PHENO[cyp2c9_allele]
    cyp2d6_pheno = random.choices(CYP2D6_PHENOTYPES, weights=CYP2D6_WEIGHTS)[0]
    vkorc1_geno = random.choices(VKORC1_GENOS, weights=VKORC1_WEIGHTS)[0]
    vkorc1_sens = VKORC1_SENSITIVITY[vkorc1_geno]
    slco1b1 = random.choices(SLCO1B1_FUNCS, weights=SLCO1B1_WEIGHTS)[0]
    cyp2c19 = random.choices(CYP2C19_PHENOTYPES, weights=CYP2C19_WEIGHTS)[0]
    tpmt = random.choices(TPMT_ACTIVITIES, weights=TPMT_WEIGHTS)[0]
    dpyd = random.choices(DPYD_FUNCS, weights=DPYD_WEIGHTS)[0]
    hla = random.choices(HLA_B5701, weights=HLA_WEIGHTS)[0]
    zygosity = random.choices(ZYGOSITY_OPTIONS, weights=[60, 40])[0]
    rsid_data = RSID_MAP.get(cyp2c9_allele, ("rs1799853", "96702047", "C", "C"))

    # ── Condition & comorbidities ─────────────────────────────────────────────
    condition = random.choices(CONDITIONS, weights=CONDITION_WEIGHTS)[0]
    comorbid_htn = 1 if random.random() < 0.228 else 0
    comorbid_t2d = 1 if random.random() < 0.096 else 0
    comorbid_hld = 1 if random.random() < 0.126 else 0
    comorbid_af  = 1 if random.random() < 0.052 else 0
    comorbid_ckd = 1 if random.random() < 0.102 else 0
    comorbid_obs = 1 if random.random() < 0.271 else 0

    # ── Lab values (stable with slight variation across visits) ───────────────
    gfr_base = round(random.uniform(5.0, 130.0), 1)
    alt_base = round(random.uniform(7.0, 100.0), 1)
    creatinine = round(random.uniform(0.5, 3.5), 2)

    # ── Previous ADR (pre-existing, before study) ─────────────────────────────
    prev_adr = random.choices(ADR_TYPES, weights=ADR_WEIGHTS)[0]

    # ── Current medications ────────────────────────────────────────────────────
    current_meds = random.choice(CURRENT_MEDS_POOL)
    drug_interaction_risk = random.choices(DRUG_INTERACTION_RISK, weights=DRUG_RISK_WEIGHTS)[0]

    # ── Decide number of visits (1, 2, or 3) ─────────────────────────────────
    # Weights: ~55% have 1 visit, ~30% have 2, ~15% have 3
    # (matches real-world follow-up rates for chronic conditions)
    n_visits = random.choices([1, 2, 3], weights=[55, 30, 15])[0]

    rows = []
    prev_drug = None
    prev_outcome = None
    prev_dose_cat = None

    for visit_num in range(1, n_visits + 1):
        # Lab values vary slightly each visit (realistic)
        gfr = round(max(5.0, min(130.0, gfr_base + random.uniform(-5, 5))), 1)
        alt = round(max(7.0, min(350.0, alt_base + random.uniform(-5, 15))), 1)
        phred = round(random.uniform(31.5, 60.0), 1)

        # Get drug recommendation for this visit
        drug, dose_cat, dose_min, dose_max, unit, flag, confidence = get_drug_recommendation(
            condition, cyp2c9_pheno, cyp2d6_pheno, vkorc1_geno, vkorc1_sens,
            slco1b1, cyp2c19, tpmt, dpyd, hla,
            prev_drug=prev_drug, prev_outcome=prev_outcome,
            visit=visit_num, gfr=gfr, alt=alt, age=age, weight=weight
        )

        # Simulate outcome (used for next visit)
        if visit_num < n_visits:
            outcome = simulate_outcome(
                drug, dose_cat, cyp2c9_pheno, cyp2d6_pheno, vkorc1_geno,
                slco1b1, tpmt, dpyd, hla, flag, prev_adr
            )
        else:
            outcome = "First Visit" if visit_num == 1 else random.choices(
                ["Stable", "Improved", "No Change", "Adverse Reaction",
                 "Switched Drug", "Lost to Follow-up"],
                weights=[35, 30, 20, 8, 5, 2]
            )[0]

        # DATA ENTRY ERROR rows: ~5% of all rows
        data_error = random.random() < 0.05
        if data_error:
            flag = "DATA ENTRY ERROR – VERIFY"
            confidence = round(random.uniform(0.10, 0.45), 3)

        # Med flags
        med_flags = {col: 0 for col in MED_COLS}
        if current_meds != "None":
            for med_str in current_meds.split(";"):
                med_str = med_str.strip()
                col_name = "Med_" + med_str.replace(" ", "")
                if col_name in med_flags:
                    med_flags[col_name] = 1

        row = {
            "Patient_ID": pid,
            "Age": age,
            "Sex": sex,
            "Ethnicity": ethnicity,
            "Weight_kg": maybe_missing(weight, "Weight_kg"),
            "Height_cm": maybe_missing(height, "Height_cm"),
            "BMI": maybe_missing(round(bmi, 1), "BMI"),
            "Primary_Condition": maybe_missing(condition, "Primary_Condition"),
            "Comorbid_Hypertension": comorbid_htn,
            "Comorbid_T2Diabetes": comorbid_t2d,
            "Comorbid_Hyperlipidemia": comorbid_hld,
            "Comorbid_AtrialFibrillation": comorbid_af,
            "Comorbid_ChronicKidneyDis": comorbid_ckd,
            "Comorbid_Obesity": comorbid_obs,
            "GFR_mL_min": maybe_missing(gfr, "GFR_mL_min"),
            "ALT_U_L": maybe_missing(alt, "ALT_U_L"),
            "rsID_Primary": rsid_data[0],
            "Chromosome": "10",
            "Genomic_Position": rsid_data[1],
            "Ref_Allele": rsid_data[2],
            "Alt_Allele": rsid_data[3],
            "Zygosity": zygosity,
            "Quality_Score_PHRED": maybe_missing(phred, "Quality_Score_PHRED"),
            "CYP2C9_Alleles": cyp2c9_allele,
            "CYP2C9_Phenotype": cyp2c9_pheno,
            "CYP2D6_Phenotype": maybe_missing(cyp2d6_pheno, "CYP2D6_Phenotype"),
            "VKORC1_Genotype": vkorc1_geno,
            "VKORC1_Sensitivity": vkorc1_sens,
            "SLCO1B1_Function": maybe_missing(slco1b1, "SLCO1B1_Function"),
            "CYP2C19_Phenotype": maybe_missing(cyp2c19, "CYP2C19_Phenotype"),
            "TPMT_Activity": maybe_missing(tpmt, "TPMT_Activity"),
            "DPYD_Function": maybe_missing(dpyd, "DPYD_Function"),
            "HLA_B5701": maybe_missing(hla, "HLA_B5701"),
            "Previous_ADR": prev_adr,
            "Current_Medications": current_meds,
            "Drug_Interaction_Risk": drug_interaction_risk,
            "Visit_Number": visit_num,
            "Previous_Drug": prev_drug if prev_drug else "None",
            "Previous_Outcome": prev_outcome if prev_outcome else "None",  # NEW COLUMN
            "Treatment_Outcome": outcome,
            "Recommended_Drug": drug,
            "Dose_Category": dose_cat,
            "Dose_Min_mg": dose_min,
            "Dose_Max_mg": dose_max,
            "Dose_Unit": unit,
            "Clinical_Flag": flag,
            "Recommendation_Confidence": confidence,
            "Creatinine_mg_dL": maybe_missing(creatinine, "Creatinine_mg_dL"),
            **med_flags,
        }
        rows.append(row)

        # Carry forward for next visit
        prev_drug = drug
        prev_outcome = outcome

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: GENERATE ALL PATIENTS
# ─────────────────────────────────────────────────────────────────────────────

print("Generating longitudinal dataset...")
print("Target: ~5000 patients, ~13000–14000 rows, 1–3 visits per patient")

all_rows = []
patient_num = 1
target_patients = 5000  # Experiment 1: scale patient count only; preserve generation rules

while patient_num <= target_patients:
    patient_rows = generate_patient_visits(patient_num)
    all_rows.extend(patient_rows)
    patient_num += 1

print(f"Generated {patient_num - 1} unique patients, {len(all_rows)} total rows")

# Shuffle within patient groups (keep visit order within patient)
# Sort by Patient_ID then Visit_Number for clean output
all_rows.sort(key=lambda r: (r["Patient_ID"], r["Visit_Number"]))

# ─────────────────────────────────────────────────────────────────────────────
# WRITE CSV
# ─────────────────────────────────────────────────────────────────────────────

fieldnames = list(all_rows[0].keys())
output_path = "longitudinal_dataset.csv"

with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Saved: {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION REPORT
# ─────────────────────────────────────────────────────────────────────────────
from collections import Counter

print("\n=== DATASET VALIDATION ===")
print(f"Total rows: {len(all_rows)}")
print(f"Unique patients: {len(set(r['Patient_ID'] for r in all_rows))}")

visit_dist = Counter(str(r['Visit_Number']) for r in all_rows)
print(f"Visit distribution: {dict(sorted(visit_dist.items()))}")

# Verify longitudinal links
patient_visits = defaultdict(list)
for r in all_rows:
    patient_visits[r['Patient_ID']].append(r)
multi = {p: vs for p, vs in patient_visits.items() if len(vs) > 1}
print(f"Patients with multiple visits: {len(multi)}")

# Verify drug continuity
continuity_ok = 0
escalation_ok = 0
for pid, visits in multi.items():
    visits_sorted = sorted(visits, key=lambda r: r['Visit_Number'])
    for i in range(1, len(visits_sorted)):
        v_prev = visits_sorted[i-1]
        v_curr = visits_sorted[i]
        # Previous_Drug in current visit should match Recommended_Drug of previous visit
        if v_curr['Previous_Drug'] == v_prev['Recommended_Drug']:
            continuity_ok += 1
        # If prev outcome was No Change, check dose escalated
        if v_prev['Treatment_Outcome'] == 'No Change':
            if v_curr['Dose_Category'] in ['Increased Dose', 'Standard Dose']:
                escalation_ok += 1

total_transitions = sum(len(vs)-1 for vs in multi.values())
print(f"Drug continuity check: {continuity_ok}/{total_transitions} transitions correct "
      f"({100*continuity_ok/max(total_transitions,1):.1f}%)")

# Missing value rates
print("\nMissing value rates:")
for col in ["Weight_kg","Height_cm","BMI","GFR_mL_min","ALT_U_L",
            "CYP2D6_Phenotype","SLCO1B1_Function","TPMT_Activity","DPYD_Function",
            "Creatinine_mg_dL"]:
    missing = sum(1 for r in all_rows if str(r.get(col, "")).strip() == "")
    print(f"  {col}: {missing/len(all_rows)*100:.1f}%")

# Drug distribution
print("\nTop 10 recommended drugs:")
for drug, cnt in Counter(r['Recommended_Drug'] for r in all_rows).most_common(10):
    print(f"  {drug}: {cnt}")

# Outcome distribution
print("\nTreatment outcomes:")
for outcome, cnt in Counter(r['Treatment_Outcome'] for r in all_rows).most_common():
    print(f"  {outcome}: {cnt}")

# Dose category distribution  
print("\nDose categories:")
for cat, cnt in Counter(r['Dose_Category'] for r in all_rows).most_common():
    print(f"  {repr(cat)}: {cnt}")

# Dose escalation verification
v2_escalated = [r for r in all_rows
                if str(r['Visit_Number']) == '2' and r['Previous_Outcome'] == 'No Change'
                and r['Dose_Category'] == 'Increased Dose']
v2_no_change = [r for r in all_rows
                if str(r['Visit_Number']) == '2' and r['Previous_Outcome'] == 'No Change']
print(f"\nDose escalation when Visit 1 outcome = No Change:")
print(f"  {len(v2_escalated)}/{len(v2_no_change)} patients got Increased Dose at Visit 2")

print("\n✅ Dataset generation complete: longitudinal_dataset.csv")