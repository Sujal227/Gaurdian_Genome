"""
GUARDIAN GENOME — PHASE 5: Web App (Full Role-Based System)
============================================================
Architecture:
  - SQLite database (guardian_genome.db) — persists across restarts
  - 4 tables: users, patients, visit_records, login_log
  - Doctor: register/login → patient list → predict → save records → audit log
  - Patient: register/login → home → chatbot → doctors list → settings

Bottom nav (both roles): Home · [Patients/Chat] · [Predict/Doctors] · Settings · Sign out
Settings (both): Change password (optional) · Light/Dark mode · Font size

Run: python phase5_app.py
Visit: http://localhost:5000
"""

from flask import (Flask, request, jsonify, render_template_string,
                   session, redirect, url_for)
import sqlite3, hashlib, json, os, datetime, random, re, difflib, html
import numpy as np, joblib
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = os.environ.get("GUARDIAN_SECRET_KEY", "guardian_genome_dev_key_change_me")
DB_PATH = "guardian_genome.db"
# Domain-structured assistant: deterministic, contextual and explainable.
CHATBOT_READY = True

# ── Load ML artifacts ─────────────────────────────────────────────────────────
try:
    # IMPORTANT: use the NEW tuned Random Forest models.
    # rf_v1.pkl is the OLD HistGradientBoosting model (72 features).
    # The revised V1 model expects 78 features and the revised Full model
    # expects 82 features.
    rf_v1 = joblib.load("rf_v1_tuned.joblib")
    rf_full = joblib.load("rf_full_tuned.joblib")

    rf_v1_imputer = joblib.load("rf_v1_imputer.joblib")
    rf_full_imputer = joblib.load("rf_full_imputer.joblib")

    with open("feature_list_v1.json", encoding="utf-8") as f:
        FEATURES_V1 = json.load(f)
    with open("feature_list_full.json", encoding="utf-8") as f:
        FEATURES_FULL = json.load(f)
    with open("drug_label_map.json", encoding="utf-8") as f:
        drug_map = json.load(f)

    # Sanity-check the model/feature contracts at startup.
    if getattr(rf_v1, "n_features_in_", None) != len(FEATURES_V1):
        raise ValueError(
            f"V1 model expects {getattr(rf_v1, 'n_features_in_', '?')} "
            f"features but feature_list_v1.json contains {len(FEATURES_V1)}."
        )
    if getattr(rf_full, "n_features_in_", None) != len(FEATURES_FULL):
        raise ValueError(
            f"Full model expects {getattr(rf_full, 'n_features_in_', '?')} "
            f"features but feature_list_full.json contains {len(FEATURES_FULL)}."
        )

    ML_READY = True
    print(
        f"[ML] Loaded tuned RF models: "
        f"V1={len(FEATURES_V1)} features, "
        f"Full={len(FEATURES_FULL)} features"
    )
except Exception as e:
    ML_READY = False
    print(f"[WARN] ML not loaded: {e}. Run phases 1-3 first.")
    FEATURES_V1 = FEATURES_FULL = []
    drug_map = {}
    rf_v1_imputer = rf_full_imputer = None

LONG_FEATS    = ["Previous_Outcome_encoded","Previous_ADR_flag","Is_Return_Visit"]
GENOMIC_FEATS = ["CYP2C9_Phenotype_encoded","CYP2D6_Phenotype_encoded","VKORC1_Genotype_encoded",
                 "VKORC1_Sensitivity_encoded","SLCO1B1_Function_encoded","CYP2C19_Phenotype_encoded",
                 "TPMT_Activity_encoded","DPYD_Function_encoded","HLA_B5701_encoded","CYP2C9_AlleleScore"]
OUTCOME_ORDINAL = {"None":0,"First Visit":0,"Lost to Follow-up":0,"No Change":1,
                   "Adverse Reaction":-1,"Switched Drug":-1,"Stable":2,"Improved":3}
ORDINAL_MAPS = {
    "CYP2C9_Phenotype_encoded":    {"Poor Metabolizer":0,"Intermediate Metabolizer":1,"Normal Metabolizer":2,"Rapid Metabolizer":3,"Ultrarapid Metabolizer":4},
    "CYP2D6_Phenotype_encoded":    {"Poor Metabolizer":0,"Intermediate Metabolizer":1,"Normal Metabolizer":2,"Rapid Metabolizer":3,"Ultrarapid Metabolizer":4},
    "CYP2C19_Phenotype_encoded":   {"Poor Metabolizer":0,"Intermediate Metabolizer":1,"Normal Metabolizer":2,"Rapid Metabolizer":3,"Ultrarapid Metabolizer":4},
    "VKORC1_Sensitivity_encoded":  {"High Sensitivity":2,"Moderate Sensitivity":1,"Low Sensitivity":0},
    "SLCO1B1_Function_encoded":    {"Non-functional":0,"Poor Function":0,"Decreased Function":1,"Normal Function":2},
    "DPYD_Function_encoded":       {"Non-functional":0,"Decreased Function":1,"Normal Function":2},
    "TPMT_Activity_encoded":       {"Deficient Activity":0,"Low Activity":1,"Intermediate Activity":2,"Normal Activity":3},
    "Drug_Interaction_Risk_encoded":{"Low":0,"Moderate":1,"High":2},
    "VKORC1_Genotype_encoded":     {"GG":0,"GA":1,"AA":2},
    "Zygosity_encoded":            {"Homozygous":0,"Heterozygous":1},
    "HLA_B5701_encoded":           {"Negative":0,"Positive":1},
}

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('doctor','patient')),
        full_name TEXT, email TEXT, specialisation TEXT, hospital TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active INTEGER DEFAULT 1,
        password_changed_at TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT UNIQUE NOT NULL,
        user_id INTEGER REFERENCES users(id),
        registered_by INTEGER REFERENCES users(id),
        full_name TEXT NOT NULL, age INTEGER,
        primary_condition TEXT,
        registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        assigned_doctor INTEGER REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS visit_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        doctor_id INTEGER REFERENCES users(id),
        visit_number INTEGER,
        recommended_drug TEXT, dose_category TEXT,
        treatment_outcome TEXT, clinical_flag TEXT,
        confidence REAL, visit_action TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS login_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER REFERENCES users(id),
        login_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        patient_id_accessed TEXT, action TEXT, ip_address TEXT
    );
    """)
    conn.commit(); conn.close()

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def gen_patient_id():
    conn = get_db()
    existing = {r["patient_id"] for r in conn.execute("SELECT patient_id FROM patients")}
    conn.close()
    while True:
        pid = f"PGX-{random.randint(1000,9999)}"
        if pid not in existing: return pid

def log_action(user_id, action, patient_id=None):
    conn = get_db()
    conn.execute("INSERT INTO login_log (user_id,action,patient_id_accessed,ip_address) VALUES (?,?,?,?)",
                 (user_id, action, patient_id, request.remote_addr))
    conn.commit(); conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# ML HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def encode_patient(form, visit_number=1, prev_outcome="None"):
    """
    Build the exact 82-feature representation used by the revised Full RF.

    This mirrors the feature construction in phase1_preprocessing.py:
      - ordinal genomic encodings
      - clinical values
      - disease/ethnicity one-hot features
      - pharmacogenomic × condition interactions
      - longitudinal history
      - previous-drug one-hot features

    The V1 prediction later selects only FEATURES_V1 from this dictionary.
    """

    enc = {f: 0 for f in FEATURES_FULL}

    # -----------------------------
    # Numeric clinical features
    # -----------------------------
    for field in ["Age", "Weight_kg", "BMI", "GFR_mL_min", "ALT_U_L"]:
        try:
            enc[field] = float(form.get(field, 0) or 0)
        except (TypeError, ValueError):
            enc[field] = 0.0

    enc["Sex_encoded"] = 1 if form.get("Sex") == "Male" else 0
    enc["Visit_Number"] = int(visit_number)

    # -----------------------------
    # Genomic ordinal encodings
    # -----------------------------
    for col, mapping in ORDINAL_MAPS.items():
        raw_key = col.replace("_encoded", "")
        raw_value = form.get(raw_key, "") or form.get(col, "")
        enc[col] = mapping.get(raw_value, 0)

    # -----------------------------
    # CYP2C9 allele score
    # Same scoring used in Phase 1.
    # -----------------------------
    allele_scores = {"*1": 1.0, "*2": 0.5, "*3": 0.0}
    alleles = str(form.get("CYP2C9_Alleles", "*1/*1") or "*1/*1")

    try:
        a1, a2 = [a.strip() for a in alleles.split("/", 1)]
        enc["CYP2C9_AlleleScore"] = (
            allele_scores.get(a1, 0.5)
            + allele_scores.get(a2, 0.5)
        )
    except (ValueError, AttributeError):
        enc["CYP2C9_AlleleScore"] = 2.0

    # -----------------------------
    # Normalize ethnicity labels from
    # the app UI to dataset labels.
    # -----------------------------
    ethnicity_ui_to_data = {
        "Asian": "East Asian",
        "Black": "African",
        "Other": "Middle Eastern",
    }

    eth = form.get("Ethnicity", "Caucasian")
    eth = ethnicity_ui_to_data.get(eth, eth)

    eth_feature = f"Ethnicity_{eth}"
    if eth_feature in enc:
        enc[eth_feature] = 1

    # -----------------------------
    # Primary condition one-hot
    # -----------------------------
    cond = form.get("Primary_Condition", "Hypertension")
    cond_feature = f"Primary_Condition_{cond}"
    if cond_feature in enc:
        enc[cond_feature] = 1

    # -----------------------------
    # Pharmacogenomic × condition
    # interactions — same definitions
    # used by phase1_preprocessing.py.
    # -----------------------------
    cyp2c9 = form.get("CYP2C9_Phenotype", "")
    slco = form.get("SLCO1B1_Function", "")
    cyp2c19 = form.get("CYP2C19_Phenotype", "")
    tpmt = form.get("TPMT_Activity", "")
    dpyd = form.get("DPYD_Function", "")
    hla = form.get("HLA_B5701", "")

    enc["CYP2C9_Diabetes_interaction"] = int(
        cyp2c9 == "Poor Metabolizer"
        and cond == "Type 2 Diabetes"
    )

    enc["SLCO1B1_Lipid_interaction"] = int(
        slco in [
            "Poor Function",
            "Non-functional",
            "Decreased Function",
        ]
        and cond == "Hyperlipidemia"
    )

    enc["CYP2C19_Depression_interaction"] = int(
        cyp2c19 == "Poor Metabolizer"
        and cond == "Depression"
    )

    enc["TPMT_RA_interaction"] = int(
        tpmt in ["Deficient Activity", "Low Activity"]
        and cond == "Rheumatoid Arthritis"
    )

    enc["DPYD_CRC_interaction"] = int(
        dpyd in ["Non-functional", "Decreased Function"]
        and cond == "Colorectal Cancer"
    )

    enc["HLA_HIV_interaction"] = int(
        hla == "Positive"
        and cond == "HIV/AIDS"
    )

    # -----------------------------
    # Longitudinal features
    # -----------------------------
    enc["Previous_Outcome_encoded"] = OUTCOME_ORDINAL.get(
        prev_outcome, 0
    )

    enc["Previous_ADR_flag"] = int(
        prev_outcome == "Adverse Reaction"
    )

    enc["Is_Return_Visit"] = int(
        int(visit_number) > 1
    )

    # The app does not ask for a current dose before prediction,
    # so this remains 0 rather than inventing a dose-escalation event.
    enc["Dose_Escalated_flag"] = 0

    # -----------------------------
    # Previous-drug one-hot features
    # -----------------------------
    prev_drug = str(
        form.get("prev_drug", "None") or "None"
    )

    # Make sure the app's "None" state matches Phase 1.
    if prev_drug.strip() == "":
        prev_drug = "None"

    prev_feature = f"Previous_Drug_{prev_drug}"
    if prev_feature in enc:
        enc[prev_feature] = 1

    return enc

def run_rules(form, prev_outcome="None"):
    flags, dose, conf_mod = [], "Standard Dose", 0.0
    cyp2c9=form.get("CYP2C9_Phenotype",""); cyp2d6=form.get("CYP2D6_Phenotype","")
    vkorc1=form.get("VKORC1_Genotype","");  slco=form.get("SLCO1B1_Function","")
    tpmt=form.get("TPMT_Activity","");      dpyd=form.get("DPYD_Function","")
    hla=form.get("HLA_B5701","");           cyp2c19=form.get("CYP2C19_Phenotype","")
    if cyp2c9=="Poor Metabolizer":          flags.append("Avoid Glipizide – CYP2C9 reduced function"); conf_mod-=0.05
    if cyp2d6=="Poor Metabolizer":          flags.append("Avoid Metoprolol – CYP2D6 Poor Metabolizer"); conf_mod-=0.05
    if cyp2d6=="Ultrarapid Metabolizer":    flags.append("Ultrarapid CYP2D6 – may need higher dose"); conf_mod-=0.03
    if vkorc1=="AA":   flags.append("High Bleeding Risk – Monitor INR closely"); dose="Reduced Dose"; conf_mod-=0.02
    elif vkorc1=="GA": flags.append("Moderate Risk – Routine INR monitoring"); dose="Standard-Low Dose"
    if slco=="Poor Function":               flags.append("Avoid Simvastatin – SLCO1B1 Poor Function"); conf_mod-=0.05
    if slco=="Decreased Function":          flags.append("Reduced SLCO1B1 – consider low-dose statin"); conf_mod-=0.03
    if cyp2c19=="Poor Metabolizer":         flags.append("CYP2C19 Poor – Reduce SSRI dose"); conf_mod-=0.02
    if tpmt=="Deficient Activity":          flags.append("CONTRAINDICATED: TPMT Deficient – Azathioprine Myelotoxicity Risk"); conf_mod-=0.10
    if tpmt=="Low Activity":                flags.append("TPMT Low Activity – Reduce Azathioprine dose"); conf_mod-=0.03
    if dpyd=="Non-functional":              flags.append("CONTRAINDICATED: DPYD Non-functional – 5-FU Severe Toxicity Risk"); conf_mod-=0.10
    if dpyd=="Decreased Function":          flags.append("DPYD Decreased – Reduce fluoropyrimidine 50%"); conf_mod-=0.03
    if hla=="Positive":                     flags.append("CONTRAINDICATED: HLA-B*5701 Positive – Abacavir hypersensitivity"); conf_mod-=0.10
    if prev_outcome=="Adverse Reaction":    flags.append("Previous ADR – consider switching drug class"); conf_mod-=0.07
    if prev_outcome=="No Change":           flags.append("No response to previous dose – consider dose escalation"); dose="Increased Dose"; conf_mod-=0.03
    if prev_outcome in ["Stable","Improved"]: flags.append("Previous treatment effective – continuing regimen"); conf_mod+=0.03
    return {"flags":flags if flags else ["None"],"dose":dose,"conf_mod":conf_mod,
            "visit_action":("SWITCH" if prev_outcome=="Adverse Reaction" else
                            "ESCALATE" if prev_outcome=="No Change" else
                            "CONTINUE" if prev_outcome in ["Stable","Improved"] else "FIRST_VISIT")}

# ─────────────────────────────────────────────────────────────────────────────
# SHARED STYLES
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
:root{--bg:#f4f7fb;--bg2:#fff;--bg3:#eef3f8;--panel:#fff;--bd:#dce5ee;--tx:#172536;--tx2:#617386;--tx3:#8a9aaa;--ac:#167c68;--ac2:#e5f5f0;--doc:#2b67c7;--doc2:#eaf1ff;--pat:#167c68;--danger:#c94a4a;--danger2:#fff0f0;--warn:#a66a00;--warn2:#fff6df;--ok:#257a4b;--ok2:#eaf7ef;--shadow:0 8px 30px rgba(24,43,65,.07);--shadow2:0 2px 10px rgba(24,43,65,.05);--r:16px;--rs:10px}
body.light{--bg:#f4f7fb;--bg2:#fff;--bg3:#eef3f8}
body:not(.light){--bg:#0d1724;--bg2:#132234;--bg3:#0b1622;--panel:#132234;--bd:#24384d;--tx:#e7eef5;--tx2:#94a8ba;--tx3:#60758a;--shadow:0 12px 32px rgba(0,0,0,.20);--shadow2:0 3px 12px rgba(0,0,0,.16)}
*{box-sizing:border-box;margin:0;padding:0}html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--tx);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px;line-height:1.5}
body.lf{--fs:18px;font-size:var(--fs)}body.lf h1{font-size:2.25rem}body.lf h2{font-size:1.8rem}body.lf h3{font-size:1.45rem}
a{color:inherit}.card{background:var(--panel);border:1px solid var(--bd);border-radius:var(--r);padding:1.15rem;box-shadow:var(--shadow2)}
label{display:block;color:var(--tx2);font-size:.76rem;font-weight:600;margin-bottom:.34rem}
select,input[type=text],input[type=number],input[type=password],input[type=email],textarea{width:100%;padding:.65rem .78rem;background:var(--bg3);border:1px solid var(--bd);border-radius:10px;color:var(--tx);font-size:.86rem;transition:.18s}
select:focus,input:focus,textarea:focus{outline:none;border-color:var(--ac);box-shadow:0 0 0 3px rgba(22,124,104,.12)}
.btn{padding:.64rem 1rem;border:0;border-radius:10px;cursor:pointer;font-size:.82rem;font-weight:700;transition:.18s;display:inline-flex;align-items:center;justify-content:center;gap:.35rem}.btn:hover{transform:translateY(-1px)}.btn:disabled{opacity:.55;cursor:not-allowed;transform:none}
.bp{background:var(--ac);color:#fff}.bd2{background:transparent;border:1px solid var(--bd);color:var(--tx2)}.bdoc{background:var(--doc);color:#fff}.bpat{background:var(--pat);color:#fff}.bdanger{background:var(--danger);color:#fff}
.badge{display:inline-flex;align-items:center;gap:.25rem;padding:.23rem .58rem;border-radius:999px;font-size:.68rem;font-weight:800}.bdoc-b{background:var(--doc2);color:var(--doc)}.bpat-b{background:var(--ac2);color:var(--pat)}.bwarn-b{background:var(--warn2);color:var(--warn)}.bdanger-b{background:var(--danger2);color:var(--danger)}.bok-b{background:var(--ok2);color:var(--ok)}
.fr{margin-bottom:.8rem}.g2{display:grid;grid-template-columns:1fr 1fr;gap:.7rem}.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem}
.sh{font-size:.68rem;font-weight:800;color:var(--tx3);text-transform:uppercase;letter-spacing:.1em;margin:1.15rem 0 .5rem;padding-top:.85rem;border-top:1px solid var(--bd)}.sh:first-child{border-top:0;margin-top:0;padding-top:0}
.hdr{height:66px;background:var(--bg2);border-bottom:1px solid var(--bd);padding:.75rem max(1rem,calc((100vw - 1180px)/2));display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:50}.logo{font-weight:800;font-size:1rem;color:var(--ac)}
.pw{max-width:1180px;margin:0 auto;padding:1.2rem 1rem 6rem}.bnav{position:fixed;bottom:0;left:50%;transform:translateX(-50%);width:min(760px,100%);background:var(--bg2);border:1px solid var(--bd);border-bottom:0;border-radius:18px 18px 0 0;display:flex;z-index:100;box-shadow:0 -8px 25px rgba(20,35,50,.06)}
.ni{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:.58rem .2rem;cursor:pointer;color:var(--tx3);font-size:.67rem;gap:3px;border:0;background:transparent}.ni:hover,.ni.adoc{color:var(--doc)}.ni.apat{color:var(--pat)}.ni.so:hover{color:var(--danger)}.ni-ic{font-size:1.05rem;line-height:1}
.alert{padding:.72rem .9rem;border-radius:11px;font-size:.79rem;margin-bottom:.7rem;border:1px solid var(--bd)}.aerr{background:var(--danger2);border-color:#f0c4c4;color:var(--danger)}.aok{background:var(--ok2);border-color:#c8e8d4;color:var(--ok)}.ainfo{background:var(--doc2);border-color:#cbdafa;color:var(--doc)}
.hero{border:1px solid var(--bd);background:linear-gradient(135deg,var(--bg2),var(--ac2));border-radius:22px;padding:1.35rem;margin-bottom:.9rem;position:relative;overflow:hidden}.eyebrow{font-size:.68rem;font-weight:800;letter-spacing:.11em;text-transform:uppercase;color:var(--ac);margin-bottom:.35rem}.hero h1{font-size:1.7rem;letter-spacing:-.035em;line-height:1.15;margin-bottom:.35rem}.hero p{color:var(--tx2);max-width:720px;font-size:.86rem}
.metric{padding:1rem;border:1px solid var(--bd);background:var(--panel);border-radius:15px}.metric .num{font-size:1.7rem;font-weight:800;letter-spacing:-.04em}.metric .lbl{font-size:.71rem;color:var(--tx2);margin-top:.15rem}
.section-title{display:flex;align-items:center;justify-content:space-between;margin:1.15rem 0 .55rem}.section-title h3{font-size:.88rem}.section-title span{font-size:.7rem;color:var(--tx3)}
.quick-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem}.quick{padding:.9rem;border:1px solid var(--bd);border-radius:14px;background:var(--panel);cursor:pointer;transition:.18s}.quick:hover{border-color:var(--ac);transform:translateY(-1px)}.quick .qi{font-size:1.25rem;margin-bottom:.45rem}.quick b{font-size:.8rem}.quick p{font-size:.7rem;color:var(--tx2);margin-top:.15rem}
.timeline{position:relative;padding-left:1.15rem}.timeline:before{content:"";position:absolute;left:.31rem;top:.35rem;bottom:.35rem;width:1px;background:var(--bd)}.tl-item{position:relative;padding:0 0 .85rem .65rem}.tl-dot{position:absolute;left:-.98rem;top:.25rem;width:9px;height:9px;border-radius:50%;background:var(--ac);box-shadow:0 0 0 4px var(--ac2)}.tl-date{font-size:.67rem;color:var(--tx3)}.tl-main{font-size:.82rem;font-weight:700}.tl-sub{font-size:.72rem;color:var(--tx2);margin-top:.08rem}
.chat-shell{height:calc(100vh - 145px);min-height:560px;display:grid;grid-template-rows:auto 1fr auto;border:1px solid var(--bd);border-radius:20px;background:var(--panel);overflow:hidden;box-shadow:var(--shadow)}.chat-head{padding:.9rem 1rem;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:.65rem;background:linear-gradient(90deg,var(--ac2),transparent)}.chat-avatar{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:var(--ac);color:#fff;font-weight:800}.chat-title{font-weight:800;font-size:.88rem}.chat-sub{font-size:.68rem;color:var(--tx2)}
.chat-msgs{overflow-y:auto;padding:1rem;display:flex;flex-direction:column;gap:.75rem}.mw{display:flex;flex-direction:column}.mw.u{align-items:flex-end}.bubble{padding:.72rem .88rem;border-radius:14px;font-size:.82rem;line-height:1.65;max-width:min(720px,88%)}.bu{background:var(--doc);color:#fff;border-bottom-right-radius:4px}.bb{background:var(--bg3);border:1px solid var(--bd);color:var(--tx);border-bottom-left-radius:4px}.mt{font-size:.62rem;color:var(--tx3);margin-top:2px}.qps{display:flex;flex-wrap:wrap;gap:.38rem;margin-top:.5rem}.qp{font-size:.7rem;padding:.38rem .68rem;border-radius:999px;border:1px solid var(--bd);background:var(--bg2);color:var(--tx2);cursor:pointer}.qp:hover{border-color:var(--ac);color:var(--ac)}.chat-input{padding:.7rem;border-top:1px solid var(--bd);display:flex;gap:.5rem;background:var(--bg2)}.chat-input input{flex:1}
.vc{border:1px solid var(--bd);border-radius:14px;padding:.85rem;margin-bottom:.55rem;background:var(--bg2)}.dc{border:1px solid var(--bd);border-radius:14px;padding:.9rem;margin-bottom:.6rem;display:flex;gap:.7rem;background:var(--panel)}.dc:hover{border-color:var(--ac)}.av{width:42px;height:42px;border-radius:13px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.82rem;flex-shrink:0}.av-d{background:var(--doc2);color:var(--doc)}.av-p{background:var(--ac2);color:var(--ac)}
.sr{display:flex;align-items:center;justify-content:space-between;padding:.8rem 0;border-bottom:1px solid var(--bd);font-size:.84rem}.sr:last-child{border-bottom:none}.tgl{position:relative;width:40px;height:22px;display:inline-block}.tgl input{opacity:0;width:0;height:0}.tsl{position:absolute;cursor:pointer;inset:0;background:var(--bd);border-radius:22px;transition:.3s}.tsl:before{content:"";position:absolute;width:16px;height:16px;left:3px;bottom:3px;background:#fff;border-radius:50%;transition:.3s}input:checked+.tsl{background:var(--ac)}input:checked+.tsl:before{transform:translateX(18px)}
.pr{display:flex;align-items:center;gap:9px;padding:.8rem 0;border-bottom:1px solid var(--bd);cursor:pointer}.pr:hover{color:var(--ac)}.pr:last-child{border-bottom:none}
.predict-layout{display:grid;grid-template-columns:minmax(320px,430px) 1fr;gap:1rem;align-items:start}.sticky-card{position:sticky;top:82px;max-height:calc(100vh - 105px);overflow:auto}.stepbar{display:flex;gap:.35rem;margin-bottom:1rem}.step{height:4px;flex:1;border-radius:9px;background:var(--bd)}.step.on{background:var(--ac)}
.result-hero{border:1px solid var(--bd);border-radius:18px;padding:1.15rem;background:linear-gradient(135deg,var(--bg2),var(--ac2));margin-bottom:.75rem}.result-drug{font-size:1.55rem;font-weight:850;letter-spacing:-.04em}.confidence{height:9px;border-radius:99px;background:var(--bg3);overflow:hidden;margin-top:.5rem}.confidence>span{display:block;height:100%;background:var(--ac);border-radius:99px}.model-grid{display:grid;grid-template-columns:1fr 1fr;gap:.65rem}.model-card{padding:.8rem;border:1px solid var(--bd);border-radius:14px;background:var(--bg2)}.model-card.primary{border-color:var(--ac);box-shadow:0 0 0 2px var(--ac2)}
.explain-grid{display:grid;grid-template-columns:1fr 1fr;gap:.65rem}.why-card{border:1px solid var(--bd);border-radius:14px;padding:.85rem;background:var(--panel)}.why-row{display:grid;grid-template-columns:minmax(120px,1.15fr) 1fr 58px;gap:.5rem;align-items:center;margin:.5rem 0}.why-name{font-size:.7rem;color:var(--tx2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.why-track{height:9px;background:var(--bg3);border-radius:99px;overflow:hidden}.why-fill{height:100%;border-radius:99px}.why-fill.pos{background:var(--ac)}.why-fill.neg{background:var(--danger)}.why-val{font-size:.67rem;text-align:right;color:var(--tx3)}.legend{display:flex;gap:.8rem;flex-wrap:wrap;font-size:.68rem;color:var(--tx2)}.legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}.dot-pos{background:var(--ac)}.dot-neg{background:var(--danger)}
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:.65rem}.kpi{padding:.8rem;border:1px solid var(--bd);border-radius:13px;background:var(--bg2)}.kpi b{font-size:1.05rem}.kpi span{display:block;color:var(--tx2);font-size:.66rem;margin-top:.1rem}.empty{padding:2rem 1rem;text-align:center;color:var(--tx3)}.muted{color:var(--tx2)}.small{font-size:.72rem}.divider{height:1px;background:var(--bd);margin:.8rem 0}
@media(max-width:850px){.predict-layout{grid-template-columns:1fr}.sticky-card{position:static;max-height:none}.quick-grid{grid-template-columns:1fr}.kpi-row{grid-template-columns:1fr 1fr}.explain-grid{grid-template-columns:1fr}.pw{padding:.9rem .75rem 5.8rem}.hero h1{font-size:1.45rem}}
@media(max-width:560px){.g2,.g3,.model-grid{grid-template-columns:1fr}.hdr{padding:.7rem .8rem}.bubble{max-width:94%}.chat-shell{min-height:calc(100vh - 145px)}}

/* ── Guardian Genome landing page ─────────────────────────────────────── */
.gg-home{min-height:100vh;background:#071522;color:#edf5fb;overflow:hidden}
.gg-home *{box-sizing:border-box}
.gg-top{max-width:1240px;margin:0 auto;padding:28px 34px;display:flex;align-items:center;justify-content:space-between;position:relative;z-index:2}
.gg-brand{display:flex;align-items:center;gap:11px;font-weight:800;letter-spacing:-.02em}
.gg-brand-mark{width:36px;height:36px;border-radius:11px;background:linear-gradient(135deg,#19b897,#2b67c7);display:grid;place-items:center;box-shadow:0 8px 25px rgba(25,184,151,.18)}
.gg-brand small{display:block;font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#7891a5;font-weight:700;margin-top:1px}
.gg-top-link{color:#a9bdcc;text-decoration:none;font-size:12px;padding:9px 13px;border:1px solid #20374a;border-radius:10px;background:#0d2030}
.gg-main{max-width:1240px;margin:0 auto;padding:52px 34px 70px;display:grid;grid-template-columns:minmax(0,1.08fr) minmax(380px,.92fr);gap:48px;align-items:center;position:relative;z-index:1}
.gg-copy .eyebrow{color:#28c3a4;font-size:11px;letter-spacing:.17em;margin-bottom:18px}
.gg-copy h1{font-size:clamp(48px,6vw,82px);line-height:.94;letter-spacing:-.065em;margin:0 0 25px;max-width:760px;color:#f2f7fb}
.gg-copy h1 span{color:#20b99b}
.gg-copy p{font-size:17px;line-height:1.75;color:#91aabd;max-width:700px;margin-bottom:28px}
.gg-pills{display:flex;flex-wrap:wrap;gap:9px}
.gg-pill{padding:8px 12px;border:1px solid #203a4f;background:#0d2233;color:#9db2c2;border-radius:999px;font-size:11px}
.gg-visual-wrap{position:relative;min-height:540px;display:flex;align-items:center;justify-content:center}
.gg-orb{position:absolute;width:420px;height:420px;border-radius:50%;background:radial-gradient(circle,rgba(31,188,157,.22),rgba(31,188,157,0) 68%);filter:blur(2px)}
.gg-visual{position:relative;width:min(540px,100%);height:520px;border:1px solid #294458;border-radius:30px;background:linear-gradient(145deg,#10283a,#081b2a);overflow:hidden;box-shadow:0 30px 80px rgba(0,0,0,.32)}
.gg-visual img{width:100%;height:100%;object-fit:cover;display:block;opacity:.92}
.gg-visual:after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(7,21,34,.03),rgba(7,21,34,.32));pointer-events:none}
.gg-portal{position:absolute;right:20px;bottom:20px;width:min(390px,calc(100% - 40px));padding:22px;border:1px solid #294458;border-radius:20px;background:rgba(13,30,45,.94);backdrop-filter:blur(18px);box-shadow:0 20px 50px rgba(0,0,0,.3);z-index:3}
.gg-portal-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:15px}
.gg-portal h2{font-size:20px;letter-spacing:-.025em}
.gg-secure{font-size:10px;font-weight:800;color:#12644f;background:#dff7ef;padding:6px 9px;border-radius:999px}
.gg-role{display:flex;align-items:center;gap:13px;padding:15px;border:1px solid #294458;border-radius:15px;background:#0c2030;text-decoration:none;color:#edf5fb;margin-top:9px;transition:.18s}
.gg-role:hover{border-color:#20b99b;transform:translateY(-1px)}
.gg-role-icon{width:43px;height:43px;border-radius:13px;display:grid;place-items:center;font-weight:850;font-size:11px;background:#18385e;color:#a8c9ff;flex:none}
.gg-role-icon.pat{background:#0d443e;color:#8de4d2}
.gg-role b{font-size:13px}.gg-role span{display:block;color:#7891a5;font-size:10px;margin-top:3px}.gg-arrow{margin-left:auto;color:#5e8196;font-size:16px}
.gg-note{border-top:1px solid #233c4e;margin-top:15px;padding-top:14px;color:#7891a5;font-size:10px;line-height:1.65}
.gg-bottom{max-width:1240px;margin:0 auto;padding:0 34px 35px;display:flex;justify-content:space-between;gap:20px;color:#607a8e;font-size:10px;position:relative;z-index:1}
@media(max-width:900px){.gg-main{grid-template-columns:1fr;padding-top:30px}.gg-visual-wrap{min-height:460px}.gg-visual{height:460px}.gg-copy h1{font-size:clamp(46px,11vw,70px)}}
@media(max-width:560px){.gg-top{padding:20px}.gg-main{padding:28px 20px 50px;gap:28px}.gg-copy p{font-size:14px}.gg-visual-wrap{min-height:390px}.gg-visual{height:390px;border-radius:22px}.gg-portal{right:12px;bottom:12px;width:calc(100% - 24px);padding:16px}.gg-portal h2{font-size:17px}.gg-bottom{padding:0 20px 24px;display:block}}

/* GUARDIAN GENOME — VISUAL SYSTEM V2 */
:root{--gg-blue:#2d7ff9;--gg-teal:#13b8a2;--gg-cyan:#58d9e8;--gg-line:rgba(126,177,214,.18)}
body:not(.light){background:radial-gradient(circle at 12% 0%,rgba(30,117,183,.14),transparent 30%),radial-gradient(circle at 88% 10%,rgba(19,184,162,.10),transparent 28%),linear-gradient(180deg,#071321 0%,#0a1726 52%,#07121f 100%)}
body:not(.light) .hdr{height:74px;background:rgba(7,19,33,.88);backdrop-filter:blur(18px);border-bottom:1px solid rgba(107,177,213,.14);box-shadow:0 8px 28px rgba(0,0,0,.14)}
.hdr .logo{display:flex;align-items:center;gap:.65rem;color:#f2f8fc;font-size:1.04rem;letter-spacing:-.02em}.hdr .logo:before{content:'GG';display:grid;place-items:center;width:35px;height:35px;border-radius:11px;background:linear-gradient(145deg,#0e8f82,#1ac7ba);color:white;font-size:.75rem;font-weight:900;box-shadow:0 8px 24px rgba(19,184,162,.22)}
body:not(.light) .pw{max-width:1260px;padding:1.5rem 1.15rem 7rem}
body:not(.light) .hero{background:radial-gradient(circle at 92% 10%,rgba(50,151,255,.15),transparent 28%),radial-gradient(circle at 10% 100%,rgba(19,184,162,.10),transparent 34%),linear-gradient(135deg,rgba(17,39,59,.96),rgba(10,28,43,.94));border-color:rgba(100,175,216,.16);box-shadow:0 0 0 1px rgba(89,210,229,.06),0 22px 55px rgba(0,0,0,.22);padding:1.55rem 1.65rem}
body:not(.light) .hero:after{content:'✧';position:absolute;right:28px;top:20px;color:rgba(88,217,232,.18);font-size:4rem;line-height:1}body:not(.light) .hero h1{font-size:2rem;color:#f4f9fc}body:not(.light) .hero p{color:#9fb6c9;max-width:780px}body:not(.light) .eyebrow{color:#42d3c0}
body:not(.light) .card,body:not(.light) .metric,body:not(.light) .quick{background:linear-gradient(145deg,rgba(17,38,57,.94),rgba(11,27,43,.94));border-color:rgba(105,171,209,.15);box-shadow:0 12px 34px rgba(0,0,0,.13),inset 0 1px 0 rgba(255,255,255,.025)}
body:not(.light) .card:hover{border-color:rgba(77,195,215,.24)}body:not(.light) .metric{padding:1.15rem 1.2rem;position:relative;overflow:hidden}body:not(.light) .metric:after{content:'';position:absolute;right:-25px;bottom:-32px;width:100px;height:100px;border-radius:50%;background:rgba(19,184,162,.07)}body:not(.light) .metric .num{font-size:1.9rem;color:#f2f8fc}body:not(.light) .metric .lbl{color:#8ea8bc}
body:not(.light) .quick{padding:1.05rem;transition:.22s;border-radius:17px}body:not(.light) .quick:hover{transform:translateY(-3px);border-color:rgba(62,210,194,.42);box-shadow:0 16px 34px rgba(0,0,0,.22)}body:not(.light) .quick .qi{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:rgba(22,184,162,.10);color:#5ce0cf;font-size:1.1rem}body:not(.light) .quick b{color:#edf5fa;font-size:.84rem}body:not(.light) .quick p{color:#829bad}
body:not(.light) .section-title{margin-top:1.45rem}body:not(.light) .section-title h3{color:#eaf3f8;font-size:.94rem}body:not(.light) .section-title span{color:#718ba0}
body:not(.light) .bnav{bottom:14px;width:min(760px,calc(100% - 28px));border:1px solid rgba(107,177,213,.16);border-radius:18px;background:rgba(8,23,37,.88);backdrop-filter:blur(20px);box-shadow:0 18px 48px rgba(0,0,0,.32)}body:not(.light) .ni{padding:.68rem .2rem;color:#6f899d}body:not(.light) .ni:hover,body:not(.light) .ni.adoc{color:#58a4ff}body:not(.light) .ni.apat{color:#35d1bd}body:not(.light) .ni.so:hover{color:#ff7777}
body:not(.light) input,body:not(.light) select,body:not(.light) textarea{background:rgba(6,20,33,.74);border-color:rgba(104,169,208,.18);color:#edf5fa;box-shadow:inset 0 1px 0 rgba(255,255,255,.025)}body:not(.light) input:focus,body:not(.light) select:focus,body:not(.light) textarea:focus{border-color:#32cfc0;box-shadow:0 0 0 3px rgba(50,207,192,.10)}body:not(.light) label{color:#9bb2c4}
body:not(.light) .pr{border-color:rgba(100,170,207,.10)!important;background:rgba(8,24,39,.38);border-radius:13px;transition:.18s}body:not(.light) .pr:hover{background:rgba(22,50,71,.58)!important;border-color:rgba(64,202,191,.22)!important}body:not(.light) .divider{border-color:rgba(100,170,207,.12)}body:not(.light) .chat-shell{background:linear-gradient(145deg,rgba(13,32,48,.96),rgba(8,22,35,.96));border-color:rgba(104,170,209,.16)}body:not(.light) .chat-head{background:linear-gradient(90deg,rgba(17,91,100,.26),transparent)}body:not(.light) .chat-avatar{background:linear-gradient(145deg,#128d81,#18c3b2);box-shadow:0 8px 22px rgba(18,141,129,.22)}
/* Rich authentication experience */
.gg-auth{min-height:100vh;display:grid;grid-template-columns:minmax(0,1.08fr) minmax(460px,.92fr);background:#071321;color:#eaf4f8;overflow:hidden}.gg-auth-visual{position:relative;min-height:100vh;padding:clamp(2rem,5vw,5rem);display:flex;flex-direction:column;justify-content:space-between;overflow:hidden;background:radial-gradient(circle at 72% 32%,rgba(36,128,195,.23),transparent 34%),radial-gradient(circle at 25% 78%,rgba(19,184,162,.13),transparent 32%),linear-gradient(145deg,#071321,#0b2132)}.gg-auth-visual:before{content:'';position:absolute;width:620px;height:620px;right:-220px;top:-130px;border-radius:50%;border:1px solid rgba(85,211,229,.13);box-shadow:0 0 0 70px rgba(85,211,229,.025),0 0 0 140px rgba(85,211,229,.018)}.gg-auth-visual:after{content:'DNA';position:absolute;right:9%;top:22%;font-size:12rem;font-weight:900;letter-spacing:-.12em;color:rgba(80,205,223,.035);transform:rotate(-14deg)}
.gg-brand{position:relative;z-index:2;display:flex;align-items:center;gap:.8rem;font-size:1.25rem;font-weight:850;letter-spacing:-.025em}.gg-brand-mark{width:48px;height:48px;border-radius:15px;display:grid;place-items:center;background:linear-gradient(145deg,#0e8f82,#18c8b6);color:#fff;font-weight:900;box-shadow:0 12px 30px rgba(19,184,162,.2)}.gg-brand em{font-style:normal;color:#39cbbb}.gg-auth-copy{position:relative;z-index:2;max-width:660px;margin:auto 0}.gg-auth-kicker{color:#43d8c6;font-size:.7rem;letter-spacing:.17em;text-transform:uppercase;font-weight:850;margin-bottom:.9rem}.gg-auth-copy h1{font-size:clamp(2.7rem,5vw,5.2rem);line-height:.98;letter-spacing:-.065em;margin-bottom:1.15rem;color:#f3f8fb;max-width:720px}.gg-auth-copy p{font-size:1rem;line-height:1.75;color:#91aabc;max-width:600px}.gg-auth-pills{display:flex;flex-wrap:wrap;gap:.55rem;margin-top:1.5rem}.gg-auth-pill{padding:.48rem .72rem;border-radius:999px;background:rgba(21,50,69,.58);border:1px solid rgba(91,177,213,.14);color:#a9bfce;font-size:.72rem}.gg-auth-note{position:relative;z-index:2;color:#668397;font-size:.68rem;max-width:580px}
.gg-auth-panel{display:flex;align-items:center;justify-content:center;padding:clamp(1.5rem,5vw,5rem);background:linear-gradient(180deg,#0a1827,#0a1725);border-left:1px solid rgba(107,177,213,.10)}.gg-auth-card{width:min(470px,100%);padding:2.2rem;border-radius:26px;background:linear-gradient(145deg,rgba(19,43,62,.96),rgba(10,27,42,.98));border:1px solid rgba(104,179,215,.17);box-shadow:0 30px 90px rgba(0,0,0,.34),inset 0 1px 0 rgba(255,255,255,.035)}.gg-auth-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:1.8rem}.gg-back{color:#7d98ab;font-size:.75rem;text-decoration:none}.gg-back:hover{color:#c9d9e3}.gg-role{display:flex;align-items:center;gap:.5rem;font-size:.78rem;font-weight:800}.gg-role-dot{width:9px;height:9px;border-radius:50%;background:var(--auth-accent);box-shadow:0 0 14px var(--auth-accent)}.gg-auth-card h2{font-size:1.65rem;letter-spacing:-.035em;margin-bottom:.35rem}.gg-auth-sub{font-size:.76rem;color:#7893a7;margin-bottom:1.35rem}
.gg-tabs{display:grid;grid-template-columns:1fr 1fr;padding:.25rem;background:rgba(5,17,29,.65);border:1px solid rgba(101,168,207,.13);border-radius:12px;margin-bottom:1.35rem}.gg-tab{padding:.65rem;border:0;background:transparent;color:#7690a4;border-radius:9px;font-weight:750;cursor:pointer;font-size:.78rem}.gg-tab.active{background:rgba(42,117,218,.13);color:#dfeef7;box-shadow:inset 0 0 0 1px rgba(71,148,234,.15)}.gg-auth-card .fr{margin-bottom:1rem}.gg-auth-card label{color:#9db4c5}.gg-auth-card input{padding:.78rem .9rem;border-radius:12px}.gg-submit{width:100%;padding:.82rem 1rem;border-radius:12px;border:0;color:white;font-weight:800;cursor:pointer;margin-top:.25rem;transition:.2s;background:var(--auth-accent);box-shadow:0 12px 26px rgba(0,0,0,.18)}.gg-submit:hover{transform:translateY(-1px);filter:brightness(1.08)}.gg-auth-foot{display:flex;align-items:center;gap:.5rem;margin-top:1.25rem;color:#627d91;font-size:.67rem}.gg-auth-foot:before{content:'';width:24px;height:1px;background:rgba(100,170,207,.2)}.gg-alert{padding:.7rem .8rem;border-radius:11px;margin-bottom:1rem;background:rgba(201,74,74,.10);border:1px solid rgba(240,126,126,.22);color:#ffaaa8;font-size:.73rem}
@media(max-width:900px){.gg-auth{grid-template-columns:1fr}.gg-auth-visual{min-height:340px;padding:2rem}.gg-auth-copy{margin-top:3rem}.gg-auth-copy h1{font-size:3rem}.gg-auth-note{display:none}.gg-auth-panel{padding:1.2rem;border-left:0}.gg-auth-card{padding:1.5rem}}@media(max-width:520px){.gg-auth-visual{min-height:280px}.gg-auth-copy h1{font-size:2.35rem}.gg-auth-copy p{font-size:.82rem}.gg-auth-pills{display:none}.gg-auth-card{border-radius:20px}}

"""

JS = """
function applyTheme(){const d=localStorage.getItem('gg_d')!=='0';document.body.classList.toggle('light',!d);const e=document.getElementById('tt');if(e)e.checked=d;}
function applyFont(){const l=localStorage.getItem('gg_f')==='1';document.body.classList.toggle('lf',l);const e=document.getElementById('ft');if(e)e.checked=l;}
document.addEventListener('DOMContentLoaded',()=>{applyTheme();applyFont();});
function toggleTheme(v){localStorage.setItem('gg_d',v?'1':'0');applyTheme();}
function toggleFont(v){localStorage.setItem('gg_f',v?'1':'0');applyFont();}
"""


def hdr(name, role):
    badge = f'<span class="badge {"bdoc-b" if role=="doctor" else "bpat-b"}">{"Doctor" if role=="doctor" else "Patient"}</span>'
    return f'<header class="hdr"><div class="logo">🧬 Guardian Genome</div><div style="display:flex;align-items:center;gap:.5rem">{badge}<span style="font-size:.78rem;color:var(--tx2)">{name}</span></div></header>'

def bnav(role, active):
    doc_items=[("🏠","Home","home"),("👥","Patients","patients"),("🔬","Predict","predict"),("⚙️","Settings","settings"),("🚪","Sign out","so")]
    pat_items=[("🏠","Home","home"),("💬","Chat","chat"),("👨‍⚕️","Doctors","doctors"),("⚙️","Settings","settings"),("🚪","Sign out","so")]
    items = doc_items if role=="doctor" else pat_items
    html = '<nav class="bnav">'
    for icon,label,key in items:
        is_so = key=="so"
        cls = f'ni so' if is_so else f'ni {"adoc" if role=="doctor" and active==key else "apat" if role=="patient" and active==key else ""}'
        act = "location.href='/logout'" if is_so else f"location.href='/{role}/{key}'"
        html += f'<button class="{cls}" onclick="{act}"><span class="ni-ic">{icon}</span>{label}</button>'
    return html+"</nav>"

def page(title, body, role, active):
    return f"""<!DOCTYPE html><html><head><title>{title}</title><style>{CSS}</style></head><body>
<script>{JS}</script>{hdr(session.get('full_name',''), role)}<div class="pw">{body}</div>{bnav(role,active)}</body></html>"""

def require(role=None):
    if "user_id" not in session: return redirect("/")
    if role and session.get("role")!=role: return redirect("/")

# ─────────────────────────────────────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(f"/{session['role']}/home")

    # The landing page is intentionally image-led.  The supplied 16:9
    # Guardian Genome visual already contains the hero and portal artwork.
    # Transparent clickable regions preserve the two portal actions without
    # adding duplicate/un-styled HTML below or above the artwork.
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Guardian Genome</title>
    <style>
        * { box-sizing: border-box; }
        html, body {
            width: 100%;
            height: 100%;
            margin: 0;
            overflow: hidden;
            background: #eef7fb;
        }
        body {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .landing {
            position: relative;
            width: 100vw;
            height: 100vh;
            max-width: none;
            max-height: none;
            flex: 0 0 auto;
        }
        .landing img {
            display: block;
            width: 100vw;
            height: 100vh;
            object-fit: fill;
            user-select: none;
            -webkit-user-drag: none;
        }
        .portal-link {
            position: absolute;
            display: block;
            z-index: 5;
            border-radius: 22px;
            text-decoration: none;
        }
        .portal-link.doctor {
            left: 36.2%;
            top: 62.7%;
            width: 28.1%;
            height: 34.0%;
        }
        .portal-link.patient {
            left: 65.2%;
            top: 62.7%;
            width: 28.1%;
            height: 34.0%;
        }
        .portal-link:focus-visible {
            outline: 3px solid #1976ff;
            outline-offset: 3px;
        }
        @media (max-width: 700px) {
            .landing, .landing img { width:100vw; height:100vh; }
        }
    </style>
</head>
<body>
    <main class="landing" aria-label="Guardian Genome landing page">
        <img
            src="/static/guardian_genome_hero_visual.png"
            alt="Guardian Genome — From genes to better decisions. Choose your workspace."
        >
        <a class="portal-link doctor" href="/auth/doctor" aria-label="Enter clinical portal"></a>
        <a class="portal-link patient" href="/auth/patient" aria-label="Enter patient portal"></a>
    </main>
</body>
</html>
""")

@app.route("/auth/<role>")
def auth_page(role):
    if role not in ("doctor", "patient"):
        return redirect("/")
    accent = "#2d7ff9" if role == "doctor" else "#13b8a2"
    label = "Doctor" if role == "doctor" else "Patient"
    kicker = "Clinical decision workspace" if role == "doctor" else "Personal care workspace"
    headline = "Clinical intelligence,\nclearly explained." if role == "doctor" else "Your care,\nmade easier to understand."
    sub = ("Manage patients, run visit-adaptive pharmacogenomic predictions, and review explainable model insights in one focused workspace." if role == "doctor" else "View your treatment history, understand your pharmacogenomic information, and stay connected with your care team.")
    pills = ("Patient management", "Visit-adaptive predictions", "SHAP explanations") if role == "doctor" else ("Treatment overview", "Visit history", "Guardian Assistant")
    err = f'<div class="gg-alert">{html.escape(request.args["e"])}</div>' if "e" in request.args else ""
    tab = request.args.get("tab", "signin")
    extra = ('''<div class="fr"><label>Specialisation</label><input type="text" name="specialisation" placeholder="e.g. Cardiology"></div><div class="fr"><label>Hospital / Clinic</label><input type="text" name="hospital" placeholder="Hospital or clinic name"></div>''' if role == "doctor" else '''<div class="fr"><label>Date of Birth</label><input type="text" name="dob" placeholder="DD/MM/YYYY"></div>''')
    pills_html = ''.join(f'<span class="gg-auth-pill">✓ {html.escape(x)}</span>' for x in pills)
    return render_template_string(f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>{label} Portal · Guardian Genome</title><style>{CSS}</style></head><body>
<div class="gg-auth" style="--auth-accent:{accent}"><section class="gg-auth-visual"><div class="gg-brand"><div class="gg-brand-mark">GG</div><div>Guardian <em>Genome</em></div></div><div class="gg-auth-copy"><div class="gg-auth-kicker">{kicker}</div><h1>{headline.replace(chr(10), "<br>")}</h1><p>{sub}</p><div class="gg-auth-pills">{pills_html}</div></div></section>
<section class="gg-auth-panel"><div class="gg-auth-card"><div class="gg-auth-top"><a href="/" class="gg-back">← Back to workspace</a><div class="gg-role"><span class="gg-role-dot"></span>{label} portal</div></div>{err}<h2>Welcome back</h2><div class="gg-auth-sub">Sign in to your {label.lower()} workspace.</div><div class="gg-tabs"><button type="button" class="gg-tab {'active' if tab=='signin' else ''}" id="tabSignin" onclick="showTab('signin')">Sign in</button><button type="button" class="gg-tab {'active' if tab=='signup' else ''}" id="tabSignup" onclick="showTab('signup')">Create account</button></div>
<form id="if" method="POST" action="/auth/{role}/signin"><div class="fr"><label>Username</label><input type="text" name="username" autocomplete="username" placeholder="Enter your username" required></div><div class="fr"><label>Password</label><input type="password" name="password" autocomplete="current-password" placeholder="Enter your password" required></div><button type="submit" class="gg-submit">Sign in to {label.lower()} portal <span>→</span></button></form>
<form id="sf" method="POST" action="/auth/{role}/signup" style="display:none"><div class="fr"><label>Full name</label><input type="text" name="full_name" placeholder="Your full name" required></div><div class="fr"><label>Username</label><input type="text" name="username" placeholder="Choose a username" required></div><div class="fr"><label>Email</label><input type="email" name="email" placeholder="name@example.com"></div>{extra}<div class="fr"><label>Password</label><input type="password" name="password" placeholder="Create a password" required></div><div class="fr"><label>Confirm password</label><input type="password" name="confirm" placeholder="Repeat your password" required></div><button type="submit" class="gg-submit">Create {label.lower()} account <span>→</span></button></form>
</div></section></div>
<script>function showTab(t){{const sign=document.getElementById('if'),signup=document.getElementById('sf'),a=document.getElementById('tabSignin'),b=document.getElementById('tabSignup');sign.style.display=t==='signin'?'block':'none';signup.style.display=t==='signup'?'block':'none';a.classList.toggle('active',t==='signin');b.classList.toggle('active',t==='signup');}}showTab('{"signin" if tab=="signin" else "signup"}');</script></body></html>''')

@app.route("/auth/<role>/signin", methods=["POST"])
def signin(role):
    u=request.form.get("username","").strip(); p=request.form.get("password","")
    conn=get_db()
    user=conn.execute("SELECT * FROM users WHERE username=? AND role=? AND is_active=1",(u,role)).fetchone()
    conn.close()
    if not user or user["password_hash"]!=hash_pw(p):
        return redirect(f"/auth/{role}?e=Invalid+username+or+password")
    session.update({"user_id":user["id"],"username":user["username"],"full_name":user["full_name"],"role":role})
    log_action(user["id"],"login")
    return redirect(f"/{role}/home")

@app.route("/auth/<role>/signup", methods=["POST"])
def signup(role):
    f=request.form
    if f.get("password")!=f.get("confirm"):
        return redirect(f"/auth/{role}?e=Passwords+do+not+match&tab=signup")
    conn=get_db()
    try:
        conn.execute("INSERT INTO users (username,password_hash,role,full_name,email,specialisation,hospital) VALUES (?,?,?,?,?,?,?)",
                     (f["username"],hash_pw(f["password"]),role,f.get("full_name",""),f.get("email",""),f.get("specialisation",""),f.get("hospital","")))
        conn.commit()
        user=conn.execute("SELECT * FROM users WHERE username=?",(f["username"],)).fetchone()
        if role=="patient":
            pgx=gen_patient_id()
            conn.execute("INSERT INTO patients (patient_id,user_id,registered_by,full_name,age,primary_condition) VALUES (?,?,?,?,?,?)",
                         (pgx,user["id"],None,f.get("full_name",""),0,""))
            conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return redirect(f"/auth/{role}?e=Username+already+taken&tab=signup")
    conn.close()
    session.update({"user_id":user["id"],"username":user["username"],"full_name":user["full_name"],"role":role})
    log_action(user["id"],"login")
    return redirect(f"/{role}/home")

@app.route("/logout")
def logout():
    session.clear(); return redirect("/")

# ─────────────────────────────────────────────────────────────────────────────
# DOCTOR ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/doctor/home")
def doctor_home():
    if require("doctor"):
        return require("doctor")

    uid = session["user_id"]
    conn = get_db()
    np_ = conn.execute(
        "SELECT COUNT(*) FROM patients WHERE registered_by=? OR assigned_doctor=?",
        (uid, uid)
    ).fetchone()[0]
    nv = conn.execute(
        "SELECT COUNT(*) FROM visit_records WHERE doctor_id=?",
        (uid,)
    ).fetchone()[0]
    nt = conn.execute(
        "SELECT COUNT(*) FROM visit_records WHERE doctor_id=? AND date(created_at)=date('now')",
        (uid,)
    ).fetchone()[0]
    avg_conf = conn.execute(
        "SELECT AVG(confidence) FROM visit_records WHERE doctor_id=?",
        (uid,)
    ).fetchone()[0]
    recent = conn.execute(
        """
        SELECT v.created_at, v.recommended_drug, v.patient_id,
               v.confidence, v.visit_action, p.full_name
        FROM visit_records v
        LEFT JOIN patients p ON p.patient_id=v.patient_id
        WHERE v.doctor_id=?
        ORDER BY v.created_at DESC
        LIMIT 6
        """,
        (uid,)
    ).fetchall()
    conn.close()

    first = (session.get("full_name") or "Doctor").split()[0]
    rows = ""
    for r in recent:
        initials = "".join(w[0].upper() for w in (r["full_name"] or "P").split()[:2])
        rows += f"""
        <div class="pr" onclick="location.href='/doctor/predict?pid={html.escape(r['patient_id'])}'">
          <div class="av av-p">{html.escape(initials)}</div>
          <div style="flex:1">
            <div style="font-weight:750;font-size:.82rem">{html.escape(r['full_name'] or 'Patient')}</div>
            <div class="small muted">{html.escape(r['patient_id'])} · {html.escape(r['recommended_drug'] or '—')}</div>
          </div>
          <div style="text-align:right">
            <div class="small">{float(r['confidence'] or 0):.0%}</div>
            <div class="small muted">{str(r['created_at'])[:10]}</div>
          </div>
        </div>
        """
    if not rows:
        rows = '<div class="empty">No prediction activity yet.</div>'

    body = f"""
    <div class="hero">
      <div class="eyebrow">Clinical workspace</div>
      <h1>Good to see you, {html.escape(first)}.</h1>
      <p>Review patients, generate pharmacogenomic recommendations, and explain each model decision clearly during clinical review.</p>
    </div>
    <div class="kpi-row" style="margin-bottom:1rem">
      <div class="kpi"><b>{np_}</b><span>Patients under care</span></div>
      <div class="kpi"><b>{nv}</b><span>Total predictions</span></div>
      <div class="kpi"><b>{nt}</b><span>Predictions today</span></div>
      <div class="kpi"><b>{float(avg_conf or 0):.0%}</b><span>Average confidence</span></div>
    </div>
    <div class="section-title"><h3>Quick actions</h3><span>Common clinical workflows</span></div>
    <div class="quick-grid" style="margin-bottom:1rem">
      <div class="quick" onclick="location.href='/doctor/patients'"><div class="qi">＋</div><b>Add patient</b><p>Link an existing account or create a new patient.</p></div>
      <div class="quick" onclick="location.href='/doctor/predict'"><div class="qi">✦</div><b>New prediction</b><p>Run the full recommendation workflow.</p></div>
      <div class="quick" onclick="location.href='/doctor/patients'"><div class="qi">♙</div><b>Patient registry</b><p>Search patients and review visits.</p></div>
    </div>
    <div class="card">
      <div class="section-title" style="margin-top:0"><h3>Recent clinical activity</h3><span>Latest 6 records</span></div>
      {rows}
    </div>
    """
    return render_template_string(page("Doctor Home", body, "doctor", "home"))

@app.route("/doctor/patients")
def doctor_patients():
    if require("doctor"):
        return require("doctor")

    conn = get_db()
    pts = conn.execute(
        """
        SELECT p.*, u.username
        FROM patients p
        LEFT JOIN users u ON p.user_id = u.id
        WHERE p.registered_by = ? OR p.assigned_doctor = ?
        ORDER BY p.registered_at DESC
        """,
        (session["user_id"], session["user_id"])
    ).fetchall()
    conn.close()

    rows = "".join(
        f"""
        <div class="pr" onclick="location.href='/doctor/predict?pid={html.escape(p['patient_id'])}'">
          <div class="av av-p">{html.escape(''.join(w[0].upper() for w in (p['full_name'] or 'P').split()[:2]))}</div>
          <div style="flex:1">
            <div style="font-weight:750;font-size:.84rem">{html.escape(p['full_name'] or 'Patient')}</div>
            <div class="small muted">{html.escape(p['primary_condition'] or 'No condition recorded')} · {html.escape(p['patient_id'])}</div>
          </div>
          <div style="text-align:right">
            <div class="small">{html.escape(p['username'] or '—')}</div>
            <div class="small muted">{str(p['registered_at'])[:10]}</div>
          </div>
        </div>
        """
        for p in pts
    )
    if not rows:
        rows = '<div class="empty">No patients are currently linked to your care.</div>'

    plural = "s" if len(pts) != 1 else ""
    body = f"""
    <div class="hero">
      <div class="eyebrow">Clinical workspace · patient registry</div>
      <h1>My Patients</h1>
      <p>Manage patient relationships, review longitudinal records, and open the pharmacogenomic prediction workflow.</p>
    </div>

    <div style="display:flex;justify-content:space-between;align-items:center;gap:.8rem;margin:1rem 0 .65rem">
      <div class="small muted">{len(pts)} patient{plural} under your care</div>
      <button class="btn bdoc" onclick="document.getElementById('patientModal').style.display='flex'">＋ Add patient</button>
    </div>

    <div class="card">
      <input type="text" id="patientSearch" placeholder="Search by name, username or PGX-ID..." onkeyup="filterPatients()" style="margin-bottom:.75rem">
      <div id="patientList">{rows}</div>
    </div>

    <div id="patientModal" style="display:none;position:fixed;inset:0;background:rgba(5,15,25,.74);backdrop-filter:blur(7px);z-index:250;align-items:center;justify-content:center;padding:1rem">
      <div class="card" style="width:min(560px,100%);max-height:90vh;overflow:auto;padding:1.35rem">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem">
          <div><div class="eyebrow">Patient management</div><div style="font-size:1.1rem;font-weight:850">Add patient</div><div class="small muted" style="margin-top:.2rem">Link an existing Guardian Genome account or create a new one.</div></div>
          <button class="btn bd2" onclick="closePatientModal()">Close</button>
        </div>

        <div style="display:flex;gap:.4rem;padding:.3rem;background:var(--bg3);border-radius:12px;margin-bottom:1rem">
          <button id="linkTab" class="btn bdoc" style="flex:1;font-size:.72rem" onclick="showPatientMode('link')">Link existing</button>
          <button id="createTab" class="btn bd2" style="flex:1;font-size:.72rem" onclick="showPatientMode('create')">Create new</button>
        </div>

        <form id="linkPatientForm" method="POST" action="/doctor/patients/register">
          <input type="hidden" name="mode" value="link">
          <div class="alert ainfo">If the patient already created an account, do not create another account. Search by their username, PGX-ID, or email.</div>
          <div class="fr"><label>Patient username, PGX-ID or email</label><input type="text" name="existing_identifier" placeholder="e.g. rida123 or PGX-4821" required></div>
          <button type="submit" class="btn bdoc" style="width:100%">Link patient to my care</button>
        </form>

        <form id="createPatientForm" method="POST" action="/doctor/patients/register" style="display:none">
          <input type="hidden" name="mode" value="create">
          <div class="fr"><label>Full name *</label><input type="text" name="full_name" required></div>
          <div class="g2"><div class="fr"><label>Age</label><input type="number" name="age" min="1" max="120"></div><div class="fr"><label>Primary condition</label><select name="condition"><option>Hypertension</option><option>Type 2 Diabetes</option><option>Hyperlipidemia</option><option>Atrial Fibrillation</option><option>Depression</option><option>Rheumatoid Arthritis</option><option>HIV/AIDS</option><option>Epilepsy</option><option>Colorectal Cancer</option></select></div></div>
          <div class="fr"><label>Patient username</label><input type="text" name="pat_username" required></div>
          <div class="fr"><label>Temporary password</label><input type="password" name="pat_password" required></div>
          <button type="submit" class="btn bdoc" style="width:100%">Create patient account</button>
        </form>
      </div>
    </div>

    <script>
    function filterPatients() {{
      const q=document.getElementById('patientSearch').value.toLowerCase();
      document.querySelectorAll('#patientList .pr').forEach(r=>r.style.display=r.textContent.toLowerCase().includes(q)?'flex':'none');
    }}
    function closePatientModal() {{ document.getElementById('patientModal').style.display='none'; }}
    function showPatientMode(mode) {{
      const link=document.getElementById('linkPatientForm'), create=document.getElementById('createPatientForm');
      const lt=document.getElementById('linkTab'), ct=document.getElementById('createTab');
      const isLink=mode==='link';
      link.style.display=isLink?'block':'none'; create.style.display=isLink?'none':'block';
      lt.className='btn '+(isLink?'bdoc':'bd2'); ct.className='btn '+(isLink?'bd2':'bdoc');
    }}
    </script>
    """
    return render_template_string(page("Patients", body, "doctor", "patients"))

@app.route("/doctor/patients/register", methods=["POST"])
def register_patient():
    if require("doctor"):
        return require("doctor")

    f = request.form
    mode = f.get("mode", "create")
    doctor_id = session["user_id"]
    conn = get_db()

    if mode == "link":
        identifier = f.get("existing_identifier", "").strip()
        if not identifier:
            conn.close()
            return redirect("/doctor/patients?error=patient_not_found")

        patient = conn.execute(
            """
            SELECT p.*, u.username, u.email
            FROM patients p
            JOIN users u ON u.id = p.user_id
            WHERE u.role='patient'
              AND (u.username=? OR p.patient_id=? OR LOWER(u.email)=LOWER(?))
            LIMIT 1
            """,
            (identifier, identifier, identifier)
        ).fetchone()

        if not patient:
            conn.close()
            return redirect("/doctor/patients?error=patient_not_found")

        if patient["assigned_doctor"] == doctor_id:
            conn.close()
            return redirect("/doctor/patients?error=already_linked")

        if patient["assigned_doctor"] is not None and patient["assigned_doctor"] != doctor_id:
            conn.close()
            return redirect("/doctor/patients?error=assigned_elsewhere")

        conn.execute("UPDATE patients SET assigned_doctor=? WHERE id=?", (doctor_id, patient["id"]))
        conn.commit()
        log_action(doctor_id, "link_existing_patient", patient["patient_id"])
        conn.close()
        return redirect("/doctor/patients?linked=1")

    if mode == "create":
        username = f.get("pat_username", "").strip()
        if not username:
            conn.close()
            return redirect("/doctor/patients?error=registration_failed")

        existing = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if existing:
            conn.close()
            return redirect("/doctor/patients?error=username_exists")

        try:
            conn.execute(
                "INSERT INTO users (username,password_hash,role,full_name) VALUES (?,?,?,?)",
                (username, hash_pw(f.get("pat_password", "")), "patient", f.get("full_name", ""))
            )
            uid = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()["id"]
            pgx = gen_patient_id()
            conn.execute(
                """
                INSERT INTO patients
                (patient_id,user_id,registered_by,assigned_doctor,full_name,age,primary_condition)
                VALUES (?,?,?,?,?,?,?)
                """,
                (pgx, uid, doctor_id, doctor_id, f.get("full_name", ""), int(f.get("age", 0) or 0), f.get("condition", ""))
            )
            conn.commit()
            log_action(doctor_id, "register_patient", pgx)
        except sqlite3.IntegrityError:
            conn.rollback()
            conn.close()
            return redirect("/doctor/patients?error=registration_failed")

        conn.close()
        return redirect("/doctor/patients?created=1")

    conn.close()
    return redirect("/doctor/patients")

@app.route("/doctor/predict")
def doctor_predict():
    if require("doctor"): return require("doctor")
    conn=get_db(); pts=conn.execute("SELECT patient_id,full_name FROM patients WHERE registered_by=? OR assigned_doctor=? ORDER BY full_name",(session["user_id"],session["user_id"])).fetchall(); conn.close()
    opts="".join(f'<option value="{html.escape(p["patient_id"])}">{html.escape(p["full_name"])} ({html.escape(p["patient_id"])})</option>' for p in pts)
    pid=request.args.get("pid","")
    drugs=["None","Metoprolol","Amlodipine","Warfarin","Metformin","Metformin + Glipizide","Simvastatin","Rosuvastatin","Pravastatin","Escitalopram","Sertraline","Phenytoin","Azathioprine","Methotrexate","Abacavir + Lamivudine","Tenofovir + Emtricitabine","5-Fluorouracil (5-FU)","Capecitabine","Irinotecan"]
    drug_opts="".join(f"<option>{html.escape(d)}</option>" for d in drugs)
    body=f"""<div class="hero"><div class="eyebrow">Decision support</div><h1>Pharmacogenomic prediction</h1><p>Generate a recommendation from clinical, genomic, and longitudinal information. The explanation panel shows which features contributed to the model output.</p></div>
    <div class="predict-layout"><div class="card sticky-card"><div class="stepbar"><span class="step on"></span><span class="step on"></span><span class="step" id="step3"></span></div>
      <div class="sh">Patient context</div><div class="fr"><label>Select patient</label><select id="psel" onchange="loadPat(this.value)"><option value="">— choose a patient —</option>{opts}</select></div><div id="historyPreview"></div>
      <div style="display:flex;gap:.4rem;margin-bottom:.7rem"><button class="btn bdoc" id="v1b" onclick="sv(1)" style="flex:1;font-size:.72rem">Visit 1</button><button class="btn bd2" id="v2b" onclick="sv(2)" style="flex:1;font-size:.72rem">Visit 2</button><button class="btn bd2" id="v3b" onclick="sv(3)" style="flex:1;font-size:.72rem">Visit 3</button></div>
      <div id="hs" style="display:none"><div class="sh">Previous visit</div><div class="fr"><label>Previous drug</label><select id="pd">{drug_opts}</select></div><div class="fr"><label>Previous outcome</label><select id="po"><option>None</option><option>Stable</option><option>Improved</option><option>No Change</option><option>Adverse Reaction</option><option>Switched Drug</option><option>Lost to Follow-up</option></select></div></div>
      <div class="sh">Clinical profile</div><div class="fr"><label>Primary condition</label><select id="fc"><option>Hypertension</option><option>Type 2 Diabetes</option><option>Hyperlipidemia</option><option>Atrial Fibrillation</option><option>Depression</option><option>Rheumatoid Arthritis</option><option>HIV/AIDS</option><option>Epilepsy</option><option>Colorectal Cancer</option></select></div>
      <div class="g2"><div class="fr"><label>Age</label><input type="number" id="fa" value="52"></div><div class="fr"><label>Sex</label><select id="fs"><option>Male</option><option>Female</option></select></div><div class="fr"><label>Weight (kg)</label><input type="number" id="fw" value="70" step="0.1"></div><div class="fr"><label>BMI</label><input type="number" id="fb" value="26.5" step="0.1"></div><div class="fr"><label>GFR (mL/min)</label><input type="number" id="fg" value="74"></div><div class="fr"><label>ALT (U/L)</label><input type="number" id="falt" value="25" step="0.1"></div></div>
      <div class="fr"><label>Ethnicity</label><select id="fe"><option>Caucasian</option><option>Asian</option><option>Black</option><option>Hispanic</option><option>Other</option></select></div>
      <div class="sh">Genomic profile</div><div class="fr"><label>CYP2C9 phenotype</label><select id="f1"><option>Normal Metabolizer</option><option>Intermediate Metabolizer</option><option>Poor Metabolizer</option></select></div><div class="fr"><label>CYP2C9 alleles</label><select id="f1a"><option>*1/*1</option><option>*1/*2</option><option>*1/*3</option><option>*2/*2</option><option>*2/*3</option><option>*3/*3</option></select></div>
      <div class="fr"><label>CYP2D6</label><select id="f2"><option>Normal Metabolizer</option><option>Intermediate Metabolizer</option><option>Poor Metabolizer</option><option>Rapid Metabolizer</option><option>Ultrarapid Metabolizer</option></select></div>
      <div class="g2"><div class="fr"><label>VKORC1 genotype</label><select id="f3"><option>GG</option><option>GA</option><option>AA</option></select></div><div class="fr"><label>VKORC1 sensitivity</label><select id="f3s"><option>Low Sensitivity</option><option>Moderate Sensitivity</option><option>High Sensitivity</option></select></div></div>
      <div class="fr"><label>SLCO1B1 function</label><select id="f4"><option>Normal Function</option><option>Decreased Function</option><option>Poor Function</option></select></div><div class="fr"><label>CYP2C19 phenotype</label><select id="f5"><option>Normal Metabolizer</option><option>Intermediate Metabolizer</option><option>Poor Metabolizer</option><option>Rapid Metabolizer</option><option>Ultrarapid Metabolizer</option></select></div>
      <div class="g2"><div class="fr"><label>TPMT activity</label><select id="f6"><option>Normal Activity</option><option>Intermediate Activity</option><option>Low Activity</option><option>Deficient Activity</option></select></div><div class="fr"><label>DPYD function</label><select id="f7"><option>Normal Function</option><option>Decreased Function</option><option>Non-functional</option></select></div></div><div class="fr"><label>HLA-B*57:01</label><select id="f8"><option>Negative</option><option>Positive</option></select></div>
      <button class="btn bdoc" id="predictBtn" style="width:100%;margin-top:.25rem" onclick="runP()">✦ Generate recommendation</button>
    </div><div id="rp"><div class="card empty" style="min-height:520px;display:grid;place-items:center"><div><div style="font-size:2rem;margin-bottom:.4rem">✦</div><b>Ready for analysis</b><div class="small muted" style="margin-top:.25rem">Select a patient, review the profile, and generate a recommendation.</div></div></div></div></div>
<script>
let cv=1,cpid='',lastPrediction=null;const esc=s=>String(s??'').replace(/[&<>"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]));function sv(n){{cv=n;['v1b','v2b','v3b'].forEach((id,i)=>document.getElementById(id).className='btn '+(i+1===n?'bdoc':'bd2'));document.getElementById('hs').style.display=n>1?'block':'none';}}sv(1);
function loadPat(pid){{cpid=pid;if(!pid){{document.getElementById('historyPreview').innerHTML='';return;}}fetch('/doctor/patient_info/'+encodeURIComponent(pid)).then(r=>r.json()).then(d=>{{if(d.condition)document.getElementById('fc').value=d.condition;if(d.age)document.getElementById('fa').value=d.age;const nv=d.next_visit&&d.next_visit<=3?d.next_visit:(d.next_visit>3?3:1);sv(nv);if(d.last_drug&&d.last_drug!=='None')document.getElementById('pd').value=d.last_drug;if(d.last_outcome&&d.last_outcome!=='None')document.getElementById('po').value=d.last_outcome;let h='';if(d.total_visits){{h='<div class="small" style="font-weight:800;margin:.5rem 0 .35rem">Previous visits</div><div class="timeline">';d.visits.slice(0,3).forEach(v=>{{h+=`<div class="tl-item"><span class="tl-dot"></span><div class="tl-date">${{esc(v.created_at)}} · Visit ${{v.visit_number}}</div><div class="tl-main">${{esc(v.recommended_drug)}}</div><div class="tl-sub">${{esc(v.dose_category)}} · ${{esc(v.treatment_outcome||'—')}}</div></div>`;}});h+='</div>';}}document.getElementById('historyPreview').innerHTML=h;}});}}
function gf(){{return{{Primary_Condition:document.getElementById('fc').value,Age:document.getElementById('fa').value,Sex:document.getElementById('fs').value,Weight_kg:document.getElementById('fw').value,BMI:document.getElementById('fb').value,GFR_mL_min:document.getElementById('fg').value,ALT_U_L:document.getElementById('falt').value,Ethnicity:document.getElementById('fe').value,CYP2C9_Phenotype:document.getElementById('f1').value,CYP2D6_Phenotype:document.getElementById('f2').value,VKORC1_Genotype:document.getElementById('f3').value,VKORC1_Sensitivity:document.getElementById('f3s').value,SLCO1B1_Function:document.getElementById('f4').value,CYP2C19_Phenotype:document.getElementById('f5').value,TPMT_Activity:document.getElementById('f6').value,DPYD_Function:document.getElementById('f7').value,HLA_B5701:document.getElementById('f8').value,CYP2C9_Alleles:document.getElementById('f1a').value,visit_number:cv,prev_outcome:cv>1?(document.getElementById('po')?.value||'None'):'None',prev_drug:cv>1?(document.getElementById('pd')?.value||'None'):'None',patient_id:cpid}};}}
async function runP(){{const btn=document.getElementById('predictBtn');btn.disabled=true;btn.textContent='Analysing profile…';document.getElementById('rp').innerHTML='<div class="card empty" style="min-height:520px;display:grid;place-items:center"><div><div style="font-size:2rem">◌</div><b>Analysing genomic + clinical profile</b><div class="small muted">Computing model prediction and explanation…</div></div></div>';try{{const r=await fetch('/predict',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(gf())}});const d=await r.json();if(d.error){{document.getElementById('rp').innerHTML='<div class="alert aerr">'+esc(d.error)+'</div>';return;}}document.getElementById('step3').className='step on';lastPrediction=d;const actionLabel={{FIRST_VISIT:'First visit · genetics + clinical profile',CONTINUE:'Continue · previous response was favourable',ESCALATE:'Escalate · previous response was limited',SWITCH:'Switch · previous adverse reaction'}};const actionClass={{FIRST_VISIT:'bdoc-b',CONTINUE:'bok-b',ESCALATE:'bwarn-b',SWITCH:'bdanger-b'}};const pct=Math.round(d.full_confidence*100);
const bars=(d.shap_features||[]).map(x=>{{const sign=x.value>=0?'pos':'neg';const width=Math.max(3,Math.round(x.relative*100));return `<div class="why-row"><div class="why-name" title="${{esc(x.name)}}">${{esc(x.name)}}</div><div class="why-track"><div class="why-fill ${{sign}}" style="width:${{width}}%"></div></div><div class="why-val">${{x.value>=0?'+':''}}${{Number(x.value).toFixed(3)}}</div></div>`;}}).join('');
const factors=(d.shap_features||[]).slice(0,4).map(x=>`<div style="padding:.62rem 0;border-bottom:1px solid var(--bd)"><b style="font-size:.75rem">${{esc(x.name)}}</b><div class="small muted" style="line-height:1.55;margin-top:.18rem">${{esc(x.reason)}}</div></div>`).join('');
const flags=(d.flags||[]).filter(x=>x!=='None').map(f=>`<div class="alert ${{f.includes('CONTRAINDICATED')?'aerr':f.includes('effective')?'aok':'ainfo'}}">${{esc(f)}}</div>`).join('');
const saveSec=cpid?`<div class="divider"></div><div class="small muted" style="margin-bottom:.4rem">Save this recommendation to <b>${{esc(cpid)}}</b></div><div style="display:flex;gap:.5rem"><select id="osel" style="flex:1"><option>Stable</option><option>Improved</option><option>No Change</option><option>Adverse Reaction</option><option>First Visit</option></select><button class="btn bpat" onclick="saveR()">Save record</button></div>`:'<div class="alert ainfo">Select a patient to enable saving this recommendation.</div>';
document.getElementById('rp').innerHTML=`<div class="result-hero"><div style="display:flex;justify-content:space-between;gap:.7rem;align-items:flex-start"><div><div class="eyebrow">Model recommendation</div><div class="result-drug">${{esc(d.full_drug)}}</div><div class="small muted">${{esc(d.dose)}} · ${{esc(actionLabel[d.visit_action]||d.visit_action)}}</div></div><span class="badge ${{actionClass[d.visit_action]||'bdoc-b'}}">${{pct}}% confidence</span></div><div class="confidence"><span style="width:${{pct}}%"></span></div><div class="small muted" style="margin-top:.35rem">Confidence is the model's estimated probability for the selected class after the rule-based adjustment.</div></div>
<div class="model-grid" style="margin-bottom:.75rem"><div class="model-card"><div class="small muted">V1 · genetics-focused</div><b>${{esc(d.v1_drug)}}</b><div class="small muted">${{Math.round(d.v1_confidence*100)}}% confidence</div></div><div class="model-card primary"><div class="small" style="color:var(--ac)">Full model · longitudinal</div><b>${{esc(d.full_drug)}}</b><div class="small muted">${{d.models_agree?'Models agree':'Models differ'}} · history included</div></div></div>
${{flags}}<div class="explain-grid"><div class="why-card"><div style="font-weight:800;font-size:.82rem">Why this recommendation?</div><div class="small muted" style="line-height:1.55">These are the strongest patient-relevant SHAP contributors to the selected class. Green bars support the model's choice; red bars push against it. SHAP explains model behaviour — it does not prove medical causation.</div><div class="legend" style="margin:.65rem 0"><span><i class="dot-pos"></i>supports class</span><span><i class="dot-neg"></i>pushes against class</span></div>${{bars||'<div class="empty">No SHAP explanation available.</div>'}}</div><div class="why-card"><div style="font-weight:800;font-size:.82rem">What this means</div>${{factors||'<div class="small muted">No patient-specific explanation available.</div>'}}</div></div>
<div class="card" style="margin-top:.75rem;background:var(--bg3)"><div style="font-weight:800;font-size:.8rem;margin-bottom:.25rem">Clinical interpretation</div><div class="small" style="line-height:1.75">${{d.nl_explanation}}</div></div>
<div class="card" style="margin-top:.75rem"><div style="font-weight:800;font-size:.8rem">Decision summary</div><div class="small muted" style="margin-top:.2rem">The recommendation combines the trained model with the project's pharmacogenomic safety rules and, for return visits, longitudinal treatment information. It should support — not replace — clinician judgement.</div>${{saveSec}}<div id="sm"></div></div>`;}}catch(e){{document.getElementById('rp').innerHTML='<div class="alert aerr">Could not complete the prediction. Please check the server and try again.</div>';}}finally{{btn.disabled=false;btn.textContent='✦ Generate recommendation';}}}}
async function saveR(){{if(!cpid||!lastPrediction)return;const outcome=document.getElementById('osel')?.value||'Stable';const p=lastPrediction;const r=await fetch('/doctor/save_record',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{patient_id:cpid,visit_number:cv,recommended_drug:p.full_drug,dose_category:p.dose,visit_action:p.visit_action,confidence:p.full_confidence,clinical_flag:(p.flags||[]).join('|'),treatment_outcome:outcome}})}});const x=await r.json();document.getElementById('sm').innerHTML=x.ok?'<div class="alert aok" style="margin-top:.55rem">Recommendation saved to the patient record.</div>':'<div class="alert aerr" style="margin-top:.55rem">'+esc(x.error)+'</div>';}}
{f'document.getElementById("psel").value="{html.escape(pid)}";loadPat("{html.escape(pid)}");' if pid else ''}
</script>"""
    return render_template_string(page("Predict", body, "doctor", "predict"))

@app.route("/doctor/patient_info/<pid>")
def patient_info_api(pid):
    if require("doctor"): return require("doctor")
    conn=get_db()
    p=conn.execute("SELECT * FROM patients WHERE patient_id=?",(pid,)).fetchone()
    if not p: conn.close(); return jsonify({})
    # Fetch all visits ordered — auto-fill uses the latest saved visit
    visits=conn.execute(
        "SELECT * FROM visit_records WHERE patient_id=? ORDER BY visit_number DESC",
        (pid,)
    ).fetchall()
    conn.close()
    last=visits[0] if visits else None
    next_visit=int(last["visit_number"])+1 if last else 1
    return jsonify({
        "condition":  p["primary_condition"],
        "age":        p["age"],
        "name":       p["full_name"],
        # Auto-fill for visit history section
        "next_visit":       next_visit,              # which visit tab to auto-select
        "last_drug":        last["recommended_drug"] if last else "None",
        "last_outcome":     last["treatment_outcome"] if last else "None",
        "last_dose":        last["dose_category"]    if last else "",
        "last_visit_num":   int(last["visit_number"]) if last else 0,
        "total_visits":     len(visits),
        "visits": [
            {
                "visit_number":     v["visit_number"],
                "recommended_drug": v["recommended_drug"],
                "dose_category":    v["dose_category"],
                "treatment_outcome":v["treatment_outcome"],
                "visit_action":     v["visit_action"],
                "created_at":       str(v["created_at"])[:10],
            } for v in visits
        ],
    })

@app.route("/doctor/save_record", methods=["POST"])
def save_record():
    if require("doctor"): return require("doctor")
    d=request.get_json(); conn=get_db()
    p=conn.execute("SELECT id FROM patients WHERE patient_id=?",(d["patient_id"],)).fetchone()
    if not p: conn.close(); return jsonify({"error":"Patient not found"})
    conn.execute("INSERT INTO visit_records (patient_id,doctor_id,visit_number,recommended_drug,dose_category,treatment_outcome,clinical_flag,confidence,visit_action) VALUES (?,?,?,?,?,?,?,?,?)",
                 (d["patient_id"],session["user_id"],d["visit_number"],d["recommended_drug"],d["dose_category"],d["treatment_outcome"],d["clinical_flag"],d["confidence"],d["visit_action"]))
    conn.commit(); log_action(session["user_id"],"save_record",d["patient_id"]); conn.close()
    return jsonify({"ok":True})

@app.route("/doctor/settings", methods=["GET"])
def doctor_settings():
    if require("doctor"): return require("doctor")
    conn=get_db()
    user=conn.execute("SELECT * FROM users WHERE id=?",(session["user_id"],)).fetchone()
    logs=conn.execute("SELECT login_at,action,patient_id_accessed FROM login_log WHERE user_id=? ORDER BY login_at DESC LIMIT 5",(session["user_id"],)).fetchall()
    conn.close()
    lh="".join(f'<div style="display:flex;justify-content:space-between;padding:.38rem 0;border-bottom:1px solid var(--bd);font-size:.76rem"><span style="color:var(--tx3)">{l["login_at"][:16]}</span><span style="color:var(--tx2)">{l["action"]}</span><span>{l["patient_id_accessed"] or "—"}</span></div>' for l in logs) or '<div style="color:var(--tx3);font-size:.78rem">No history</div>'
    msg=f'<div class="alert aok">{request.args["msg"]}</div>' if "msg" in request.args else ""
    err=f'<div class="alert aerr">{request.args["err"]}</div>' if "err" in request.args else ""
    body=f"""{msg}{err}
<div class="card" style="margin-bottom:.75rem">
  <div style="font-size:.75rem;color:var(--tx2);font-weight:700;margin-bottom:.7rem">APPEARANCE</div>
  <div class="sr"><span>Dark mode</span><label class="tgl"><input type="checkbox" id="tt" onchange="toggleTheme(this.checked)"><span class="tsl"></span></label></div>
  <div class="sr"><span>Large font</span><label class="tgl"><input type="checkbox" id="ft" onchange="toggleFont(this.checked)"><span class="tsl"></span></label></div>
</div>
<div class="card" style="margin-bottom:.75rem">
  <div style="font-size:.75rem;color:var(--tx2);font-weight:700;margin-bottom:.7rem">PROFILE</div>
  <div class="sr"><span style="color:var(--tx2)">Name</span><span>{user["full_name"] or "—"}</span></div>
  <div class="sr"><span style="color:var(--tx2)">Specialisation</span><span>{user["specialisation"] or "—"}</span></div>
  <div class="sr"><span style="color:var(--tx2)">Hospital</span><span>{user["hospital"] or "—"}</span></div>
</div>
<div class="card" style="margin-bottom:.75rem">
  <div style="font-size:.75rem;color:var(--tx2);font-weight:700;margin-bottom:.7rem">CHANGE PASSWORD</div>
  <form method="POST" action="/doctor/settings/change_password">
    <div class="fr"><label>Current password</label><input type="password" name="current" required></div>
    <div class="fr"><label>New password</label><input type="password" name="new_pw" required></div>
    <div class="fr"><label>Confirm</label><input type="password" name="confirm" required></div>
    <button type="submit" class="btn bdoc">Update password</button>
  </form>
</div>
<div class="card"><div style="font-size:.75rem;color:var(--tx2);font-weight:700;margin-bottom:.7rem">RECENT LOGINS</div>{lh}</div>"""
    return render_template_string(page("Settings", body, "doctor", "settings"))

@app.route("/doctor/settings/change_password", methods=["POST"])
def doc_change_pw():
    if require("doctor"): return require("doctor")
    f=request.form; conn=get_db()
    user=conn.execute("SELECT * FROM users WHERE id=?",(session["user_id"],)).fetchone()
    if user["password_hash"]!=hash_pw(f["current"]): conn.close(); return redirect("/doctor/settings?err=Current+password+incorrect")
    if f["new_pw"]!=f["confirm"]: conn.close(); return redirect("/doctor/settings?err=Passwords+do+not+match")
    conn.execute("UPDATE users SET password_hash=?,password_changed_at=CURRENT_TIMESTAMP WHERE id=?",(hash_pw(f["new_pw"]),session["user_id"]))
    conn.commit(); conn.close(); return redirect("/doctor/settings?msg=Password+updated")

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT ROUTES
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED CHATBOT KNOWLEDGE BASE
# ADD THIS ABOVE: @app.route("/patient/chat_api")
# ─────────────────────────────────────────────────────────────────────────────

MEDICATION_INFO = {

    "Metformin": {
        "common": [
            "Nausea",
            "Diarrhea",
            "Stomach discomfort",
            "Loss of appetite"
        ],
        "serious": [
            "Lactic acidosis (rare)",
            "Severe weakness",
            "Breathing difficulty"
        ],
        "advice": "Take with meals and stay hydrated.",
        "genetic_note": "Usually not strongly affected by CYP metabolism."
    },

    "Metoprolol": {
        "common": [
            "Dizziness",
            "Fatigue",
            "Slow heart rate",
            "Cold hands or feet"
        ],
        "serious": [
            "Very low heart rate",
            "Fainting",
            "Breathing problems"
        ],
        "advice": "Do not stop suddenly without consulting your doctor.",
        "genetic_note": "CYP2D6 poor metabolizers may experience stronger drug effects."
    },

    "Warfarin": {
        "common": [
            "Easy bruising",
            "Minor bleeding",
            "Nosebleeds"
        ],
        "serious": [
            "Internal bleeding",
            "Blood in urine",
            "Severe headache"
        ],
        "advice": "Regular INR monitoring is important.",
        "genetic_note": "VKORC1 variants can significantly affect dose sensitivity."
    },

    "Simvastatin": {
        "common": [
            "Muscle pain",
            "Headache",
            "Fatigue"
        ],
        "serious": [
            "Severe muscle breakdown",
            "Dark urine",
            "Liver toxicity"
        ],
        "advice": "Report unexplained muscle pain immediately.",
        "genetic_note": "SLCO1B1 reduced function increases myopathy risk."
    },

    "Rosuvastatin": {
        "common": [
            "Muscle aches",
            "Fatigue",
            "Headache"
        ],
        "serious": [
            "Severe muscle injury",
            "Kidney complications",
            "Liver toxicity"
        ],
        "advice": "Routine monitoring may be needed.",
        "genetic_note": "Some SLCO1B1 variants may increase statin sensitivity."
    },

    "Pravastatin": {
        "common": [
            "Headache",
            "Muscle pain",
            "Nausea"
        ],
        "serious": [
            "Liver problems",
            "Muscle breakdown"
        ],
        "advice": "Usually safer in SLCO1B1-sensitive patients.",
        "genetic_note": "Often preferred for reduced SLCO1B1 function."
    },

    "Azathioprine": {
        "common": [
            "Nausea",
            "Fatigue",
            "Reduced appetite"
        ],
        "serious": [
            "Bone marrow suppression",
            "Infection risk",
            "Liver damage"
        ],
        "advice": "Routine blood monitoring is recommended.",
        "genetic_note": "TPMT deficiency greatly increases toxicity risk."
    },

    "Methotrexate": {
        "common": [
            "Mouth ulcers",
            "Fatigue",
            "Nausea"
        ],
        "serious": [
            "Liver toxicity",
            "Lung inflammation",
            "Low blood counts"
        ],
        "advice": "Avoid alcohol and attend regular blood tests.",
        "genetic_note": "Genetic factors can influence toxicity risk."
    },

    "Escitalopram": {
        "common": [
            "Sleepiness",
            "Dry mouth",
            "Nausea"
        ],
        "serious": [
            "Serotonin syndrome",
            "Severe mood changes"
        ],
        "advice": "Do not stop abruptly.",
        "genetic_note": "CYP2C19 poor metabolizers may require lower doses."
    },

    "Sertraline": {
        "common": [
            "Insomnia",
            "Nausea",
            "Dizziness"
        ],
        "serious": [
            "Serotonin syndrome",
            "Suicidal thoughts"
        ],
        "advice": "Improvement may take several weeks.",
        "genetic_note": "CYP2C19 status can affect response."
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT-AWARE PATIENT ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────

MEDICAL_INTENTS={
 "medication":["current medication","what medicine","what medication","which medicine","what am i taking","my drug","prescribed","what am i prescribed"],
 "side_effects":["side effect","side effects","adverse effect","reaction to","medicine causing","symptom from","feeling after taking","feeling after medicine","problem after taking"],
 "dose_reason":["why was my dose","why was the dose","why was my dosage","why did my dose","why did the dose","why did my dosage","why was my dosage escalated","why was my dose increased","why was the dose increased","why increase my dose","why increase the dose","why was it increased","why was it escalated","why did my dose change","why did my dosage change"],
 "dose":["dose","dosage","how much","how many mg","increase my dose","decrease my dose","take more","take less","how often should i take","how often do i take","when should i take","what time should i take","morning or night","morning and night","day or night","day and night","once a day","twice a day"],
 "missed_dose":["skip a dose","skip one dose","miss a dose","missed a dose","forgot a dose","forgot my medicine","forgot to take","can i skip","should i skip"],
 "followup":["when should i visit","when do i visit","when is my next visit","next follow up","next follow-up","follow up visit","follow-up visit","when should i see my doctor","when do i see my doctor","how soon should i see my doctor","when should i come back","next appointment","appointment scheduled"],
 "symptom":["i am having","i'm having","i have","i feel","i am feeling","i am facing","i'm facing","facing this issue","experiencing","what should i do","what can i do","is this normal","is this okay","is it normal","after taking","after i took","pain","dizzy","dizziness","nausea","vomiting","rash","swelling","breathing","shortness of breath","heart beating","palpitations","diarrhea","constipation","sleepy","insomnia","itching"],
 "history":["visit history","past visits","previous visit","medical history","treatment history","what happened before","my visits"],
 "doctor":["my doctor","assigned doctor","who is my doctor","doctor name","which doctor","my physician","who do i see"],
 "recommendation":["why was","why did","why this medicine","why this drug","recommendation","recommended","why recommended","how was this chosen"],
 "genetics":["gene","genes","genetic","genome","pharmacogenomic","cyp2","vkorc1","slco1b1","tpmt","dpyd","hla-b"],
 "outcome":["treatment outcome","did my medicine work","response to treatment","previous outcome","stable","improved","no change","adverse reaction"],
 "safety":["safe","safety","can i stop","stop taking","double dose","overdose","too much medicine","took extra"],
 "food":["food","eat","diet","meal","before food","after food","with food","without food"],
 "exercise":["exercise","gym","workout","running","physical activity"],
 "headache":["headache","head pain","migraine"],
 "fatigue":["tired","fatigue","weak","sleepy","low energy"],
 "emergency":["chest pain","can't breathe","cannot breathe","severe bleeding","collapse","unconscious","seizure","face droop","difficulty speaking","severe allergic reaction","swollen throat"],
 "help":["help","what can you do","what can i ask","options","commands"]
}

def _normalise_text(text): return re.sub(r"\s+"," ",(text or "").strip().lower())
def _intent_for(message):
    msg=_normalise_text(message)
    for intent,phrases in MEDICAL_INTENTS.items():
        if any(p in msg for p in phrases): return intent
    words=set(re.findall(r"[a-z0-9]+",msg));scores={}
    for intent,phrases in MEDICAL_INTENTS.items():
        best=0.0
        for phrase in phrases:
            pw=set(re.findall(r"[a-z0-9]+",phrase))
            overlap=len(words&pw)/max(1,len(pw));seq=difflib.SequenceMatcher(None,msg,phrase).ratio()
            best=max(best,overlap*.85+seq*.15)
        scores[intent]=best
    return max(scores,key=scores.get) if scores and max(scores.values())>=.62 else "help"
def _drug_from_message(msg,current_drug):
    low=_normalise_text(msg)
    for name in sorted(MEDICATION_INFO.keys(),key=len,reverse=True):
        if name.lower() in low:return name
    return current_drug
def _pretty_feature(name):
    overrides={"CYP2D6_Phenotype_encoded":"CYP2D6 Phenotype","CYP2C9_Phenotype_encoded":"CYP2C9 Phenotype","CYP2C19_Phenotype_encoded":"CYP2C19 Phenotype","VKORC1_Genotype_encoded":"VKORC1 Genotype","VKORC1_Sensitivity_encoded":"VKORC1 Sensitivity","SLCO1B1_Function_encoded":"SLCO1B1 Function","TPMT_Activity_encoded":"TPMT Activity","DPYD_Function_encoded":"DPYD Function","HLA_B5701_encoded":"HLA-B*57:01 Status","Previous_Outcome_encoded":"Previous Treatment Outcome","Previous_ADR_flag":"Previous Adverse Reaction","Is_Return_Visit":"Return Visit Status","CYP2C9_AlleleScore":"CYP2C9 Allele Score","Age":"Age","BMI":"BMI","Weight_kg":"Body Weight","GFR_mL_min":"Kidney Function (GFR)","ALT_U_L":"ALT","Drug_Interaction_Risk_encoded":"Drug Interaction Risk"}
    if name in overrides:return overrides[name]
    name=re.sub(r"_encoded$","",name)
    name=name.replace("Primary_Condition_","Condition: ").replace("Previous_Drug_","Previous Drug: ")
    name=name.replace("_"," ")
    # Keep model/medical abbreviations readable while presenting labels professionally.
    words=[]
    for word in name.split():
        if word.upper() in {"BMI", "ALT", "GFR", "HLA"} or word.startswith("CYP"):
            words.append(word)
        else:
            words.append(word[:1].upper()+word[1:])
    return " ".join(words)
def _feature_reason(name, value, patient):
    """Brief, patient-specific explanation for the clinician-facing SHAP panel."""
    condition = patient.get("condition") or "the recorded condition"
    outcome = patient.get("outcome") or "None"
    current_drug = patient.get("current_drug") or "the recommended treatment"
    previous_drug = patient.get("previous_drug") or "None"
    dose = patient.get("dose") or "Standard Dose"
    visit_action = patient.get("visit_action") or "FIRST_VISIT"
    visit_number = int(patient.get("visit_number") or 1)
    value = float(value)
    direction = "supports" if value >= 0 else "pushes against"
    shap_note = f"SHAP {value:+.3f}: {direction} the recommendation of {current_drug}."

    if name.startswith("Primary_Condition_"):
        return f"{condition} is the patient's recorded main condition. {shap_note}"

    if name.startswith("Ethnicity_"):
        return f"The patient's recorded demographic profile {direction} this recommendation. {shap_note}"

    if name.startswith("Previous_Drug_"):
        drug = name.replace("Previous_Drug_", "").replace("_", " ")
        if outcome == "Adverse Reaction":
            msg = f"{drug} was used previously, but an adverse reaction was recorded, so the workflow considers switching away from it."
        elif drug == current_drug:
            if outcome == "No Change":
                msg = f"{drug} was used previously and no adverse reaction was recorded. The workflow keeps the same drug and moves to {dose}."
            else:
                msg = f"{drug} was used previously and is also the current recommendation, so the workflow continues it."
        else:
            msg = f"{drug} was used previously, while the current recommendation is {current_drug}; the workflow is therefore considering a different treatment."
        return f"{msg} {shap_note}"

    if name == "Previous_Outcome_encoded":
        if outcome == "No Change":
            if previous_drug == current_drug and previous_drug not in {"", "None"}:
                msg = f"The previous treatment produced no recorded change. With no adverse reaction recorded, the same drug is maintained and the workflow moves to {dose}."
            else:
                msg = f"The previous treatment produced no recorded change, so the workflow moves to {dose} for the current recommendation."
        elif outcome in {"Stable", "Improved"}:
            msg = f"The previous treatment was {outcome.lower()}, so the workflow supports continuing the current treatment approach."
        elif outcome == "Adverse Reaction":
            msg = "A previous adverse reaction was recorded, so the workflow considers switching treatment."
        elif outcome == "Switched Drug":
            msg = "The previous treatment was switched, so that history is considered in the current recommendation."
        else:
            msg = "No previous treatment response was recorded, so this feature adds little historical context."
        return f"{msg} {shap_note}"

    if name == "Previous_ADR_flag":
        if outcome == "Adverse Reaction":
            msg = "A previous adverse reaction was recorded, so the workflow considers treatment safety and a possible switch."
        else:
            msg = "No previous adverse reaction is recorded, so there is no safety-related switch trigger from treatment history."
        return f"{msg} {shap_note}"

    if name == "Is_Return_Visit":
        if visit_number > 1:
            return f"This is a follow-up visit, so previous treatment history is included. {shap_note}"
        return f"This is the first visit, so no previous treatment response is available. {shap_note}"

    if any(x in name for x in ["Phenotype", "Function", "Activity", "Genotype", "HLA", "Allele"]):
        return f"The patient's genetic result {direction} the recommendation of {current_drug}. {shap_note}"

    if name == "Drug_Interaction_Risk_encoded":
        return f"The recorded interaction risk {direction} the recommendation. {shap_note}"

    if name in {"GFR_mL_min", "ALT_U_L", "BMI", "Weight_kg", "Age"}:
        labels = {"GFR_mL_min": "kidney function", "ALT_U_L": "liver function", "BMI": "BMI", "Weight_kg": "body weight", "Age": "age"}
        return f"The patient's {labels[name]} {direction} the recommendation. {shap_note}"

    return f"This patient detail {direction} the recommendation. {shap_note}"

def _shap_feature_is_active(name, encoded, patient):
    """Keep explanations focused on information that is actually present for this patient.

    The trained model contains one-hot columns for every possible condition, drug, and
    ethnicity. A tree model can assign SHAP values to inactive (zero-valued) one-hot
    columns as part of its mathematical decomposition. Showing all of those columns to
    a clinician, however, makes it look as though the patient has every condition.
    We therefore omit inactive categorical columns from the human-facing explanation.
    """
    value=float(encoded.get(name,0) or 0)

    if name.startswith("Primary_Condition_"):
        return value != 0
    if name.startswith("Ethnicity_"):
        return value != 0
    if name.startswith("Previous_Drug_"):
        return value != 0
    if name in {"Previous_Outcome_encoded","Previous_ADR_flag","Is_Return_Visit"}:
        return value != 0
    if name.endswith("_interaction") or name.endswith("_flag"):
        return value != 0

    # Ordinal genomic features remain meaningful even when their encoded value is 0
    # (for example, Poor Metabolizer or Negative status).
    return True
def _chat_context(patient,visits):
    latest=visits[0] if visits else None;doctor=None
    if patient and patient["assigned_doctor"]:
        conn=get_db();doctor=conn.execute("SELECT full_name,specialisation,hospital FROM users WHERE id=?",(patient["assigned_doctor"],)).fetchone();conn.close()
    return {"name":patient["full_name"] if patient else "there","condition":patient["primary_condition"] if patient else "","drug":latest["recommended_drug"] if latest else "","dose":latest["dose_category"] if latest else "","outcome":latest["treatment_outcome"] if latest else "","visit_number":latest["visit_number"] if latest else 0,"doctor":doctor}
def _chat_visit_reason(ctx, visits):
    """Short, patient-specific explanation for recommendation/dose questions."""
    latest = visits[0] if visits else None
    previous = visits[1] if len(visits) > 1 else None
    if not latest:
        return "There is no saved treatment record yet."

    drug = latest["recommended_drug"] or "the recorded treatment"
    dose = latest["dose_category"] or "Dose not recorded"
    action = latest["visit_action"] or ""

    if previous:
        outcome = previous["treatment_outcome"] or "None"
        prev_drug = previous["recommended_drug"] or "the previous treatment"
        if outcome == "No Change":
            if prev_drug == drug:
                return (f"At Visit {previous['visit_number']}, {html.escape(prev_drug)} produced No Change. "
                        f"With no adverse reaction recorded, the same drug was continued and the current visit moved to "
                        f"{html.escape(dose)}.")
            return (f"At Visit {previous['visit_number']}, the previous treatment produced No Change. "
                    f"The current workflow therefore moved to {html.escape(dose)} and selected {html.escape(drug)}.")
        if outcome == "Adverse Reaction":
            return (f"At Visit {previous['visit_number']}, an adverse reaction was recorded with {html.escape(prev_drug)}. "
                    f"The current workflow therefore moved away from that treatment and selected {html.escape(drug)}.")
        if outcome in {"Stable", "Improved"}:
            return (f"At Visit {previous['visit_number']}, the previous treatment was recorded as {html.escape(outcome)}. "
                    f"The latest workflow therefore continues the treatment approach with {html.escape(drug)} at {html.escape(dose)}.")

    if action == "ESCALATE":
        return f"The latest visit is marked for dose escalation, with {html.escape(dose)} recorded."
    if action == "SWITCH":
        return f"The latest visit is marked as a treatment switch, with {html.escape(drug)} as the recommendation."
    return (f"The latest record recommends {html.escape(drug)} at {html.escape(dose)} "
            "using the patient's current profile and available treatment history.")



def _chat_previous_pair(visits):
    if len(visits) < 2:
        return None, None
    return visits[0], visits[1]

def _chat_followup_reply(ctx, visits):
    return ("I don't have a scheduled follow-up date in your Guardian Genome record. "
            "If your doctor has not given you a date, contact the clinic to confirm when you should return. "
            "If you develop severe symptoms or a serious reaction, seek medical attention sooner.")

def _chat_timing_reply(drug):
    info = MEDICATION_INFO.get(drug)
    if info and info.get("advice"):
        return (f"For {html.escape(drug)}, the app's medication guidance says: "
                f"{html.escape(info['advice'])} The exact morning/night or tablet schedule is not stored here, "
                "so follow the timing written on your prescription or ask your pharmacist.")
    return (f"I don't have the exact timing schedule for {html.escape(drug)} in your record. "
            "Take it exactly as prescribed rather than choosing morning or night based on this chat. "
            "If the prescription is unclear, ask your pharmacist or doctor.")

def _chat_symptom_reply(msg, drug):
    info = MEDICATION_INFO.get(drug)
    serious = info.get("serious", []) if info else []
    if any(k in msg for k in ["chest pain","can't breathe","cannot breathe","collapse","unconscious","seizure","swollen throat","severe bleeding"]):
        return "This could need urgent medical attention. Seek emergency care now rather than waiting for a chat response."
    if info:
        common = ", ".join(info.get("common", [])[:4])
        return (f"Some symptoms can occur with {html.escape(drug)}, including {html.escape(common)}. "
                "I can't tell from chat whether your symptom is caused by the medicine. If it is severe, worsening, or worrying you, contact your doctor or pharmacist promptly. "
                "Do not stop or change the medicine on your own.")
    return ("I can't safely tell from chat what is causing that symptom. "
            "If it is mild, monitor it and contact your doctor or pharmacist if it persists or worsens. "
            "If you have severe breathing difficulty, collapse, a seizure, severe bleeding, or sudden stroke-like symptoms, seek emergency care now.")

def _chat_missed_dose_reply(drug):
    return (f"For a missed dose of {html.escape(drug)}, the correct action depends on how long ago you missed it and the medicine's instructions. "
            "Check the prescription or patient leaflet, or ask your pharmacist. Do not automatically take two doses together.")

def _chat_reply(message,patient,visits,history=None):
    ctx = _chat_context(patient, visits)
    name = html.escape(str(ctx["name"]).split()[0])
    msg = _normalise_text(message)
    drug = _drug_from_message(msg, ctx["drug"])
    intent = _intent_for(msg)
    latest = visits[0] if visits else None

    # Emergency questions always take priority over ordinary symptom questions.
    if intent == "emergency":
        return ("<b>This may need urgent medical attention.</b><br><br>"
                "If you have severe chest pain, major bleeding, severe breathing difficulty, collapse, a seizure, "
                "sudden weakness or difficulty speaking, or a severe allergic reaction, seek emergency medical care now. "
                "This assistant cannot assess an emergency remotely.")

    if intent == "followup":
        return _chat_followup_reply(ctx, visits)

    if intent == "missed_dose":
        return _chat_missed_dose_reply(drug)

    if intent == "symptom":
        return _chat_symptom_reply(msg, drug)

    if intent == "help":
        return (f"Hi {name} — you can ask me about your current medicine, why it was recommended, dose changes, "
                "when to take it, missed doses, side effects or symptoms, follow-up visits, treatment history, "
                "your genes, or your doctor. I can explain your saved record, but I can't replace your clinician or pharmacist.")

    if intent == "medication":
        if not latest:
            return "There is no medication record in your account yet. Your doctor can add a treatment record after your visit."
        return (f"Your latest saved treatment is <b>{html.escape(latest['recommended_drug'] or 'Medication not recorded')}</b> · "
                f"<b>{html.escape(latest['dose_category'] or 'Dose not recorded')}</b>. "
                f"Visit {latest['visit_number']} · outcome: <b>{html.escape(latest['treatment_outcome'] or 'Not recorded')}</b>.")

    if intent == "side_effects":
        info = MEDICATION_INFO.get(drug)
        if not info:
            return (f"I don't have a medicine-specific side-effect list for {html.escape(drug)} in this project's knowledge base. "
                    "If you're having a new or worrying symptom, contact your doctor or pharmacist.")
        common = ", ".join(info["common"][:4])
        serious = ", ".join(info["serious"][:4])
        return (f"For <b>{html.escape(drug)}</b>, common effects can include {html.escape(common)}. "
                f"More serious symptoms can include {html.escape(serious)}. "
                "If you develop a severe or rapidly worsening symptom, seek medical attention. Do not stop the medicine on your own.")

    if intent == "dose_reason":
        if not latest:
            return "There is no saved treatment record yet, so I cannot explain a dose change."
        return (_chat_visit_reason(ctx, visits) +
                "<br><br>The exact tablet or mg schedule is not stored in Guardian Genome; follow the prescription from your clinician.")

    if intent == "dose":
        if not latest:
            return "I don't have a current treatment record to interpret. Ask your doctor or pharmacist for the exact dose and timing."
        if any(p in msg for p in ["when should i take","how often","what time","morning or night","morning and night","day or night","day and night","once a day","twice a day"]):
            return _chat_timing_reply(drug)
        return (f"Your latest record shows <b>{html.escape(latest['dose_category'] or 'a dose category not recorded')}</b> for "
                f"<b>{html.escape(latest['recommended_drug'] or 'your medication')}</b>. "
                "Guardian Genome does not store the exact tablet/mg schedule, so follow the prescription rather than changing the dose yourself.")

    if intent == "history":
        if not visits:
            return "No visit records are stored in your account yet."
        items = [f"• <b>Visit {v['visit_number']}</b> · {str(v['created_at'])[:10]} — "
                 f"{html.escape(v['recommended_drug'] or 'No drug')} · {html.escape(v['dose_category'] or 'Dose not recorded')} · "
                 f"{html.escape(v['treatment_outcome'] or 'Outcome not recorded')}" for v in visits[:6]]
        return "<b>Your recent treatment history</b><br><br>" + "<br>".join(items)

    if intent == "doctor":
        d = ctx["doctor"]
        if d:
            return (f"<b>Your assigned doctor</b><br>Dr. {html.escape(d['full_name'])}<br>"
                    f"{html.escape(d['specialisation'] or 'Specialisation not recorded')} · "
                    f"{html.escape(d['hospital'] or 'Hospital/clinic not recorded')}")
        return "No assigned doctor is currently stored. Open the Doctors section to choose a doctor."

    if intent == "outcome":
        if not latest:
            return "There is no treatment outcome recorded yet."
        return (f"Your latest recorded outcome is <b>{html.escape(latest['treatment_outcome'] or 'not recorded')}</b> for Visit {latest['visit_number']}. "
                "Ask why your dose changed if you want to understand how that outcome affected the next visit.")

    if intent == "recommendation":
        if not latest:
            return "There is no saved recommendation yet. A doctor can run a prediction from the clinical and genomic profile."
        return (f"<b>{html.escape(latest['recommended_drug'] or 'Not recorded')}</b> is the latest recommendation at "
                f"<b>{html.escape(latest['dose_category'] or 'dose not recorded')}</b>.<br><br>{_chat_visit_reason(ctx, visits)}")

    if intent == "genetics":
        return ("Your genetic results are one part of the medication prediction. "
                "For example, CYP2D6 can affect how some medicines are processed. "
                "The doctor's prediction screen shows which genetic features influenced your recommendation.")

    if intent == "safety":
        if "missed" in msg or "forgot" in msg or "skip" in msg:
            return _chat_missed_dose_reply(drug)
        if "stop" in msg:
            return "Do not stop a prescribed medicine based only on this chat. Contact your doctor or pharmacist first."
        if "double" in msg or "extra" in msg or "overdose" in msg:
            return "If you may have taken more than prescribed, contact a medical professional or poison/emergency service promptly. Do not take another dose until you have received appropriate advice."
        return "Tell me the medicine and the safety concern, and I can explain the relevant information stored in this project."

    if intent == "food":
        info = MEDICATION_INFO.get(drug)
        advice = info["advice"] if info else "Food instructions depend on the medicine and your prescription."
        return f"<b>Food and medication</b><br>{html.escape(advice)} If your prescription gives different instructions, follow your prescription."

    if intent == "exercise":
        return "Activity advice depends on your condition and treatment. If exercise causes chest pain, severe breathlessness, fainting, or severe dizziness, stop and seek medical attention. For routine activity questions, your doctor can give advice based on your condition."

    if intent == "headache":
        return "Headache can have many causes. If it is sudden or severe, or comes with weakness, confusion, fainting, vision changes, or repeated vomiting, seek urgent medical care. Otherwise, discuss a persistent or new headache with your doctor or pharmacist, especially if it started after a medicine change."

    if intent == "fatigue":
        return "Tiredness can have many causes, including illness, sleep problems, or medication effects. If it is severe, persistent, or started after a treatment change, discuss it with your doctor rather than changing the medicine yourself."

    return (f"I can help with your saved medication and treatment information. You can ask things like "
            "'Why was my dose changed?', 'When should I take this medicine?', 'What if I miss a dose?', "
            "'When is my next follow-up?', or 'I'm having a new symptom — what should I do?'")

@app.route("/patient/chat_api",methods=["POST"])
def patient_chat_api():
    if "user_id" not in session:return jsonify({"reply":"Please log in again."}),401
    data=request.get_json(silent=True) or {};message=(data.get("message") or "").strip()
    if not message:return jsonify({"reply":"Please type a question and I'll help you."})
    patient,visits=get_pat(session["user_id"]);return jsonify({"reply":_chat_reply(message,patient,visits,data.get("history") or []),"intent":_intent_for(message)})

def get_pat(uid):
    conn=get_db()
    p=conn.execute("SELECT * FROM patients WHERE user_id=?",(uid,)).fetchone()
    vs=conn.execute("SELECT * FROM visit_records WHERE patient_id=? ORDER BY created_at DESC",(p["patient_id"],)).fetchall() if p else []
    conn.close(); return p,vs

@app.route("/patient/home")
def patient_home():
    if require("patient"): return require("patient")
    p,vs=get_pat(session["user_id"]);pgx=p["patient_id"] if p else "—";fn=(session.get("full_name") or "there").split()[0];latest=vs[0] if vs else None;total=len(vs)
    if latest:
        status_cls="bok-b" if latest["treatment_outcome"] in ["Stable","Improved","First Visit"] else "bwarn-b" if latest["treatment_outcome"]=="No Change" else "bdanger-b"
        med=f"""<div class="card" style="margin-bottom:.75rem"><div style="display:flex;justify-content:space-between;gap:.7rem;align-items:flex-start"><div><div class="eyebrow">Latest treatment</div><div style="font-size:1.3rem;font-weight:850">{html.escape(latest["recommended_drug"] or "Not recorded")}</div><div class="small muted">{html.escape(p["primary_condition"] or "Condition not recorded")} · Visit {latest["visit_number"]}</div></div><span class="badge {status_cls}">{html.escape(latest["treatment_outcome"] or "Recorded")}</span></div><div class="divider"></div><div class="g2"><div><div class="small muted">Dose category</div><b>{html.escape(latest["dose_category"] or "Not recorded")}</b></div><div><div class="small muted">Recorded</div><b>{str(latest["created_at"])[:10]}</b></div></div></div>"""
    else:med='<div class="card empty" style="margin-bottom:.75rem"><div style="font-size:1.5rem">◎</div><b>No treatment record yet</b><div class="small" style="margin-top:.25rem">Your doctor can add a record after your next visit.</div></div>'
    timeline="".join(f"""<div class="tl-item"><span class="tl-dot"></span><div class="tl-date">{str(v["created_at"])[:10]} · Visit {v["visit_number"]}</div><div class="tl-main">{html.escape(v["recommended_drug"] or "No medication recorded")}</div><div class="tl-sub">{html.escape(v["dose_category"] or "Dose not recorded")} · {html.escape(v["treatment_outcome"] or "Outcome not recorded")}</div></div>""" for v in vs[:5]) or '<div class="empty">Your visit timeline will appear here after your first recorded visit.</div>'
    body=f"""<div class="hero"><div class="eyebrow">Your care overview</div><h1>Hi, {html.escape(fn)}.</h1><p>Your pharmacogenomic care information, treatment history, and care-team tools are in one place.</p><div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.8rem"><span class="badge bpat-b">PGX-ID · {html.escape(pgx)}</span><span class="badge" style="background:var(--bg2);color:var(--tx2)">{total} recorded visit{"s" if total!=1 else ""}</span></div></div>
    <div class="g3" style="margin-bottom:.9rem"><div class="metric"><div class="num">{total}</div><div class="lbl">Recorded visits</div></div><div class="metric"><div class="num">{html.escape(str(p["age"] or "—"))}</div><div class="lbl">Age on profile</div></div><div class="metric"><div class="num" style="font-size:1.05rem">{html.escape((p["primary_condition"] or "—")[:18])}</div><div class="lbl">Primary condition</div></div></div>
    {med}<div class="section-title"><h3>What would you like to do?</h3><span>Patient portal</span></div><div class="quick-grid" style="margin-bottom:1rem"><div class="quick" onclick="location.href='/patient/chat'"><div class="qi">✦</div><b>Ask Guardian Assistant</b><p>Understand your medication, history, or recommendation.</p></div><div class="quick" onclick="location.href='/patient/doctors'"><div class="qi">♙</div><b>My care team</b><p>View or choose your assigned doctor.</p></div><div class="quick" onclick="location.href='/patient/settings'"><div class="qi">⚙</div><b>Account & accessibility</b><p>Appearance, font size, and password.</p></div></div>
    <div class="card"><div class="section-title" style="margin-top:0"><h3>Recent treatment timeline</h3><span>Latest 5 visits</span></div><div class="timeline">{timeline}</div></div>"""
    return render_template_string(page("Patient Home",body,"patient","home"))

@app.route("/patient/chat")
def patient_chat():
    if require("patient"):return require("patient")
    fn=(session.get("full_name") or "there").split()[0]
    body=f"""<div class="chat-shell"><div class="chat-head"><div class="chat-avatar">✦</div><div><div class="chat-title">Guardian Assistant</div><div class="chat-sub">Patient-record explainer · educational support, not a replacement for your clinician</div></div></div>
      <div id="msgs" class="chat-msgs"><div class="mw"><div class="bubble bb">Hi {html.escape(fn)}. I can explain the information stored in your Guardian Genome record in plain language. You can ask a follow-up question instead of choosing from a fixed menu.</div><div class="qps"><div class="qp" onclick="sq('What medicine am I currently taking?')">What medicine am I on?</div><div class="qp" onclick="sq('Why was this medicine recommended?')">Why this medicine?</div><div class="qp" onclick="sq('What side effects should I know about?')">Side effects</div><div class="qp" onclick="sq('Explain my visit history')">Visit history</div><div class="qp" onclick="sq('How are my genes used?')">How are my genes used?</div><div class="qp" onclick="sq('Who is my doctor?')">My doctor</div></div><div class="mt">Guardian Assistant</div></div></div>
      <div class="chat-input"><input type="text" id="ci" placeholder="Ask a question about your record…" autocomplete="off" onkeydown="if(event.key==='Enter')sm()"><button class="btn bpat" id="sendBtn" onclick="sm()">Send</button></div></div>
<script>
const hist=[];function sq(q){{document.getElementById('ci').value=q;sm();}}function am(txt,role){{const w=document.createElement('div');w.className='mw'+(role==='u'?' u':'');w.innerHTML=`<div class="bubble ${{role==='u'?'bu':'bb'}}">${{txt}}</div><div class="mt">${{new Date().toLocaleTimeString([],{{hour:'2-digit',minute:'2-digit'}})}}</div>`;document.getElementById('msgs').appendChild(w);w.scrollIntoView({{behavior:'smooth',block:'end'}});}}function at(){{const d=document.createElement('div');d.id='typ';d.className='mw';d.innerHTML='<div class="bubble bb" style="color:var(--tx3)">Thinking through your record…</div>';document.getElementById('msgs').appendChild(d);d.scrollIntoView({{behavior:'smooth'}});}}function rt(){{const t=document.getElementById('typ');if(t)t.remove();}}
async function sm(){{const inp=document.getElementById('ci'),btn=document.getElementById('sendBtn'),msg=inp.value.trim();if(!msg)return;inp.value='';btn.disabled=true;am(msg,'u');hist.push({{role:'user',content:msg}});at();try{{const r=await fetch('/patient/chat_api',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{message:msg,history:hist}})}});const d=await r.json();rt();am(d.reply||'I could not generate a response. Please try again.','b');hist.push({{role:'assistant',content:d.reply||''}});}}catch(e){{rt();am('The assistant is temporarily unavailable. Please try again or contact your doctor for urgent questions.','b');}}finally{{btn.disabled=false;inp.focus();}}}}
</script>"""
    return render_template_string(page("Guardian Assistant",body,"patient","chat"))

@app.route("/patient/doctors")
def patient_doctors():
    if require("patient"): return require("patient")
    conn=get_db()
    docs=conn.execute("SELECT id,full_name,specialisation,hospital FROM users WHERE role='doctor' AND is_active=1 ORDER BY full_name").fetchall()
    pr=conn.execute("SELECT assigned_doctor FROM patients WHERE user_id=?",(session["user_id"],)).fetchone()
    assigned=pr["assigned_doctor"] if pr else None; conn.close()
    cards=""
    for d in docs:
        ini="".join(w[0].upper() for w in (d["full_name"] or "DR").split()[:2])
        mine=assigned==d["id"]
        btn='<span class="badge bok-b">My doctor</span>' if mine else f'<button class="btn bpat" style="font-size:.73rem;padding:.28rem .68rem;margin-top:.25rem" onclick="rd({d["id"]},&quot;{d["full_name"]}&quot;)">Choose as my doctor</button>'
        cards+=f'<div class="dc"><div class="av av-d">{ini}</div><div style="flex:1"><div style="font-weight:600;font-size:.87rem">{d["full_name"]}</div><div style="font-size:.75rem;color:var(--tx2);margin:.08rem 0">{d["specialisation"] or "General"} · {d["hospital"] or "—"}</div>{btn}</div></div>'
    body=f"""<h2 style="font-size:1.05rem;margin-bottom:.35rem;color:var(--ac)">Find a doctor</h2>
<p style="font-size:.8rem;color:var(--tx2);margin-bottom:.9rem">Browse registered doctors and choose one to manage your care.</p>
<input type="text" id="ds" placeholder="Search by name or specialisation..." onkeyup="filterD()" style="margin-bottom:.75rem">
<div id="dl">{cards or '<div style="color:var(--tx3);font-size:.8rem">No doctors registered yet</div>'}</div>
<div id="rm3"></div>
<script>
function filterD(){{const q=document.getElementById('ds').value.toLowerCase();document.querySelectorAll('.dc').forEach(c=>c.style.display=c.textContent.toLowerCase().includes(q)?'flex':'none');}}
async function rd(id,name){{const r=await fetch('/patient/request_doctor',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{doctor_id:id}})}});const d=await r.json();document.getElementById('rm3').innerHTML=d.ok?`<div class="alert aok" style="margin-top:.65rem">Dr. ${{name}} set as your doctor.</div>`:`<div class="alert aerr" style="margin-top:.65rem">${{d.error}}</div>`;setTimeout(()=>location.reload(),1200);}}
</script>"""
    return render_template_string(page("Doctors", body, "patient", "doctors"))

@app.route("/patient/request_doctor", methods=["POST"])
def request_doctor():
    if require("patient"): return require("patient")
    did=request.get_json().get("doctor_id"); conn=get_db()
    conn.execute("UPDATE patients SET assigned_doctor=? WHERE user_id=?",(did,session["user_id"]))
    conn.commit(); conn.close(); return jsonify({"ok":True})

@app.route("/patient/settings")
def patient_settings():
    if require("patient"): return require("patient")
    conn=get_db()
    user=conn.execute("SELECT * FROM users WHERE id=?",(session["user_id"],)).fetchone()
    pr=conn.execute("SELECT patient_id FROM patients WHERE user_id=?",(session["user_id"],)).fetchone()
    conn.close()
    pwc=str(user["password_changed_at"])[:16] if user["password_changed_at"] else "Not changed yet"
    msg=f'<div class="alert aok">{request.args["msg"]}</div>' if "msg" in request.args else ""
    err=f'<div class="alert aerr">{request.args["err"]}</div>' if "err" in request.args else ""
    body=f"""{msg}{err}
<div class="card" style="margin-bottom:.75rem">
  <div style="font-size:.75rem;color:var(--tx2);font-weight:700;margin-bottom:.7rem">APPEARANCE</div>
  <div class="sr"><span>Dark mode</span><label class="tgl"><input type="checkbox" id="tt" onchange="toggleTheme(this.checked)"><span class="tsl"></span></label></div>
  <div class="sr"><span>Large font</span><label class="tgl"><input type="checkbox" id="ft" onchange="toggleFont(this.checked)"><span class="tsl"></span></label></div>
</div>
<div class="card" style="margin-bottom:.75rem">
  <div style="font-size:.75rem;color:var(--tx2);font-weight:700;margin-bottom:.7rem">MY ACCOUNT</div>
  <div class="sr"><span style="color:var(--tx2)">PGX-ID</span><code style="color:var(--ac)">{pr["patient_id"] if pr else "—"}</code></div>
  <div class="sr"><span style="color:var(--tx2)">Name</span><span>{user["full_name"] or "—"}</span></div>
  <div class="sr"><span style="color:var(--tx2)">Password last changed</span><span style="font-size:.78rem;color:var(--tx3)">{pwc}</span></div>
</div>
<div class="card">
  <div style="font-size:.75rem;color:var(--tx2);font-weight:700;margin-bottom:.3rem">CHANGE PASSWORD <span style="font-weight:400;color:var(--tx3)">(optional)</span></div>
  <p style="font-size:.76rem;color:var(--tx3);margin-bottom:.7rem">Your password is private — your doctor cannot see it.</p>
  <form method="POST" action="/patient/settings/change_password">
    <div class="fr"><label>Current password</label><input type="password" name="current" required></div>
    <div class="fr"><label>New password</label><input type="password" name="new_pw" required></div>
    <div class="fr"><label>Confirm</label><input type="password" name="confirm" required></div>
    <button type="submit" class="btn bpat">Update password</button>
  </form>
</div>"""
    return render_template_string(page("Settings", body, "patient", "settings"))

@app.route("/patient/settings/change_password", methods=["POST"])
def pat_change_pw():
    if require("patient"): return require("patient")
    f=request.form; conn=get_db()
    user=conn.execute("SELECT * FROM users WHERE id=?",(session["user_id"],)).fetchone()
    if user["password_hash"]!=hash_pw(f["current"]): conn.close(); return redirect("/patient/settings?err=Current+password+incorrect")
    if f["new_pw"]!=f["confirm"]: conn.close(); return redirect("/patient/settings?err=Passwords+do+not+match")
    conn.execute("UPDATE users SET password_hash=?,password_changed_at=CURRENT_TIMESTAMP WHERE id=?",(hash_pw(f["new_pw"]),session["user_id"]))
    conn.commit(); conn.close(); return redirect("/patient/settings?msg=Password+updated+successfully")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT API
# ─────────────────────────────────────────────────────────────────────────────
def _shap_rows(model,X,predicted_class_index,patient_context,encoded_features):
    rows=[]
    try:
        import shap
        sv=shap.TreeExplainer(model).shap_values(X)
        if isinstance(sv,list): vals=np.asarray(sv[predicted_class_index])[0]
        else:
            arr=np.asarray(sv)
            vals=arr[0,:,predicted_class_index] if arr.ndim==3 else arr[0] if arr.ndim==2 else np.ravel(arr)
        vals=np.asarray(vals,dtype=float)
        if len(vals)!=len(FEATURES_FULL):return []

        # Keep only patient-relevant / active features for the clinician-facing view.
        # The model still receives the complete feature vector; this only changes what
        # is displayed in the explanation panel.
        candidates=[]
        for i,value in enumerate(vals):
            feature=FEATURES_FULL[i]
            if not _shap_feature_is_active(feature,encoded_features,patient_context):
                continue
            candidates.append((i,float(value)))

        candidates.sort(key=lambda item:abs(item[1]),reverse=True)
        candidates=candidates[:8]
        denom=float(sum(abs(value) for _,value in candidates)) or 1.0

        for i,value in candidates:
            feature=FEATURES_FULL[i]
            rows.append({
                "feature":feature,
                "name":_pretty_feature(feature),
                "value":value,
                "relative":abs(value)/denom,
                "direction":"supports" if value>=0 else "opposes",
                "reason":_feature_reason(feature,value,patient_context)
            })
    except Exception as e:print(f"[XAI] SHAP unavailable: {e}")
    return rows

@app.route("/predict",methods=["POST"])
def predict():
    if require("doctor"): return require("doctor")
    if not ML_READY:return jsonify({"error":"ML models not loaded. Run phases 1–3 first."})
    try:
        form=request.get_json(silent=True) or {};vn=int(form.get("visit_number",1));po=form.get("prev_outcome","None");pid=form.get("patient_id","");enc=encode_patient(form,vn,po)
        missing_v1=[f for f in FEATURES_V1 if f not in enc];missing_full=[f for f in FEATURES_FULL if f not in enc]
        if missing_v1 or missing_full:raise ValueError(f"Feature mismatch. Missing V1={missing_v1}; Missing Full={missing_full}")
        Xv1=np.asarray([[enc[f] for f in FEATURES_V1]],dtype=float);Xf=np.asarray([[enc[f] for f in FEATURES_FULL]],dtype=float);Xv1=rf_v1_imputer.transform(Xv1);Xf=rf_full_imputer.transform(Xf)
        v1l=int(rf_v1.predict(Xv1)[0]);v1d=drug_map.get(str(v1l),"Unknown");v1_probs=rf_v1.predict_proba(Xv1)[0];v1c=float(v1_probs[list(rf_v1.classes_).index(v1l)])
        fl=int(rf_full.predict(Xf)[0]);fd=drug_map.get(str(fl),"Unknown");fp=rf_full.predict_proba(Xf)[0];fc=float(fp[list(rf_full.classes_).index(fl)]);rules=run_rules(form,po);fc=max(0.0,min(1.0,fc+rules["conf_mod"]))
        sf=_shap_rows(
            rf_full,
            Xf,
            list(rf_full.classes_).index(fl),
            {
                "condition":form.get("Primary_Condition",""),
                "outcome":po,
                "visit_number":vn,
                "previous_drug":form.get("prev_drug","None"),
                "current_drug":fd,
                "dose":rules["dose"],
                "visit_action":rules["visit_action"],
            },
            enc
        )
        if vn==1:lead="This is a first visit, so the recommendation does not use a previous treatment outcome."
        elif po=="No Change":lead="The previous visit recorded limited response, so the longitudinal component supports considering an escalation."
        elif po=="Adverse Reaction":lead="The previous visit recorded an adverse reaction, so the workflow supports considering a switch."
        elif po in ["Stable","Improved"]:lead=f"The previous treatment was recorded as {po.lower()}, so the longitudinal component supports continuing the regimen."
        else:lead=f"This is a return visit with previous outcome recorded as {po}."
        confidence_note="High model confidence" if fc>=.90 else "Moderate model confidence" if fc>=.75 else "Lower model confidence — clinical review is especially important."
        nl=f"{lead}<br><br><b>{html.escape(fd)}</b> is the full-model recommendation with {fc:.0%} adjusted confidence. {confidence_note}."
        if rules["flags"] and rules["flags"]!=["None"]:nl+="<br><br>"+"<br>".join(html.escape(f) for f in rules["flags"])
        if pid:log_action(session["user_id"],"predict",pid)
        return jsonify({"v1_drug":v1d,"v1_confidence":round(v1c,3),"full_drug":fd,"full_confidence":round(fc,3),"dose":rules["dose"],"flags":rules["flags"],"visit_action":rules["visit_action"],"nl_explanation":nl,"shap_features":sf,"models_agree":v1d==fd,"xai_note":"SHAP values describe how features contributed to the selected model class for this prediction; they do not establish clinical causation."})
    except Exception as e:return jsonify({"error":str(e)}),500

@app.route("/health")
def health():
    return jsonify({"status":"ok","ml_ready":ML_READY,"db":os.path.exists(DB_PATH)})

if __name__=="__main__":
    init_db()
    print("="*52)
    print("  Guardian Genome — Full Role-Based System")
    print("="*52)
    print(f"  DB:  {DB_PATH}")
    print(f"  ML:  {'ready' if ML_READY else 'NOT LOADED — run phases 1-3 first'}")
    print(f"  URL: http://localhost:5000")
    print("="*52)
    app.run(debug=True,port=5000)
