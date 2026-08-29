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

try:
    from transformers import pipeline
except Exception as e:
    pipeline = None
    print(f"[WARN] Transformers unavailable: {e}")
from flask import (Flask, request, jsonify, render_template_string,
                   session, redirect, url_for)
import sqlite3, hashlib, json, os, datetime, random
import numpy as np, joblib
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "guardian_genome_v3_2025"
DB_PATH = "guardian_genome.db"
chatbot_ai = None
CHATBOT_READY = False
if pipeline is not None:
    try:
        chatbot_ai = pipeline(
            "text-generation",
            model="distilgpt2"
        )
        CHATBOT_READY = True
    except Exception as e:
        print(f"[WARN] Chatbot unavailable: {e}")

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
:root{--bg:#0f1c2e;--bg2:#1a2e45;--bg3:#0d1826;--bd:#2a4060;--tx:#d0e8f5;--tx2:#7a9ab5;--tx3:#4a6a85;--ac:#00d4aa;--doc:#378add;--pat:#1d9e75;--danger:#e74c3c;--warn:#f39c12;--r:10px;--rs:6px}
body.light{--bg:#f5f7fa;--bg2:#fff;--bg3:#eef1f5;--bd:#d0dae8;--tx:#1a2e45;--tx2:#4a6a85;--tx3:#7a9ab5;--ac:#0f6e56}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:'Segoe UI',sans-serif;font-size:var(--fs,14px)}
body.lf{
    --fs:22px;
}

body.lf h1{
    font-size:2.6rem;
}

body.lf h2{
    font-size:2.2rem;
}

body.lf h3{
    font-size:1.8rem;
}

body.lf h4{
    font-size:1.5rem;
}

body.lf p,
body.lf span,
body.lf label,
body.lf input,
body.lf select,
body.lf textarea,
body.lf button,
body.lf a,
body.lf div{
    font-size:1.15rem !important;
    line-height:1.7;
}

body.lf .btn{
    padding:.8rem 1.3rem;
}

body.lf .badge{
    font-size:.9rem;
}

body.lf .ni{
    font-size:.9rem;
}

body.lf .ni-ic{
    font-size:1.4rem;
}

body.lf .logo{
    font-size:1.3rem;
}

body.lf .alert{
    font-size:1rem;
}

body.lf .sh{
    font-size:.9rem;
}

body.lf .sr,
body.lf .pr{
    font-size:1rem;
}
.card{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);padding:1.1rem}
label{display:block;color:var(--tx2);font-size:.77rem;margin-bottom:.28rem}
select,input[type=text],input[type=number],input[type=password],input[type=email]{width:100%;padding:.48rem .72rem;background:var(--bg3);border:1px solid var(--bd);border-radius:var(--rs);color:var(--tx);font-size:.84rem}
select:focus,input:focus{outline:none;border-color:var(--ac)}
.btn{padding:.55rem 1.1rem;border:none;border-radius:var(--rs);cursor:pointer;font-size:.83rem;font-weight:600}
.btn:hover{opacity:.85}
.bp{background:var(--ac);color:#0f1c2e}
.bd2{background:transparent;border:1px solid var(--bd);color:var(--tx2)}
.bdoc{background:var(--doc);color:#fff}
.bpat{background:var(--pat);color:#fff}
.bdanger{background:var(--danger);color:#fff}
.badge{display:inline-block;padding:.18rem .55rem;border-radius:20px;font-size:.7rem;font-weight:600}
.bdoc-b{background:#1a3a5a;color:#60a8e8}
.bpat-b{background:#0d3a2a;color:#3fc090}
.bwarn-b{background:#3a2a0a;color:var(--warn)}
.bdanger-b{background:#3a0f0f;color:var(--danger)}
.bok-b{background:#0a3a25;color:#2ecc71}
.fr{margin-bottom:.75rem}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:.65rem}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.65rem}
.sh{font-size:.67rem;font-weight:700;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em;margin:1rem 0 .45rem;padding-top:.7rem;border-top:1px solid var(--bd)}
.sh:first-child{border-top:none;margin-top:0}
.bnav{position:fixed;bottom:0;left:0;right:0;background:var(--bg2);border-top:1px solid var(--bd);display:flex;z-index:100}
.ni{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:.55rem .2rem;cursor:pointer;color:var(--tx3);font-size:.67rem;gap:2px;border:none;background:transparent}
.ni:hover,.ni.adoc{color:var(--doc)}
.ni.apat{color:var(--pat)}
.ni.so:hover{color:var(--danger)}
.ni-ic{font-size:1.05rem;line-height:1}
.hdr{background:var(--bg2);border-bottom:1px solid var(--bd);padding:.7rem 1.2rem;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:50}
.logo{font-weight:700;font-size:.95rem;color:var(--ac)}
.pw{padding:.9rem 1.1rem 5rem}
.alert{padding:.65rem .9rem;border-radius:var(--rs);font-size:.8rem;margin-bottom:.7rem}
.aerr{background:#3a0f0f;border-left:3px solid var(--danger);color:#f5b7b1}
.aok{background:#0a3a25;border-left:3px solid #2ecc71;color:#a9dfbf}
.ainfo{background:#0d2040;border-left:3px solid var(--doc);color:#aed6f1}
.bubble{padding:.7rem .95rem;border-radius:var(--r);font-size:.84rem;line-height:1.65;max-width:88%}
.bu{background:var(--doc);color:#fff;align-self:flex-end;border-bottom-right-radius:3px}
.bb{background:var(--bg2);border:1px solid var(--bd);color:var(--tx);border-bottom-left-radius:3px}
.mw{display:flex;flex-direction:column}.mw.u{align-items:flex-end}
.mt{font-size:.67rem;color:var(--tx3);margin-top:2px}
.qps{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.35rem}
.qp{font-size:.73rem;padding:.28rem .7rem;border-radius:20px;border:1px solid var(--bd);background:var(--bg2);color:var(--tx2);cursor:pointer}
.qp:hover{border-color:var(--pat);color:var(--pat)}
.vc{border:1px solid var(--bd);border-radius:var(--r);padding:.9rem;margin-bottom:.65rem}
.dc{border:1px solid var(--bd);border-radius:var(--r);padding:.9rem;margin-bottom:.65rem;display:flex;gap:.7rem}
.av{width:38px;height:38px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.82rem;flex-shrink:0}
.av-d{background:#1a3a5a;color:#60a8e8}
.av-p{background:#0d3a2a;color:#3fc090}
.sr{display:flex;align-items:center;justify-content:space-between;padding:.8rem 0;border-bottom:1px solid var(--bd);font-size:.86rem}
.sr:last-child{border-bottom:none}
.tgl{position:relative;width:40px;height:22px;display:inline-block}
.tgl input{opacity:0;width:0;height:0}
.tsl{position:absolute;cursor:pointer;inset:0;background:var(--bd);border-radius:22px;transition:.3s}
.tsl:before{content:"";position:absolute;width:16px;height:16px;left:3px;bottom:3px;background:#fff;border-radius:50%;transition:.3s}
input:checked+.tsl{background:var(--ac)}
input:checked+.tsl:before{transform:translateX(18px)}
.pr{display:flex;align-items:center;gap:7px;padding:.7rem 0;border-bottom:1px solid var(--bd);cursor:pointer}
.pr:hover{color:var(--ac)}
.pr:last-child{border-bottom:none}
.cg{display:grid;grid-template-columns:1fr 1fr;gap:.55rem;margin-bottom:.7rem}
.cc{background:var(--bg3);border-radius:var(--rs);padding:.7rem;border:1px solid var(--bd)}
.cc.win{border-color:var(--ac)}
.sr2{display:flex;align-items:center;gap:7px;margin-bottom:4px}
.sl{font-size:.7rem;color:var(--tx2);width:155px;flex-shrink:0;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.st{flex:1;height:9px;background:var(--bg3);border-radius:4px;overflow:hidden}
.sf{height:100%;border-radius:4px}
.sv{font-size:.69rem;color:var(--tx3);width:32px}
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
    return render_template_string(f"""<!DOCTYPE html><html><head><title>Guardian Genome</title><style>{CSS}</style></head><body>
<script>{JS}</script>
<div style="min-height:100vh;display:flex;align-items:center;justify-content:center;padding:1rem">
  <div style="width:340px">
    <div style="text-align:center;margin-bottom:2rem">
      <div style="font-size:2.2rem;margin-bottom:.5rem">🧬</div>
      <div style="font-size:1.4rem;font-weight:700;color:var(--ac)">Guardian Genome</div>
      <div style="font-size:.8rem;color:var(--tx2);margin-top:.3rem">Personalised Medicine · Visit-Adaptive System</div>
    </div>
    <a href="/auth/doctor" style="text-decoration:none"><div style="background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);padding:1rem;text-align:center;margin-bottom:.75rem;cursor:pointer;transition:border-color .2s" onmouseover="this.style.borderColor='var(--doc)'" onmouseout="this.style.borderColor='var(--bd)'">
      <div style="font-size:1.5rem;margin-bottom:.3rem">👨‍⚕️</div>
      <div style="font-weight:600;color:var(--doc)">Doctor</div>
      <div style="font-size:.75rem;color:var(--tx2);margin-top:.2rem">Clinical dashboard · Predictions · Patient management</div>
    </div></a>
    <a href="/auth/patient" style="text-decoration:none"><div style="background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);padding:1rem;text-align:center;cursor:pointer;transition:border-color .2s" onmouseover="this.style.borderColor='var(--pat)'" onmouseout="this.style.borderColor='var(--bd)'">
      <div style="font-size:1.5rem;margin-bottom:.3rem">🧑‍💼</div>
      <div style="font-weight:600;color:var(--pat)">Patient</div>
      <div style="font-size:.75rem;color:var(--tx2);margin-top:.2rem">My records · Chat assistant · Find a doctor</div>
    </div></a>
  </div>
</div></body></html>""")

@app.route("/auth/<role>")
def auth_page(role):
    if role not in ("doctor","patient"): return redirect("/")
    color = "var(--doc)" if role=="doctor" else "var(--pat)"
    label = "Doctor" if role=="doctor" else "Patient"
    err = f'<div class="alert aerr">{request.args["e"]}</div>' if "e" in request.args else ""
    tab = request.args.get("tab","signin")
    extra = ("""<div class="fr"><label>Specialisation</label><input type="text" name="specialisation" placeholder="e.g. Cardiology"></div>
      <div class="fr"><label>Hospital / Clinic</label><input type="text" name="hospital"></div>""" if role=="doctor" else
             """<div class="fr"><label>Date of Birth</label><input type="text" name="dob" placeholder="DD/MM/YYYY"></div>""")
    return render_template_string(f"""<!DOCTYPE html><html><head><title>{label} Login</title><style>{CSS}
.aw{{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:1rem}}
.ac{{width:375px}}.tr{{display:flex;margin-bottom:1.1rem;border-bottom:1px solid var(--bd)}}
.at{{flex:1;padding:.55rem;text-align:center;cursor:pointer;color:var(--tx2);font-size:.86rem;border-bottom:2px solid transparent}}
.at.a{{color:{color};border-color:{color};font-weight:600}}
#sf{{display:none}}
</style></head><body><script>{JS}</script>
<div class="aw"><div class="ac">
  <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:1.4rem">
    <a href="/" style="color:var(--tx3);font-size:.78rem;text-decoration:none">← Back</a>
    <div style="margin-left:auto;font-weight:700;color:{color}">{label} Portal</div>
  </div>
  {err}
  <div class="tr">
    <div class="at {'a' if tab=='signin' else ''}" onclick="showTab('signin')">Sign in</div>
    <div class="at {'a' if tab=='signup' else ''}" onclick="showTab('signup')">Sign up</div>
  </div>
  <form id="if" method="POST" action="/auth/{role}/signin">
    <div class="fr"><label>Username</label><input type="text" name="username" required></div>
    <div class="fr"><label>Password</label><input type="password" name="password" required></div>
    <button type="submit" class="btn {'bdoc' if role=='doctor' else 'bpat'}" style="width:100%;margin-top:.4rem">Sign in</button>
  </form>
  <form id="sf" method="POST" action="/auth/{role}/signup">
    <div class="fr"><label>Full name</label><input type="text" name="full_name" required></div>
    <div class="fr"><label>Username</label><input type="text" name="username" required></div>
    <div class="fr"><label>Email</label><input type="email" name="email"></div>
    {extra}
    <div class="fr"><label>Password</label><input type="password" name="password" required></div>
    <div class="fr"><label>Confirm password</label><input type="password" name="confirm" required></div>
    <button type="submit" class="btn {'bdoc' if role=='doctor' else 'bpat'}" style="width:100%;margin-top:.4rem">Create account</button>
  </form>
</div></div>
<script>{JS}
function showTab(t){{document.getElementById('if').style.display=t==='signin'?'block':'none';document.getElementById('sf').style.display=t==='signup'?'block':'none';document.querySelectorAll('.at').forEach((e,i)=>e.classList.toggle('a',i===(t==='signin'?0:1)));}}
showTab('{"signin" if tab=="signin" else "signup"}');
</script></body></html>""")

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
                         (pgx,user["id"],user["id"],f.get("full_name",""),0,""))
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
    if require("doctor"): return require("doctor")
    uid=session["user_id"]; conn=get_db()
    np_=conn.execute("SELECT COUNT(*) FROM patients WHERE registered_by=?",(uid,)).fetchone()[0]
    nv=conn.execute("SELECT COUNT(*) FROM visit_records WHERE doctor_id=?",(uid,)).fetchone()[0]
    nt=conn.execute("SELECT COUNT(*) FROM visit_records WHERE doctor_id=? AND date(created_at)=date('now')",(uid,)).fetchone()[0]
    recent=conn.execute("SELECT created_at,recommended_drug,patient_id FROM visit_records WHERE doctor_id=? ORDER BY created_at DESC LIMIT 5",(uid,)).fetchall()
    conn.close()
    rows="".join(f'<div style="display:flex;justify-content:space-between;padding:.38rem 0;border-bottom:1px solid var(--bd);font-size:.8rem"><span style="color:var(--tx2)">{r["patient_id"]}</span><span>{r["recommended_drug"]}</span><span style="color:var(--tx3)">{str(r["created_at"])[:10]}</span></div>' for r in recent) or '<div style="color:var(--tx3);font-size:.8rem">No visits yet</div>'
    body=f"""<h2 style="font-size:1.05rem;margin-bottom:.9rem;color:var(--ac)">Dashboard</h2>
<div class="g3" style="margin-bottom:.9rem">
  <div class="card" style="text-align:center"><div style="font-size:1.7rem;font-weight:700;color:var(--doc)">{np_}</div><div style="font-size:.73rem;color:var(--tx2);margin-top:.2rem">Patients</div></div>
  <div class="card" style="text-align:center"><div style="font-size:1.7rem;font-weight:700;color:var(--ac)">{nv}</div><div style="font-size:.73rem;color:var(--tx2);margin-top:.2rem">Total visits</div></div>
  <div class="card" style="text-align:center"><div style="font-size:1.7rem;font-weight:700;color:var(--warn)">{nt}</div><div style="font-size:.73rem;color:var(--tx2);margin-top:.2rem">Today</div></div>
</div>
<div class="card" style="margin-bottom:.75rem"><div style="font-size:.75rem;color:var(--tx2);font-weight:700;margin-bottom:.6rem">QUICK ACTIONS</div>
  <div style="display:flex;gap:.5rem;flex-wrap:wrap">
    <button class="btn bdoc" onclick="document.getElementById('rm').style.display='flex'">+ Register patient</button>
    <button class="btn bd2" onclick="location.href='/doctor/predict'">New prediction</button>
  </div>
</div>
<div class="card"><div style="font-size:.75rem;color:var(--tx2);font-weight:700;margin-bottom:.6rem">RECENT ACTIVITY</div>{rows}</div>
<div id="rm" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:200;align-items:center;justify-content:center">
  <div class="card" style="width:340px;max-height:88vh;overflow-y:auto">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.9rem">
      <div style="font-weight:700;color:var(--ac)">Register new patient</div>
      <button class="btn bd2" onclick="document.getElementById('rm').style.display='none'" style="padding:.25rem .55rem">✕</button>
    </div>
    <form method="POST" action="/doctor/patients/register">
      <div class="fr"><label>Full name *</label><input type="text" name="full_name" required></div>
      <div class="g2">
        <div class="fr"><label>Age</label><input type="number" name="age" min="1" max="120"></div>
        <div class="fr"><label>Condition</label><select name="condition"><option>Hypertension</option><option>Type 2 Diabetes</option><option>Hyperlipidemia</option><option>Atrial Fibrillation</option><option>Depression</option><option>Rheumatoid Arthritis</option><option>HIV/AIDS</option><option>Epilepsy</option><option>Colorectal Cancer</option></select></div>
      </div>
      <div class="fr"><label>Patient username</label><input type="text" name="pat_username" required></div>
      <div class="fr"><label>Patient password</label><input type="password" name="pat_password" required></div>
      <button type="submit" class="btn bdoc" style="width:100%;margin-top:.4rem">Register & generate PGX-ID</button>
    </form>
  </div>
</div>"""
    return render_template_string(page("Doctor Home", body, "doctor", "home"))

@app.route("/doctor/patients")
def doctor_patients():
    if require("doctor"): return require("doctor")
    conn=get_db()
    pts=conn.execute(
    "SELECT p.*,u.username FROM patients p LEFT JOIN users u ON p.user_id=u.id WHERE p.registered_by=? OR p.assigned_doctor=? ORDER BY p.registered_at DESC",
    (session["user_id"], session["user_id"])
    ).fetchall()
    conn.close()
    rows="".join(f'<div class="pr" onclick="location.href=\'/doctor/predict?pid={p["patient_id"]}\'">'
                 f'<div style="flex:1"><div style="font-weight:600;font-size:.86rem">{p["full_name"]}</div>'
                 f'<div style="font-size:.73rem;color:var(--tx2)">{p["primary_condition"] or "—"} · {p["patient_id"]}</div></div>'
                 f'<div style="font-size:.73rem;color:var(--tx3)">{str(p["registered_at"])[:10]}</div></div>' for p in pts) or '<div style="color:var(--tx3);font-size:.8rem;padding:.5rem 0">No patients yet</div>'
    body=f"""<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.9rem">
  <h2 style="font-size:1.05rem;color:var(--ac)">My Patients</h2>
  <button class="btn bdoc" onclick="document.getElementById('rm2').style.display='flex'" style="font-size:.78rem;padding:.45rem .9rem">+ New patient</button>
</div>
<div class="card">
  <input type="text" id="srch" placeholder="Search name or PGX-ID..." onkeyup="filterP()" style="margin-bottom:.75rem">
  <div id="pl">{rows}</div>
</div>
<div id="rm2" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:200;align-items:center;justify-content:center">
  <div class="card" style="width:340px;max-height:88vh;overflow-y:auto">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.9rem">
      <div style="font-weight:700;color:var(--ac)">Register new patient</div>
      <button class="btn bd2" onclick="document.getElementById('rm2').style.display='none'" style="padding:.25rem .55rem">✕</button>
    </div>
    <form method="POST" action="/doctor/patients/register">
      <div class="fr"><label>Full name *</label><input type="text" name="full_name" required></div>
      <div class="g2">
        <div class="fr"><label>Age</label><input type="number" name="age" min="1" max="120"></div>
        <div class="fr"><label>Condition</label><select name="condition"><option>Hypertension</option><option>Type 2 Diabetes</option><option>Hyperlipidemia</option><option>Atrial Fibrillation</option><option>Depression</option><option>Rheumatoid Arthritis</option><option>HIV/AIDS</option><option>Epilepsy</option><option>Colorectal Cancer</option></select></div>
      </div>
      <div class="fr"><label>Patient username</label><input type="text" name="pat_username" required></div>
      <div class="fr"><label>Patient password</label><input type="password" name="pat_password" required></div>
      <button type="submit" class="btn bdoc" style="width:100%;margin-top:.4rem">Register & generate PGX-ID</button>
    </form>
  </div>
</div>
<script>function filterP(){{const q=document.getElementById('srch').value.toLowerCase();document.querySelectorAll('.pr').forEach(r=>r.style.display=r.textContent.toLowerCase().includes(q)?'flex':'none');}}</script>"""
    return render_template_string(page("Patients", body, "doctor", "patients"))

@app.route("/doctor/patients/register", methods=["POST"])
def register_patient():

    if require("doctor"):
        return require("doctor")

    f = request.form
    conn = get_db()

    try:

        # Create patient login user
        conn.execute(
            """
            INSERT INTO users
            (username, password_hash, role, full_name)
            VALUES (?, ?, ?, ?)
            """,
            (
                f["pat_username"],
                hash_pw(f["pat_password"]),
                "patient",
                f["full_name"]
            )
        )

        conn.commit()

        # Get created user id
        uid = conn.execute(
            "SELECT id FROM users WHERE username=?",
            (f["pat_username"],)
        ).fetchone()["id"]

        # Generate PGX ID
        pgx = gen_patient_id()

        # Insert patient record
        conn.execute(
            """
            INSERT INTO patients
            (
                patient_id,
                user_id,
                registered_by,
                assigned_doctor,
                full_name,
                age,
                primary_condition
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pgx,
                uid,
                session["user_id"],
                session["user_id"],   # auto assign doctor
                f["full_name"],
                int(f.get("age", 0) or 0),
                f.get("condition", "")
            )
        )

        conn.commit()

        # Save user credentials into file
        with open("user_login_details.txt", "a", encoding="utf-8") as file:

            file.write("NEW PATIENT CREATED\n")
            file.write("===================\n")
            file.write(f"Patient Name: {f['full_name']}\n")
            file.write(f"Username: {f['pat_username']}\n")
            file.write(f"Password: {f['pat_password']}\n")
            file.write(f"Assigned Doctor ID: {session['user_id']}\n")
            file.write(f"PGX ID: {pgx}\n")
            file.write("---------------------------\n\n")

        # Log action
        log_action(
            session["user_id"],
            "register_patient",
            pgx
        )

    except sqlite3.IntegrityError:
        conn.close()
        return redirect("/doctor/patients")

    conn.close()
    return redirect("/doctor/patients")

@app.route("/doctor/predict")
def doctor_predict():
    if require("doctor"): return require("doctor")
    conn=get_db()
    pts=conn.execute(
    "SELECT patient_id,full_name FROM patients WHERE registered_by=? OR assigned_doctor=? ORDER BY full_name",
    (session["user_id"], session["user_id"])
    ).fetchall()
    conn.close()
    opts="".join(f'<option value="{p["patient_id"]}">{p["full_name"]} ({p["patient_id"]})</option>' for p in pts)
    pid=request.args.get("pid","")
    drugs=["None","Metoprolol","Amlodipine","Warfarin","Metformin","Metformin + Glipizide","Simvastatin","Rosuvastatin","Pravastatin","Escitalopram","Sertraline","Phenytoin","Azathioprine","Methotrexate","Abacavir + Lamivudine","Tenofovir + Emtricitabine","5-Fluorouracil (5-FU)","Capecitabine","Irinotecan"]
    drug_opts="".join(f"<option>{d}</option>" for d in drugs)
    body=f"""<div style="display:grid;grid-template-columns:310px 1fr;gap:.9rem;min-height:calc(100vh - 120px)">
<div class="card" style="overflow-y:auto;max-height:calc(100vh - 120px)">
  <div class="sh" style="margin-top:0">Patient</div>
  <div class="fr"><label>Select patient</label>
    <select id="psel" onchange="loadPat(this.value)"><option value="">-- choose --</option>{opts}</select>
  </div>
  <div style="display:flex;gap:.4rem;margin-bottom:.7rem">
    <button class="btn bd2" id="v1b" onclick="sv(1)" style="flex:1;font-size:.73rem">Visit 1</button>
    <button class="btn bd2" id="v2b" onclick="sv(2)" style="flex:1;font-size:.73rem">Visit 2</button>
    <button class="btn bd2" id="v3b" onclick="sv(3)" style="flex:1;font-size:.73rem">Visit 3</button>
  </div>
  <div id="hs" style="display:none">
    <div class="sh">Previous visit</div>
    <div class="fr"><label>Previous drug</label><select id="pd">{drug_opts}</select></div>
    <div class="fr"><label>Previous outcome</label><select id="po"><option>None</option><option>Stable</option><option>Improved</option><option>No Change</option><option>Adverse Reaction</option><option>Switched Drug</option><option>Lost to Follow-up</option></select></div>
  </div>
  <div class="sh">Clinical</div>
  <div class="fr"><label>Primary condition</label><select id="fc"><option>Hypertension</option><option>Type 2 Diabetes</option><option>Hyperlipidemia</option><option>Atrial Fibrillation</option><option>Depression</option><option>Rheumatoid Arthritis</option><option>HIV/AIDS</option><option>Epilepsy</option><option>Colorectal Cancer</option></select></div>
  <div class="g2">
    <div class="fr"><label>Age</label><input type="number" id="fa" value="52"></div>
    <div class="fr"><label>Sex</label><select id="fs"><option>Male</option><option>Female</option></select></div>
    <div class="fr"><label>Weight (kg)</label><input type="number" id="fw" value="70" step="0.1"></div>
    <div class="fr"><label>BMI</label><input type="number" id="fb" value="26.5" step="0.1"></div>
    <div class="fr"><label>GFR</label><input type="number" id="fg" value="74"></div>
    <div class="fr"><label>ALT (U/L)</label><input type="number" id="falt" value="25" step="0.1"></div>
    <div class="fr"><label>Ethnicity</label><select id="fe">
      <option>Caucasian</option><option>Asian</option><option>Black</option><option>Hispanic</option><option>Other</option>
    </select></div>
  </div>
  <div class="sh">Genomic variants</div>
  <div class="fr"><label>CYP2C9 Phenotype</label><select id="f1"><option>Normal Metabolizer</option><option>Intermediate Metabolizer</option><option>Poor Metabolizer</option></select></div>
  <div class="fr"><label>CYP2C9 Alleles</label><select id="f1a">
    <option>*1/*1</option><option>*1/*2</option><option>*1/*3</option>
    <option>*2/*2</option><option>*2/*3</option><option>*3/*3</option>
  </select></div>
  <div class="fr"><label>CYP2D6</label><select id="f2"><option>Normal Metabolizer</option><option>Intermediate Metabolizer</option><option>Poor Metabolizer</option><option>Rapid Metabolizer</option><option>Ultrarapid Metabolizer</option></select></div>
  <div class="fr"><label>VKORC1 Genotype</label><select id="f3"><option>GG</option><option>GA</option><option>AA</option></select></div>
  <div class="fr"><label>VKORC1 Sensitivity</label><select id="f3s"><option>Low Sensitivity</option><option>Moderate Sensitivity</option><option>High Sensitivity</option></select></div>
  <div class="fr"><label>SLCO1B1</label><select id="f4"><option>Normal Function</option><option>Decreased Function</option><option>Poor Function</option></select></div>
  <div class="fr"><label>CYP2C19</label><select id="f5"><option>Normal Metabolizer</option><option>Intermediate Metabolizer</option><option>Poor Metabolizer</option><option>Rapid Metabolizer</option><option>Ultrarapid Metabolizer</option></select></div>
  <div class="g2">
    <div class="fr"><label>TPMT</label><select id="f6"><option>Normal Activity</option><option>Intermediate Activity</option><option>Low Activity</option><option>Deficient Activity</option></select></div>
    <div class="fr"><label>DPYD</label><select id="f7"><option>Normal Function</option><option>Decreased Function</option><option>Non-functional</option></select></div>
  </div>
  <div class="fr"><label>HLA-B*5701</label><select id="f8"><option>Negative</option><option>Positive</option></select></div>
  <button class="btn bdoc" style="width:100%;margin-top:.5rem" onclick="runP()">🔬 Generate recommendation</button>
</div>
<div id="rp"><div class="card" style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--tx3)">Fill patient profile and click Generate</div></div>
</div>
<script>
let cv=1,cpid='';
function sv(n){{cv=n;['v1b','v2b','v3b'].forEach((id,i)=>document.getElementById(id).className='btn '+(i+1===n?'bdoc':'bd2'));document.getElementById('hs').style.display=n>1?'block':'none';}}
sv(1);
function loadPat(pid){{
  cpid=pid; if(!pid)return;
  fetch('/doctor/patient_info/'+pid).then(r=>r.json()).then(d=>{{
    if(d.condition)document.getElementById('fc').value=d.condition;
    if(d.age)document.getElementById('fa').value=d.age;
    const nv=d.next_visit&&d.next_visit<=3?d.next_visit:(d.next_visit>3?3:1);
    sv(nv);
    const pdEl=document.getElementById('pd'), poEl=document.getElementById('po');
    if(d.last_drug&&d.last_drug!=='None'&&pdEl) pdEl.value=d.last_drug;
    if(d.last_outcome&&d.last_outcome!=='None'&&poEl) poEl.value=d.last_outcome;
    if(d.total_visits>0){{
      const oclr={{Stable:'bok-b',Improved:'bok-b','No Change':'bwarn-b','Adverse Reaction':'bdanger-b'}};
      let h='<div class="card" style="margin-bottom:.7rem">';
      h+='<div style="font-size:.69rem;color:var(--tx3);font-weight:700;margin-bottom:.5rem">VISIT HISTORY FROM DATABASE</div>';
      d.visits.forEach(v=>{{
        h+='<div style="display:flex;justify-content:space-between;align-items:center;padding:.35rem 0;border-bottom:1px solid var(--bd);font-size:.78rem">';
        h+='<span style="color:var(--tx2)">Visit '+v.visit_number+' &middot; '+v.created_at+'</span>';
        h+='<span style="font-weight:600">'+v.recommended_drug+'</span>';
        h+='<span class="badge '+(oclr[v.treatment_outcome]||'bdoc-b')+'">'+( v.treatment_outcome||'&mdash;')+'</span></div>';
      }});
      h+='<div style="font-size:.75rem;color:var(--ac);margin-top:.5rem;font-weight:600">Auto-set to Visit '+nv+' &middot; Previous drug + outcome pre-filled from database</div></div>';
      document.getElementById('rp').innerHTML=h;
    }}
  }});
}}
function gf(){{return{{Primary_Condition:document.getElementById('fc').value,Age:document.getElementById('fa').value,Sex:document.getElementById('fs').value,Weight_kg:document.getElementById('fw').value,BMI:document.getElementById('fb').value,GFR_mL_min:document.getElementById('fg').value,ALT_U_L:document.getElementById('falt').value,Ethnicity:document.getElementById('fe').value,CYP2C9_Phenotype:document.getElementById('f1').value,CYP2D6_Phenotype:document.getElementById('f2').value,VKORC1_Genotype:document.getElementById('f3').value,VKORC1_Sensitivity:document.getElementById('f3s').value,SLCO1B1_Function:document.getElementById('f4').value,CYP2C19_Phenotype:document.getElementById('f5').value,TPMT_Activity:document.getElementById('f6').value,DPYD_Function:document.getElementById('f7').value,HLA_B5701:document.getElementById('f8').value,CYP2C9_Alleles:document.getElementById('f1a').value,visit_number:cv,prev_outcome:cv>1?(document.getElementById('po')?.value||'None'):'None',prev_drug:cv>1?(document.getElementById('pd')?.value||'None'):'None',patient_id:cpid}};}}
async function runP(){{
  document.getElementById('rp').innerHTML='<div class="card" style="color:var(--tx2);padding:2rem;text-align:center">Analysing genomic profile...</div>';
  const r=await fetch('/predict',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(gf())}});
  const d=await r.json();
  if(d.error){{document.getElementById('rp').innerHTML='<div class="alert aerr">'+d.error+'</div>';return;}}
  const al={{FIRST_VISIT:'bdoc-b',CONTINUE:'bok-b',ESCALATE:'bwarn-b',SWITCH:'bdanger-b'}};
  const lb={{FIRST_VISIT:'First visit — genetics-only',CONTINUE:'Previous treatment effective',ESCALATE:'No response — dose escalated',SWITCH:'Previous ADR — drug switched'}};
  let fh=d.flags.map(f=>f==='None'?'':
    `<div class="alert ${{f.includes('CONTRAINDICATED')?'aerr':f.includes('effective')?'aok':'ainfo'}}" style="margin-bottom:.35rem">${{f}}</div>`
  ).join('');
  const LF=['Previous_Outcome_encoded','Previous_ADR_flag','Is_Return_Visit'];
  const GF=['CYP2C9_Phenotype_encoded','CYP2D6_Phenotype_encoded','VKORC1_Genotype_encoded','VKORC1_Sensitivity_encoded','SLCO1B1_Function_encoded','CYP2C19_Phenotype_encoded','TPMT_Activity_encoded','DPYD_Function_encoded','HLA_B5701_encoded','CYP2C9_AlleleScore'];
  let sh=d.shap_features?d.shap_features.map(([n,v])=>{{const c=LF.includes(n)?'#e74c3c':GF.includes(n)?'#1d9e75':'#378add';const w=Math.min(Math.abs(v)*400,100);return`<div class="sr2"><div class="sl">${{n.replace('_encoded','').replace(/_/g,' ')}}</div><div class="st"><div class="sf" style="width:${{w}}%;background:${{c}}"></div></div><div class="sv">${{v.toFixed(3)}}</div></div>`;}}
  ).join(''):'';
  const saveSec=cpid?`<div style="margin-top:.7rem;display:flex;gap:.5rem;align-items:center"><span style="font-size:.78rem;color:var(--tx2)">Save to ${{cpid}}</span>
    <select id="osel" style="flex:1;font-size:.78rem;padding:.38rem .55rem;background:var(--bg3);border:1px solid var(--bd);border-radius:var(--rs);color:var(--tx)"><option>Stable</option><option>Improved</option><option>No Change</option><option>Adverse Reaction</option><option>First Visit</option></select>
    <button class="btn bpat" onclick="saveR('${{d.full_drug}}','${{d.dose}}','${{d.visit_action}}',${{d.full_confidence}},'${{d.flags.join("|")}}')" style="padding:.38rem .75rem;font-size:.78rem">Save</button></div>`:'<div class="alert ainfo" style="margin-top:.7rem">Select a patient to save</div>';
  document.getElementById('rp').innerHTML=`<div class="card">
    <span class="badge ${{al[d.visit_action]||'bdoc-b'}}" style="margin-bottom:.7rem;display:inline-block">${{lb[d.visit_action]||d.visit_action}}</span>
    <div class="cg">
      <div class="cc"><div style="font-size:.69rem;color:var(--tx3);margin-bottom:.25rem">V1 MODEL — genetics only</div><div style="font-weight:600">${{d.v1_drug}}</div><div style="font-size:.7rem;color:var(--tx2)">${{(d.v1_confidence*100).toFixed(0)}}%</div></div>
      <div class="cc win"><div style="font-size:.69rem;color:var(--tx3);margin-bottom:.25rem">FULL MODEL — + history</div><div style="font-weight:600;color:var(--ac)">${{d.full_drug}}</div><div style="font-size:.7rem;color:var(--ac)">${{(d.full_confidence*100).toFixed(0)}}% · ${{d.v1_drug===d.full_drug?'both agree':'models differ'}}</div></div>
    </div>
    <div style="margin-bottom:.7rem;padding:.7rem;background:var(--bg3);border-radius:var(--rs)"><div style="font-size:.69rem;color:var(--tx3);margin-bottom:.25rem">RECOMMENDATION</div>
      <div style="font-size:1.35rem;font-weight:700">${{d.full_drug}}</div><span class="badge bwarn-b" style="margin-top:.25rem">${{d.dose}}</span></div>
    ${{fh}}
    ${{sh?`<div style="margin:.65rem 0"><div style="font-size:.69rem;color:var(--tx3);margin-bottom:.4rem">SHAP — 🟢 Gene &nbsp; 🔴 History &nbsp; 🔵 Clinical</div>${{sh}}</div>`:''}}
    <div style="margin-top:.65rem;padding:.7rem;background:var(--bg3);border-radius:var(--rs);font-size:.8rem;line-height:1.7;color:var(--tx2)">${{d.nl_explanation}}</div>
    ${{saveSec}}<div id="sm"></div></div>`;
}}
async function saveR(drug,dose,action,conf,flags){{
  const outcome=document.getElementById('osel')?.value||'Stable';
  const r=await fetch('/doctor/save_record',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{patient_id:cpid,visit_number:cv,recommended_drug:drug,dose_category:dose,visit_action:action,confidence:conf,clinical_flag:flags,treatment_outcome:outcome}})}});
  const d=await r.json();
  document.getElementById('sm').innerHTML=d.ok?'<div class="alert aok" style="margin-top:.45rem">Saved successfully</div>':'<div class="alert aerr" style="margin-top:.45rem">'+d.error+'</div>';
}}
{f'document.getElementById("psel").value="{pid}";loadPat("{pid}");' if pid else ''}
</script>"""
    return render_template_string(page("Predict", body, "doctor", "predict"))

@app.route("/doctor/patient_info/<pid>")
def patient_info_api(pid):
    if "user_id" not in session: return jsonify({}),401
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
    if "user_id" not in session: return jsonify({"error":"Unauthorized"}),401
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
# CHATBOT HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def generate_side_effect_response(drug_name, patient_name="", genetics_flags=None):

    if genetics_flags is None:
        genetics_flags = []

    info = MEDICATION_INFO.get(drug_name)

    if not info:
        return (
            f"I could not find detailed side-effect information for "
            f"{drug_name}. Please consult your doctor or pharmacist."
        )

    response = []

    response.append(
        f"{patient_name}, here is detailed information about "
        f"your medication: {drug_name}."
    )

    response.append("\nCommon side effects:")

    for effect in info["common"]:
        response.append(f"• {effect}")

    response.append(
        "\nSerious side effects requiring medical attention:"
    )

    for effect in info["serious"]:
        response.append(f"• {effect}")

    response.append(
        f"\nMedication advice: {info['advice']}"
    )

    response.append(
        f"\nPharmacogenomic note: {info['genetic_note']}"
    )

    if genetics_flags:

        response.append("\nImportant genomic considerations:")

        for warning in genetics_flags:
            response.append(f"• {warning}")

    response.append(
        "\nDo not stop or change medication without "
        "consulting your healthcare provider."
    )

    return "\n".join(response)


def get_genetic_warnings():

    warnings = []

    warnings.append(
        "Your pharmacogenomic profile may influence medication response and toxicity risk."
    )

    warnings.append(
        "Genetic biomarkers are being used to personalise your treatment recommendations."
    )

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# REPLACE YOUR EXISTING /patient/chat_api ROUTE WITH THIS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/patient/chat_api", methods=["POST"])
def patient_chat_api():

    if "user_id" not in session:
        return jsonify({
            "reply": "Please log in again."
        }), 401

    data = request.get_json()

    msg = data.get("message", "").lower()
        # SMART DOSAGE RESPONSE

    patient, visits = get_pat(session["user_id"])

    latest_visit = visits[0] if visits else None

    if any(word in msg for word in ["dosage", "dose", "tablet", "mg"]):

        if latest_visit:

            visit_date = latest_visit.get("created_at")

            try:
                visit_datetime = datetime.datetime.strptime(
                    visit_date[:19],
                    "%Y-%m-%d %H:%M:%S"
                )

                days_since = (
                    datetime.datetime.now() - visit_datetime
                ).days

            except:
                days_since = 999

            if days_since <= 2:

                return jsonify({
                    "reply": "Your medication was recently prescribed or adjusted. Please continue the current dosage and revisit your doctor before changing the dose."
                })

            else:

                return jsonify({
                    "reply": "Dosage adjustments depend on your treatment response and genomic profile. Please consult your doctor before increasing or changing medication dosage."
                })

        else:

            return jsonify({
                "reply": "No recent medical visit was found. Please consult your doctor regarding dosage instructions."
            })
        
    patient, visits = get_pat(session["user_id"])

    patient_name = session.get(
        "full_name",
        "Patient"
    ).split()[0]

    latest_visit = visits[0] if visits else None

    current_drug = (
        latest_visit["recommended_drug"]
        if latest_visit
        else "No medication recorded"
    )

    previous_outcome = (
        latest_visit["treatment_outcome"]
        if latest_visit
        else "None"
    )

    genetic_warnings = get_genetic_warnings()

    # ─────────────────────────────────────────
    # SIDE EFFECTS
    # ─────────────────────────────────────────

    if any(keyword in msg for keyword in [
        "side effect",
        "side effects",
        "effects",
        "reaction",
        "symptom",
        "medicine issue",
        "drug issue"
    ]):

        reply = generate_side_effect_response(
            current_drug,
            patient_name,
            genetic_warnings
        )

        return jsonify({
            "reply": reply
        })

    # ─────────────────────────────────────────
    # CURRENT MEDICATION
    # ─────────────────────────────────────────

    elif any(keyword in msg for keyword in [
        "medicine",
        "medication",
        "drug",
        "taking"
    ]):

        if latest_visit:

            reply = (
                f"{patient_name}, you are currently prescribed "
                f"{current_drug}.\n\n"

                f"Dose category: "
                f"{latest_visit['dose_category']}\n"

                f"Treatment outcome from last visit: "
                f"{previous_outcome}\n"

                f"Visit number: "
                f"{latest_visit['visit_number']}\n\n"

                f"This recommendation was personalised "
                f"using your pharmacogenomic and "
                f"clinical profile."
            )

        else:

            reply = (
                "No medication records were found yet. "
                "Please consult your doctor."
            )

        return jsonify({
            "reply": reply
        })

    # ─────────────────────────────────────────
    # VISIT HISTORY
    # ─────────────────────────────────────────

    elif any(keyword in msg for keyword in [
        "visit",
        "history",
        "previous",
        "past"
    ]):

        if not visits:

            return jsonify({
                "reply": "No visit history found yet."
            })

        history_text = []

        history_text.append(
            f"{patient_name}, here is your recent visit history:\n"
        )

        for visit in visits[:5]:

            history_text.append(
                f"• Visit {visit['visit_number']} — "
                f"{visit['recommended_drug']} "
                f"({visit['dose_category']}) — "
                f"Outcome: {visit['treatment_outcome']}"
            )

        return jsonify({
            "reply": "\n".join(history_text)
        })
    
       
    # ─────────────────────────────────────────
    # ASSIGNED DOCTOR
    # ─────────────────────────────────────────

    elif any(keyword in msg for keyword in [
        "my doctor",
        "assigned doctor",
        "doctor name",
        "who is my doctor",
        "which doctor",
        "doctor assigned"
    ]):

        conn = get_db()

        doctor = conn.execute("""
            SELECT u.full_name, u.specialisation, u.hospital
            FROM patients p
            LEFT JOIN users u
            ON p.assigned_doctor = u.id
            WHERE p.user_id = ?
        """, (session["user_id"],)).fetchone()

        conn.close()

        if doctor and doctor["full_name"]:

            reply = (
                f"Your assigned doctor is Dr. {doctor['full_name']}.\n\n"
                f"Specialisation: {doctor['specialisation'] or 'Not available'}\n"
                f"Hospital/Clinic: {doctor['hospital'] or 'Not available'}\n\n"
                f"You can consult your doctor for medication guidance, "
                f"adverse reactions, and treatment follow-up."
            )

        else:

            reply = (
                "No assigned doctor information was found in the system. "
                "Please go to the Doctors section and choose a doctor."
            )

        return jsonify({
            "reply": reply
        })

    # ─────────────────────────────────────────
    # DOSE QUESTIONS
    # ─────────────────────────────────────────

    elif "dose" in msg:

        if previous_outcome == "No Change":

            reply = (
                "Your previous treatment response "
                "indicated limited improvement. "
                "The system may recommend dose escalation "
                "or therapy adjustment based on your "
                "genomic profile and longitudinal history."
            )

        elif previous_outcome == "Adverse Reaction":

            reply = (
                "Your previous visit recorded an "
                "adverse reaction. The recommendation "
                "engine may switch medication class "
                "to reduce future risk."
            )

        else:

            reply = (
                "Your current dose appears consistent "
                "with your recent clinical response "
                "and genomic profile."
            )

        return jsonify({
            "reply": reply
        })

    # ─────────────────────────────────────────
    # EMERGENCY DETECTION
    # ─────────────────────────────────────────

    elif any(keyword in msg for keyword in [
        "chest pain",
        "severe bleeding",
        "breathing problem",
        "can't breathe",
        "collapse"
    ]):

        return jsonify({
            "reply":
            "Your symptoms may require urgent "
            "medical attention. Please contact "
            "emergency services or your doctor immediately."
        })

#general questions
    elif any(word in msg for word in [
            "headache",
            "headaches",
            "migraine",
            "pain in head"
    ]):

        reply = (
                "Headaches can happen due to stress, dehydration, "
                "lack of sleep, eye strain, or sometimes as a side "
                "effect of medication.\n\n"

                "Please make sure you:\n"
                "• Drink enough water\n"
                "• Get proper rest\n"
                "• Avoid skipping meals\n"
                "• Reduce screen exposure if possible\n\n"

                "Please contact your doctor immediately if you have:\n"
                "• Severe or sudden headache\n"
                "• Blurred vision\n"
                "• Vomiting\n"
                "• Dizziness\n"
                "• Chest pain\n"
                "• Weakness or numbness"
        )

        return jsonify({
                "reply": reply
        })

    elif any(word in msg for word in [
            "tired",
            "weak",
            "fatigue",
            "sleepy"
    ]):

        reply = (
                "Feeling tired or weak may happen due to stress, "
                "poor sleep, low hydration, infection, anemia, "
                "or as a side effect of medication.\n\n"

                "Try to:\n"
                "• Rest properly\n"
                "• Drink enough water\n"
                "• Eat balanced meals\n"
                "• Avoid skipping medication\n\n"

                "If tiredness becomes severe or continues for many days, "
                "please consult your doctor for proper evaluation."
        )

        return jsonify({
                "reply": reply
        })

    elif any(word in msg for word in [
            "food",
            "diet",
            "eat",
            "spicy food",
            "after food",
            "before food"
    ]):

        reply = (
                "Food instructions can vary depending on the medicine "
                "you are taking.\n\n"

                "Some medicines are taken:\n"
                "• After food to reduce stomach irritation\n"
                "• Before food for better absorption\n\n"

                "Please follow the prescription given by your doctor.\n\n"

                "In general:\n"
                "• Stay hydrated\n"
                "• Avoid excessive junk food\n"
                "• Limit alcohol and smoking\n"
                "• Maintain balanced nutrition"
        )

        return jsonify({
                "reply": reply
        })

    elif any(word in msg for word in [
            "exercise",
            "gym",
            "workout",
            "running"
    ]):

        reply = (
                "Light to moderate exercise is generally beneficial "
                "for overall health.\n\n"

                "However, your activity level may depend on:\n"
                "• Your medical condition\n"
                "• Current treatment\n"
                "• Medication side effects\n\n"

                "Stop exercising and contact your doctor if you feel:\n"
                "• Chest pain\n"
                "• Severe dizziness\n"
                "• Shortness of breath\n"
                "• Extreme weakness"
        )

        return jsonify({
                "reply": reply
        })

    else:

        reply = (
                f"{patient_name}, I can help explain:\n\n"

                "• Medication guidance\n"
                "• Side effects\n"
                "• Visit history\n"
                "• Assigned doctor details\n"
                "• Dosage safety\n"
                "• General health questions\n\n"

                "For serious symptoms or treatment changes, "
                "please consult your doctor directly."
        )

        return jsonify({
                "reply": reply
        })


def get_pat(uid):
    conn=get_db()
    p=conn.execute("SELECT * FROM patients WHERE user_id=?",(uid,)).fetchone()
    vs=conn.execute("SELECT * FROM visit_records WHERE patient_id=? ORDER BY created_at DESC",(p["patient_id"],)).fetchall() if p else []
    conn.close(); return p,vs

@app.route("/patient/home")
def patient_home():
    if require("patient"): return require("patient")
    p,vs=get_pat(session["user_id"])
    pgx=p["patient_id"] if p else "—"; fn=(session["full_name"] or "").split()[0]
    if vs:
        lv=vs[0]
        med=f'<div class="card" style="margin-bottom:.75rem;border-color:var(--ac)"><div style="font-size:.69rem;color:var(--tx3);margin-bottom:.25rem">CURRENT MEDICATION</div><div style="font-size:1.3rem;font-weight:700">{lv["recommended_drug"]}</div><span class="badge bwarn-b" style="margin-top:.25rem">{lv["dose_category"]}</span><div style="font-size:.76rem;color:var(--tx2);margin-top:.45rem">Condition: {p["primary_condition"] or "—"} · Visit {lv["visit_number"]} · {str(lv["created_at"])[:10]}</div></div>'
        vsum="".join(f'<div class="vc"><div style="font-size:.68rem;color:var(--tx3);font-weight:700;margin-bottom:.25rem">VISIT {v["visit_number"]} · {str(v["created_at"])[:10]}</div><div style="font-weight:600">{v["recommended_drug"]}</div><div style="font-size:.76rem;color:var(--tx2);margin-top:.18rem">{v["dose_category"]} · <span class="badge {"bok-b" if v["treatment_outcome"] in ["Stable","Improved","First Visit"] else "bwarn-b" if v["treatment_outcome"]=="No Change" else "bdanger-b"}">{v["treatment_outcome"] or "—"}</span></div></div>' for v in vs[:3])
    else:
        med='<div class="card" style="margin-bottom:.75rem;color:var(--tx3);font-size:.83rem">No records yet. Your doctor will add records after your visit.</div>'
        vsum='<div style="color:var(--tx3);font-size:.8rem">No visits recorded yet</div>'
    body=f"""<div style="margin-bottom:.9rem"><div style="font-size:1.05rem;font-weight:600">Hi, {fn}</div><div style="font-size:.8rem;color:var(--tx2)">PGX-ID: <code style="color:var(--ac)">{pgx}</code></div></div>
{med}<div class="card" style="margin-bottom:.75rem"><div style="font-size:.75rem;color:var(--tx2);font-weight:700;margin-bottom:.7rem">MY VISIT HISTORY</div>{vsum}</div>
<button class="btn bpat" style="width:100%" onclick="location.href='/patient/chat'">💬 Ask about my medication</button>"""
    return render_template_string(page("Home", body, "patient", "home"))

@app.route("/patient/chat")
def patient_chat():
    if require("patient"): return require("patient")
    fn=(session["full_name"] or "").split()[0]
    body=f"""<div style="display:flex;flex-direction:column;height:calc(100vh - 130px)">
<div id="msgs" style="flex:1;overflow-y:auto;padding:.5rem 0;display:flex;flex-direction:column;gap:.65rem">
  <div class="mw"><div class="bubble bb">Hi {fn}! I can help you understand your medication and visit history. What would you like to know?</div>
  <div class="qps"><div class="qp" onclick="sq('What medicine am I currently taking?')">What medicine am I on?</div><div class="qp" onclick="sq('Why was my dose changed?')">Why was my dose changed?</div><div class="qp" onclick="sq('What are side effects of my medication?')">Side effects?</div><div class="qp" onclick="sq('How many visits have I had?')">My visit history</div></div></div>
</div>
<div style="padding:.65rem 0;border-top:1px solid var(--bd);display:flex;gap:.45rem;background:var(--bg)">
  <input type="text" id="ci" placeholder="Ask about your health, medication..." style="flex:1" onkeydown="if(event.key==='Enter')sm()">
  <button class="btn bpat" onclick="sm()">Send</button>
</div>
</div>
<script>
const hist=[];
async function sm(){{const inp=document.getElementById('ci');const msg=inp.value.trim();if(!msg)return;inp.value='';am(msg,'u');hist.push({{role:'user',content:msg}});at();const r=await fetch('/patient/chat_api',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{message:msg,history:hist}})}});const d=await r.json();rt();am(d.reply,'b');hist.push({{role:'assistant',content:d.reply}});}}
function sq(q){{document.getElementById('ci').value=q;sm();}}
function am(txt,role){{const w=document.createElement('div');w.className='mw'+(role==='u'?' u':'');w.innerHTML=`<div class="bubble ${{role==='u'?'bu':'bb'}}">${{txt}}</div><div class="mt">${{new Date().toLocaleTimeString([],{{hour:'2-digit',minute:'2-digit'}})}}</div>`;document.getElementById('msgs').appendChild(w);w.scrollIntoView({{behavior:'smooth'}});}}
function at(){{const d=document.createElement('div');d.id='typ';d.className='mw';d.innerHTML='<div class="bubble bb" style="color:var(--tx3)">Thinking...</div>';document.getElementById('msgs').appendChild(d);d.scrollIntoView({{behavior:'smooth'}});}}
function rt(){{const t=document.getElementById('typ');if(t)t.remove();}}
</script>"""
    return render_template_string(page("Chat", body, "patient", "chat"))


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
    if "user_id" not in session: return jsonify({"error":"Unauthorized"}),401
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
@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session: return jsonify({"error":"Unauthorized"}),401
    if not ML_READY: return jsonify({"error":"ML models not loaded. Run phases 1–3 first."})
    try:
        form=request.get_json()
        vn=int(form.get("visit_number",1)); po=form.get("prev_outcome","None"); pid=form.get("patient_id","")
        enc=encode_patient(form,vn,po)

        # Build vectors strictly in the exact feature order used during training.
        # No silent .get(..., 0) fallback is allowed here.
        missing_v1 = [f for f in FEATURES_V1 if f not in enc]
        missing_full = [f for f in FEATURES_FULL if f not in enc]
        if missing_v1 or missing_full:
            raise ValueError(
                f"Feature mismatch. Missing V1={missing_v1}; "
                f"Missing Full={missing_full}"
            )

        Xv1=np.asarray([[enc[f] for f in FEATURES_V1]],dtype=float)
        Xf=np.asarray([[enc[f] for f in FEATURES_FULL]],dtype=float)

        # Apply the same training-time median imputers used in Phase 3.
        Xv1 = rf_v1_imputer.transform(Xv1)
        Xf = rf_full_imputer.transform(Xf)

        v1l=int(rf_v1.predict(Xv1)[0])
        v1d=drug_map.get(str(v1l),"Unknown")
        v1_probs=rf_v1.predict_proba(Xv1)[0]
        v1_class_index=list(rf_v1.classes_).index(v1l)
        v1c=float(v1_probs[v1_class_index])

        fl=int(rf_full.predict(Xf)[0])
        fd=drug_map.get(str(fl),"Unknown")
        fp=rf_full.predict_proba(Xf)[0]
        full_class_index=list(rf_full.classes_).index(fl)
        fc=float(fp[full_class_index])
        rules=run_rules(form,po); fc=max(0.0,min(1.0,fc+rules["conf_mod"]))
        sf=[]
        try:
            import shap; exp=shap.TreeExplainer(rf_full); sv=exp.shap_values(Xf)
            csv=sv[fl][0]; ti=np.argsort(np.abs(csv))[::-1][:10]
            sf=[(FEATURES_FULL[i],float(csv[i])) for i in ti]
        except: pass
        nl=[f"[Visit {vn}] "]
        if vn==1: nl[0]+="First visit — recommendation based on genetic profile."
        elif po=="No Change": nl[0]+="No response to previous dose — dose escalated."
        elif po=="Adverse Reaction": nl[0]+="Previous drug caused adverse reaction — switching."
        elif po in ["Stable","Improved"]: nl[0]+=f"Previous treatment {po.lower()} — continuing regimen."
        else: nl[0]+=f"Return visit — previous outcome: {po}."
        nl.append(f"<strong>{fd}</strong> — {rules['dose']}. Confidence: {fc:.0%}.")
        if fc>=0.90: nl.append("High confidence recommendation.")
        elif fc<0.75: nl.append("Moderate confidence — clinical review advised.")
        if pid: log_action(session["user_id"],"predict",pid)
        return jsonify({"v1_drug":v1d,"v1_confidence":round(v1c,3),"full_drug":fd,"full_confidence":round(fc,3),"dose":rules["dose"],"flags":rules["flags"],"visit_action":rules["visit_action"],"nl_explanation":"<br>".join(nl),"shap_features":sf,"models_agree":v1d==fd})
    except Exception as e:
        return jsonify({"error":str(e)}),500

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