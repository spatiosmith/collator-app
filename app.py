# app.py
# Data Collator
# - Tag-based mapping
# - Mapping persistence
# - Canonical schema
# - Conflict resolution
# - All canonical fields uploaded to Database
# - Email auto-generation from domain-based patterns

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pymongo import MongoClient
from dateutil.parser import parse
import sqlite3
from typing import Dict, Optional, Tuple
import os
import requests

# ---------------------------
# Configuration & Model
# ---------------------------

# Path to your local SQLite DB with email patterns
# Make sure this file exists in the same directory as app.py (or adjust the path).
PATTERN_DB_URL = "https://pub-17d84e44759e445b88b8626c227d8cf4.r2.dev/email_patterns.db"
PATTERN_DB_LOCAL = "email_patterns.db"
PATTERN_DB_PATH = PATTERN_DB_LOCAL

def ensure_local_pattern_db():
    """Download the .db file from remote URL if not already local."""
    if os.path.exists(PATTERN_DB_LOCAL):
        return PATTERN_DB_LOCAL
    try:
        r = requests.get(PATTERN_DB_URL, timeout=10)
        r.raise_for_status()
        with open(PATTERN_DB_LOCAL, "wb") as f:
            f.write(r.content)
        return PATTERN_DB_LOCAL
    except Exception as e:
        st.error(f"Failed to download email pattern DB: {e}")
        return None

# Try loading AI model; fallback to fuzzy if unavailable.
try:
    from sentence_transformers import SentenceTransformer, util
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    AI_AVAILABLE = True
except Exception:
    MODEL = None
    AI_AVAILABLE = False

# Canonical headers (user-provided)
CANONICAL_HEADERS_RAW = [
    "First Name", "Last Name", "Job Title", "Seniority", "Company Name",
    "Email Address", "Status", "Phone Number", "Employee Size",
    "Industry", "Prospect Linkedin", "Company Website", "Company Linkedin",
    "City", "State", "Country", "Company Address",
    "Company City", "Company State", "Postal Code", "Company Country", "SIC", "NAICS",
]

def normalize(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")

CANONICAL_HEADERS = [normalize(h) for h in CANONICAL_HEADERS_RAW]
PRETTY_MAP = {normalize(k): k for k in CANONICAL_HEADERS_RAW}

# Candidate name columns (normalized)
FIRSTNAME_CANDIDATES = ["first_name", "firstname", "first", "given_name"]
LASTNAME_CANDIDATES  = ["last_name", "lastname", "last", "surname"]

# ---------------------------
# MongoDB (use st.secrets in Streamlit Cloud)
# ---------------------------
def get_mongo_client():
    try:
        uri = st.secrets["MONGO_URI"]
        return MongoClient(uri)
    except Exception:
        st.error("MongoDB connection unavailable. Configure MONGO_URI in Streamlit secrets.")
        st.stop()

# ---------------------------
# Utilities: File Reading
# ---------------------------
def read_csv_with_fallback(fobj):
    try:
        return pd.read_csv(fobj, dtype=str, keep_default_na=False, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(fobj, dtype=str, keep_default_na=False, encoding="latin1")
        except Exception:
            return pd.read_csv(fobj, dtype=str, keep_default_na=False, encoding="ISO-8859-1", engine="python", errors="replace")

def read_file(uploaded):
    # support passing a local path string for quick testing
    if isinstance(uploaded, str):
        if uploaded.lower().endswith('.csv'):
            with open(uploaded, 'rb') as f:
                return read_csv_with_fallback(f)
        else:
            return pd.read_excel(uploaded, dtype=str)

    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return read_csv_with_fallback(uploaded)
    else:
        return pd.read_excel(uploaded, dtype=str)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize(c) for c in df.columns]
    df.replace({"": pd.NA, " ": pd.NA}, inplace=True)
    return df

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf.getvalue()

# ---------------------------
# Utilities: Type Detection
# ---------------------------
def detect_type(series: pd.Series) -> str:
    values = series.dropna().astype(str).str.strip()
    if len(values) == 0:
        return "empty"
    tot = len(values)
    ints = floats = bools = dates = 0
    for v in values:
        vl = v.lower()
        if vl in ("true","false","yes","no","0","1"):
            bools += 1; continue
        if v.isdigit():
            ints += 1; continue
        try:
            float(v); floats += 1; continue
        except: 
            pass
        try:
            parse(v, fuzzy=True); dates += 1; continue
        except: 
            pass
    if dates/tot > 0.6: return "date"
    if ints/tot > 0.6: return "integer"
    if floats/tot > 0.6: return "float"
    if bools/tot > 0.6: return "boolean"
    return "string"

# ---------------------------
# Utilities: Fuzzy / AI matching
# ---------------------------
def fuzzy_match(col, candidates):
    import difflib
    best, score = None, 0.0
    for c in candidates:
        s = difflib.SequenceMatcher(None, col, c).ratio()
        if s > score:
            best, score = c, s
    return best, score

def ai_map_columns(detected_cols, canonical_cols):
    if not AI_AVAILABLE or MODEL is None:
        return {c: fuzzy_match(c, canonical_cols) for c in detected_cols}
    src_emb = MODEL.encode(detected_cols, convert_to_tensor=True)
    tgt_emb = MODEL.encode(canonical_cols, convert_to_tensor=True)
    from sentence_transformers import util as st_util
    sims = st_util.cos_sim(src_emb, tgt_emb).cpu().numpy()
    mapping = {}
    for i, c in enumerate(detected_cols):
        j = int(np.argmax(sims[i]))
        mapping[c] = (canonical_cols[j], float(sims[i, j]))
    return mapping

# ---------------------------
# Mongo Preparation
# ---------------------------
def prepare_for_mongo(df: pd.DataFrame):
    df2 = df.where(pd.notnull(df), None)
    out = []
    for _, row in df2.iterrows():
        rec = {}
        for k, v in row.items():
            if v is None:
                rec[k] = None
            elif isinstance(v, (np.integer, np.int64)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                rec[k] = float(v)
            else:
                rec[k] = str(v)
        out.append(rec)
    return out

# ---------------------------
# Email Pattern Helpers
# ---------------------------
def normalize_domain(raw: Optional[str]) -> Optional[str]:
    """Extract and normalize domain from a URL-like string."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip().lower()
    if not s:
        return None

    # Remove protocol
    if "://" in s:
        s = s.split("://", 1)[1]

    # Remove path / query / fragment
    for sep in ["/", "?", "#"]:
        if sep in s:
            s = s.split(sep, 1)[0]

    # Strip leading 'www.'
    if s.startswith("www."):
        s = s[4:]

    s = s.strip().strip("/")
    if not s or "." not in s:
        return None
    return s

def load_email_patterns_from_db(path: str) -> Dict[str, str]:
    """
    Load {domain -> email_pattern} from a SQLite .db file.
    Uses the first non-system table it finds.
    """
    patterns: Dict[str, str] = {}
    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()

        # Find first user table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1;")
        row = cur.fetchone()
        if not row:
            st.warning("Email pattern DB has no user tables. Skipping email generation.")
            conn.close()
            return patterns

        table_name = row[0]

        # Expect columns: domain_name, email_pattern
        try:
            cur.execute(f"SELECT domain_name, email_pattern FROM {table_name}")
        except Exception as e:
            st.warning(f"Email pattern table missing required columns (domain_name, email_pattern): {e}")
            conn.close()
            return patterns

        for domain_name, email_pattern in cur.fetchall():
            dom_key = normalize_domain(domain_name)
            if not dom_key:
                continue
            # Keep first pattern for a domain; or overwrite as needed
            if dom_key not in patterns and email_pattern:
                patterns[dom_key] = str(email_pattern).strip()

        conn.close()
    except FileNotFoundError:
        st.info("email_patterns.db not found. Skipping email auto-generation.")
    except Exception as e:
        st.warning(f"Could not load email pattern DB: {e}")
    return patterns

def get_email_pattern_cache() -> Dict[str, str]:
    """Cache patterns in session_state to avoid reloading DB."""
    if "email_pattern_cache" not in st.session_state:
        st.session_state.email_pattern_cache = load_email_patterns_from_db(PATTERN_DB_PATH)
    return st.session_state.email_pattern_cache

def find_name_in_row(row: pd.Series, candidates) -> str:
    """Return first non-empty candidate column from row."""
    for cand in candidates:
        if cand in row.index:
            val = row[cand]
            if pd.notna(val) and str(val).strip():
                return str(val).strip()
    return ""

def can_generate_from_pattern(pattern: str, first_full: str, last_full: str) -> bool:
    """Check if required pieces exist for this pattern."""
    if "{firstname" in pattern and not first_full:
        return False
    if "{lastname" in pattern and not last_full:
        return False
    return True

def generate_email_from_pattern(pattern: str, first: str, last: str, domain: str) -> Optional[str]:
    """
    Generate email from pattern using:
    {firstname_full}, {firstname_initial}, {lastname_full}, {lastname_initial}, {domain}
    """
    if not domain:
        return None

    first_full = first.strip().lower() if first else ""
    last_full = last.strip().lower() if last else ""
    first_initial = first_full[0] if first_full else ""
    last_initial = last_full[0] if last_full else ""

    if not can_generate_from_pattern(pattern, first_full, last_full):
        return None

    fmt_kwargs = {
        "firstname_full": first_full,
        "firstname_initial": first_initial,
        "lastname_full": last_full,
        "lastname_initial": last_initial,
        "domain": domain,
    }

    try:
        email = pattern.format(**fmt_kwargs)
    except Exception:
        return None

    email = email.strip()
    if "@" not in email:
        return None
    return email

def apply_email_patterns_to_df(df: pd.DataFrame, patterns: Dict[str, str]) -> pd.DataFrame:
    """
    For each row in df:
      - If email_address is empty
      - And company_website gives a known domain in patterns
      - Generate email using stored pattern & first/last name.
    """
    if not patterns:
        return df

    if "company_website" not in df.columns or "email_address" not in df.columns:
        return df

    df = df.copy()

    def gen_email(row: pd.Series):
        # If email already present, keep it
        existing = row.get("email_address")
        if pd.notna(existing) and str(existing).strip():
            return existing

        website = row.get("company_website")
        domain = normalize_domain(website)
        if not domain:
            return existing

        pattern = patterns.get(domain)
        if not pattern:
            return existing

        first = find_name_in_row(row, FIRSTNAME_CANDIDATES)
        last = find_name_in_row(row, LASTNAME_CANDIDATES)

        new_email = generate_email_from_pattern(pattern, first, last, domain)
        return new_email if new_email else existing

    df["email_address"] = df.apply(gen_email, axis=1)
    return df

# ---------------------------
# UI: dark theme + layout (compact)
# ---------------------------
st.set_page_config(page_title="Collator (Deploy-Ready)", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#07101c,#071427); color: #e6eef8; }
    .topbar { position: sticky; top: 0; padding: 12px; background: rgba(5,10,20,0.9); border-bottom: 1px solid rgba(255,255,255,0.03); z-index:999; }
    .brand { font-weight:700; color:#cfe6ff; font-size:18px; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:14px; border-radius:10px; border:1px solid rgba(255,255,255,0.03); margin-bottom:12px; }
    .map-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap:12px; margin-top:12px; }
    .map-tag { background:#0f1a2b; padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.03); }
    .map-title { color:#dbeeff; font-weight:600; }
    .map-selected { margin-top:8px; display:inline-block; background:#162235; padding:6px 10px; border-radius:8px; color:#9fc7ff; border:1px solid rgba(255,255,255,0.03); cursor:pointer; }
    .map-score { color:#9fb8ff; font-size:12px; margin-top:6px; }
    .stepper { display:flex; gap:10px; margin-bottom:12px; }
    .step { flex:1; padding:10px; border-radius:8px; background:#081223; color:#a7c8ff; text-align:center; font-weight:600; border:1px solid rgba(255,255,255,0.02); }
    .muted { color:#9fb8ff; font-size:13px; }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<div class='topbar'><div class='brand'>Merge & Store Manager</div></div>", unsafe_allow_html=True)

# ---------------------------
# Upload Step
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 1) Upload CSV / Excel files")
uploaded_files = st.file_uploader(
    "Upload CSV or XLSX files (multiple)", 
    accept_multiple_files=True, 
    type=["csv", "xlsx"]
)
st.markdown("</div>", unsafe_allow_html=True)

# Allow quick testing with SAMPLE_FILE_PATH if no interactive upload provided
if not uploaded_files:
    st.info("No files uploaded — using sample file for local testing if present.")
    try:
        # SAMPLE_FILE_PATH should be defined if you want this behavior
        _df_test = read_file(SAMPLE_FILE_PATH)  # noqa: F821 (user-defined if needed)
        st.info(f"Loaded sample file: {SAMPLE_FILE_PATH}")
        uploaded_files = [SAMPLE_FILE_PATH]
    except Exception:
        st.stop()

# Read files robustly and collect detected columns
dfs = []
detected_all = []
read_errors = []
for f in uploaded_files:
    try:
        df = read_file(f)
        df = clean_columns(df)
        # If duplicate columns in the file, rename duplicates to keep unique internal names
        if df.columns.duplicated().any():
            new_cols = []
            seen = {}
            for c in df.columns:
                if c not in seen:
                    seen[c] = 1
                    new_cols.append(c)
                else:
                    seen[c] += 1
                    new_cols.append(f"{c}__dup{seen[c]}")
            df.columns = new_cols
        dfs.append(df)
        st.write(
            f"Loaded **{f if isinstance(f,str) else f.name}** — "
            f"rows: {df.shape[0]}, columns: {len(df.columns)}"
        )
        detected_all.extend(df.columns)
    except Exception as e:
        read_errors.append((f if isinstance(f,str) else f.name, str(e)))
        st.error(f"Failed to read {f if isinstance(f,str) else f.name}: {e}")

if read_errors:
    st.warning("Some files failed to load — see messages above.")

detected_unique = sorted(set(detected_all))
if len(detected_unique) == 0:
    st.error("No columns detected in uploaded files.")
    st.stop()

# ---------------------------
# Mapping Step (Tag-Based) — fixed persistence
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 2) Tag-based mapping — quick and persistent")

# Compute AI/fuzzy suggestions
suggested_map = ai_map_columns(detected_unique, CANONICAL_HEADERS)

# Initialize mapping in session state if not present; preserve existing user choices
if "mapping" not in st.session_state:
    st.session_state.mapping = {}

# Add new columns (or update scores) without overwriting user's mapped choices
for col in detected_unique:
    tgt, score = suggested_map.get(col, (None, 0.0))
    if col not in st.session_state.mapping:
        initial = tgt if (tgt and score >= 0.50) else "--ignore--"
        st.session_state.mapping[col] = {"mapped": initial, "score": float(score)}
    else:
        # update score and ensure entry has mapped key
        st.session_state.mapping[col]["score"] = float(score)
        if "mapped" not in st.session_state.mapping[col]:
            st.session_state.mapping[col]["mapped"] = tgt if (tgt and score >= 0.50) else "--ignore--"

# Remove stale columns
for col in list(st.session_state.mapping.keys()):
    if col not in detected_unique:
        del st.session_state.mapping[col]

threshold = st.slider(
    "AI similarity threshold", 
    min_value=0.0, 
    max_value=0.95, 
    value=0.55, 
    step=0.01
)

# helper to update mapping from selectbox
def update_mapping(col):
    st.session_state.mapping[col]["mapped"] = st.session_state.get(f"sel__{col}")

# Render tag grid
st.markdown("<div class='map-grid'>", unsafe_allow_html=True)
for col in detected_unique:
    info = st.session_state.mapping.get(col, {"mapped": "--ignore--", "score": 0.0})
    score = info["score"]
    ai_tgt, ai_score = suggested_map.get(col, (None, 0.0))
    current_mapped = st.session_state.mapping[col]["mapped"]
    display_label = PRETTY_MAP.get(current_mapped, current_mapped)
    icon = "✓" if current_mapped != "--ignore--" else "⚠"

    st.markdown(f"""
        <div class='map-tag'>
            <div class='map-title'>{col} 
                <span style='float:right;color:{"#4beb9b" if icon=="✓" else "#ffd36b"}; font-weight:700'>{icon}</span>
            </div>
            <div class='muted'>Suggested: {PRETTY_MAP.get(ai_tgt, ai_tgt) if ai_tgt else '--'}</div>
            <div class='map-selected' onclick="document.getElementById('sel__{col}').click();">
                {display_label}
            </div>
            <div class='map-score'>AI score: {score:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

    st.selectbox(
        f"select_map_{col}",
        ["--ignore--"] + CANONICAL_HEADERS,
        index=(0 if current_mapped == "--ignore--" else (CANONICAL_HEADERS.index(current_mapped) + 1)),
        key=f"sel__{col}",
        label_visibility="collapsed",
        on_change=update_mapping,
        args=(col,)
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Merge Step
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 3) Merge into canonical schema")

if st.button("Merge Now"):
    with st.spinner("Merging files..."):
        final_frames = []
        problems = []

        # Load email patterns once for the whole merge operation
        local_db = ensure_local_pattern_db()
        email_patterns = get_email_pattern_cache() if local_db else {}

        for df in dfs:
            dfw = df.copy()
            # Build rename map for this df based on mapping
            rename_map = {}
            for old_col in list(dfw.columns):
                mapped = st.session_state.mapping.get(old_col, {"mapped": "--ignore--"})["mapped"]
                if mapped != "--ignore--":
                    # collision handling: if canonical already exists in this df, write to a temp name
                    if mapped in dfw.columns:
                        rename_map[old_col] = f"{mapped}__from__{old_col}"
                    else:
                        rename_map[old_col] = mapped
            if rename_map:
                dfw.rename(columns=rename_map, inplace=True)

            # Keep only canonical columns and temp variants
            cols_keep = [
                c for c in dfw.columns 
                if (c in CANONICAL_HEADERS) or any(c.startswith(t + "__from__") for t in CANONICAL_HEADERS)
            ]
            dfw = dfw[cols_keep].copy()

            # Collapse temp variants: prefer left-to-right first non-null
            for canonical in CANONICAL_HEADERS:
                temp_cols = [c for c in dfw.columns if c == canonical or c.startswith(canonical + "__from__")]
                if len(temp_cols) > 1:
                    dfw[canonical] = dfw[temp_cols].bfill(axis=1).iloc[:, 0]
                    for c in temp_cols:
                        if c != canonical:
                            dfw.drop(columns=[c], inplace=True)
                elif len(temp_cols) == 1 and temp_cols[0] != canonical:
                    dfw.rename(columns={temp_cols[0]: canonical}, inplace=True)

            # Ensure all canonical columns present
            for c in CANONICAL_HEADERS:
                if c not in dfw.columns:
                    dfw[c] = pd.NA

            # Reorder to canonical schema and drop duplicates
            dfw = dfw[CANONICAL_HEADERS]
            dfw = dfw.loc[:, ~dfw.columns.duplicated()]

            if dfw.columns.duplicated().any():
                problems.append("Duplicate columns remain after processing a file.")

            # 🔹 NEW STEP: apply email patterns here
            dfw = apply_email_patterns_to_df(dfw, email_patterns)

            final_frames.append(dfw)

        if not final_frames:
            st.error("No frames to merge. Check mappings.")
        else:
            try:
                merged = pd.concat(final_frames, ignore_index=True, sort=False)
            except Exception as e:
                st.error(f"Merge failed: {e}")
                st.stop()

            st.session_state.merged_df = merged
            st.success(f"Merged {len(final_frames)} files → {merged.shape[0]} rows × {merged.shape[1]} cols")
            st.dataframe(merged.rename(columns=PRETTY_MAP).head(200))

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Preview & Schema Report
# ---------------------------
if "merged_df" in st.session_state:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Preview")
    st.dataframe(st.session_state.merged_df.rename(columns=PRETTY_MAP).head(200))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Schema Report")
    schema = pd.DataFrame({
        "Field": CANONICAL_HEADERS_RAW,
        "internal": CANONICAL_HEADERS,
        "Detected Type": [detect_type(st.session_state.merged_df[c]) for c in CANONICAL_HEADERS],
        "% Empty": [st.session_state.merged_df[c].isna().mean()*100 for c in CANONICAL_HEADERS]
    })
    st.dataframe(schema)
    st.download_button("Download schema_report.xlsx", data=to_excel_bytes(schema), file_name="schema_report.xlsx")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Upload to MongoDB (safe)
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 4) Upload to Database")

if "merged_df" not in st.session_state:
    st.info("Merge first to enable upload.")
else:
    st.markdown("<div style='display:flex;gap:12px;align-items:center'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='muted'>DB:</div>"
        f"<div style='font-weight:700'>"
        f"{st.secrets.get('DB_NAME','<set DB_NAME>')}."
        f"{st.secrets.get('COLLECTION_NAME','<set COLLECTION_NAME>')}"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Upload to MongoDB"):
        with st.spinner("Uploading to MongoDB..."):
            try:
                client = get_mongo_client()
                db = client[st.secrets["DB_NAME"]]
                coll = db[st.secrets["COLLECTION_NAME"]]
                records = prepare_for_mongo(st.session_state.merged_df)
                if len(records) == 0:
                    st.warning("No records to insert.")
                else:
                    res = coll.insert_many(records)
                    st.success(f"Uploaded {len(res.inserted_ids)} documents to MongoDB.")
            except Exception as e:
                st.error(f"Upload failed: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Downloads
# ---------------------------
if "merged_df" in st.session_state:
    st.download_button(
        "Download merged_output.xlsx", 
        data=to_excel_bytes(st.session_state.merged_df.rename(columns=PRETTY_MAP)), 
        file_name="merged_output.xlsx"
    )

# End of app





