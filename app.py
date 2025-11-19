# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pymongo import MongoClient
from dateutil.parser import parse

# ---------------------------
# Configuration & Model
# ---------------------------

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
    "First Name", "Last Name", "Job Title", "Company Name", "Seniority",
    "Email Address", "Status", "Phone Number", "Employee Size",
    "Industry", "Prospect Linkedin", "Company Website", "Company Linkedin",
    "City", "State", "Country", "Company Address",
    "Company City", "Company State", "Postal Code", "Company Country", "SIC", "NAICS",
]

def normalize(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")

CANONICAL_HEADERS = [normalize(h) for h in CANONICAL_HEADERS_RAW]
PRETTY_MAP = {normalize(k): k for k in CANONICAL_HEADERS_RAW}

# ---------------------------
# MongoDB (use st.secrets in Streamlit Cloud)
# ---------------------------
def get_mongo_client():
    # Requires MONGO_URI in st.secrets on Streamlit Cloud
    # Example keys in Secrets: MONGO_URI, DB_NAME, COLLECTION_NAME
    try:
        uri = st.secrets["MONGO_URI"]
        return MongoClient(uri)
    except Exception as e:
        st.error("MongoDB connection unavailable. Configure MONGO_URI in Streamlit secrets.")
        st.stop()

# ---------------------------
# Utilities
# ---------------------------
def read_csv_with_fallback(fobj):
    # Try common encodings
    try:
        return pd.read_csv(fobj, dtype=str, keep_default_na=False, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(fobj, dtype=str, keep_default_na=False, encoding="latin1")
        except Exception:
            return pd.read_csv(fobj, dtype=str, keep_default_na=False, encoding="ISO-8859-1", engine="python", errors="replace")

def read_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return read_csv_with_fallback(uploaded)
    else:
        # Excel
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
        except: pass
        try:
            parse(v, fuzzy=True); dates += 1; continue
        except: pass
    if dates/tot > 0.6: return "date"
    if ints/tot > 0.6: return "integer"
    if floats/tot > 0.6: return "float"
    if bools/tot > 0.6: return "boolean"
    return "string"

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
    sims = util.cos_sim(src_emb, tgt_emb).cpu().numpy()
    mapping = {}
    for i, c in enumerate(detected_cols):
        j = int(np.argmax(sims[i]))
        mapping[c] = (canonical_cols[j], float(sims[i, j]))
    return mapping

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
uploaded_files = st.file_uploader("Upload CSV or XLSX files (multiple)", accept_multiple_files=True, type=["csv", "xlsx"])
st.markdown("</div>", unsafe_allow_html=True)

if not uploaded_files:
    st.info("Upload one or more files to start.")
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
        detected_all.extend(df.columns)
        st.write(f"Loaded **{f.name}** — rows: {df.shape[0]}, columns: {len(df.columns)}")
    except Exception as e:
        read_errors.append((f.name, str(e)))
        st.error(f"Failed to read {f.name}: {e}")

if read_errors:
    st.warning("Some files failed to load — see messages above.")

detected_unique = sorted(set(detected_all))
if len(detected_unique) == 0:
    st.error("No columns detected in uploaded files.")
    st.stop()

# ---------------------------
# Mapping Initialization (Persist user mapping)
# ---------------------------

if "mapping" not in st.session_state:
    st.session_state.mapping = {}
    for col in detected_unique:
        tgt, score = suggested_map.get(col, ("--ignore--", 0.0))
        initial = tgt if (tgt and score >= 0.55) else "--ignore--"
        st.session_state.mapping[col] = {
            "mapped": initial,
            "score": float(score)
        }
else:
    # Ensure new columns are added if new file uploaded
    for col in detected_unique:
        if col not in st.session_state.mapping:
            tgt, score = suggested_map.get(col, ("--ignore--", 0.0))
            initial = tgt if (tgt and score >= 0.55) else "--ignore--"
            st.session_state.mapping[col] = {
                "mapped": initial,
                "score": float(score)
            }

# Remove stale columns (columns that no longer exist)
for col in list(st.session_state.mapping.keys()):
    if col not in detected_unique:
        del st.session_state.mapping[col]

# Render tag grid
st.markdown("<div class='map-grid'>", unsafe_allow_html=True)
for col in detected_unique:
    info = st.session_state.mapping.get(col, {"mapped": "--ignore--", "score": 0.0})
    score = info["score"]
    ai_tgt, ai_score = suggested_map.get(col, (None, 0.0))
    # determine display value: current mapping in session_state (user editable)
    current_mapped = st.session_state.mapping[col]["mapped"]
    display_label = PRETTY_MAP.get(current_mapped, current_mapped)

    icon = "✓" if current_mapped != "--ignore--" else "⚠"

    # Tag card
    st.markdown(f"""
        <div class='map-tag'>
            <div class='map-title'>{col} <span style='float:right;color:{"#4beb9b" if icon=="✓" else "#ffd36b"}; font-weight:700'>{icon}</span></div>
            <div class='muted'>Suggested: {PRETTY_MAP.get(ai_tgt, ai_tgt) if ai_tgt else '--'}</div>
            <div class='map-selected' onclick="document.getElementById('sel__{col}').click();">{display_label}</div>
            <div class='map-score'>AI score: {score:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

    def update_mapping(col):
    st.session_state.mapping[col]["mapped"] = st.session_state[f"sel__{col}"]

    st.selectbox(
        f"select_map_{col}",
        ["--ignore--"] + CANONICAL_HEADERS,
        index=(0 if current_mapped == "--ignore--" else CANONICAL_HEADERS.index(current_mapped) + 1),
        key=f"sel__{col}",
        label_visibility="collapsed",
        on_change=update_mapping,
        args=(col,)
    )

# ---------------------------
# Merge Step
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 3) Merge into canonical schema")

if st.button("Merge Now"):
    with st.spinner("Merging files..."):
        final_frames = []
        problems = []
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
            cols_keep = [c for c in dfw.columns if (c in CANONICAL_HEADERS) or any(c.startswith(t + "__from__") for t in CANONICAL_HEADERS)]
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
st.markdown("### 4) Upload to MongoDB")

if "merged_df" not in st.session_state:
    st.info("Merge first to enable upload.")
else:
    st.markdown("<div style='display:flex;gap:12px;align-items:center'>", unsafe_allow_html=True)
    st.markdown(f"<div class='muted'>DB:</div><div style='font-weight:700'>{st.secrets.get('DB_NAME','<set DB_NAME>')}.{st.secrets.get('COLLECTION_NAME','<set COLLECTION_NAME>')}</div>", unsafe_allow_html=True)
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
# Duplicate Manager (Prospect LinkedIn)
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### Manage Duplicates")

if st.button("Scan Database for duplicates"):
    with st.spinner("Scanning..."):
        try:
            client = get_mongo_client()
            db = client[st.secrets["DB_NAME"]]
            coll = db[st.secrets["COLLECTION_NAME"]]
            pipeline = [
                {"$group": {"_id": "$prospect_linkedin", "count": {"$sum": 1}, "ids": {"$push": "$_id"}, "docs": {"$push": "$$ROOT"}}},
                {"$match": {"_id": {"$ne": None}, "count": {"$gt": 1}}}
            ]
            dup_groups = list(coll.aggregate(pipeline))
            if not dup_groups:
                st.success("No duplicates found by prospect_linkedin.")
                st.session_state.duplicates = []
            else:
                st.warning(f"Found {len(dup_groups)} duplicate groups by prospect_linkedin.")
                st.session_state.duplicates = dup_groups
        except Exception as e:
            st.error(f"Scan failed: {e}")

# Show duplicates if present
if "duplicates" in st.session_state and st.session_state.duplicates:
    for group in st.session_state.duplicates:
        key = group["_id"]
        docs = group["docs"]
        ids = group["ids"]
        st.markdown(f"#### LinkedIn: `{key}` — {len(ids)} records")
        df_dup = pd.DataFrame(docs)
        st.dataframe(df_dup)
        st.download_button(f"Download group {key}", data=to_excel_bytes(df_dup), file_name=f"duplicates_{key}.xlsx")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button(f"Keep FIRST and delete others ({key})"):
                try:
                    client = get_mongo_client()
                    db = client[st.secrets["DB_NAME"]]
                    coll = db[st.secrets["COLLECTION_NAME"]]
                    to_remove = ids[1:]
                    if to_remove:
                        coll.delete_many({"_id": {"$in": to_remove}})
                        st.success(f"Deleted {len(to_remove)} docs (kept first).")
                    else:
                        st.info("Only one doc present.")
                except Exception as e:
                    st.error(f"Delete failed: {e}")
        with c2:
            if st.button(f"Keep LAST and delete others ({key})"):
                try:
                    client = get_mongo_client()
                    db = client[st.secrets["DB_NAME"]]
                    coll = db[st.secrets["COLLECTION_NAME"]]
                    to_remove = ids[:-1]
                    if to_remove:
                        coll.delete_many({"_id": {"$in": to_remove}})
                        st.success(f"Deleted {len(to_remove)} docs (kept last).")
                    else:
                        st.info("Only one doc present.")
                except Exception as e:
                    st.error(f"Delete failed: {e}")
        with c3:
            if st.button(f"Delete ALL ({key})"):
                try:
                    client = get_mongo_client()
                    db = client[st.secrets["DB_NAME"]]
                    coll = db[st.secrets["COLLECTION_NAME"]]
                    coll.delete_many({"_id": {"$in": ids}})
                    st.success(f"Deleted all {len(ids)} docs in group.")
                except Exception as e:
                    st.error(f"Delete failed: {e}")
        st.markdown("---")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Downloads
# ---------------------------
if "merged_df" in st.session_state:
    st.download_button("Download merged_output.xlsx", data=to_excel_bytes(st.session_state.merged_df.rename(columns=PRETTY_MAP)), file_name="merged_output.xlsx")

# End of app




