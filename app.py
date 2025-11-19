import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pymongo import MongoClient
from dateutil.parser import parse

# Try AI for auto column matching
try:
    from sentence_transformers import SentenceTransformer, util
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    AI_AVAILABLE = True
except:
    MODEL = None
    AI_AVAILABLE = False

# ---------------------------
# Canonical headers (from user)
# ---------------------------
CANONICAL_HEADERS_RAW = [
    "First Name", "Last Name", "Job Title", "Seniority" ,"Company Name",
    "Email Address", "Status", "Phone Number", "Employee Size",
    "Industry", "Prospect Linkedin", "Company Website", "Company Linkedin",
    "City", "State", "Country", "Company Address",
    "Company City", "Company State", "Postal Code", "Company Country", "SIC", "NAICS"
]

def normalize(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

CANONICAL_HEADERS = [normalize(h) for h in CANONICAL_HEADERS_RAW]
PRETTY_MAP = {normalize(k): k for k in CANONICAL_HEADERS_RAW}

# ---------------------------
# Mongo Settings
# ---------------------------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "master_data"
COLLECTION_NAME = "records"

# ---------------------------
# Utils
# ---------------------------

def read_file(uploaded):
    if f.name.lower().endswith(".csv"):
        # Try utf-8 first, fallback automatically
        try:
            return pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return pd.read_csv(f, dtype=str, keep_default_na=False, encoding="latin1")
            except:
                return pd.read_csv(f, dtype=str, keep_default_na=False, encoding="ISO-8859-1")
    else:
        return pd.read_excel(f, dtype=str)


def clean_columns(df):
    df = df.copy()
    df.columns = [normalize(c) for c in df.columns]
    df.replace({"": pd.NA, " ": pd.NA}, inplace=True)
    return df

def to_excel_bytes(df):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

def detect_type(series):
    values = series.dropna().astype(str).str.strip()
    if len(values) == 0:
        return "empty"
    tot = len(values)
    ints = floats = bools = dates = 0
    for v in values:
        v_low = v.lower()
        if v_low in ("true", "false", "yes", "no", "0", "1"):
            bools += 1
            continue
        if v.isdigit():
            ints += 1
            continue
        try:
            float(v)
            floats += 1
            continue
        except:
            pass
        try:
            parse(v, fuzzy=True)
            dates += 1
            continue
        except:
            pass
    if dates / tot > 0.6: return "date"
    if ints / tot > 0.6: return "integer"
    if floats / tot > 0.6: return "float"
    if bools / tot > 0.6: return "boolean"
    return "string"

# AI or fuzzy mapping
def fuzzy_match(c, candidates):
    import difflib
    best, score = None, 0
    for tgt in candidates:
        s = difflib.SequenceMatcher(None, c, tgt).ratio()
        if s > score:
            best, score = tgt, s
    return best, score

def ai_map_columns(detected, canonical):
    if not AI_AVAILABLE:
        return {c: fuzzy_match(c, canonical) for c in detected}
    src = MODEL.encode(detected, convert_to_tensor=True)
    tgt = MODEL.encode(canonical, convert_to_tensor=True)
    sims = util.cos_sim(src, tgt).cpu().numpy()
    mapping = {}
    for i, col in enumerate(detected):
        j = int(np.argmax(sims[i]))
        mapping[col] = (canonical[j], float(sims[i, j]))
    return mapping

def prepare_for_mongo(df):
    df = df.where(pd.notnull(df), None)
    cleaned = []
    for _, row in df.iterrows():
        d = {}
        for k, v in row.items():
            if isinstance(v, (np.int64, np.int32)): v = int(v)
            elif isinstance(v, (np.float64, np.float32)): v = float(v)
            elif v is not None: v = str(v)
            d[k] = v
        cleaned.append(d)
    return cleaned


# ---------------------------
# Modern UI
# ---------------------------
st.set_page_config(page_title="Modern Collator", layout="wide")

# Sticky Top Navigation
st.markdown("""
<style>
.bento-box {
    padding: 18px;
    border-radius: 14px;
    background: rgba(10, 10, 10, 0.05);
    margin-bottom: 20px;
    border: 1px solid #eee;
}
.header {
    position: sticky;
    top: 0;
    background: white;
    padding: 14px;
    z-index: 999;
    border-bottom: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'><h2>⚡ Modern AI Collator + MongoDB</h2></div>", unsafe_allow_html=True)

# Show canonical schema in a bento box
#with st.container():
 #   st.markdown("<div class='bento-box'>", unsafe_allow_html=True)
  #  st.markdown("### 📘 Canonical Schema")
   # st.dataframe(pd.DataFrame({"Final Fields": CANONICAL_HEADERS_RAW}))
    #st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# Upload Section
# ---------------------------
with st.container():
    st.markdown("<div class='bento-box'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload CSV / Excel Files")
    files = st.file_uploader("Upload multiple files", type=["csv", "xlsx"], accept_multiple_files=True)
    st.markdown("</div>", unsafe_allow_html=True)

if not files:
    st.stop()

# ---------------------------
# Load files
# ---------------------------
cleaned_dfs = []
detected_cols_all = []

for f in files:
    df = read_file(f)
    df = clean_columns(df)
    cleaned_dfs.append(df)
    detected_cols_all.extend(df.columns)

detected_unique = sorted(set(detected_cols_all))

# ---------------------------
# Mapping Section
# ---------------------------
with st.container():
    st.markdown("<div class='bento-box'>", unsafe_allow_html=True)
    st.markdown("### 🤖 AI Column Mapping")
    
    threshold = st.slider("Similarity Threshold", 0.0, 0.95, 0.50, 0.01)

    suggested = ai_map_columns(detected_unique, CANONICAL_HEADERS)

    mapping = {}
    for col in detected_unique:
        target, score = suggested[col]
        if score < threshold:
            target = "--ignore--"
        mapping[col] = st.selectbox(
            f"Map '{col}' →",
            ["--ignore--"] + CANONICAL_HEADERS,
            index=(0 if target == "--ignore--" else CANONICAL_HEADERS.index(target) + 1)
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Merge Section
# ---------------------------
with st.container():
    st.markdown("<div class='bento-box'>", unsafe_allow_html=True)
    st.markdown("### 🔄 Merge Files")

    if st.button("Merge Files"):
        final_frames = []

        for df in cleaned_dfs:
            dfw = df.copy()
            # rename
            rename_map = {}
            for old in dfw.columns:
                new = mapping.get(old, "--ignore--")
                if new != "--ignore--":
                    rename_map[old] = new
            dfw.rename(columns=rename_map, inplace=True)

            # keep only canonical fields
            dfw = dfw[[c for c in dfw.columns if c in CANONICAL_HEADERS]]

            # ensure all canonical fields exist
            for c in CANONICAL_HEADERS:
                if c not in dfw.columns:
                    dfw[c] = pd.NA

            dfw = dfw[CANONICAL_HEADERS]
            dfw = dfw.loc[:, ~dfw.columns.duplicated()]

            final_frames.append(dfw)

        merged = pd.concat(final_frames, ignore_index=True)
        st.session_state["merged_df"] = merged
        
        st.success(f"Merged {len(files)} file(s) → {merged.shape[0]} rows.")
        st.dataframe(merged.rename(columns=PRETTY_MAP).head(200))

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# Mongo Upload (GLOBAL BUTTON)
# ---------------------------
with st.container():
    st.markdown("<div class='bento-box'>", unsafe_allow_html=True)
    st.markdown("### 🗄 Upload to MongoDB")

    if "merged_df" not in st.session_state:
        st.info("Merge files first.")
    else:
        merged_df = st.session_state["merged_df"]

        if st.button("📥 Upload to MongoDB"):
            try:
                records = prepare_for_mongo(merged_df)
                client = MongoClient(MONGO_URI)
                coll = client[DB_NAME][COLLECTION_NAME]
                res = coll.insert_many(records)

                st.success(f"Uploaded {len(res.inserted_ids)} records to MongoDB.")
                st.toast("Upload Successful!", icon="✔️")
            except Exception as e:
                st.error(f"Upload failed: {e}")
                st.toast("Upload Failed!", icon="❌")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# Download Outputs
# ---------------------------
with st.container():
    st.markdown("<div class='bento-box'>", unsafe_allow_html=True)
    st.markdown("### ⬇ Download Outputs")

    if "merged_df" in st.session_state:
        merged_pretty = st.session_state["merged_df"].rename(columns=PRETTY_MAP)
        
        st.download_button(
            "Download Merged Excel",
            data=to_excel_bytes(merged_pretty),
            file_name="merged_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        schema_df = pd.DataFrame({
            "Field": CANONICAL_HEADERS_RAW,
            "Type": [detect_type(st.session_state["merged_df"][normalize(h)]) for h in CANONICAL_HEADERS_RAW],
            "% Empty": [st.session_state["merged_df"][normalize(h)].isna().mean() * 100 for h in CANONICAL_HEADERS_RAW]
        })
        st.download_button(
            "Download Schema Report",
            data=to_excel_bytes(schema_df),
            file_name="schema_report.xlsx"
        )

    st.markdown("</div>", unsafe_allow_html=True)
