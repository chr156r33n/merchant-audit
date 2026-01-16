import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
import re
from pathlib import Path
import google.generativeai as genai
import time

# -------------------------------------------------
# App setup
# -------------------------------------------------
st.set_page_config(page_title="Merchant XML Field Utilization", layout="wide")
st.title("Merchant XML Field Utilization PoC")
st.caption("Validates a merchant XML feed against a canonical Google Merchant field list.")

# -------------------------------------------------
# Load canonical field list and prompts
# -------------------------------------------------
FIELDS_FILE = Path("fields.txt")
SYSTEM_PROMPT_FILE = Path("enrichment_system_prompt.txt")
PROMPT_FILE = Path("enrichment_prompt.txt")

if not FIELDS_FILE.exists():
    st.error("fields.txt not found in repo root.")
    st.stop()

if not SYSTEM_PROMPT_FILE.exists():
    st.error("enrichment_system_prompt.txt not found in repo root.")
    st.stop()

if not PROMPT_FILE.exists():
    st.error("enrichment_prompt.txt not found in repo root.")
    st.stop()

canonical_fields = [
    line.strip()
    for line in FIELDS_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]

canonical_set = set(canonical_fields)

# Load prompts
system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

# -------------------------------------------------
# Helpers
# -------------------------------------------------
_whitespace = re.compile(r"\s+")

def clean_text(v):
    if v is None:
        return ""
    return _whitespace.sub(" ", str(v)).strip()

def build_ns_reverse_map(root):
    # Pre-seed Google Merchant namespace
    ns = {"http://base.google.com/ns/1.0": "g"}
    for k, v in root.attrib.items():
        if k.startswith("{http://www.w3.org/2000/xmlns/}"):
            ns[v] = k.split("}", 1)[1]
    return ns

def tag_name(tag, ns_map):
    if tag.startswith("{"):
        uri, name = tag[1:].split("}", 1)
        prefix = ns_map.get(uri)
        return f"{prefix}:{name}" if prefix else name
    return tag

def flatten_item(item, ns_map):
    out = defaultdict(list)

    def walk(node, prefix=""):
        children = list(node)
        name = tag_name(node.tag, ns_map)

        if children:
            for c in children:
                child_name = tag_name(c.tag, ns_map)
                new_prefix = f"{prefix}.{child_name}" if prefix else child_name
                walk(c, new_prefix)
        else:
            field = prefix if prefix else name
            out[field].append(clean_text(node.text))

    for child in list(item):
        walk(child, tag_name(child.tag, ns_map))

    return {k: " | ".join(set(v)) for k, v in out.items()}

def find_products(root):
    items = root.findall(".//item")
    if items:
        return items
    entries = root.findall(".//entry")
    if entries:
        return entries
    return []

def pick_product_id(row):
    for k in (
        "g:id", "id",
        "g:gtin", "gtin",
        "g:mpn", "mpn",
        "g:title", "title",
        "link", "g:link",
    ):
        if k in row and row[k]:
            return row[k][:120]
    return "(no identifier)"

def get_gemini_model_name(version, model_type):
    """Get Gemini model name based on version and type selection"""
    if version == "2.5":
        if model_type == "pro":
            return "gemini-2.0-flash-exp"  # Using experimental model for 2.5 pro
        else:  # fast
            return "gemini-2.0-flash-exp"  # Using experimental model for 2.5 fast
    else:  # version 3
        if model_type == "pro":
            return "gemini-1.5-pro"
        else:  # fast
            return "gemini-1.5-flash"

def enrich_product_highlight(row, model, prompt_template):
    """Enrich a single product's highlight using Gemini"""
    try:
        # Get field values
        title = row.get("g:title", "")
        description = row.get("g:description", "")
        short = row.get("g:short", "")
        material = row.get("g:material", "")
        color = row.get("g:color", "")
        sort_title = row.get("g:sort_title", "")
        product_type = row.get("g:product_type", "")
        current_highlight = row.get("g:product_highlight", "")
        
        # Format prompt
        prompt = prompt_template.format(
            g_title=title or "Not provided",
            g_description=description or "Not provided",
            g_short=short or "Not provided",
            g_material=material or "Not provided",
            g_color=color or "Not provided",
            g_sort_title=sort_title or "Not provided",
            g_product_type=product_type or "Not provided",
            g_product_highlight=current_highlight or "None"
        )
        
        # Generate response
        response = model.generate_content(prompt)
        highlight = clean_text(response.text)
        
        return highlight
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------------------------
# Sidebar - All controls
# -------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    
    uploaded = st.file_uploader("Upload merchant XML feed", type=["xml"])
    max_products = st.number_input(
        "Max products to process",
        min_value=10,
        max_value=200000,
        value=5000,
        step=500
    )
    
    st.divider()
    st.header("Enrichment Settings")
    
    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Enter your Google Gemini API key"
    )
    
    gemini_version = st.selectbox(
        "Gemini Version",
        options=["2.5", "3"],
        index=0
    )
    
    gemini_type = st.selectbox(
        "Model Type",
        options=["fast", "pro"],
        index=0
    )
    
    if gemini_api_key:
        model_name = get_gemini_model_name(gemini_version, gemini_type)
        st.info(f"Using model: {model_name}")
    
    st.divider()
    st.header("Enrichment Actions")
    
    run_enrichment = st.button(
        "Run Enrichment",
        type="primary",
        disabled=not (uploaded and gemini_api_key)
    )

# -------------------------------------------------
# Main content area
# -------------------------------------------------
if not uploaded:
    st.info("Please upload an XML feed using the sidebar.")
    st.stop()

# -------------------------------------------------
# Parse XML
# -------------------------------------------------
try:
    root = ET.fromstring(uploaded.read())
except Exception as e:
    st.error(f"XML parsing failed: {e}")
    st.stop()

ns_map = build_ns_reverse_map(root)
products = find_products(root)[: int(max_products)]

st.write(f"Products parsed: **{len(products)}**")

rows = []
observed_fields = set()

for p in products:
    flat = flatten_item(p, ns_map)
    flat["_product"] = pick_product_id(flat)
    rows.append(flat)
    observed_fields.update(flat.keys())

df = pd.DataFrame(rows)
df.insert(0, "_product", df.pop("_product"))

total = len(df)

# Initialize enrichment results in session state
if "enrichment_results" not in st.session_state:
    st.session_state.enrichment_results = None
if "enrichment_df" not in st.session_state:
    st.session_state.enrichment_df = None

# -------------------------------------------------
# Run Enrichment
# -------------------------------------------------
if run_enrichment and gemini_api_key:
    if "g:product_highlight" not in df.columns:
        df["g:product_highlight"] = ""
    
    # Filter products that have been processed (respect max_products)
    products_to_enrich = df.copy()
    
    st.header("Running Enrichment")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    enrichment_results = []
    model_name = get_gemini_model_name(gemini_version, gemini_type)
    
    # Configure API and create model once
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
    
    for idx, (_, row) in enumerate(products_to_enrich.iterrows()):
        status_text.text(f"Processing product {idx + 1} of {len(products_to_enrich)}")
        progress_bar.progress((idx + 1) / len(products_to_enrich))
        
        suggested_highlight = enrich_product_highlight(
            row, model, prompt_template
        )
        
        enrichment_results.append({
            "g:mpn": row.get("g:mpn", ""),
            "g:gtin": row.get("g:gtin", ""),
            "g:id": row.get("g:id", ""),
            "g:title": row.get("g:title", ""),
            "g:link": row.get("g:link", ""),
            "g:product_highlight": row.get("g:product_highlight", ""),
            "suggested_highlight": suggested_highlight
        })
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    st.session_state.enrichment_results = enrichment_results
    st.session_state.enrichment_df = pd.DataFrame(enrichment_results)
    
    progress_bar.empty()
    status_text.empty()
    st.success(f"Enrichment complete! Processed {len(enrichment_results)} products.")

# -------------------------------------------------
# Display Enrichment Results
# -------------------------------------------------
if st.session_state.enrichment_df is not None:
    st.header("Enrichment Results")
    
    display_df = st.session_state.enrichment_df.copy()
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download button for GMC supplemental feed format
    gmc_df = st.session_state.enrichment_df[["g:id", "suggested_highlight"]].copy()
    gmc_df.rename(columns={"suggested_highlight": "g:product_highlight"}, inplace=True)
    gmc_df = gmc_df[["g:id", "g:product_highlight"]]
    
    csv_output = gmc_df.to_csv(index=False)
    st.download_button(
        "Download GMC Supplemental Feed CSV",
        csv_output,
        "gmc_product_highlights.csv",
        "text/csv",
        help="This CSV is formatted for Google Merchant Center supplemental feed upload"
    )

# -------------------------------------------------
# Canonical utilization summary
# -------------------------------------------------
summary = []

for field in sorted(canonical_fields):
    if field in df.columns:
        series = df[field]
        present = series.notna()
        empty = present & (series.astype(str).map(clean_text) == "")
        non_empty = present & ~empty
        missing = ~present
    else:
        non_empty = empty = pd.Series([False] * total)
        missing = pd.Series([True] * total)

    summary.append({
        "field": field,
        "non_empty": int(non_empty.sum()),
        "empty": int(empty.sum()),
        "missing": int(missing.sum()),
        "utilization_%": round((non_empty.sum() / total) * 100, 2) if total else 0
    })

summary_df = pd.DataFrame(summary).sort_values(
    by=["non_empty", "utilization_%"],
    ascending=False
)

# -------------------------------------------------
# Extra (non-canonical) fields
# -------------------------------------------------
extra_fields = sorted(observed_fields - canonical_set)

extra_df = pd.DataFrame(
    [{"field": f, "products_using_field": df[f].notna().sum()} for f in extra_fields]
).sort_values("products_using_field", ascending=False)

# -------------------------------------------------
# Field → products pivot
# -------------------------------------------------
field_products = []

for field in canonical_fields:
    if field not in df.columns:
        field_products.append({
            "field": field,
            "products_using_field": 0,
            "sample_products": ""
        })
        continue

    mask = df[field].notna() & (df[field].astype(str).map(clean_text) != "")
    used = df.loc[mask, "_product"].tolist()

    field_products.append({
        "field": field,
        "products_using_field": len(used),
        "sample_products": ", ".join(used[:10])
    })

field_products_df = pd.DataFrame(field_products).sort_values(
    "products_using_field", ascending=False
)

# -------------------------------------------------
# Output
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Canonical field utilization",
    "Field → products",
    "Extra fields in feed"
])

with tab1:
    st.dataframe(summary_df, use_container_width=True, height=520)
    st.download_button(
        "Download utilization CSV",
        summary_df.to_csv(index=False),
        "field_utilization.csv",
        "text/csv"
    )

with tab2:
    st.dataframe(field_products_df, use_container_width=True, height=520)

    field = st.selectbox("Inspect field", field_products_df["field"])
    if field in df.columns:
        mask = df[field].notna() & (df[field].astype(str).map(clean_text) != "")
        st.dataframe(
            df.loc[mask, ["_product"]],
            use_container_width=True,
            height=300
        )
    else:
        st.info("Field not present in feed at all.")

with tab3:
    if extra_df.empty:
        st.success("No non-canonical fields found.")
    else:
        st.warning("Fields present in feed but NOT in canonical list")
        st.dataframe(extra_df, use_container_width=True, height=400)
