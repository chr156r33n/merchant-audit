import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
import re
from pathlib import Path

# -------------------------------------------------
# App setup
# -------------------------------------------------
st.set_page_config(page_title="Merchant XML Field Utilization", layout="wide")
st.title("Merchant XML Field Utilization PoC")
st.caption("Validates a merchant XML feed against a canonical Google Merchant field list.")

# -------------------------------------------------
# Load canonical field list
# -------------------------------------------------
FIELDS_FILE = Path("fields.txt")

if not FIELDS_FILE.exists():
    st.error("fields.txt not found in repo root.")
    st.stop()

canonical_fields = [
    line.strip()
    for line in FIELDS_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]

canonical_set = set(canonical_fields)

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

# -------------------------------------------------
# UI
# -------------------------------------------------
uploaded = st.file_uploader("Upload merchant XML feed", type=["xml"])
max_products = st.number_input(
    "Max products to process",
    min_value=10,
    max_value=200000,
    value=5000,
    step=500
)

if not uploaded:
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
