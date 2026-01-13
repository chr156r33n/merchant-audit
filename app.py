import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import re

st.set_page_config(page_title="Merchant Feed Field Utilization PoC", layout="wide")
st.title("Merchant XML Feed Field Utilization (PoC)")
st.caption("Upload an XML feed. Get counts of missing/empty/non-empty per field, plus which products actually use each field.")

# ----------------------------
# Helpers
# ----------------------------
WHITESPACE_RE = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s

def local_name(tag: str) -> str:
    # {namespace}tag -> tag
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def tag_with_prefix(tag: str, nsmap: dict) -> str:
    """
    Convert '{uri}price' to 'g:price' if nsmap reverse maps uri -> prefix,
    otherwise fallback to local name.
    """
    if tag.startswith("{") and "}" in tag:
        uri, name = tag[1:].split("}", 1)
        prefix = nsmap.get(uri)
        return f"{prefix}:{name}" if prefix else name
    return tag

def build_ns_reverse_map(root: ET.Element) -> dict:
    """
    ElementTree doesn't preserve prefixes directly, but root.attrib may include xmlns declarations
    in some parsers. We'll also attempt to infer 'g' from common Google namespace if present.
    """
    # Best-effort: map known Google namespace URIs to 'g'
    ns_reverse = {}
    known_google_uris = {
        "http://base.google.com/ns/1.0": "g",
        "http://base.google.com/ns/1.0/": "g",
    }
    ns_reverse.update(known_google_uris)

    # Try to read xmlns declarations if present (varies by parser)
    for k, v in root.attrib.items():
        # e.g. '{http://www.w3.org/2000/xmlns/}g' : 'http://base.google.com/ns/1.0'
        if k.startswith("{http://www.w3.org/2000/xmlns/}"):
            prefix = k.split("}", 1)[1]
            ns_reverse[v] = prefix
        elif k == "xmlns":
            ns_reverse[v] = None

    return ns_reverse

def flatten_item(elem: ET.Element, ns_reverse: dict, include_attributes: bool = False) -> dict:
    """
    Flattens an <item> (or <entry>) into {field_path: value}.
    - Nested elements become dotted paths: 'shipping.country'
    - Repeated fields are joined with ' | ' (for PoC readability)
    """
    out = defaultdict(list)

    def walk(node: ET.Element, prefix: str = ""):
        # Include attributes optionally
        if include_attributes and node.attrib:
            for ak, av in node.attrib.items():
                out[f"{prefix}@{ak}" if prefix else f"@{ak}"].append(clean_text(str(av)))

        # Node text can be a value if it has no children (or even if it does, but we usually ignore mixed content)
        children = list(node)
        node_tag = tag_with_prefix(node.tag, ns_reverse)

        if children:
            for child in children:
                child_tag = tag_with_prefix(child.tag, ns_reverse)
                new_prefix = f"{prefix}.{child_tag}" if prefix else child_tag
                walk(child, new_prefix)
        else:
            val = clean_text(node.text or "")
            # use the prefix as the field name (already includes this node's name)
            # if prefix is empty, fallback to node tag
            field = prefix if prefix else node_tag
            out[field].append(val)

    # Start with each direct child of item as a top-level field
    for child in list(elem):
        child_tag = tag_with_prefix(child.tag, ns_reverse)
        walk(child, child_tag)

    # Collapse repeated values
    collapsed = {}
    for k, vals in out.items():
        # Keep empties too (important for "present but empty")
        if len(vals) == 1:
            collapsed[k] = vals[0]
        else:
            # join distinct values (keep empty if all empty)
            distinct = []
            for v in vals:
                if v not in distinct:
                    distinct.append(v)
            collapsed[k] = " | ".join(distinct)

    return collapsed

def find_items(root: ET.Element) -> list:
    """
    Find item nodes for RSS (<item>) or Atom-like (<entry>) feeds.
    """
    items = root.findall(".//item")
    if items:
        return items
    entries = root.findall(".//entry")
    if entries:
        return entries
    # Fallback: some feeds use <product> or other structures
    # We'll consider direct children of <channel> if present
    channel = root.find(".//channel")
    if channel is not None:
        return list(channel)
    return []

def pick_product_id(flat: dict) -> str:
    """
    Best-effort identifier for listing products in the "Field -> products" view.
    """
    candidates = [
        "g:id", "id",
        "g:item_group_id", "item_group_id",
        "g:mpn", "mpn",
        "g:gtin", "gtin",
        "g:title", "title",
        "link", "g:link",
    ]
    for c in candidates:
        if c in flat and clean_text(flat[c]):
            return clean_text(flat[c])[:120]
    return "(no id/title/link found)"

# ----------------------------
# UI
# ----------------------------
uploaded = st.file_uploader("Upload merchant feed XML", type=["xml"])
max_products = st.number_input("Max products to process (PoC speed guard)", min_value=10, max_value=200000, value=5000, step=100)
show_raw = st.checkbox("Show flattened products table (debug)", value=False)
include_attributes = st.checkbox("Include XML attributes as fields", value=False)

if not uploaded:
    st.info("Upload an XML feed to begin.")
    st.stop()

data = uploaded.read()
try:
    root = ET.fromstring(data)
except Exception as e:
    st.error(f"XML parse failed: {e}")
    st.stop()

ns_reverse = build_ns_reverse_map(root)
items = find_items(root)

if not items:
    st.warning("No <item> or <entry> nodes found. Feed structure might be custom.")
    st.stop()

items = items[: int(max_products)]
st.write(f"Products detected: **{len(items)}**")

# Flatten each product
rows = []
all_fields = set()

for it in items:
    flat = flatten_item(it, ns_reverse, include_attributes=include_attributes)
    pid = pick_product_id(flat)
    flat["_product_id"] = pid
    rows.append(flat)
    all_fields.update([k for k in flat.keys() if k != "_product_id"])

# Build dataframe with union of fields
df = pd.DataFrame(rows)
if "_product_id" in df.columns:
    df.insert(0, "_product_id", df.pop("_product_id"))

# Utilization counts
total = len(df)
counts = []
for field in sorted([c for c in df.columns if c != "_product_id"]):
    series = df[field] if field in df.columns else pd.Series([None] * total)
    present = series.notna()
    # present-but-empty = present and cleaned string == ""
    present_empty = present & (series.astype(str).map(lambda x: clean_text(x)) == "")
    present_nonempty = present & ~present_empty
    missing = ~present

    counts.append({
        "field": field,
        "non_empty": int(present_nonempty.sum()),
        "empty": int(present_empty.sum()),
        "missing": int(missing.sum()),
        "utilization_%": round((present_nonempty.sum() / total) * 100, 2) if total else 0.0
    })

summary = pd.DataFrame(counts).sort_values(by=["non_empty", "utilization_%"], ascending=False)

# Field -> products that use it
field_to_products = {}
for field in [c for c in df.columns if c != "_product_id"]:
    vals = df[field]
    used_mask = vals.notna() & (vals.astype(str).map(lambda x: clean_text(x)) != "")
    field_to_products[field] = df.loc[used_mask, "_product_id"].tolist()

products_view = pd.DataFrame({
    "field": list(field_to_products.keys()),
    "products_using_field": [len(v) for v in field_to_products.values()],
    "sample_products": [", ".join(v[:10]) for v in field_to_products.values()],
}).sort_values(by="products_using_field", ascending=False)

# ----------------------------
# Output
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Field utilization counts", "Field -> products", "Notes"])

with tab1:
    st.subheader("Counts by field")
    st.dataframe(summary, use_container_width=True, height=520)

    st.download_button(
        "Download field utilization CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="field_utilization.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("Which products actually use each field")
    st.dataframe(products_view, use_container_width=True, height=520)

    selected_field = st.selectbox("Inspect a field (list products using it)", options=products_view["field"].tolist())
    prod_list = field_to_products.get(selected_field, [])
    st.write(f"Products using **{selected_field}**: **{len(prod_list)}**")
    st.dataframe(pd.DataFrame({"_product_id": prod_list}), use_container_width=True, height=300)

    st.download_button(
        "Download field->products CSV",
        data=products_view.to_csv(index=False).encode("utf-8"),
        file_name="field_products.csv",
        mime="text/csv",
    )

with tab3:
    st.markdown(
        """
**What this PoC does**
- Parses RSS-style `<item>` or Atom-style `<entry>` nodes.
- Flattens nested elements into dotted paths (e.g. `shipping.country`).
- Treats *present but empty* separately from *missing*.
- Joins repeated fields using ` | ` so you can at least see they exist.

**Common gotchas**
- Some feeds hide fields inside CDATA or use HTML blobs. This will still count them as non-empty, because it is.
- If your feed is enormous, bump `Max products` carefully unless you enjoy watching browsers die.
"""
    )

if show_raw:
    st.subheader("Flattened product table (debug)")
    st.dataframe(df, use_container_width=True, height=520)
