"""
Google Merchant Center feed parsing: RSS 2.0, Atom 1.0, and tab-delimited .txt/.tsv.
Used by app.py and tests without importing Streamlit.
"""

from __future__ import annotations

import io
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import xml.etree.ElementTree as ET

ATOM_NS = "http://www.w3.org/2005/Atom"
GMC_NS = "http://base.google.com/ns/1.0"
CNS_NS = "http://base.google.com/cns/1.0"

_whitespace = re.compile(r"\s+")


def clean_text(v):
    if v is None:
        return ""
    return _whitespace.sub(" ", str(v)).strip()


def load_canonical_fields(fields_path: Path) -> tuple[list[str], set[str]]:
    """Load fields.txt lines (non-comment, non-empty)."""
    if not fields_path.exists():
        raise FileNotFoundError(f"Missing fields file: {fields_path}")
    lines = [
        line.strip()
        for line in fields_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return lines, set(lines)


def build_ns_reverse_map(root):
    """Map namespace URIs to prefixes from the root and all descendants (GMC allows xmlns on nested elements)."""
    ns = {
        GMC_NS: "g",
        CNS_NS: "cns",
    }
    for el in [root, *root.iter()]:
        for k, v in el.attrib.items():
            if k.startswith("{http://www.w3.org/2000/xmlns/}"):
                ns[v] = k.split("}", 1)[1]
    return ns


def tag_name(tag, ns_map):
    if tag.startswith("{"):
        uri, name = tag[1:].split("}", 1)
        prefix = ns_map.get(uri)
        return f"{prefix}:{name}" if prefix else name
    return tag


def _leaf_text_and_attrs(node, ns_map):
    """Text for a leaf node, including Atom link href and common empty-element patterns."""
    t = clean_text(node.text)
    if t:
        return t
    if node.tag == f"{{{ATOM_NS}}}link":
        href = node.get("href")
        if href:
            return clean_text(href)
    if node.tag.endswith("}link") and not t:
        href = node.get("href")
        if href:
            return clean_text(href)
    return ""


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
            val = _leaf_text_and_attrs(node, ns_map)
            out[field].append(val)

    for child in list(item):
        walk(child, tag_name(child.tag, ns_map))

    return {k: " | ".join(set(v)) for k, v in out.items()}


def find_products(root):
    items = root.findall(".//item") or root.findall(".//{*}item")
    if items:
        return items

    entries = root.findall(".//entry") or root.findall(".//{*}entry")
    if entries:
        return entries

    products = root.findall(".//product") or root.findall(".//{*}product")
    if products:
        return products

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


def normalize_header(name):
    """Strip whitespace and optional bracket notation used in some exports."""
    s = str(name).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    return s


def normalize_gmc_row(flat_dict, canonical_fields: list[str], canonical_set: set[str]):
    """
    Merge plain RSS/Atom element names, TSV column headers, and cns: attributes
    into canonical g: keys so one code path handles all valid GMC shapes.
    """
    merged = {}
    for k, v in flat_dict.items():
        if k == "_product":
            continue
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        val = clean_text(v)
        if not val:
            continue
        key = normalize_header(k)
        merged[key] = val

    def set_if_missing(gkey, *candidates):
        cur = merged.get(gkey)
        if cur is not None and str(cur).strip() != "":
            return
        for c in candidates:
            if c in merged and merged[c] is not None and str(merged[c]).strip() != "":
                merged[gkey] = merged[c]
                return

    for field in canonical_fields:
        if not field.startswith("g:"):
            continue
        plain = field[2:]
        set_if_missing(field, plain)

    set_if_missing("g:id", "id")
    set_if_missing("g:title", "title")
    set_if_missing("g:description", "description", "summary", "content")
    set_if_missing("g:link", "link")
    set_if_missing("g:image_link", "image_link")
    set_if_missing("g:additional_image_link", "additional_image_link")
    set_if_missing("g:mobile_link", "mobile_link")
    set_if_missing("g:short_title", "short_title")
    set_if_missing("g:product_type", "product_type")
    set_if_missing("g:product_highlight", "product_highlight")

    for k in list(merged.keys()):
        if k.startswith("cns:"):
            rest = k[4:]
            gkey = f"g:{rest}"
            if gkey in canonical_set:
                set_if_missing(gkey, k)

    return merged


def detect_feed_format(raw: bytes) -> str:
    """Sniff XML (RSS/Atom) vs tab-delimited text."""
    s = raw
    if s.startswith(b"\xef\xbb\xbf"):
        s = s[3:]
    head = s[: min(800, len(s))].lstrip().lower()
    if (
        head.startswith(b"<?xml")
        or head.startswith(b"<rss")
        or head.startswith(b"<feed")
        or head.startswith(b"<rdf")
        or head.startswith(b"<channel")
    ):
        return "xml"
    return "tabular"


def parse_tab_delimited(raw: bytes) -> pd.DataFrame:
    """Parse Merchant Center tab-delimited .txt / .tsv (comma fallback if a single column)."""
    text = raw.decode("utf-8-sig")
    df_tab = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, keep_default_na=False)
    if df_tab.shape[1] > 1:
        df_tab.columns = [normalize_header(c) for c in df_tab.columns]
        return df_tab
    df_comma = pd.read_csv(io.StringIO(text), sep=",", dtype=str, keep_default_na=False)
    if df_comma.shape[1] > 1:
        df_comma.columns = [normalize_header(c) for c in df_comma.columns]
        return df_comma
    df_tab.columns = [normalize_header(c) for c in df_tab.columns]
    return df_tab


def parse_xml_feed_rows(
    raw: bytes,
    max_products: int,
    canonical_fields: list[str],
    canonical_set: set[str],
) -> tuple[list[dict], set[str]]:
    """Parse XML bytes into normalized product row dicts."""
    raw_xml = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
    root = ET.fromstring(raw_xml)
    ns_map = build_ns_reverse_map(root)
    products = find_products(root)[:max_products]
    rows = []
    observed: set[str] = set()
    for p in products:
        flat = flatten_item(p, ns_map)
        merged = normalize_gmc_row(flat, canonical_fields, canonical_set)
        merged["_product"] = pick_product_id(merged)
        rows.append(merged)
        observed.update(merged.keys())
    return rows, observed


def parse_tabular_feed_rows(
    raw: bytes,
    max_products: int,
    canonical_fields: list[str],
    canonical_set: set[str],
) -> tuple[list[dict], set[str]]:
    """Parse tab-delimited bytes into normalized product row dicts."""
    tdf = parse_tab_delimited(raw).head(max_products)
    rows = []
    observed: set[str] = set()
    for _, r in tdf.iterrows():
        merged = normalize_gmc_row(r.to_dict(), canonical_fields, canonical_set)
        merged["_product"] = pick_product_id(merged)
        rows.append(merged)
        observed.update(merged.keys())
    return rows, observed
