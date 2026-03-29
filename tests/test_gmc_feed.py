"""Tests for GMC feed parsing (fixtures mirror Merchant Center RSS, Atom, TSV)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gmc_feed import (
    detect_feed_format,
    load_canonical_fields,
    parse_tab_delimited,
    parse_tabular_feed_rows,
    parse_xml_feed_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIELDS_PATH = REPO_ROOT / "fields.txt"


@pytest.fixture(scope="module")
def canonical():
    fields, s = load_canonical_fields(FIELDS_PATH)
    return fields, s


def test_detect_format_xml_vs_tabular():
    rss = (FIXTURES / "rss_sample.xml").read_bytes()
    tab = (FIXTURES / "tab_sample.tsv").read_bytes()
    assert detect_feed_format(rss) == "xml"
    assert detect_feed_format(tab) == "tabular"


def test_rss_fixture_row_count_and_g_title(canonical):
    fields, cset = canonical
    raw = (FIXTURES / "rss_sample.xml").read_bytes()
    rows, _obs = parse_xml_feed_rows(raw, 100, fields, cset)
    assert len(rows) == 2
    assert rows[0]["g:title"] == "RSS Product A"
    assert rows[0]["g:id"] == "rss-001"
    assert rows[1]["g:title"] == "RSS Product B with g:title"
    assert rows[1]["g:id"] == "rss-002"


def test_atom_fixture_row_count_and_link_href(canonical):
    fields, cset = canonical
    raw = (FIXTURES / "atom_sample.xml").read_bytes()
    assert detect_feed_format(raw) == "xml"
    rows, _obs = parse_xml_feed_rows(raw, 100, fields, cset)
    assert len(rows) == 2
    assert rows[0]["g:title"] == "Atom Product One"
    assert "example.com/p1" in (rows[0].get("g:link") or rows[0].get("link") or "")
    assert rows[1]["g:id"] == "atom-002"


def test_tab_fixture_row_count_and_g_title(canonical):
    fields, cset = canonical
    raw = (FIXTURES / "tab_sample.tsv").read_bytes()
    assert detect_feed_format(raw) == "tabular"
    rows, _obs = parse_tabular_feed_rows(raw, 100, fields, cset)
    assert len(rows) == 3
    assert rows[0]["g:title"] == "TSV Widget One"
    assert rows[0]["g:id"] == "tsv-001"
    assert rows[2]["g:link"] == "https://example.com/t3"


def test_parse_tab_delimited_bracket_headers():
    fields, cset = load_canonical_fields(FIELDS_PATH)
    raw = b"[id]\t[title]\nbracket-1\tBracket Title\n"
    df = parse_tab_delimited(raw)
    assert list(df.columns) == ["id", "title"]
    rows, _ = parse_tabular_feed_rows(raw, 10, fields, cset)
    assert len(rows) == 1
    assert rows[0]["g:id"] == "bracket-1"
    assert rows[0]["g:title"] == "Bracket Title"
