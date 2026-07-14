"""Inspect a used-car CSV to see if it fits the CarListing / LLM pipeline.

Answers one question: does this dataset carry the fields to_model_text() needs
to turn a row into a sentence for the pricer? Uses only the standard library,
so it runs with no extra dependencies.

Usage:
    python scripts/inspect_dataset.py path/to/dataset.csv
    python scripts/inspect_dataset.py path/to/dataset.csv --rows 5
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# The fields a car pricer really needs, and the column names a dataset might
# use for each (lowercased, matched loosely). Extend the aliases as needed.
CORE_FIELDS = {
    "make": ["make", "manufacturer", "brand", "manufactor", "יצרן"],
    "model": ["model", "commercial_name", "דגם"],
    "year": ["year", "prod_year", "yearofproduction", "שנה"],
    "mileage_km": ["km", "kilometers", "mileage", "kilometrage", "ק\"מ", "קילומטראז'"],
    "fuel_type": ["fuel", "fuel_type", "engine_type", "enginetype", "סוג מנוע"],
    "price": ["price", "asking_price", "מחיר"],
}
BONUS_FIELDS = {
    "description": ["description", "desc", "text", "notes", "remarks", "תיאור"],
    "transmission": ["gear", "gearbox", "transmission", "gear_box", "תיבת הילוכים"],
    "engine_size_cc": ["engine", "capacity", "engine_volume", "enginevolume", "נפח"],
    "location": ["city", "area", "location", "region", "אזור", "עיר"],
    "owners": ["hand", "owners", "previous_owners", "יד"],
}
# Columns that look like free text get flagged - they add signal for the LLM.
TEXT_HINTS = ["desc", "text", "note", "remark", "comment", "תיאור"]


def _tokens(name: str) -> list[str]:
    """Split a column name into lowercased alphanumeric tokens (Hebrew-aware)."""
    return [t for t in re.split(r"[^a-z0-9֐-׿]+", name.lower()) if t]


def _nosep(name: str) -> str:
    """Lowercase a name with all separators removed (e.g. 'Engine_type' -> 'enginetype')."""
    return re.sub(r"[^a-z0-9֐-׿]+", "", name.lower())


def match_column(columns: list[str], aliases: list[str]) -> str | None:
    """Return the dataset column that matches an alias, avoiding substring traps.

    An alias matches a column when their separator-stripped names are equal
    (so 'engine_type' matches 'Engine_type'), or when the alias appears as a
    whole token of the column name (so 'city' matches 'City' but NOT the 'city'
    hidden inside 'capacity_Engine').
    """
    for alias in aliases:
        alias_nosep = _nosep(alias)
        for col in columns:
            if alias_nosep == _nosep(col) or alias in _tokens(col):
                return col
    return None


def inspect(path: Path, sample_rows: int) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = []
        for i, row in enumerate(reader):
            if i < sample_rows:
                rows.append(row)
            elif i > 100_000:
                break
        total = i + 1 if columns else 0

    print(f"File: {path}")
    print(f"Columns ({len(columns)}): {', '.join(columns)}\n")

    print("Core fields needed by to_model_text():")
    missing = []
    for field, aliases in CORE_FIELDS.items():
        col = match_column(columns, aliases)
        if col:
            print(f"  OK    {field:12s} <- '{col}'")
        else:
            missing.append(field)
            print(f"  MISS  {field:12s} (none of: {', '.join(aliases[:4])})")

    print("\nBonus fields (nice to have):")
    for field, aliases in BONUS_FIELDS.items():
        col = match_column(columns, aliases)
        print(f"  {'OK  ' if col else '--  '}{field:14s} {'<- ' + repr(col) if col else ''}")

    text_cols = [c for c in columns if any(h in c.lower() for h in TEXT_HINTS)]
    print(f"\nFree-text columns detected: {text_cols or 'none'}")
    if not text_cols:
        print("  -> structured-only: fine for a car pricer, to_model_text() builds the sentence.")

    if rows:
        print(f"\nSample of {len(rows)} row(s):")
        for row in rows:
            preview = {k: v for k, v in list(row.items())[:8]}
            print(f"  {preview}")

    print("\nVerdict:", end=" ")
    if not missing:
        print("usable now - all core fields present. Paste this output back and I'll write the loader.")
    elif missing == ["price"]:
        print("no price column found - a pricer needs a target price; check for a differently named one.")
    else:
        print(f"missing core fields {missing} - may be too thin; consider a richer dataset.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a used-car CSV for CarListing compatibility.")
    parser.add_argument("csv_path", type=Path, help="path to the dataset CSV")
    parser.add_argument("--rows", type=int, default=3, help="sample rows to print")
    args = parser.parse_args()
    if not args.csv_path.exists():
        parser.error(f"file not found: {args.csv_path}")
    inspect(args.csv_path, args.rows)


if __name__ == "__main__":
    main()
