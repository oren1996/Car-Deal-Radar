"""Convert the Kaggle 'Vehicle Pricing Data (Israel)' CSV into CarListing JSON.

The dataset is tabular (Yad2-sourced, prices in NIS) with no free-text
description and no mileage column. This converter:
  - maps the structured columns onto the CarListing schema;
  - SYNTHESIZES a description field from the feature flags (horsepower, 4x4,
    cruise control, etc.) so the LLM pricer has real prose to read, not just
    numbers - directly addressing the "no text" gap of a tabular dataset;
  - leaves mileage_km as null (the dataset has none; we do not fabricate it);
  - deliberately IGNORES the leakage columns log_price, cluster and
    cluster_label, which are derived from price and must never reach the model.

Usage (after `uv sync`):
    python scripts/convert_dataset.py path/to/dataset.csv --out data/israel_cars.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from car_deal_radar.data import validate_listing
from car_deal_radar.models import CarListing

# Never expose these to the model: all are derived from the target price.
LEAKAGE_COLUMNS = {"log_price", "cluster", "cluster_label"}

# Common English/Hebrew fuel spellings -> the values BaselinePricer expects.
FUEL_NORMALIZATION = {
    "petrol": "petrol", "gasoline": "petrol", "בנזין": "petrol",
    "diesel": "diesel", "דיזל": "diesel",
    "hybrid": "hybrid", "היברידי": "hybrid", "היברid": "hybrid",
    "electric": "electric", "electricity": "electric", "חשמלי": "electric",
    "gas": "lpg", "lpg": "lpg", "גז": "lpg",
}

# Boolean-ish flag columns -> the phrase added to the description when truthy.
FEATURE_PHRASES = {
    "4x4": "4x4",
    "adaptive_cruise_control": "adaptive cruise control",
    "cruise_control": "cruise control",
    "distance_control": "distance control",
    "magnesium_wheels": "magnesium wheels",
    "economical": "economical",
}
TRUE_TOKENS = {"1", "1.0", "true", "yes", "y", "כן"}


def _get(row: dict[str, str], *names: str) -> str | None:
    """Return the first non-empty value among the given column names."""
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (ValueError, TypeError):
        return None


def _is_true(value: str | None) -> bool:
    return value is not None and value.strip().lower() in TRUE_TOKENS


def _engine_cc(value: str | None) -> int | None:
    """Normalize engine volume to cc, whether the source is liters or cc."""
    if value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    if number <= 0:
        return None
    # Values below ~100 are almost certainly liters (e.g. 1.6 -> 1600 cc).
    return int(round(number * 1000)) if number < 100 else int(round(number))


def build_description(row: dict[str, str]) -> str:
    """Synthesize prose from the dataset's structured feature flags."""
    parts: list[str] = []
    car_name = _get(row, "car_name")
    if car_name:
        parts.append(car_name)
    hp = _to_int(_get(row, "horse_power"))
    if hp:
        parts.append(f"{hp} hp")
    if _is_true(_get(row, "electric_or_not")):
        parts.append("electric")
    features = [phrase for col, phrase in FEATURE_PHRASES.items() if _is_true(_get(row, col))]
    if features:
        parts.append("with " + ", ".join(features))
    if _is_true(_get(row, "valid_test")):
        parts.append("valid test")
    return ", ".join(parts)


def row_to_listing_dict(row: dict[str, str], index: int) -> dict[str, Any] | None:
    """Map one CSV row onto the CarListing schema, or None if unusable."""
    make = _get(row, "brand", "brand_normalized")
    model = _get(row, "model")
    year = _to_int(_get(row, "year"))
    price = _to_int(_get(row, "price"))
    if not make or not model or not year or not price or price <= 0:
        return None
    fuel_raw = _get(row, "fuel_type")
    fuel = FUEL_NORMALIZATION.get(fuel_raw.lower(), fuel_raw.lower()) if fuel_raw else None
    if fuel is None and _is_true(_get(row, "electric_or_not")):
        fuel = "electric"
    return {
        "listing_id": f"kaggle-il-{index}",
        "source": "kaggle-vehicle-pricing-israel",
        "url": None,
        "make": make,
        "model": model,
        "year": year,
        "mileage_km": None,  # dataset has no mileage column - do not fabricate
        "ownership_type": None,
        "previous_owners": _to_int(_get(row, "hand_num")),
        "transmission": None,
        "fuel_type": fuel,
        "engine_size_cc": _engine_cc(_get(row, "engine_volume")),
        "location": None,
        "description": build_description(row),
        "asking_price_ils": float(price),
    }


def convert(csv_path: Path, out_path: Path) -> int:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        leak = LEAKAGE_COLUMNS.intersection(c.lower() for c in columns)
        if leak:
            print(f"Ignoring leakage columns (derived from price): {', '.join(sorted(leak))}")
        listings: list[dict[str, Any]] = []
        skipped = 0
        for index, row in enumerate(reader):
            entry = row_to_listing_dict(row, index)
            if entry is None:
                skipped += 1
                continue
            listings.append(entry)

    problems = sum(1 for e in listings if validate_listing(CarListing(**e)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(listings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Converted {len(listings)} listings ({skipped} skipped, {problems} with warnings).")
    print(f"Wrote {out_path}")
    if listings:
        sample = CarListing(**listings[0])
        print(f"\nExample model text:\n  {sample.to_model_text()}")
    return len(listings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the Israel car CSV into CarListing JSON.")
    parser.add_argument("csv_path", type=Path, help="path to the Kaggle dataset CSV")
    parser.add_argument("--out", type=Path, default=Path("data/israel_cars.json"), help="output JSON path")
    args = parser.parse_args()
    if not args.csv_path.exists():
        parser.error(f"file not found: {args.csv_path}")
    count = convert(args.csv_path, args.out)
    if count == 0:
        print("No listings converted - check the column names against the mapping.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
