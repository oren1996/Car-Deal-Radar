"""Loading, validating, splitting and exporting car listings.

Adapts week 6's data preparation (pricer/items.py, pricer/loaders.py) and the
week 7 prompt/completion format to used-car listings priced in shekels.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from car_deal_radar.models import CarListing

# Adapted from week 7 pricer/items.py, where
# QUESTION = "What does this cost to the nearest dollar?" and PREFIX = "Price is $".
QUESTION = "How much does this car cost to the nearest shekel?"
PREFIX = "Price is ₪"

# Sanity bounds used by validate_listing.
MIN_PLAUSIBLE_YEAR = 1950
MAX_PLAUSIBLE_YEAR = 2027
MAX_PLAUSIBLE_PRICE_ILS = 2_000_000


def load_car_listings(path: str | Path) -> list[CarListing]:
    """Load listings from a JSON file containing a list of listing objects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Listings file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list of listings in {path}, got {type(raw).__name__}")
    listings = []
    for index, entry in enumerate(raw):
        try:
            listings.append(CarListing(**entry))
        except TypeError as exc:
            raise ValueError(f"Listing at index {index} in {path} has invalid fields: {exc}") from exc
    return listings


def validate_listing(listing: CarListing) -> list[str]:
    """Return a list of detected problems; empty means the listing looks sane.

    Never raises: the caller decides what to do with problematic listings.
    """
    problems: list[str] = []
    if not listing.listing_id:
        problems.append("missing listing_id")
    if not listing.make:
        problems.append("missing make")
    if not listing.model:
        problems.append("missing model")
    if not MIN_PLAUSIBLE_YEAR <= listing.year <= MAX_PLAUSIBLE_YEAR:
        problems.append(f"implausible year: {listing.year}")
    if listing.mileage_km < 0:
        problems.append(f"negative mileage: {listing.mileage_km}")
    if listing.asking_price_ils <= 0:
        problems.append(f"asking price is zero or negative: {listing.asking_price_ils}")
    elif listing.asking_price_ils > MAX_PLAUSIBLE_PRICE_ILS:
        problems.append(f"asking price is implausibly high: {listing.asking_price_ils}")
    return problems


def split_listings(
    listings: list[CarListing],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[CarListing], list[CarListing]]:
    """Split listings into (train, test) with a deterministic shuffle.

    Same approach as week 6 day 2: seed, shuffle, slice. No scikit-learn
    needed for this.
    """
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
    shuffled = list(listings)
    random.Random(seed).shuffle(shuffled)
    test_size = int(len(shuffled) * test_ratio)
    return shuffled[test_size:], shuffled[:test_size]


def create_fine_tuning_records(listings: list[CarListing]) -> list[dict[str, object]]:
    """Build simple records for the future week 7 style fine-tuning phase.

    The prompt/completion fields follow week 7 pricer/items.py, where the
    prompt is "{QUESTION}\\n\\n{summary}\\n\\n{PREFIX}" and the completion is
    the rounded price as "{price}.00". No tokenization, Hugging Face Dataset
    or Hub upload happens here yet.
    """
    records: list[dict[str, object]] = []
    for listing in listings:
        text = listing.to_model_text()
        records.append(
            {
                "text": text,
                "price_ils": listing.asking_price_ils,
                "prompt": f"{QUESTION}\n\n{text}\n\n{PREFIX}",
                "completion": f"{round(listing.asking_price_ils)}.00",
            }
        )
    return records
