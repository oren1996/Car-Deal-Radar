"""Scrape car listings from Yad2 by reading the page's embedded Next.js JSON.

Yad2 is a Next.js app: each search-results page embeds all its listing data in
a <script id="__NEXT_DATA__"> tag. Yad2 sits behind Radware Bot Manager, which
serves a JavaScript-challenge "Loader page" to plain HTTP clients and even to
automated Chrome (tested). Scrapling's StealthyFetcher (a Camoufox stealth
browser) clears the challenge, after which the __NEXT_DATA__ JSON is present as
usual. Note the search feed omits mileage (km); it lives on each item page.

Usage (from the repo root, after `uv sync --extra scraping`):

    python scripts/scrape_yad2.py --pages 2 --delay 3 --out data/yad2_cars.json

Keep the volume small and the delay polite: this is for personal, educational
use. Do not republish the data (it contains seller information).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from car_deal_radar.data import validate_listing
from car_deal_radar.models import CarListing

CARS_URL = "https://www.yad2.co.il/vehicles/cars"
ITEM_URL = "https://www.yad2.co.il/vehicles/item/{token}"
NEXT_DATA_MARKER = '<script id="__NEXT_DATA__" type="application/json">'
# Yad2 sits behind Radware Bot Manager. To a plain-HTTP client it returns a
# "Loader page" (a JS challenge) whose server header is "rdwr" and whose title
# is "Radware Page" - not the old "Are you for real" captcha. Detect all of it.
ANTIBOT_MARKERS = ("Are you for real", "Radware Page", "Loader page.")

# Fallback translations for common Hebrew field values (textEng is preferred
# when Yad2 provides it). Fuel values match BaselinePricer.fuel_factors.
HEBREW_FUEL_TYPES = {
    "בנזין": "petrol",
    "דיזל": "diesel",
    "טורבו דיזל": "diesel",
    "היברידי": "hybrid",
    "היבריד": "hybrid",
    "היברידי חשמל / בנזין": "hybrid",
    "היברידי חשמל / דיזל": "hybrid",
    "חשמלי": "electric",
    "חשמל": "electric",
    "גז": "lpg",
}
HEBREW_TRANSMISSIONS = {
    "אוטומטית": "automatic",
    "אוטומט": "automatic",
    "ידנית": "manual",
    "רובוטית": "automatic",
    "טיפטרוניק": "automatic",
}


def _page_html(page: Any) -> str:
    """Return a Scrapling fetch result's HTML as a string, across versions."""
    for attr in ("html_content", "body"):
        value = getattr(page, attr, None)
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        if isinstance(value, str) and value:
            return value
    return str(page)


def fetch_page(page: int, timeout: float = 120.0) -> str:
    """Fetch one search-results page with Scrapling's stealth browser.

    StealthyFetcher (Camoufox) executes Yad2's Radware JavaScript challenge and
    returns the real page. `network_idle` lets client-side hydration settle so
    the __NEXT_DATA__ payload is fully present.
    """
    # Imported lazily so the parsing helpers stay usable/testable without the
    # heavy scrapling + Camoufox browser dependency installed.
    from scrapling.fetchers import StealthyFetcher

    result = StealthyFetcher.fetch(
        f"{CARS_URL}?page={page}",
        headless=True,
        network_idle=True,
        timeout=int(timeout * 1000),  # Scrapling expects milliseconds
    )
    return _checked_html(result)


def _checked_html(result: Any) -> str:
    """Return a fetch result's HTML, raising if Yad2 served a Radware bot page."""
    html = _page_html(result)
    if getattr(result, "status", 200) != 200 or any(m in html for m in ANTIBOT_MARKERS):
        raise RuntimeError(
            "Yad2 returned a Radware bot-protection page even through Scrapling's "
            "stealth browser. Retry later with a longer --delay; if it persists, "
            "Yad2 may have tightened its bot protection."
        )
    return html


def extract_next_data(html: str) -> dict[str, Any]:
    """Pull the JSON out of the page's __NEXT_DATA__ script tag."""
    start = html.find(NEXT_DATA_MARKER)
    if start == -1:
        raise RuntimeError(
            "No __NEXT_DATA__ script tag found - Yad2 may have changed its page "
            "structure, or served an unexpected page."
        )
    start += len(NEXT_DATA_MARKER)
    end = html.find("</script>", start)
    if end == -1:
        raise RuntimeError("Malformed page: unterminated __NEXT_DATA__ script tag.")
    return json.loads(html[start:end])


def iter_vehicles(next_data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield raw vehicle dicts from the Next.js dehydrated state.

    The state holds queries whose data maps feed-section names to lists of
    listings; anything dict-shaped inside those lists is a vehicle.
    """
    queries = next_data.get("props", {}).get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
    for query in queries:
        data = query.get("state", {}).get("data")
        if not isinstance(data, dict):
            continue
        for section in data.values():
            if not isinstance(section, list):
                continue
            for vehicle in section:
                if isinstance(vehicle, dict) and "price" in vehicle:
                    yield vehicle


def _field_text(value: Any, translations: dict[str, str] | None = None) -> str | None:
    """Read a Yad2 {text, textEng, id} field, preferring English."""
    if not isinstance(value, dict):
        return None
    text_eng = value.get("textEng")
    if isinstance(text_eng, str) and text_eng.strip():
        return text_eng.strip()
    text = value.get("text")
    if isinstance(text, str) and text.strip():
        text = text.strip()
        if translations:
            return translations.get(text, text)
        return text
    return None


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _field_lower(value: Any, translations: dict[str, str] | None = None) -> str | None:
    """Like _field_text but lowercased, so 'Petrol'/'Automatic' match the
    lowercase keys BaselinePricer and DealFinder expect."""
    text = _field_text(value, translations)
    return text.lower() if text else None


def build_description(vehicle: dict[str, Any]) -> str:
    """Synthesize the listing text from the item-page attributes, followed by
    the seller's own description.

    Weaves in the extra price-relevant fields (trim, horsepower, body type,
    colour, seats/doors, fuel economy, safety, turbo, test validity, equipment)
    rather than adding CarListing columns. All of these live on the item page,
    not the search feed, so this degrades gracefully to the seller description
    alone when a field is absent.
    """
    lead: list[str] = []

    sub_model = vehicle.get("subModel")
    if isinstance(sub_model, dict) and (sub_model.get("text") or "").strip():
        lead.append(sub_model["text"].strip())

    hp = _int_or_none(vehicle.get("horsePower"))
    if hp:
        lead.append(f"{hp} hp")

    if vehicle.get("specification", {}).get("isTurbo") is True:
        lead.append("turbo")

    color = _field_text(vehicle.get("color"))
    if color:
        lead.append(color)

    # Prefer the English car-family labels (Crossover, Jeep) over Hebrew bodyType.
    family = vehicle.get("carFamilyType")
    if isinstance(family, list):
        families = [f.get("textEng") for f in family if isinstance(f, dict) and f.get("textEng")]
        if families:
            lead.append("/".join(families))
    body = _field_text(vehicle.get("bodyType"))
    if body and not (isinstance(family, list) and family):
        lead.append(body)

    seats = _int_or_none(vehicle.get("seats"))
    doors = _int_or_none(vehicle.get("numberOfDoors"))
    if seats:
        lead.append(f"{seats} seats")
    if doors:
        lead.append(f"{doors} doors")

    consumption = vehicle.get("combinedFuelConsumption")
    if isinstance(consumption, (int, float)) and not isinstance(consumption, bool) and consumption:
        lead.append(f"{consumption} km/l combined")

    safety = vehicle.get("specification", {}).get("safetyPoints")
    if isinstance(safety, int) and not isinstance(safety, bool):
        lead.append(f"safety {safety}")

    test_date = vehicle.get("vehicleDates", {}).get("testDate")
    if isinstance(test_date, str) and test_date.strip():
        lead.append(f"test valid until {test_date[:10]}")

    car_tags = vehicle.get("carTag")
    if isinstance(car_tags, list):
        features = [t.get("textEng") for t in car_tags if isinstance(t, dict) and t.get("textEng")]
        if features:
            lead.append(", ".join(features))

    seller = str((vehicle.get("metaData") or {}).get("description") or "").strip()
    lead_text = " · ".join(lead)
    if lead_text and seller:
        return f"{lead_text}\n{seller}"
    return lead_text or seller


def to_listing_dict(vehicle: dict[str, Any]) -> dict[str, Any] | None:
    """Map one raw Yad2 vehicle dict onto the CarListing JSON schema.

    Returns None when essential fields (token, price, year, make, model) are
    missing - same "parse or reject" pattern as week6/pricer/parser.py. Mileage
    (km) is absent from the search feed, so it is optional here and stays None.
    """
    token = vehicle.get("token")
    price = _int_or_none(vehicle.get("price"))
    year = _int_or_none((vehicle.get("vehicleDates") or {}).get("yearOfProduction"))
    km = _int_or_none(vehicle.get("km"))
    make = _field_text(vehicle.get("manufacturer"))
    model = _field_text(vehicle.get("model"))
    if not token or not price or price <= 0 or not year or not make or not model:
        return None
    hand = vehicle.get("hand")
    # adType ("private"/"commercial") is cleaner English than owner.text ("פרטית").
    ad_type = vehicle.get("adType")
    owner = vehicle.get("owner")
    ownership_type = ad_type if isinstance(ad_type, str) and ad_type.strip() else (
        _field_text(owner) if isinstance(owner, dict) else None
    )
    return {
        "listing_id": f"yad2-{token}",
        "source": "yad2",
        "url": ITEM_URL.format(token=token),
        "make": make,
        "model": model,
        "year": year,
        "mileage_km": km,
        "ownership_type": ownership_type,
        "previous_owners": _int_or_none(hand.get("id")) if isinstance(hand, dict) else None,
        "transmission": _field_lower(vehicle.get("gearBox"), HEBREW_TRANSMISSIONS),
        "fuel_type": _field_lower(vehicle.get("engineType"), HEBREW_FUEL_TYPES),
        "engine_size_cc": _int_or_none(vehicle.get("engineVolume")),
        "location": (
            _field_text((vehicle.get("address") or {}).get("city"))
            or _field_text((vehicle.get("address") or {}).get("area"))
        ),
        "description": build_description(vehicle),
        "asking_price_ils": float(price),
    }


def _looks_like_vehicle(obj: Any) -> bool:
    """True for a dict that carries a full vehicle record (item-page shape)."""
    return isinstance(obj, dict) and "manufacturer" in obj and "price" in obj and "token" in obj


def _find_vehicle_dict(obj: Any) -> dict[str, Any] | None:
    """Depth-first search for a vehicle record anywhere in a nested structure."""
    if _looks_like_vehicle(obj):
        return obj
    if isinstance(obj, dict):
        for value in obj.values():
            found = _find_vehicle_dict(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_vehicle_dict(item)
            if found is not None:
                return found
    return None


def extract_item_vehicle(next_data: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the single vehicle record from an item page's __NEXT_DATA__.

    An item page stores the car as a dict (not the feed's dict-of-lists), so we
    look through the dehydrated queries first, then fall back to a recursive
    search to stay robust against structure changes.
    """
    queries = next_data.get("props", {}).get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
    for query in queries:
        data = query.get("state", {}).get("data")
        if _looks_like_vehicle(data):
            return data
    return _find_vehicle_dict(next_data)


def merge_enrichment(base: dict[str, Any], item_vehicle: dict[str, Any]) -> dict[str, Any]:
    """Fill a search-feed listing with the richer item-page fields (km, etc.).

    The item record is a superset of the feed record, so any non-empty value it
    provides overrides the base; the base survives as a fallback.
    """
    item = to_listing_dict(item_vehicle)
    if item is None:
        return base
    merged = dict(base)
    for key, value in item.items():
        if value not in (None, "", []):
            merged[key] = value
    return merged


def enrich_listings(listings: dict[str, dict[str, Any]], delay: float) -> int:
    """Visit each listing's item page in one stealth session to add mileage etc.

    A single StealthySession keeps the Radware cookie warm, so the many item
    pages load without re-solving the challenge each time. Per-item failures are
    logged and skipped, keeping the search-feed data for that listing.
    """
    from scrapling.fetchers import StealthySession

    enriched = 0
    total = len(listings)
    with StealthySession(headless=True) as session:
        for index, (listing_id, base) in enumerate(list(listings.items()), start=1):
            print(f"Enriching {index}/{total}: {listing_id} ...")
            try:
                result = session.fetch(base["url"], network_idle=True, timeout=120_000)
                item_vehicle = extract_item_vehicle(extract_next_data(_checked_html(result)))
            except (RuntimeError, OSError) as exc:
                print(f"  skipped ({exc})")
                item_vehicle = None
            if item_vehicle is not None:
                listings[listing_id] = merge_enrichment(base, item_vehicle)
                if listings[listing_id].get("mileage_km") is not None:
                    enriched += 1
            if index < total:
                time.sleep(delay)
    print(f"Enriched {enriched}/{total} listings with mileage from item pages")
    return enriched


def scrape(pages: int, delay: float, out_path: Path, dump_raw: bool, enrich: bool) -> int:
    """Scrape the requested number of pages and write CarListing-format JSON."""
    listings: dict[str, dict[str, Any]] = {}
    for page in range(1, pages + 1):
        print(f"Fetching page {page}/{pages} (stealth browser) ...")
        html = fetch_page(page)
        next_data = extract_next_data(html)
        if dump_raw and page == 1:
            raw_path = out_path.with_suffix(".raw.json")
            raw_path.write_text(json.dumps(next_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  raw __NEXT_DATA__ of page 1 dumped to {raw_path}")
        page_count = 0
        for vehicle in iter_vehicles(next_data):
            listing = to_listing_dict(vehicle)
            if listing is not None and listing["listing_id"] not in listings:
                listings[listing["listing_id"]] = listing
                page_count += 1
        print(f"  kept {page_count} listings")
        if page < pages:
            time.sleep(delay)

    if enrich and listings:
        # The search feed omits mileage; the item page carries it (roadmap 1.1).
        enrich_listings(listings, delay)

    problems = 0
    for entry in listings.values():
        issues = validate_listing(CarListing(**entry))
        if issues:
            problems += 1
            print(f"  warning {entry['listing_id']}: {'; '.join(issues)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(list(listings.values()), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {len(listings)} listings ({problems} with validation warnings) to {out_path}")
    return len(listings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Yad2 car listings into CarListing JSON.")
    parser.add_argument("--pages", type=int, default=1, help="number of result pages to fetch")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds to sleep between pages")
    parser.add_argument("--out", type=Path, default=Path("data/yad2_cars.json"), help="output JSON path")
    parser.add_argument("--dump-raw", action="store_true", help="also dump page 1's raw __NEXT_DATA__")
    parser.add_argument(
        "--enrich",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="visit each item page to add mileage and other missing fields (default: on)",
    )
    args = parser.parse_args()
    if args.pages < 1:
        parser.error("--pages must be >= 1")
    try:
        count = scrape(args.pages, args.delay, args.out, args.dump_raw, args.enrich)
    except (RuntimeError, OSError) as exc:
        # RuntimeError: our anti-bot / parse guards. OSError: browser/network faults.
        print(f"\nScraping failed: {exc}", file=sys.stderr)
        sys.exit(1)
    if count == 0:
        print(
            "No listings extracted - the page structure may have changed. "
            "Re-run with --dump-raw and inspect the JSON.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
