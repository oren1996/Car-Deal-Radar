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

from scrapling.fetchers import StealthyFetcher

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
    result = StealthyFetcher.fetch(
        f"{CARS_URL}?page={page}",
        headless=True,
        network_idle=True,
        timeout=int(timeout * 1000),  # Scrapling expects milliseconds
    )
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
    metadata = vehicle.get("metaData") or {}
    description = metadata.get("description") or ""
    hand = vehicle.get("hand")
    owner = vehicle.get("owner")
    return {
        "listing_id": f"yad2-{token}",
        "source": "yad2",
        "url": ITEM_URL.format(token=token),
        "make": make,
        "model": model,
        "year": year,
        "mileage_km": km,
        "ownership_type": _field_text(owner) if isinstance(owner, dict) else None,
        "previous_owners": _int_or_none(hand.get("id")) if isinstance(hand, dict) else None,
        "transmission": _field_text(vehicle.get("gearBox"), HEBREW_TRANSMISSIONS),
        "fuel_type": _field_text(vehicle.get("engineType"), HEBREW_FUEL_TYPES),
        "engine_size_cc": _int_or_none(vehicle.get("engineVolume")),
        "location": (
            _field_text((vehicle.get("address") or {}).get("city"))
            or _field_text((vehicle.get("address") or {}).get("area"))
        ),
        "description": str(description).strip(),
        "asking_price_ils": float(price),
    }


def scrape(pages: int, delay: float, out_path: Path, dump_raw: bool) -> int:
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
    args = parser.parse_args()
    if args.pages < 1:
        parser.error("--pages must be >= 1")
    try:
        count = scrape(args.pages, args.delay, args.out, args.dump_raw)
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
