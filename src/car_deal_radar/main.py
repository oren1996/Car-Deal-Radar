"""Runnable demo of the full pipeline on the bundled sample data.

Works fully offline: no API key, internet access, GPU or database needed.
Run with: python -m car_deal_radar.main
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from car_deal_radar.data import load_car_listings, validate_listing
from car_deal_radar.deals import DealFinder
from car_deal_radar.evaluator import calculate_metrics, evaluate_pricer
from car_deal_radar.pricers import BaselinePricer

SAMPLE_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_cars.json"


def main() -> None:
    """Load sample listings, evaluate the baseline pricer and report deals."""
    # Same convention as every course notebook; harmless when no .env exists.
    load_dotenv(override=True)

    listings = load_car_listings(SAMPLE_DATA_PATH)
    print(f"Loaded {len(listings)} listings from {SAMPLE_DATA_PATH.name}\n")

    print("Validation:")
    clean = True
    for listing in listings:
        problems = validate_listing(listing)
        for problem in problems:
            clean = False
            print(f"  {listing.listing_id}: {problem}")
    if clean:
        print("  no problems detected")

    pricer = BaselinePricer()
    results = evaluate_pricer(pricer, listings)
    metrics = calculate_metrics(results)
    print(f"\nBaseline pricer evaluation on {int(metrics['count'])} listings:")
    print(f"  MAE:                   ₪{metrics['mae_ils']:,.0f}")
    print(f"  MAPE:                  {metrics['mape_percentage']:.1f}%")
    print(f"  Median absolute error: ₪{metrics['median_absolute_error_ils']:,.0f}")

    finder = DealFinder(pricers=[pricer])
    deals = finder.find_deals(listings)
    print(f"\nPotential deals found: {len(deals)}")
    for deal in deals:
        car = deal.listing
        print(f"\n  {car.make} {car.model} {car.year} ({car.listing_id})")
        print(f"    asking:    ₪{car.asking_price_ils:,.0f}")
        print(f"    estimated: ₪{deal.predicted_price_ils:,.0f}")
        print(f"    discount:  ₪{deal.discount_ils:,.0f} ({deal.discount_percentage:.1f}%)")
        print(f"    score:     {deal.deal_score:.1f}")
        for flag in deal.risk_flags:
            print(f"    risk flag: {flag}")
    print("\nReminder: estimates target asking prices, not sale prices - verify every listing manually.")


if __name__ == "__main__":
    main()
