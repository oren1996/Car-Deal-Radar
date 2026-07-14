"""Core tests for the Car Deal Radar skeleton.

Fast, deterministic, and independent of any API, GPU or network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from car_deal_radar.data import load_car_listings, validate_listing
from car_deal_radar.deals import DealFinder
from car_deal_radar.evaluator import calculate_metrics, evaluate_prediction
from car_deal_radar.models import CarListing, PricePrediction
from car_deal_radar.pricers import BaselinePricer, FrontierLLMPricer

SAMPLE_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_cars.json"


def make_listing(**overrides: object) -> CarListing:
    """A valid listing with sensible defaults, overridable per test."""
    fields: dict = {
        "listing_id": "test-001",
        "source": "test",
        "url": None,
        "make": "Toyota",
        "model": "Corolla",
        "year": 2020,
        "mileage_km": 82000,
        "ownership_type": "private",
        "previous_owners": 2,
        "transmission": "automatic",
        "fuel_type": "hybrid",
        "engine_size_cc": 1800,
        "location": "Haifa",
        "description": "Well maintained, full service history.",
        "asking_price_ils": 80000.0,
    }
    fields.update(overrides)
    return CarListing(**fields)


def test_to_model_text_contains_make_model_year() -> None:
    text = make_listing().to_model_text()
    assert "Toyota" in text
    assert "Corolla" in text
    assert "2020" in text


def test_to_model_text_excludes_asking_price() -> None:
    listing = make_listing(asking_price_ils=123456.0)
    text = listing.to_model_text()
    assert "123456" not in text
    assert "123,456" not in text


def test_load_car_listings_from_sample_file() -> None:
    listings = load_car_listings(SAMPLE_DATA_PATH)
    assert len(listings) >= 6
    assert all(isinstance(listing, CarListing) for listing in listings)


def test_validate_listing_detects_impossible_year() -> None:
    problems = validate_listing(make_listing(year=1890))
    assert any("year" in problem for problem in problems)


def test_evaluate_prediction_computes_absolute_error() -> None:
    listing = make_listing(asking_price_ils=80000.0)
    prediction = PricePrediction(
        listing_id=listing.listing_id, model_name="test-model", predicted_price_ils=75000.0
    )
    result = evaluate_prediction(listing, prediction)
    assert result.absolute_error_ils == 5000.0
    assert result.percentage_error == pytest.approx(6.25)


def test_calculate_metrics_computes_mae() -> None:
    listing_a = make_listing(listing_id="a", asking_price_ils=100000.0)
    listing_b = make_listing(listing_id="b", asking_price_ils=50000.0)
    results = [
        evaluate_prediction(
            listing_a,
            PricePrediction(listing_id="a", model_name="m", predicted_price_ils=110000.0),
        ),
        evaluate_prediction(
            listing_b,
            PricePrediction(listing_id="b", model_name="m", predicted_price_ils=48000.0),
        ),
    ]
    metrics = calculate_metrics(results)
    assert metrics["mae_ils"] == pytest.approx((10000.0 + 2000.0) / 2)
    assert metrics["count"] == 2.0


def test_deal_finder_detects_underpriced_listing() -> None:
    pricer = BaselinePricer()
    listing = make_listing()
    estimate = pricer.predict(listing).predicted_price_ils
    underpriced = make_listing(listing_id="cheap", asking_price_ils=estimate * 0.6)
    deals = DealFinder(pricers=[pricer]).find_deals([underpriced])
    assert len(deals) == 1
    assert deals[0].listing.listing_id == "cheap"
    assert deals[0].discount_percentage == pytest.approx(40.0)


def test_deal_finder_returns_nothing_when_price_above_estimate() -> None:
    pricer = BaselinePricer()
    listing = make_listing()
    estimate = pricer.predict(listing).predicted_price_ils
    overpriced = make_listing(listing_id="pricey", asking_price_ils=estimate * 1.5)
    assert DealFinder(pricers=[pricer]).find_deals([overpriced]) == []


def test_parse_response_accepts_valid_json() -> None:
    pricer = FrontierLLMPricer(model_name="test-frontier")
    listing = make_listing()
    raw = json.dumps({"predicted_price_ils": 85000, "confidence": 0.75})
    prediction = pricer.parse_response(listing, raw)
    assert prediction.predicted_price_ils == 85000.0
    assert prediction.confidence == 0.75
    assert prediction.listing_id == listing.listing_id


def test_parse_response_rejects_invalid_response() -> None:
    pricer = FrontierLLMPricer(model_name="test-frontier")
    listing = make_listing()
    with pytest.raises(ValueError):
        pricer.parse_response(listing, "The price is around 85,000 shekels.")
    with pytest.raises(ValueError):
        pricer.parse_response(listing, json.dumps({"predicted_price_ils": -5}))
