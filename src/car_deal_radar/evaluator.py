"""Evaluation of pricers against advertised asking prices.

Adapts week 6's pricer/evaluator.py, whose central metric is the average
absolute error between guess and truth (plus MSE/r² charts we skip here).
Only the standard library is needed for these metrics.
"""

from __future__ import annotations

import statistics

from car_deal_radar.models import CarListing, EvaluationResult, PricePrediction
from car_deal_radar.pricers import BasePricer


def evaluate_prediction(listing: CarListing, prediction: PricePrediction) -> EvaluationResult:
    """Compare one prediction with the listing's asking price."""
    if listing.asking_price_ils <= 0:
        raise ValueError(
            f"Cannot evaluate listing {listing.listing_id}: asking price is "
            f"{listing.asking_price_ils} (must be positive)"
        )
    absolute_error_ils = abs(prediction.predicted_price_ils - listing.asking_price_ils)
    percentage_error = (absolute_error_ils / listing.asking_price_ils) * 100
    return EvaluationResult(
        listing_id=listing.listing_id,
        model_name=prediction.model_name,
        actual_price_ils=listing.asking_price_ils,
        predicted_price_ils=prediction.predicted_price_ils,
        absolute_error_ils=absolute_error_ils,
        percentage_error=percentage_error,
    )


def evaluate_pricer(pricer: BasePricer, listings: list[CarListing]) -> list[EvaluationResult]:
    """Run a pricer over every listing and collect evaluation results.

    Errors from the pricer propagate with the listing id added for context.
    """
    results = []
    for listing in listings:
        try:
            prediction = pricer.predict(listing)
        except NotImplementedError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Pricer '{pricer.name}' failed on listing {listing.listing_id}: {exc}"
            ) from exc
        results.append(evaluate_prediction(listing, prediction))
    return results


def calculate_metrics(results: list[EvaluationResult]) -> dict[str, float]:
    """Aggregate evaluation results into summary metrics."""
    if not results:
        raise ValueError("Cannot calculate metrics from an empty list of results")
    absolute_errors = [r.absolute_error_ils for r in results]
    percentage_errors = [r.percentage_error for r in results]
    return {
        "mae_ils": statistics.mean(absolute_errors),
        "mape_percentage": statistics.mean(percentage_errors),
        "median_absolute_error_ils": statistics.median(absolute_errors),
        "count": float(len(results)),
    }
