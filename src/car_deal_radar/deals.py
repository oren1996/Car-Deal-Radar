"""Deal detection: find listings priced well below their estimated value.

DealFinder is a temporary stand-in for the week 8 agent system (Planning,
Scanner, Ensemble and Messaging agents). It follows the same core logic as
the course's PlanningAgent - estimate, discount = estimate - price, keep
deals above a threshold - and will only be split into separate components in
Phase 3, once that separation is actually useful.
"""

from __future__ import annotations

import statistics

from car_deal_radar.models import CarListing, DealCandidate
from car_deal_radar.pricers import BasePricer

# Thresholds for the human-review risk flags.
EXTREME_DISCOUNT_PERCENTAGE = 40.0
MIN_DESCRIPTION_CHARS = 20
LOW_MILEAGE_AGE_YEARS = 8
LOW_MILEAGE_KM_PER_YEAR = 5_000
EXTREME_MILEAGE_KM = 250_000
REFERENCE_YEAR = 2026

# Penalty per risk flag in the deal score heuristic.
RISK_FLAG_PENALTY = 5.0


class DealFinder:
    """Scores listings against model estimates and surfaces potential deals."""

    def __init__(
        self,
        pricers: list[BasePricer],
        minimum_discount_percentage: float = 10.0,
        minimum_confidence: float = 0.0,
    ) -> None:
        self.pricers = pricers
        self.minimum_discount_percentage = minimum_discount_percentage
        self.minimum_confidence = minimum_confidence

    def estimate_price(self, listing: CarListing) -> tuple[float, float | None]:
        """Aggregate the available pricers into one estimate.

        Pricers that raise NotImplementedError (not yet wired up) are skipped.
        For now the aggregate is a plain mean of prices and of the available
        confidences - no ensemble weighting yet (week 8 will refine this).
        """
        prices: list[float] = []
        confidences: list[float] = []
        for pricer in self.pricers:
            try:
                prediction = pricer.predict(listing)
            except NotImplementedError:
                continue
            prices.append(prediction.predicted_price_ils)
            if prediction.confidence is not None:
                confidences.append(prediction.confidence)
        if not prices:
            raise RuntimeError(
                f"No pricer could produce an estimate for listing {listing.listing_id}; "
                "all configured pricers are unimplemented"
            )
        confidence = statistics.mean(confidences) if confidences else None
        return statistics.mean(prices), confidence

    def detect_risk_flags(self, listing: CarListing, predicted_price_ils: float) -> list[str]:
        """Simple signals that a listing deserves manual checking.

        These are NOT fraud detection - just reasons for a human to look twice.
        """
        flags: list[str] = []
        if listing.asking_price_ils <= 0:
            flags.append("asking price is zero or negative")
        elif predicted_price_ils > 0:
            below = (predicted_price_ils - listing.asking_price_ils) / predicted_price_ils * 100
            if below > EXTREME_DISCOUNT_PERCENTAGE:
                flags.append(
                    f"asking price is more than {EXTREME_DISCOUNT_PERCENTAGE:.0f}% "
                    "below the estimated price"
                )
        if len(listing.description.strip()) < MIN_DESCRIPTION_CHARS:
            flags.append(f"description contains fewer than {MIN_DESCRIPTION_CHARS} characters")
        age = REFERENCE_YEAR - listing.year
        if age < 0 or listing.year < 1950:
            flags.append(f"implausible year: {listing.year}")
        elif (
            listing.mileage_km is not None
            and age >= LOW_MILEAGE_AGE_YEARS
            and listing.mileage_km < age * LOW_MILEAGE_KM_PER_YEAR
        ):
            flags.append("unusually low mileage for vehicle age")
        if listing.mileage_km is not None and listing.mileage_km > EXTREME_MILEAGE_KM:
            flags.append(f"extremely high mileage: {listing.mileage_km:,} km")
        if not listing.fuel_type and not listing.transmission and not listing.previous_owners:
            flags.append("key details missing (fuel type, transmission, owners)")
        return flags

    def score_listing(self, listing: CarListing) -> DealCandidate | None:
        """Return a DealCandidate when the discount clears the configured threshold."""
        predicted_price_ils, confidence = self.estimate_price(listing)
        if predicted_price_ils <= 0:
            return None
        discount_ils = predicted_price_ils - listing.asking_price_ils
        discount_percentage = (discount_ils / predicted_price_ils) * 100
        if discount_percentage < self.minimum_discount_percentage:
            return None
        if confidence is not None and confidence < self.minimum_confidence:
            return None
        risk_flags = self.detect_risk_flags(listing, predicted_price_ils)
        # Starting heuristic only: reward big, confident discounts and
        # penalize each risk flag. To be replaced by the Phase 3 ensemble.
        effective_confidence = confidence if confidence is not None else 0.5
        deal_score = discount_percentage * effective_confidence - len(risk_flags) * RISK_FLAG_PENALTY
        return DealCandidate(
            listing=listing,
            predicted_price_ils=predicted_price_ils,
            discount_ils=discount_ils,
            discount_percentage=discount_percentage,
            confidence=confidence,
            risk_flags=risk_flags,
            deal_score=deal_score,
        )

    def find_deals(self, listings: list[CarListing]) -> list[DealCandidate]:
        """Score every listing and return candidates sorted by deal score."""
        candidates = [self.score_listing(listing) for listing in listings]
        deals = [candidate for candidate in candidates if candidate is not None]
        deals.sort(key=lambda deal: deal.deal_score, reverse=True)
        return deals
