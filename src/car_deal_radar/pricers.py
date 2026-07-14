"""Price predictors: a local baseline plus skeletons for the course models.

Three kinds of pricers will eventually be compared, following the course arc:
a simple baseline (week 6 day 3), frontier LLMs (week 6 day 4) and a
fine-tuned open-source model (week 7).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from car_deal_radar.models import CarListing, PricePrediction


class BasePricer(ABC):
    """Common interface so different pricing models can be compared."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name used in predictions and reports."""

    @abstractmethod
    def predict(self, listing: CarListing) -> PricePrediction:
        """Predict the asking price in shekels for one listing."""


class BaselinePricer(BasePricer):
    """A tiny deterministic formula, no API, GPU or network needed.

    Plays the role of week 6 day 3's traditional-ML baselines: it exists to
    exercise the pipeline and give the LLMs something to beat, not to be
    accurate. Formula: a flat new-car reference price, exponential
    depreciation per year of age, a fuel-type factor, and a per-km adjustment
    for mileage above or below the expected total for the car's age.
    """

    def __init__(
        self,
        base_price_ils: float = 160_000.0,
        annual_depreciation: float = 0.88,
        fuel_factors: dict[str, float] | None = None,
        expected_km_per_year: int = 15_000,
        price_per_excess_km: float = 0.35,
        minimum_price_ils: float = 5_000.0,
        reference_year: int = 2026,
    ) -> None:
        self.base_price_ils = base_price_ils
        self.annual_depreciation = annual_depreciation
        self.fuel_factors = fuel_factors or {"hybrid": 1.1, "electric": 1.2, "diesel": 0.95}
        self.expected_km_per_year = expected_km_per_year
        self.price_per_excess_km = price_per_excess_km
        self.minimum_price_ils = minimum_price_ils
        # Fixed reference year (not "today") keeps predictions deterministic for tests.
        self.reference_year = reference_year

    @property
    def name(self) -> str:
        return "baseline"

    def predict(self, listing: CarListing) -> PricePrediction:
        age = max(self.reference_year - listing.year, 0)
        value = self.base_price_ils * (self.annual_depreciation**age)
        if listing.fuel_type:
            value *= self.fuel_factors.get(listing.fuel_type, 1.0)
        excess_km = listing.mileage_km - age * self.expected_km_per_year
        value -= excess_km * self.price_per_excess_km
        value = max(value, self.minimum_price_ils)
        return PricePrediction(
            listing_id=listing.listing_id,
            model_name=self.name,
            predicted_price_ils=round(value, 2),
            confidence=0.3,
        )


class FrontierLLMPricer(BasePricer):
    """Skeleton for the Phase 1 frontier-model pricer (week 6 day 4).

    The course calls frontier models with litellm.completion(model=...,
    messages=[...]) in week6/day4.ipynb, and with the openai client in
    week8/agents/frontier_agent.py — not with LangChain. predict() will be
    wired to one of those clients during Phase 1; build_prompt and
    parse_response are already usable and tested.
    """

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key

    @property
    def name(self) -> str:
        return self.model_name

    def build_prompt(self, listing: CarListing) -> str:
        """Build the prompt asking the model for a JSON price estimate in ILS."""
        return (
            "You are estimating prices of used cars for sale on the Israeli market.\n"
            "The advertised asking price is NOT provided. Estimate a plausible asking "
            "price for this listing in Israeli new shekels (ILS).\n"
            "Respond with a single JSON object only - no markdown, no text before or "
            "after the JSON - in exactly this format:\n"
            '{"predicted_price_ils": 85000, "confidence": 0.75}\n\n'
            f"{listing.to_model_text()}"
        )

    def parse_response(self, listing: CarListing, raw_response: str) -> PricePrediction:
        """Parse the model's JSON reply into a PricePrediction.

        Raises ValueError with the offending response when the format is invalid.
        """
        try:
            data = json.loads(raw_response.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{self.model_name} returned invalid JSON for listing "
                f"{listing.listing_id}: {raw_response!r}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"{self.model_name} returned a JSON {type(data).__name__} instead of an "
                f"object for listing {listing.listing_id}: {raw_response!r}"
            )
        price = data.get("predicted_price_ils")
        if isinstance(price, bool) or not isinstance(price, (int, float)) or price <= 0:
            raise ValueError(
                f"{self.model_name} returned a missing or non-positive price for "
                f"listing {listing.listing_id}: {raw_response!r}"
            )
        confidence = data.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            confidence = min(max(float(confidence), 0.0), 1.0)
        else:
            confidence = None
        return PricePrediction(
            listing_id=listing.listing_id,
            model_name=self.model_name,
            predicted_price_ils=float(price),
            confidence=confidence,
            raw_response=raw_response,
        )

    def predict(self, listing: CarListing) -> PricePrediction:
        raise NotImplementedError(
            "Connect this class to the frontier model provider used in the course."
        )


class FineTunedPricer(BasePricer):
    """Skeleton for the Phase 2 fine-tuned open-source pricer.

    Will be implemented during the phase matching week 7 of the course, using
    the same stack as the week 7 QLoRA Colab notebook (transformers, datasets,
    peft, trl, bitsandbytes). Those libraries are intentionally not imported
    while this class is a placeholder.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    @property
    def name(self) -> str:
        return f"fine-tuned:{self.model_path}"

    def predict(self, listing: CarListing) -> PricePrediction:
        raise NotImplementedError(
            "FineTunedPricer will be implemented in Phase 2 (week 7: QLoRA fine-tuning)."
        )
