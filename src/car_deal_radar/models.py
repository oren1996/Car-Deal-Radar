"""Core data objects for Car Deal Radar.

These adapt the course concepts to used cars:
Item (week 6/7) -> CarListing, Opportunity (week 8) -> DealCandidate.

The current edition of the course represents Item and Deal with pydantic
BaseModel; plain dataclasses are used here instead to keep the skeleton
dependency-free. Switching to pydantic can be revisited in Phase 3 when
structured outputs are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Wording for previous-owner counts in the model-facing text.
_OWNER_PHRASES = {1: "first owner", 2: "second owner", 3: "third owner"}


@dataclass
class CarListing:
    """A single used-car listing, raw or normalized."""

    listing_id: str
    source: str
    url: str | None
    make: str
    model: str
    year: int
    mileage_km: int
    ownership_type: str | None
    previous_owners: int | None
    transmission: str | None
    fuel_type: str | None
    engine_size_cc: int | None
    location: str | None
    description: str
    asking_price_ils: float

    def to_model_text(self) -> str:
        """Build the textual description of this car that is sent to an LLM.

        Mirrors how the course's Item exposes a text summary to the model.
        Deterministic, skips missing optional fields, and deliberately
        excludes the asking price so a model can be asked to predict it.
        """
        parts: list[str] = [
            f"{self.make} {self.model}",
            f"year {self.year}",
            f"{self.mileage_km:,} km",
        ]
        if self.fuel_type:
            parts.append(self.fuel_type)
        if self.transmission:
            parts.append(self.transmission)
        if self.engine_size_cc:
            parts.append(f"{self.engine_size_cc:,} cc")
        if self.previous_owners:
            parts.append(
                _OWNER_PHRASES.get(self.previous_owners, f"{self.previous_owners} previous owners")
            )
        if self.ownership_type:
            parts.append(f"{self.ownership_type} ownership")
        if self.location:
            parts.append(f"located in {self.location}")
        text = ", ".join(parts) + "."
        if self.description.strip():
            text += f" Seller description: {self.description.strip()}"
        return text


@dataclass
class PricePrediction:
    """A price estimate for one listing produced by one model."""

    listing_id: str
    model_name: str
    predicted_price_ils: float
    confidence: float | None = None
    raw_response: str | None = None


@dataclass
class EvaluationResult:
    """Comparison of a prediction against the advertised asking price."""

    listing_id: str
    model_name: str
    actual_price_ils: float
    predicted_price_ils: float
    absolute_error_ils: float
    percentage_error: float


@dataclass
class DealCandidate:
    """A listing whose asking price sits well below the estimated price."""

    listing: CarListing
    predicted_price_ils: float
    discount_ils: float
    discount_percentage: float
    confidence: float | None
    risk_flags: list[str] = field(default_factory=list)
    deal_score: float = 0.0
