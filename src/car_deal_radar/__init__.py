"""Car Deal Radar: spot undervalued used-car listings on the Israeli market.

An automotive adaptation of the capstone project from weeks 6-8 of
Ed Donner's "LLM Engineering" course (https://github.com/ed-donner/llm_engineering).
"""

from car_deal_radar.models import CarListing, DealCandidate, EvaluationResult, PricePrediction

__version__ = "0.1.0"

__all__ = [
    "CarListing",
    "PricePrediction",
    "EvaluationResult",
    "DealCandidate",
    "__version__",
]
