# Car Deal Radar

Spot potentially undervalued used-car listings on the Israeli market by comparing
advertised asking prices (in shekels) against model-estimated prices.

This is an automotive adaptation of the capstone project from **weeks 6-8 of
Ed Donner's "LLM Engineering" course**. The original project predicts Amazon
product prices with frontier models (week 6), fine-tunes an open-source model
with QLoRA to compete with them (week 7), and builds an autonomous multi-agent
system that finds deals and sends push notifications (week 8). Car Deal Radar
follows the exact same arc, but the "products" are used cars in Israel — a
domain with richer structure (year, mileage, owners, fuel type) and prices in
a different currency, which makes it a more interesting adaptation than
re-running the same dataset.

The current state is a deliberately small, fully offline skeleton: the data
model, a naive baseline pricer, evaluation, and deal detection all work
end-to-end on bundled fictional sample data. The frontier and fine-tuned
pricers are documented placeholders to be filled in during phases 1 and 2.

## Course alignment

| Phase   | Course equivalent                 | Car Deal Radar                        |
| ------- | --------------------------------- | ------------------------------------- |
| Phase 1 | Week 6 frontier price prediction  | Predict car listing prices            |
| Phase 2 | Week 7 QLoRA fine-tuning          | Fine-tune an open-source car pricer   |
| Phase 3 | Week 8 autonomous agents          | Scan listings and identify deals      |

Concept mapping from the course code to this project:

| Course (weeks 6-8)                   | Car Deal Radar          |
| ------------------------------------ | ----------------------- |
| `Item` (`pricer/items.py`)           | `CarListing`            |
| `Tester` / `evaluate` (`evaluator.py`) | `evaluator.py` functions |
| Frontier pricing (week 6 day 4)      | `FrontierLLMPricer`     |
| Fine-tuned specialist (week 7)       | `FineTunedPricer`       |
| `Opportunity` (`agents/deals.py`)    | `DealCandidate`         |
| `PlanningAgent` + agents (week 8)    | `DealFinder`            |

`DealFinder` intentionally bundles what week 8 splits into Scanner, Ensemble,
Planning and Messaging agents. It will be separated into those components
gradually during Phase 3, only when each separation becomes useful.

## Architecture

```
data/sample_cars.json      7 fictional listings (clearly fake, not market data)
src/car_deal_radar/
    models.py              CarListing, PricePrediction, EvaluationResult, DealCandidate
    data.py                load, validate, train/test split, fine-tuning records
    pricers.py             BasePricer, BaselinePricer, FrontierLLMPricer*, FineTunedPricer*
    evaluator.py           per-prediction evaluation and aggregate metrics
    deals.py               DealFinder: estimate, risk flags, scoring
    main.py                offline end-to-end demo
tests/test_core.py         fast offline tests
```

\* skeleton only — `predict()` raises `NotImplementedError` until its phase.

## Installation

Requires Python 3.11+ (same floor as the official course repo).

With [uv](https://docs.astral.sh/uv/), the package manager used by the official
course repo (which ships a `uv.lock`, as does this project):

```bash
uv sync --extra dev
```

Or with plain pip:

```bash
pip install -e ".[dev]"
```

## Running

```bash
uv run python -m car_deal_radar.main   # with uv
python -m car_deal_radar.main          # with pip
# or, via the console script:
car-deal-radar
```

The demo runs entirely offline: no API keys, internet access, GPU or database.
Copy `.env.example` to `.env` and add keys only when Phase 1 starts.

## Tests

```bash
uv run pytest   # or just: pytest
```

## Roadmap

The detailed, step-by-step plan to complete phases 1-3 lives in
[ROADMAP.md](ROADMAP.md).

## Important limitation

- The system currently predicts a listing's **asking price**, which is not
  necessarily the true final sale price.
- A very low price does **not** automatically mean a car is a good deal — it
  may reflect damage, accident history, or other problems not in the text.
- `risk_flags` are simple heuristics that suggest a human should look twice;
  they are **not** fraud detection.
- Every listing surfaced by this tool must be verified manually.
- The bundled sample prices are fictional and do not represent the actual
  Israeli market.

## Dependency policy

This project uses the same libraries and approaches as the official course,
verified against the `pyproject.toml` and source files of
[ed-donner/llm_engineering](https://github.com/ed-donner/llm_engineering):

- `python-dotenv` — the only runtime dependency of the skeleton; every course
  notebook loads keys with `load_dotenv(override=True)`.
- Optional `frontier` group (`openai`, `anthropic`, `litellm`) — week 6 day 4
  calls frontier models via `litellm.completion`; the week 8 agents use the
  `openai` client directly. No LangChain for this project, matching the course.
- Optional `finetuning` group (`datasets`, `torch`, `transformers`) — pinned
  as in the official pyproject. Week 7's QLoRA training runs in a Google Colab
  notebook that additionally installs `peft`, `trl` and `bitsandbytes`; those
  will be added when Phase 2 is implemented.
- Optional `notifications` group (`requests`) — week 8's `MessagingAgent`
  sends Pushover notifications with `requests.post`.
- `pytest` (dev group) — **not** from the course repo; needed only to run this
  project's tests.

One documented deviation: the current course edition models `Item` and `Deal`
with pydantic `BaseModel`, while this skeleton uses stdlib dataclasses to stay
dependency-free. This may switch to pydantic in Phase 3 when structured
outputs (as in `scanner_agent.py`) are introduced.

## Official references

- Udemy course: <https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/>
- Official repository: <https://github.com/ed-donner/llm_engineering>
