# Regime Portfolio: Adaptive Portfolio Allocation with Market Regime Detection

Project E448 | Stellenbosch University | Final Year Engineering Thesis

This repository holds the data pipeline for a study on regime-conditioned portfolio allocation. The question behind the project is whether detecting market regimes (bull, bear, crisis) lets a portfolio adapt its allocation to changing correlation structure, and whether that improves diversification when it matters most.

## Project structure

```
regime-portfolio/
├── README.md                   # This file
├── pyproject.toml              # Dependencies (pinned versions)
├── .gitignore
├── config/
│   └── data.yaml               # Configuration (universe, windows, FRED IDs)
├── src/regime/
│   ├── __init__.py
│   ├── config.py               # Configuration loading and validation
│   ├── data/
│   │   ├── ingest.py           # yfinance + FRED fetching with caching
│   │   ├── calendars.py        # NYSE calendar and coverage handling
│   │   ├── clean.py            # Validation, missingness, inception handling
│   │   ├── features.py         # Causal feature computation
│   │   └── pipeline.py         # Pipeline orchestration
│   ├── detection/              # Phase 3: regime detection
│   ├── allocation/             # Phase 4: regime-conditioned allocation
│   ├── backtest/               # Phase 5: walk-forward testing (stub)
│   └── utils/
│       ├── logging.py          # Logging setup
│       └── io.py               # Parquet I/O and caching
├── scripts/
│   └── build_dataset.py        # CLI entry point
├── data/
│   ├── raw/                    # Cached downloads (gitignored)
│   ├── interim/                # Aligned panels (gitignored)
│   └── processed/              # features.parquet + data_dictionary.md
├── notebooks/
│   └── 01_data_exploration.ipynb
├── tests/
│   ├── test_causality.py       # No-lookahead tests
│   ├── test_alignment.py       # Calendar alignment tests
│   └── test_inception.py       # Ragged-start handling tests
└── reports/
    ├── qa_report.md            # Pipeline QA report
    └── figures/                # Generated plots
```

## Quick start

### 1. Installation

```bash
cd regime-portfolio

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Unix

# Install dependencies
pip install -e ".[dev]"
```

### 2. Run the pipeline

```bash
python scripts/build_dataset.py --config config/data.yaml
```

This downloads ETF prices from yfinance (SPY, TLT, LQD, GLD, DBC, BIL) and macro series from FRED (VIXCLS, T10Y2Y, BAA10Y, NFCI), aligns everything to the NYSE trading calendar, computes the causal features (returns, volatility, correlations, drawdowns), and writes `data/processed/features.parquet` along with the QA reports.

### 3. Run tests

```bash
pytest tests/ -v
```

The causality test is the one to pay attention to: it checks that no feature at day t draws on information from after day t.

## Asset universe

| Ticker | Exposure | Approx. Inception |
|--------|----------|-------------------|
| SPY | US equities | 1993-01 |
| TLT | Long US Treasuries | 2002-07 |
| LQD | US investment-grade credit | 2002-07 |
| GLD | Gold | 2004-11 |
| DBC | Broad commodities | 2006-02 |
| BIL | 1-3 month T-bills (cash proxy) | 2007-05 |

BIL is the binding constraint (inception around May 2007), so the full six-asset panel only becomes complete from mid-2007. That window still spans the 2008 GFC, the 2020 COVID crash, and the 2022 rate-hike period.

## Features (all causal)

Every feature at day t uses only data available up to and including t:

| Feature Type | Description | Window |
|--------------|-------------|--------|
| `{ticker}_ret` | Daily log returns | 1 day |
| `{ticker}_vol21d` | Rolling annualised volatility | 21 days |
| `{ticker}_vol63d` | Rolling annualised volatility | 63 days |
| `{A}_{B}_corr63d` | Pairwise rolling correlation | 63 days |
| `avg_corr63d` | Average pairwise correlation | 63 days |
| `{ticker}_dd` | Rolling drawdown from peak | 252 days |
| `VIXCLS` | CBOE VIX | - |
| `T10Y2Y` | 10Y-2Y Treasury spread | - |
| `BAA10Y` | Baa-10Y credit spread | - |
| `NFCI` | Chicago Fed financial conditions | - |

## Configuration

Parameters live in `config/data.yaml`:

```yaml
features:
  windows:
    volatility_short: 21   # ~1 month
    volatility_long: 63    # ~3 months
    correlation: 63        # ~3 months
    drawdown: 252          # ~1 year

calendar:
  exchange: "NYSE"
  macro_max_staleness_days: 5  # Max forward-fill for macro
```

## Design decisions

### Causality

The causality test (`tests/test_causality.py`) is what backs the no-lookahead claim. For a sample of random dates t, it computes each feature on the full series and again on the series truncated at t, and checks the value at t matches. If the two disagree, a feature is reaching into the future, and the test fails.

### Prices are not forward-filled

Price gaps from halts or missing data are left as NaN. Forward-filling them would invent returns that never happened.

### Macro data is forward-filled, with a limit

Macro series are published with a lag, so carrying the last value forward up to five trading days is reasonable. Beyond that the value goes stale and is left missing.

### Pre-inception NaNs are kept

The period before an asset existed is not forward-filled. That is a different case from a gap inside the asset's coverage, and the two are handled separately.

### NYSE trading calendar

The master index comes from `pandas-market-calendars` rather than naive business days, so early closes and holidays are handled correctly.

## Known limitations

### FRED vintage data

FRED series here use latest-vintage values, not point-in-time. Economic data is revised after first publication, so this introduces a mild lookahead bias. The proper fix for production research is ALFRED (Archival FRED), which serves vintage data. For an undergraduate thesis the limitation is usually acceptable as long as it is documented, which it is, both in the log and in the data dictionary.

### ETF inception

Full six-asset coverage only starts in mid-2007 because of BIL. Earlier dates have fewer assets available.

## Extending to the JSE

The universe and exchange are config-driven so the same code can later cover JSE (South African) assets:

```yaml
# Future JSE configuration (not implemented)
universe:
  name: "JSE_BASKET"
  currency: "ZAR"
  exchange: "JSE"
  assets:
    SATRIX40:
      description: "Satrix 40 ETF"
      approx_inception: "2000-11-01"
```

Calendar and currency would need handling, but the feature code should not have to change.

## Regime detection (Phase 3)

Phase 3 labels each trading day as **calm**, **volatile** or **crisis** from the Phase 2 feature matrix, under the same causal, point-in-time discipline as the features themselves. It implements three detector families on a shared `RegimeDetector` interface, plus the walk-forward labelling harness:

- **HMM** — a Gaussian hidden Markov model (the probabilistic baseline), with a hand-written forward filter for causal inference.
- **Jump model** — temporal clustering with a jump penalty (Nystrup; Shu and Mulvey); a forward dynamic program gives the causal labels. Produces more persistent regimes.
- **Change-point** — assumption-light segmentation with `ruptures`, with each segment labelled by its nearest severity-ordered prototype.

```bash
python scripts/run_detection.py --method hmm           # one detector
python scripts/run_detection.py --method all           # all three + a comparison
```

Each detector standardises a small feature set (market return, market volatility, average pairwise correlation, VIX) with a causal expanding z-score, then labels every day with a walk-forward loop: refit on an expanding window at each refit point and filter forward in between, so no future information reaches any past label. Outputs go to `data/processed/regimes_{method}.parquet` and `reports/detection/` (per-detector reports, a comparison, and figures).

Key design points:

- **Causal by construction.** The filtered label at day t uses only data up to t. `tests/test_detection.py` verifies this by truncation for all three detectors, the same way the feature causality test does.
- **States are severity-ordered.** A model's internal states are unlabelled, so they are sorted by their stress-feature means and mapped to calm < volatile < crisis. Labels stay comparable across refits and across detectors.
- **Filtered vs smoothed.** Filtered (causal) inference is used for labelling; full-sample smoothed states are available for plotting and comparison only.

On the cross-asset basket the crisis regime concentrates in the 2008, 2020 and 2022 stress periods, far above the calm-period baseline, and the regimes are persistent rather than rapidly switching. The detectors differ in character as expected: the HMM is the most reactive, the jump model the most persistent, and change-point the most assumption-light.

## Regime-conditioned allocation (Phase 4)

Phase 4 turns the regime labels into portfolio weights. Following the project's methodological direction, the allocation *objective* changes with the regime rather than staying fixed: variance is a reasonable risk measure when returns are roughly symmetric, but it understates the fat-tailed losses of a crisis, where a tail measure fits better.

Five allocators share one `Allocator` interface, all long-only and fully invested (weights are non-negative and sum to one, so the BIL cash holding absorbs any de-risking):

- **Mean-variance** — the Markowitz QP, with Ledoit-Wolf shrinkage covariance and a shrunk mean (a minimum-variance mode drops the mean term entirely).
- **Risk parity** — equal risk contribution, solved by coordinate descent; uses no expected-return estimate.
- **Minimum-CVaR** — the Rockafellar-Uryasev tail-risk linear program at the 95% level; this is the crisis objective.
- **HRP** — hierarchical risk parity (López de Prado), a regime-blind robustness benchmark.
- **Equal weight** — the 1/N baseline.

The `RegimeSwitchingAllocator` dispatches by regime: mean-variance in calm, risk parity in volatile, minimum-CVaR in crisis. A causal walk-forward loop re-estimates on a trailing window at each monthly rebalance (and on a regime change) and holds the weights in between, so the weight on day t uses only returns up to t.

```bash
python scripts/run_allocation.py --regimes hmm          # switch on the HMM labels
python scripts/run_allocation.py --regimes jump --no-baselines
```

Outputs go to `data/processed/weights_*.parquet` and `reports/allocation/`. On the cross-asset basket the strategy behaves as intended: in calm it runs about 47% equities, in volatile it cuts equity and moves toward cash, and in crisis it holds almost entirely cash. The static minimum-CVaR baseline collapses to cash in every period, which is exactly why CVaR is used only as the crisis objective and not as a standalone strategy.

Each optimiser is written from its primal form with scipy (HiGHS for the CVaR LP, SLSQP for the mean-variance QP, coordinate descent for risk parity, scipy hierarchy for HRP), so there is no extra solver dependency and the formulation stays explicit. The weight series stops here; turning weights into returns, costs and risk-adjusted performance is Phase 5.

## Later phases

- **Phase 5:** walk-forward backtesting and evaluation. Apply the weights to forward returns with transaction costs, then measure Sharpe ratio and maximum drawdown against the baselines with significance testing (deflated Sharpe, probability of backtest overfitting).

A stub module sits at `src/regime/backtest/`.

## Reproducibility

Dependencies are pinned in `pyproject.toml`, raw downloads are cached in `data/raw/` with date stamps, and the features are deterministic (no random seeds needed), so a re-run from cache works offline.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only causality tests
pytest tests/test_causality.py -v

# Run with coverage
pytest tests/ --cov=regime --cov-report=html
```

## Oral exam notes

Design choices worth being ready to justify:

1. Why yfinance? Free, reliable, and it returns adjusted prices automatically.
2. Why pandas-datareader for FRED? No API key required; fredapi stays optional.
3. Why a 63-day correlation window? About three months, which trades off responsiveness against stability.
4. Why no standardization here? Raw features keep their units and stay interpretable; any standardization can be applied downstream where it is actually needed.
5. Why forward-fill macro but not prices? Macro has a publication lag; prices should reflect what was actually tradable.

## License

MIT

## Author

Cian Robertson | Stellenbosch University | 2026
