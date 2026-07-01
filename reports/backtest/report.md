# Backtest evaluation (cost 10 bps)

- Assets: SPY, TLT, LQD, GLD, DBC, BIL
- Evaluated window: 2008-10-03 to 2026-06-26 (4459 days)

## Performance (full sample, net of cost)

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | Calmar | Turnover | Deflated Sharpe |
|----------|-------------|----------|--------|--------|--------|----------|-----------------|
| cvarfloor | 5.1% | 6.0% | 0.65 | 14.3% | 0.36 | 3.83 | 0.93 |
| defcrisis minvar | 5.4% | 5.7% | 0.74 | 13.4% | 0.41 | 3.64 | 0.96 |
| defcrisis rp | 5.9% | 6.2% | 0.77 | 12.0% | 0.50 | 2.80 | 0.97 |
| regime-switch (changepoint) | 6.1% | 7.7% | 0.66 | 12.2% | 0.50 | 3.03 | 0.93 |
| regime-switch (hmm) | 4.8% | 5.6% | 0.66 | 14.3% | 0.34 | 4.22 | 0.93 |
| regime-switch (jump) | 6.1% | 8.1% | 0.62 | 13.4% | 0.45 | 2.43 | 0.90 |
| equal weight | 6.0% | 7.5% | 0.64 | 14.8% | 0.40 | 0.00 | 0.92 |
| hrp | 2.3% | 1.9% | 0.58 | 5.3% | 0.44 | 0.28 | 0.88 |
| mean variance | 8.4% | 9.6% | 0.76 | 13.9% | 0.60 | 2.18 | 0.97 |
| min cvar | 1.2% | 0.4% | 0.09 | 1.0% | 1.30 | 0.02 | 0.18 |
| risk parity | 4.1% | 4.3% | 0.67 | 9.3% | 0.44 | 0.19 | 0.94 |

## Return within stress windows (net of cost)

| Strategy | 2008 GFC | 2020 COVID | 2022 tightening |
|---|---|---|---|
| cvarfloor | -3.1% | -8.6% | -1.3% |
| defcrisis minvar | +0.6% | -0.7% | -5.9% |
| defcrisis rp | -0.1% | -1.7% | -8.2% |
| regime-switch (changepoint) | -3.9% | -6.3% | +0.4% |
| regime-switch (hmm) | -3.1% | +0.3% | -4.5% |
| regime-switch (jump) | -3.0% | -11.4% | -4.3% |
| equal weight | -5.6% | -2.7% | -11.3% |
| hrp | +1.9% | +0.1% | -4.6% |
| mean variance | -4.2% | +2.0% | +0.5% |
| min cvar | -0.6% | +0.2% | +0.7% |
| risk parity | -0.9% | -1.7% | -8.1% |

## Significance / overfitting

- Strategies compared: 11
- Probability of backtest overfitting (CSCV, 16 blocks): 0.04
- Deflated Sharpe ratio is the last column of the performance table (probability the Sharpe beats the expected best of the trials).

## Cost sensitivity (Sharpe ratio)

| Strategy | 0 bps | 5 bps | 10 bps | 20 bps |
|---|---|---|---|---|
| regime-switch (changepoint) | 0.73 | 0.70 | 0.66 | 0.58 |
| regime-switch (hmm) | 0.81 | 0.74 | 0.66 | 0.50 |
| regime-switch (jump) | 0.68 | 0.65 | 0.62 | 0.56 |
| equal weight | 0.56 | 0.56 | 0.56 | 0.56 |
| risk parity | 0.59 | 0.58 | 0.58 | 0.57 |
