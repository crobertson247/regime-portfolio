# Backtest evaluation (cost 10 bps)

- Assets: SPY, TLT, LQD, GLD, DBC, BIL
- Evaluated window: 2008-10-03 to 2026-06-26 (4459 days)

## Performance (full sample, net of cost)

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | Turnover | Deflated Sharpe |
|----------|-------------|----------|--------|--------|----------|-----------------|
| regime-switch (changepoint) | 6.1% | 7.7% | 0.66 | 12.2% | 3.03 | 0.93 |
| regime-switch (hmm) | 4.8% | 5.6% | 0.66 | 14.3% | 4.22 | 0.93 |
| regime-switch (jump) | 6.1% | 8.1% | 0.62 | 13.4% | 2.43 | 0.90 |
| equal weight | 5.4% | 7.6% | 0.56 | 20.2% | 0.03 | 0.93 |
| hrp | 2.2% | 1.9% | 0.52 | 5.3% | 0.31 | 0.88 |
| mean variance | 7.8% | 9.8% | 0.69 | 21.2% | 2.20 | 0.97 |
| min cvar | 1.2% | 0.5% | -0.06 | 1.4% | 0.05 | 0.18 |
| risk parity | 3.8% | 4.3% | 0.58 | 10.6% | 0.22 | 0.94 |

## Return within stress windows (net of cost)

| Strategy | 2008 GFC | 2020 COVID | 2022 tightening |
|---|---|---|---|
| regime-switch (changepoint) | -3.9% | -6.3% | +0.4% |
| regime-switch (hmm) | -3.1% | +0.3% | -4.5% |
| regime-switch (jump) | -3.0% | -11.4% | -4.3% |
| equal weight | -11.0% | -2.7% | -11.3% |
| hrp | +1.0% | +0.1% | -4.6% |
| mean variance | -8.1% | +2.0% | +0.5% |
| min cvar | -0.7% | +0.2% | +0.7% |
| risk parity | -4.6% | -1.7% | -8.1% |

## Significance / overfitting

- Strategies compared: 8
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
