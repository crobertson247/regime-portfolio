# Data dictionary

Generated: 2026-06-28T19:54:40.047195

## Dataset overview

- Coverage period: 2007-05-30 to 2026-06-26
- Total features: 39
- Asset universe: SPY, TLT, LQD, GLD, DBC, BIL
- Exchange calendar: NYSE

## Notes

### Causality

Each feature at day t is computed from data up to and including day t, so the
dataset carries no lookahead bias.

### FRED data

The macro series use FRED's latest-vintage values rather than point-in-time data.
FRED revises its series after first publication, so this leaves a mild lookahead
bias. Vintage (point-in-time) data is available from ALFRED if it is ever needed;
this build uses the latest-vintage values and notes the limitation here.

### Forward-fill policy

- Prices are not forward-filled. A gap means a halt or missing data, and filling
  it would invent a return that never happened.
- Macro series are forward-filled up to 5 trading days to cover publication lag.

### Companion files

- `features.parquet` (described by this dictionary): the modelling features.
- `prices.parquet`: adjusted prices plus simple returns (`_simpleret`). The backtest
  uses these for portfolio P&L, since a portfolio return is the weighted sum of its
  assets' simple returns, which is not true of log returns.

## Feature descriptions

| Column | Type | Description | Window | Unit | Source |
|--------|------|-------------|--------|------|--------|
| SPY_ret | return | Daily log return for SPY | 1 | log return (decimal) | yfinance |
| TLT_ret | return | Daily log return for TLT | 1 | log return (decimal) | yfinance |
| LQD_ret | return | Daily log return for LQD | 1 | log return (decimal) | yfinance |
| GLD_ret | return | Daily log return for GLD | 1 | log return (decimal) | yfinance |
| DBC_ret | return | Daily log return for DBC | 1 | log return (decimal) | yfinance |
| BIL_ret | return | Daily log return for BIL | 1 | log return (decimal) | yfinance |
| SPY_vol21d | volatility | 21-day rolling annualised volatility for SPY | 21 | annualised standard deviation | computed from returns |
| TLT_vol21d | volatility | 21-day rolling annualised volatility for TLT | 21 | annualised standard deviation | computed from returns |
| LQD_vol21d | volatility | 21-day rolling annualised volatility for LQD | 21 | annualised standard deviation | computed from returns |
| GLD_vol21d | volatility | 21-day rolling annualised volatility for GLD | 21 | annualised standard deviation | computed from returns |
| DBC_vol21d | volatility | 21-day rolling annualised volatility for DBC | 21 | annualised standard deviation | computed from returns |
| BIL_vol21d | volatility | 21-day rolling annualised volatility for BIL | 21 | annualised standard deviation | computed from returns |
| SPY_vol63d | volatility | 63-day rolling annualised volatility for SPY | 63 | annualised standard deviation | computed from returns |
| TLT_vol63d | volatility | 63-day rolling annualised volatility for TLT | 63 | annualised standard deviation | computed from returns |
| LQD_vol63d | volatility | 63-day rolling annualised volatility for LQD | 63 | annualised standard deviation | computed from returns |
| GLD_vol63d | volatility | 63-day rolling annualised volatility for GLD | 63 | annualised standard deviation | computed from returns |
| DBC_vol63d | volatility | 63-day rolling annualised volatility for DBC | 63 | annualised standard deviation | computed from returns |
| BIL_vol63d | volatility | 63-day rolling annualised volatility for BIL | 63 | annualised standard deviation | computed from returns |
| SPY_TLT_corr63d | correlation | 63-day rolling correlation between SPY and TLT | 63 | correlation coefficient (-1 to 1) | computed from returns |
| SPY_LQD_corr63d | correlation | 63-day rolling correlation between SPY and LQD | 63 | correlation coefficient (-1 to 1) | computed from returns |
| SPY_GLD_corr63d | correlation | 63-day rolling correlation between SPY and GLD | 63 | correlation coefficient (-1 to 1) | computed from returns |
| SPY_DBC_corr63d | correlation | 63-day rolling correlation between SPY and DBC | 63 | correlation coefficient (-1 to 1) | computed from returns |
| TLT_LQD_corr63d | correlation | 63-day rolling correlation between TLT and LQD | 63 | correlation coefficient (-1 to 1) | computed from returns |
| TLT_GLD_corr63d | correlation | 63-day rolling correlation between TLT and GLD | 63 | correlation coefficient (-1 to 1) | computed from returns |
| TLT_DBC_corr63d | correlation | 63-day rolling correlation between TLT and DBC | 63 | correlation coefficient (-1 to 1) | computed from returns |
| LQD_GLD_corr63d | correlation | 63-day rolling correlation between LQD and GLD | 63 | correlation coefficient (-1 to 1) | computed from returns |
| LQD_DBC_corr63d | correlation | 63-day rolling correlation between LQD and DBC | 63 | correlation coefficient (-1 to 1) | computed from returns |
| GLD_DBC_corr63d | correlation | 63-day rolling correlation between GLD and DBC | 63 | correlation coefficient (-1 to 1) | computed from returns |
| avg_corr63d | correlation | Average pairwise correlation across all assets | 63 | correlation coefficient (-1 to 1) | computed from pairwise correlations |
| SPY_dd | drawdown | Rolling drawdown from 252-day peak for SPY | 252 | decimal (0 = at peak, -0.1 = 10% below peak) | computed from prices |
| TLT_dd | drawdown | Rolling drawdown from 252-day peak for TLT | 252 | decimal (0 = at peak, -0.1 = 10% below peak) | computed from prices |
| LQD_dd | drawdown | Rolling drawdown from 252-day peak for LQD | 252 | decimal (0 = at peak, -0.1 = 10% below peak) | computed from prices |
| GLD_dd | drawdown | Rolling drawdown from 252-day peak for GLD | 252 | decimal (0 = at peak, -0.1 = 10% below peak) | computed from prices |
| DBC_dd | drawdown | Rolling drawdown from 252-day peak for DBC | 252 | decimal (0 = at peak, -0.1 = 10% below peak) | computed from prices |
| BIL_dd | drawdown | Rolling drawdown from 252-day peak for BIL | 252 | decimal (0 = at peak, -0.1 = 10% below peak) | computed from prices |
| VIXCLS | macro | CBOE VIX - implied volatility index (fear gauge) | - | varies by series | FRED (latest vintage, forward-filled) |
| T10Y2Y | macro | 10Y-2Y Treasury term spread (recession signal) | - | varies by series | FRED (latest vintage, forward-filled) |
| BAA10Y | macro | Baa corporate - 10Y Treasury spread (credit spread) | - | varies by series | FRED (latest vintage, forward-filled) |
| NFCI | macro | Chicago Fed National Financial Conditions Index | - | varies by series | FRED (latest vintage, forward-filled) |

## Feature categories

### Returns
Daily log returns: `log(price[t]) - log(price[t-1])`

### Volatility
Rolling realised volatility (standard deviation of returns), annualised by
multiplying by sqrt(252).

### Correlations
Rolling pairwise Pearson correlations between asset returns. `avg_corr{N}d` is the
average across all pairs, and tends to rise during crises.
Excluded from the correlation features: BIL (a near-zero-variance cash proxy, whose correlations are numerically unstable).

### Drawdowns
Rolling drawdown from the recent peak: `(price[t] - max(price[t-N:t])) / max(...)`.
Values are <= 0 (0 at the peak, -0.10 ten percent below it).

### Macro
FRED series, forward-filled onto trading days.
