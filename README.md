# WTI vs Brent Crude Oil Price Forecasting

End-to-end time series forecasting project comparing WTI and Brent crude oil prices (1986–2026), testing whether ARIMA or a Random Forest model can meaningfully outperform a naive baseline for short-horizon WTI price forecasting.

## Motivation

Crude oil pricing sits at the center of budgeting, hedging, and revenue forecasting decisions in the oil & gas sector. This project asks a practical question: **can a statistical or machine learning model meaningfully outperform a "tomorrow will look like today" baseline over a 30-day horizon?** The answer has real implications for how much forecasting effort is worth investing in short-horizon commodity price prediction.

## Data

- Daily WTI and Brent spot prices, 1986–2026 (~10,000 daily observations per series)
- Missing values handled via limited forward-fill (max 5 trading days), remaining gaps dropped

## Exploratory analysis

**WTI vs Brent, full history:**

![WTI vs Brent Crude Oil Prices](images/wti_vs_brent_oil_prices.png)

**Brent–WTI spread** — has swung from strongly negative to strongly positive over the sample, confirming the two benchmarks shouldn't be treated as interchangeable:

![Brent - WTI Spread](images/brent-wti_spread.png)

**WTI daily returns and 21-day rolling volatility** — volatility clusters visibly around known shock periods (2008, 2014–16, 2020, 2022) rather than staying constant:

![WTI Daily Returns](images/wti_daily_return.png)
![21-Day Rolling Volatility](images/21-_day_rolling_volatility.png)

## Approach

1. **Data cleaning & merging** — aligned WTI and Brent on date, engineered the Brent–WTI spread
2. **Exploratory analysis** — price trends, spread over time, daily returns, 21-day rolling volatility
3. **Stationarity testing** — ADF test confirms the price level is non-stationary and the first difference is stationary, justifying `d=1` for ARIMA
4. **Baseline model** — naive forecast (last known price repeated across the horizon)
5. **ARIMA** — order selected via AIC grid search over `(p, 1, q)` combinations rather than a manual guess; best order found: **ARIMA(0,1,2)**
6. **Random Forest** — lag and rolling-window features (1/2/3/5/10/20-day lags, 5/10/20-day rolling mean & std), forecast generated recursively over a 30-day horizon
7. **Evaluation** — MAE, RMSE, MAPE on a single 30-day holdout, then validated with a **6-fold walk-forward backtest** across independent 30-day windows

## Results

**Naive forecast vs actual:**

![Naive Forecast vs Actual - WTI](images/Naive_forcast_vs_actual_-_wti.png)

**Naive vs ARIMA, with 95% confidence interval:**

![Naive vs ARIMA Forecast - WTI](images/naive_vs_arima_forcast_-_wti.png)

**All three models compared:**

![30-Day Forecast Comparison](images/30_day_forcast_comparsion.png)

**Single 30-day test window:**

| Model          | MAE   | RMSE  | MAPE  |
|----------------|-------|-------|-------|
| Naive          | 16.98 | 18.40 | 22.73% |
| ARIMA(0,1,2)   | 17.05 | 18.47 | 22.82% |
| Random Forest  | 15.98 | 17.47 | 21.44% |

**Walk-forward backtest, averaged across 6 independent 30-day windows:**

| Model          | Avg MAE | Avg MAPE |
|----------------|---------|----------|
| Naive          | 6.55    | 8.33%    |
| ARIMA(0,1,2)   | 6.64    | 8.38%    |
| Random Forest  | **6.25** | **7.80%** |

Random Forest has the lowest average error across all 6 folds — this isn't a result from a single lucky window. The margin over the naive baseline is real but modest, consistent with WTI prices behaving close to a random walk at a 30-day horizon.

## Key takeaway

Recent price history (mainly the most recent day, per the Random Forest's feature importance) gives a small, consistent edge over "assume no change," but short-horizon crude oil forecasting has hard limits. That's a useful, honest finding for deciding how much to invest in short-term price forecasting versus scenario-based or longer-horizon approaches — not a failed experiment.

## Limitations & next steps

- Six 30-day folds is a modest backtest; more folds across a longer span would further strengthen the conclusion
- ARIMA order was selected once on the full sample and reused across folds rather than re-selected per fold
- Random Forest forecasts are generated recursively, so errors can compound over the horizon; a direct multi-output strategy is worth testing
- No macro/exogenous variables (USD index, OPEC supply data, inventory reports) included — these are known drivers of short-term oil price moves

## Repo structure

```
├── data/
│   ├── wti-daily.csv
│   └── brent-daily.csv
├── notebooks/
│   └── oil_price_forecast.ipynb
├── images/
│   ├── wti_vs_brent_oil_prices.png
│   ├── brent-wti_spread.png
│   ├── wti_daily_return.png
│   ├── 21-_day_rolling_volatility.png
│   ├── Naive_forcast_vs_actual_-_wti.png
│   ├── naive_vs_arima_forcast_-_wti.png
│   └── 30_day_forcast_comparsion.png
├── README.md
├── requirements.txt
└── .gitignore
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/oil_price_forecast.ipynb
```

## Tools

Python, pandas, NumPy, statsmodels, scikit-learn, matplotlib

---
*Part of an ongoing data analytics portfolio focused on financial and energy-sector applications, built on top of an 18-year finance background in the oil & gas industry.*
