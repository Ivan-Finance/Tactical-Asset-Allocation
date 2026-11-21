"""
Tactical Asset Allocation: Building and Comparing Portfolio Strategies
=======================================================================
Master's thesis code - University of Turin, Department of Management "Valter Cantino".
Author: Ivan Garofalo

Pipeline:
    1. Download ETF/ETC prices from Yahoo Finance (yfinance)
    2. Exploratory analysis of the five asset classes
    3. Simulation of five strategies (Buy & Hold, Passive, Protective,
       Aggressive, Gold Hedge) with transaction costs
    4. Performance metrics, market-regime analysis
    5. Robustness checks (sub-periods, transaction-cost sensitivity)

Usage:
    pip install -r requirements.txt
    python tactical_asset_allocation.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ============================================================
# GLOBAL PARAMETERS
# ============================================================

START_DATE = "2012-01-01"
END_DATE = "2026-02-01"        # yfinance treats the end date as exclusive -> data up to Jan 2026.
                               # Set to None to download up to the latest available date.
RISK_FREE = 0.01               # Annual risk-free rate (approx. 3M Euribor average 2012-2026)
INITIAL_VALUE = 100_000        # Initial portfolio value (EUR)
TC = 0.001                     # Transaction cost: 0.1% of turnover

# Use prices adjusted for distributions (total-return style series). Accumulating
# ETFs are unaffected; distributing ones (e.g. IBCN, TCBT) are adjusted for payouts.
# Set to False to use raw closing prices.
AUTO_ADJUST = True

# Number of months by which the signals are delayed before being applied.
#   0 -> weights for month t use the signal computed at the END of month t
#        (this reproduces the thesis results; see the look-ahead note below).
#   1 -> weights for month t use the signal known at the END of month t-1
#        (no look-ahead: the realistic, tradable setup).
SIGNAL_LAG = 0

# Output of the charts
SAVE_FIGURES = True
FIGURES_DIR = "figures"

# Yahoo Finance tickers (exchange suffix included).
# Order matters: it must match the order of the weight vectors below.
# Each asset can have several candidate symbols: they are tried in order and the
# first one returning data is used. The script stops with an explicit message
# if none of the candidates returns data.
# NOTE: verify the symbols on finance.yahoo.com. Quotes of the same instrument on
# different exchanges can differ slightly, so results may not match the thesis exactly.
TICKERS = {
    "EQUITY":      ["IWDA.AS"],   # iShares MSCI World UCITS ETF (Acc) - Euronext Amsterdam
    "GOV":         ["IBCN.DE"],   # iShares Euro Govt Bond 3-5yr UCITS ETF - Xetra
    "CORPORATE":   ["TCBT.AS"],   # VanEck iBoxx EUR Corporate Bond UCITS ETF - Euronext Amsterdam
    "GOLD":        ["XAD5.DE"],   # Xtrackers Physical Gold EUR Hedged ETC - Xetra
    # UBS Bloomberg Commodity Index SF UCITS ETF hEUR acc (ISIN IE00B58ZM503):
    # launched April 2011, accumulating, EUR-hedged, tracks the same index as the
    # WisdomTree ETC used in the thesis (Bloomberg Commodity EUR Hedged Total Return),
    # but with different hedging details and fees (TER 0.19%).
    # The WisdomTree ETC (00XK) is not available on Yahoo Finance.
    "COMMODITIES": ["DCEUAS.SW"],   # SIX Swiss Exchange (EUR) - verify on finance.yahoo.com
}

# Optional local CSV fallback, used only when Yahoo Finance returns no data for an asset.
# Expected format: a CSV with (at least) the columns "Date" and "Price", daily prices.
# Leave the dictionary empty to rely on Yahoo Finance only.
# WARNING: using the old WisdomTree 00XK file here would silently change the instrument.
LOCAL_FALLBACK = {}

# Static portfolio weights: EQUITY, GOV, CORPORATE, GOLD, COMMODITIES
WEIGHTS_STATIC = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

STRATEGY_COLORS = {
    "Buy & Hold": "black",
    "Passive":    "steelblue",
    "Protective": "lightgreen",
    "Aggressive": "red",
    "Gold Hedge": "goldenrod",
}

REGIME_COLORS = {
    "Bull":  "green",
    "Bear":  "yellow",
    "Shock": "red",
}

if SAVE_FIGURES:
    os.makedirs(FIGURES_DIR, exist_ok=True)


def finish_figure(name):
    """Apply tight layout, optionally save the current figure, then show it."""
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(os.path.join(FIGURES_DIR, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# DATA DOWNLOAD AND CLEANING
# ============================================================

def download_prices(tickers, start, end, auto_adjust=True):
    """Download daily closing prices from Yahoo Finance, one asset at a time.

    For each asset the candidate symbols are tried in order; the first one that
    returns data is used. Returns the price DataFrame (columns = asset-class
    names) and a dict with the symbol actually used for each asset.
    """
    series, used = {}, {}
    for name, candidates in tickers.items():
        for symbol in candidates:
            try:
                raw = yf.download(symbol, start=start, end=end,
                                  auto_adjust=auto_adjust, progress=False)
            except Exception as exc:               # network/parsing problems: try next symbol
                print(f"  {name}: {symbol} failed ({exc})")
                continue
            if raw is None or raw.empty:
                print(f"  {name}: no data for {symbol}, trying next candidate...")
                continue
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):    # recent yfinance returns one column per ticker
                close = close.iloc[:, 0]
            close = close.dropna()
            if close.empty:
                continue
            series[name] = close
            used[name] = symbol
            break
        else:
            # No Yahoo candidate worked: try the local CSV fallback, if configured
            path = LOCAL_FALLBACK.get(name)
            if path and os.path.exists(path):
                df = pd.read_csv(path)
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=False)
                s = pd.to_numeric(df.set_index("Date")["Price"], errors="coerce")
                s = s.sort_index().dropna()
                s = s[s.index >= start]
                if end is not None:
                    s = s[s.index < end]
                series[name] = s
                used[name] = f"local CSV"
                print(f"  {name}: using local file {path}")
            else:
                raise ValueError(
                    f"No data returned for {name} (tried: {candidates}) and no local "
                    f"CSV found at {path}. Add the CSV, or replace this asset with "
                    "another instrument (this would change the results)."
                )
    return pd.DataFrame(series).sort_index(), used


prices, USED_TICKERS = download_prices(TICKERS, START_DATE, END_DATE, AUTO_ADJUST)

# Report the symbol used and the first/last available observation for each instrument
print("Data availability:")
for col in prices.columns:
    s = prices[col].dropna()
    print(f"  {col:<12} {USED_TICKERS[col]:<14} {s.index.min().date()} -> {s.index.max().date()}"
          f"  ({prices[col].isna().sum()} missing daily values)")

# Monthly prices (last available price of each month) and monthly returns.
# dropna() removes the initial months in which not every instrument is available yet.
monthly_prices = prices.resample("ME").last()
returns = monthly_prices.pct_change().dropna()

print(f"\nAnalysis period: {returns.index.min().date()} -> {returns.index.max().date()}")
print(f"Monthly observations: {len(returns)}\n")


# ============================================================
# EXPLORATORY ANALYSIS OF THE ASSET CLASSES
# ============================================================

# Annualized statistics from monthly returns (arithmetic mean * 12, std * sqrt(12))
ann_return = returns.mean() * 12
ann_vol = returns.std() * np.sqrt(12)
corr_matrix = returns.corr()

print("Annualized return:\n", ann_return.round(4))
print("\nAnnualized volatility:\n", ann_vol.round(4))
print("\nCorrelation matrix:\n", corr_matrix.round(2))

# Prices normalized to base 100 (first valid observation of each series)
first_valid = monthly_prices.apply(lambda col: col.dropna().iloc[0])
normalized = monthly_prices / first_valid * 100

# Rolling statistics (12-month window)
rolling_ret = (1 + returns).rolling(12).apply(lambda x: x.prod() - 1)
rolling_vol = returns.rolling(12).std() * np.sqrt(12)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Normalized prices
ax = axes[0, 0]
for col in normalized.columns:
    ax.plot(normalized[col], label=col)
ax.set_title("Asset Classes - Normalized Prices (Base 100)")
ax.set_ylabel("Value (base 100)")
ax.legend(fontsize=12)
ax.grid(True, linestyle="--", alpha=0.4)

# Correlation matrix
ax = axes[0, 1]
im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdYlGn")
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45)
ax.set_yticklabels(corr_matrix.columns)
ax.set_title("Correlation Matrix")
plt.colorbar(im, ax=ax)

# Rolling 12-month return
ax = axes[1, 0]
for col in rolling_ret.columns:
    ax.plot(rolling_ret[col], label=col)
ax.set_title("Rolling 12M Return")
ax.set_ylabel("Return")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.legend(fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

# Rolling 12-month volatility
ax = axes[1, 1]
for col in rolling_vol.columns:
    ax.plot(rolling_vol[col], label=col)
ax.set_title("Rolling 12M Volatility (Annualized)")
ax.set_ylabel("Volatility")
ax.legend(fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

finish_figure("01_asset_class_overview")


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def max_drawdown(series):
    """Maximum peak-to-trough loss of the cumulated return series."""
    cum = (1 + series).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()


def drawdown_series(series):
    """Drawdown at each date, relative to the running maximum."""
    cum = (1 + series).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


def performance_metrics(series):
    """Compute the performance metrics used in the thesis from monthly returns."""
    ann_ret = series.mean() * 12
    ann_vol = series.std() * np.sqrt(12)
    sharpe = (ann_ret - RISK_FREE) / ann_vol
    mdd = max_drawdown(series)

    # Sortino: downside deviation estimated on negative monthly returns only
    downside_std = series[series < 0].std() * np.sqrt(12)
    sortino = (ann_ret - RISK_FREE) / downside_std

    # Calmar: annual return divided by the absolute maximum drawdown
    calmar = ann_ret / abs(mdd)

    # Historical 95% Value at Risk (monthly)
    var_95 = series.quantile(0.05)

    return {
        "Return":       round(ann_ret, 4),
        "Volatility":   round(ann_vol, 4),
        "Sharpe":       round(sharpe, 4),
        "Sortino":      round(sortino, 4),
        "Calmar":       round(calmar, 4),
        "Max Drawdown": round(mdd, 4),
        "VaR 95%":      round(var_95, 4),
        "Skewness":     round(series.skew(), 4),
        "Kurtosis":     round(series.kurtosis(), 4),   # excess kurtosis
    }


# ============================================================
# PORTFOLIO SIMULATION WITH TRANSACTION COSTS
#
# Monthly logic:
#   1. Weights drift with market returns.
#   2. Turnover = half the sum of absolute differences between the
#      drifted weights and the target weights.
#   3. Cost = TC * turnover * portfolio value.
#   4. The cost is deducted from the portfolio value, and the weights are
#      reset to the target weights.
# ============================================================

def simulate_portfolio(returns, weights_target_df, initial_value=INITIAL_VALUE, tc=TC):
    """Simulate a portfolio that is rebalanced each month to the target weights.

    Returns: net monthly returns (Series), number of months with trading costs,
             dates of those months, total transaction costs (EUR).
    """
    n = len(returns)
    portfolio_value = initial_value
    current_weights = weights_target_df.iloc[0].values.copy()   # start from first target weights
    net_returns = []
    rebalancing_dates = []
    total_costs = 0.0

    for t in range(n):
        r = returns.iloc[t].values
        w_target = weights_target_df.iloc[t].values

        # Gross return: target weights applied to this month's asset returns
        gross_ret = np.dot(w_target, r)

        # Weights after market drift, starting from last month's post-rebalancing weights
        drifted = current_weights * (1 + r)
        drifted = drifted / drifted.sum()

        # Turnover needed to go from the drifted weights to the target weights
        turnover = np.sum(np.abs(drifted - w_target)) / 2
        cost = tc * turnover * portfolio_value * (1 + gross_ret)

        portfolio_value_net = portfolio_value * (1 + gross_ret) - cost
        net_ret = (portfolio_value_net - portfolio_value) / portfolio_value

        if cost > 0:
            rebalancing_dates.append(returns.index[t])
            total_costs += cost

        portfolio_value = portfolio_value_net
        current_weights = w_target.copy()      # weights are reset to target after rebalancing
        net_returns.append(net_ret)

    return (
        pd.Series(net_returns, index=returns.index),
        len(rebalancing_dates),
        rebalancing_dates,
        total_costs,
    )


def simulate_buyhold(returns, w0=WEIGHTS_STATIC, initial_value=INITIAL_VALUE):
    """Buy & Hold: no rebalancing, so weights evolve freely with the market
    (e.g. if equity rises, its weight grows). No transaction costs."""
    portfolio_value = initial_value
    current_weights = w0.copy()
    net_returns = []

    for t in range(len(returns)):
        r = returns.iloc[t].values
        gross_ret = np.dot(current_weights, r)
        net_returns.append(gross_ret)                       # no costs
        current_weights = current_weights * (1 + r)         # drift only
        current_weights = current_weights / current_weights.sum()
        portfolio_value *= (1 + gross_ret)

    return pd.Series(net_returns, index=returns.index), 0, [], 0.0


# ============================================================
# STRATEGY DEFINITIONS (target weights)
# Weight order: EQUITY, GOV, CORPORATE, GOLD, COMMODITIES
#
# NOTE ON LOOK-AHEAD: at SIGNAL_LAG = 0, the signal computed with the price at
# the end of month t is used for the weights applied to the return of month t
# itself (the return from t-1 to t). In practice the signal would only be
# known at the end of the month, so a tradable version should use SIGNAL_LAG = 1.
# ============================================================

# --- Passive: fixed weights, rebalanced every month ---
w_passive = pd.DataFrame(
    np.tile(WEIGHTS_STATIC, (len(returns), 1)),
    index=returns.index, columns=returns.columns
)

# --- 12-month moving average signal (Protective and Aggressive) ---
# Protective:  price > MA12 -> standard weights; price <= MA12 -> equity 20%, government bonds 50%
# Aggressive:  price > MA12 -> equity 60%;       price <= MA12 -> standard weights
ma12 = monthly_prices["EQUITY"].rolling(12).mean()
signal_ma = ((monthly_prices["EQUITY"] > ma12)
             .reindex(returns.index).fillna(False)
             .shift(SIGNAL_LAG, fill_value=False))

W_PROT_BULL = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
W_PROT_BEAR = np.array([0.2, 0.5, 0.1, 0.1, 0.1])
W_AGGR_BULL = np.array([0.6, 0.2, 0.05, 0.1, 0.05])
W_AGGR_BEAR = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

w_prot = pd.DataFrame(
    np.where(signal_ma.values.reshape(-1, 1), W_PROT_BULL, W_PROT_BEAR),
    index=returns.index, columns=returns.columns
)
w_aggr = pd.DataFrame(
    np.where(signal_ma.values.reshape(-1, 1), W_AGGR_BULL, W_AGGR_BEAR),
    index=returns.index, columns=returns.columns
)

# --- Gold Hedge: rolling 3-month equity return below -5% -> hedge on ---
# Crisis:  equity 10%, government bonds 40%, gold 30%
# Normal:  standard weights
THRESHOLD_HEDGE = -0.05
LOOKBACK_HEDGE = 3

equity_roll_3m = (1 + returns["EQUITY"]).rolling(LOOKBACK_HEDGE).apply(np.prod) - 1
signal_hedge = ((equity_roll_3m < THRESHOLD_HEDGE).fillna(False)
                .shift(SIGNAL_LAG, fill_value=False))

W_HEDGE_CRISIS = np.array([0.1, 0.4, 0.1, 0.3, 0.1])
W_HEDGE_NORMAL = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

w_hedge = pd.DataFrame(
    np.where(signal_hedge.values.reshape(-1, 1), W_HEDGE_CRISIS, W_HEDGE_NORMAL),
    index=returns.index, columns=returns.columns
)


# ============================================================
# SIMULATION OF ALL STRATEGIES (net of costs)
# ============================================================

ret_bh,   _, _, cost_bh   = simulate_buyhold(returns)
ret_pass, _, _, cost_pass = simulate_portfolio(returns, w_passive)
ret_prot, _, _, cost_prot = simulate_portfolio(returns, w_prot)
ret_aggr, _, _, cost_aggr = simulate_portfolio(returns, w_aggr)
ret_hedge, _, _, cost_hedge = simulate_portfolio(returns, w_hedge)


# ============================================================
# SIGNAL ACTIVATION OVER TIME
# ============================================================

# NOTE: this block uses the signals as defined above (shifted by SIGNAL_LAG).
valid_ma = ma12.reindex(returns.index).notna()
valid_hedge = equity_roll_3m.notna()
common = valid_ma & valid_hedge          # window where both signals are defined

sig_ma = signal_ma[common]               # True = price above MA12
sig_hg = signal_hedge[common]            # True = crisis phase
idx = sig_ma.index                       # common time axis

# Signal changes (the first row is excluded)
chg_ma = sig_ma.index[sig_ma.ne(sig_ma.shift())][1:]
chg_hg = sig_hg.index[sig_hg.ne(sig_hg.shift())][1:]

# Distinct crisis episodes (entries into the Gold Hedge signal)
ep_start = sig_hg.index[sig_hg & ~sig_hg.shift(fill_value=False)]
n_ep = len(ep_start)

print("\n=== Signal activation ===")
print(f"Common window: {idx[0].strftime('%Y-%m')} -> {idx[-1].strftime('%Y-%m')}")
print(f"MA12 - months above MA12 (Aggressive overweight): {sig_ma.sum()} of {len(sig_ma)}")
print(f"MA12 - months below MA12 (Protective defensive):  {(~sig_ma).sum()} of {len(sig_ma)}")
print(f"MA12 - signal changes: {len(chg_ma)}")
print(f"Gold Hedge - crisis months: {sig_hg.sum()} of {len(sig_hg)}")
print(f"Gold Hedge - crisis episodes: {n_ep}")
print(f"Gold Hedge - signal changes: {len(chg_hg)}")
print("Gold Hedge - episode start dates:", [d.strftime('%Y-%m') for d in ep_start])


def style_grid(ax):
    """Common grid style: continuous, light grey lines."""
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="-", linewidth=0.5, color="grey", alpha=0.25)


# Text sizes for the signal chart
TITLE_SIZE = 15
LABEL_SIZE = 16
TICK_SIZE = 15
LEGEND_SIZE = 13

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

# ---------- Panel 1: MA12 signal ----------
ax = axes[0]
price = monthly_prices["EQUITY"].reindex(returns.index)[common]
ma = ma12.reindex(returns.index)[common]

# Background: green above MA12, red below (step="mid" keeps isolated months visible)
ax.fill_between(idx, 0, 1, where=sig_ma.values,
                transform=ax.get_xaxis_transform(), step="mid",
                color="green", alpha=0.12,
                label=f"Above MA12 - Aggressive ({sig_ma.sum()} of {len(sig_ma)} months)")
ax.fill_between(idx, 0, 1, where=(~sig_ma).values,
                transform=ax.get_xaxis_transform(), step="mid",
                color="red", alpha=0.25,
                label=f"Below MA12 - Protective ({(~sig_ma).sum()} of {len(sig_ma)} months)")

ax.plot(price, color="black", linewidth=1.5, label="Equity price")
ax.plot(ma, color="steelblue", linewidth=1.8, label="MA12")

ax.set_title(f"Protective and Aggressive signal ({len(chg_ma)} signal changes)",
             fontsize=TITLE_SIZE)
ax.set_ylabel("Equity price", fontsize=LABEL_SIZE)
ax.tick_params(axis="both", labelsize=TICK_SIZE)
ax.legend(fontsize=LEGEND_SIZE, loc="upper left", framealpha=0.95)
style_grid(ax)

# ---------- Panel 2: Gold Hedge signal ----------
ax = axes[1]
ax.fill_between(idx, 0, 1, where=sig_hg.values,
                transform=ax.get_xaxis_transform(), step="mid",
                color="goldenrod", alpha=0.35,
                label=f"Crisis - hedge on ({sig_hg.sum()} of {len(sig_hg)} months)")

ax.plot(equity_roll_3m[common], color="black", linewidth=1.5,
        label="Rolling 3M equity return")
ax.axhline(THRESHOLD_HEDGE, color="firebrick", linestyle="--", linewidth=1.8,
           label=f"Threshold ({THRESHOLD_HEDGE:.0%})")

ax.set_title(f"Gold Hedge signal ({n_ep} crisis episodes)", fontsize=TITLE_SIZE)
ax.set_ylabel("Rolling 3M return", fontsize=LABEL_SIZE)
ax.set_xlabel("Year", fontsize=LABEL_SIZE)
ax.tick_params(axis="both", labelsize=TICK_SIZE)
ax.legend(fontsize=LEGEND_SIZE, loc="lower left", framealpha=0.95)
style_grid(ax)

finish_figure("02_signals")


# ============================================================
# REBALANCING AND TRANSACTION COST SUMMARY
# ============================================================

# Number of months in which the target weights differ from the previous month
n_signal_changes_prot = int((w_prot != w_prot.shift()).any(axis=1).sum())
n_signal_changes_aggr = int((w_aggr != w_aggr.shift()).any(axis=1).sum())
# NOTE: for Gold Hedge this counts the months with the hedge on (crisis months),
# not the number of signal changes.
n_signal_changes_hedge = int(signal_hedge.sum())

print("\n=== Rebalancing and Transaction Cost Summary ===")
print(f"{'Strategy':<20} {'Signal Changes':>15} {'Total Costs (EUR)':>20}")
print("-" * 57)
print(f"{'Buy & Hold':<20} {'0':>15} {cost_bh:>20.2f}")
print(f"{'Passive':<20} {'every month':>15} {cost_pass:>20.2f}")
print(f"{'Protective':<20} {n_signal_changes_prot:>15} {cost_prot:>20.2f}")
print(f"{'Aggressive':<20} {n_signal_changes_aggr:>15} {cost_aggr:>20.2f}")
print(f"{'Gold Hedge (crisis months)':<20} {n_signal_changes_hedge:>15} {cost_hedge:>20.2f}")

hedge_signal_dates = returns.index[signal_hedge]
print("\nGold Hedge crisis months (hedge on):")
print(f"  Number of crisis months: {len(hedge_signal_dates)}")
print("  Dates:")
for d in hedge_signal_dates:
    print(f"    {d.strftime('%Y-%m')}")


# ============================================================
# COLLECT STRATEGIES AND COMPUTE METRICS
# ============================================================

strategies = pd.DataFrame({
    "Buy & Hold": ret_bh,
    "Passive":    ret_pass,
    "Protective": ret_prot,
    "Aggressive": ret_aggr,
    "Gold Hedge": ret_hedge,
})

metrics = pd.DataFrame(
    {name: performance_metrics(strategies[name]) for name in strategies.columns}
).T

print("\n=== Performance Metrics (net of costs) ===")
print(metrics.to_string())


# ============================================================
# PERFORMANCE CHARTS
# ============================================================

cum_all = (1 + strategies).cumprod() * INITIAL_VALUE

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Cumulative growth
ax = axes[0, 0]
for col in cum_all.columns:
    ax.plot(cum_all[col], label=col, color=STRATEGY_COLORS[col], linewidth=2)
ax.set_title(f"Cumulative Growth (initial investment: EUR {INITIAL_VALUE:,})")
ax.set_xlabel("Year")
ax.set_ylabel("Portfolio Value (EUR)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

# Drawdown
ax = axes[0, 1]
for col in strategies.columns:
    ax.plot(drawdown_series(strategies[col]), label=col,
            color=STRATEGY_COLORS[col], linewidth=1.5)
ax.set_title("Strategy Drawdowns")
ax.set_ylabel("Drawdown")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

# Rolling 12-month Sharpe ratio (risk-free rate not subtracted in this chart)
ax = axes[1, 0]
rolling_sharpe = (strategies.rolling(12).mean() * 12) / (strategies.rolling(12).std() * np.sqrt(12))
for col in rolling_sharpe.columns:
    ax.plot(rolling_sharpe[col], label=col, color=STRATEGY_COLORS[col])
ax.set_title("Rolling Sharpe Ratio (12 months)")
ax.set_ylabel("Sharpe")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

# Rolling 12-month volatility
ax = axes[1, 1]
rolling_vol_strat = strategies.rolling(12).std() * np.sqrt(12)
for col in rolling_vol_strat.columns:
    ax.plot(rolling_vol_strat[col], label=col, color=STRATEGY_COLORS[col])
ax.set_title("Rolling Volatility (12 months)")
ax.set_ylabel("Annualized Volatility")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

finish_figure("03_performance_overview")

# Distribution of monthly returns
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, col in zip(axes, strategies.columns):
    ax.hist(strategies[col], bins=35, color=STRATEGY_COLORS[col], edgecolor="black", alpha=0.8)
    ax.set_title(col)
    ax.set_xlabel("Monthly return")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.4)
plt.suptitle("Distribution of Monthly Returns", fontsize=13)
finish_figure("04_return_distributions")


# ============================================================
# WEIGHT EVOLUTION - SIMPLIFIED CHART
# Shows only the key asset weight for each tactical strategy
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# Protective - EQUITY weight
ax = axes[0]
ax.plot(w_prot["EQUITY"], color="lightgreen", linewidth=2, label="EQUITY weight")
ax.axhline(0.4, color="black", linestyle="--", linewidth=0.8, label="Default weight (40%)")
ax.fill_between(w_prot.index, w_prot["EQUITY"], 0.4,
                where=w_prot["EQUITY"] < 0.4, alpha=0.2, color="red", label="Defensive phase")
ax.set_title("Protective Strategy - EQUITY weight over time\n(below the line = negative MA12 signal)")
ax.set_ylabel("EQUITY weight")
ax.set_ylim(0, 0.7)
ax.legend(fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

# Aggressive - EQUITY weight
ax = axes[1]
ax.plot(w_aggr["EQUITY"], color="red", linewidth=2, label="EQUITY weight")
ax.axhline(0.4, color="black", linestyle="--", linewidth=0.8, label="Default weight (40%)")
ax.fill_between(w_aggr.index, w_aggr["EQUITY"], 0.4,
                where=w_aggr["EQUITY"] > 0.4, alpha=0.2, color="green", label="Aggressive phase")
ax.set_title("Aggressive Strategy - EQUITY weight over time\n(above the line = positive MA12 signal)")
ax.set_ylabel("EQUITY weight")
ax.set_ylim(0, 0.7)
ax.legend(fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

# Gold Hedge - GOLD weight
ax = axes[2]
ax.plot(w_hedge["GOLD"], color="goldenrod", linewidth=2, label="GOLD weight")
ax.axhline(0.1, color="black", linestyle="--", linewidth=0.8, label="Default weight (10%)")
ax.fill_between(w_hedge.index, w_hedge["GOLD"], 0.1,
                where=w_hedge["GOLD"] > 0.1, alpha=0.2, color="goldenrod", label="Hedge phase")
ax.set_title(f"Gold Hedge - GOLD weight over time\n"
             f"(above the line = crisis signal on, {len(hedge_signal_dates)} of {len(returns)} months)")
ax.set_ylabel("GOLD weight")
ax.set_ylim(0, 0.5)
ax.legend(fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

finish_figure("05_weight_evolution")


# ============================================================
# MARKET REGIME ANALYSIS
#
# Regime definitions (based on the equity component):
#   Bull:  rolling 12M equity return > 5%
#   Bear:  rolling 12M equity return <= 5%
#   Shock: equity drawdown below -10% from its historical maximum
#          (overrides Bull/Bear)
# ============================================================

roll_ret_12 = (1 + returns["EQUITY"]).rolling(12).apply(np.prod) - 1
cum_equity = (1 + returns["EQUITY"]).cumprod()
dd_equity = (cum_equity - cum_equity.cummax()) / cum_equity.cummax()

regime = pd.Series(index=returns.index, dtype="object")
regime[roll_ret_12 > 0.05] = "Bull"
regime[roll_ret_12 <= 0.05] = "Bear"
regime[dd_equity < -0.10] = "Shock"

# Sharpe ratio per regime.
# NOTE: here the risk-free rate is NOT subtracted (return / volatility).
# The Bull/Bear block further below subtracts it.
# The first 12 months (no rolling return available) are still unclassified at this point.
regime_sharpe = {}
for r in ["Bull", "Bear", "Shock"]:
    mask = regime == r
    subset = strategies[mask]
    ann_r = subset.mean() * 12
    ann_v = subset.std() * np.sqrt(12)
    regime_sharpe[r] = (ann_r / ann_v).round(3)

regime_df = pd.DataFrame(regime_sharpe)
print("\n=== Sharpe Ratio by Regime ===")
print(regime_df.to_string())

# Bar chart: Sharpe by regime (all regimes)
regime_df.T.plot(kind="bar", figsize=(10, 5),
                 color=[STRATEGY_COLORS[c] for c in regime_df.index])
plt.title("Sharpe Ratio by Market Regime")
plt.ylabel("Sharpe")
plt.xticks(rotation=0)
plt.axhline(0, color="black", linewidth=0.8)
plt.legend(title="Strategy")
plt.grid(axis="y", linestyle="--", alpha=0.4)
finish_figure("06_sharpe_by_regime_all")

# Bar chart: Sharpe by regime (Shock excluded)
regime_df_plot = regime_df[["Bull", "Bear"]]
ax = regime_df_plot.T.plot(kind="bar", figsize=(10, 5),
                           color=[STRATEGY_COLORS[c] for c in regime_df_plot.index])
plt.title("Sharpe Ratio by Market Regime (Bull and Bear)", fontsize=15)
plt.ylabel("Sharpe", fontsize=15)
plt.xticks(rotation=0)
plt.axhline(0, color="black", linewidth=0.8)
ax.tick_params(axis="both", labelsize=15)
plt.legend(title="Strategy", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.4)
finish_figure("07_sharpe_by_regime_bull_bear")

# Sharpe ratio for Bull and Bear regimes (risk-free rate subtracted)
print("\n=== Sharpe Ratio for Bull and Bear Regimes (risk-free subtracted) ===")
for r in ["Bull", "Bear"]:
    mask = regime == r
    subset = strategies[mask]
    ann_r = subset.mean() * 12
    ann_v = subset.std() * np.sqrt(12)
    sharpe_r = ((ann_r - RISK_FREE) / ann_v).round(4)
    print(f"\n--- {r} ({mask.sum()} months) ---")
    print(sharpe_r.to_string())

# Shock regime: return and volatility by strategy
print("\n=== Shock Regime - Return and Volatility ===")
mask_shock = regime == "Shock"
subset_shock = strategies[mask_shock]

shock_df = pd.DataFrame({
    "Annualized return":     (subset_shock.mean() * 12).round(4),
    "Annualized volatility": (subset_shock.std() * np.sqrt(12)).round(4),
})
print(shock_df.to_string())
print(f"\nMonths in Shock regime: {mask_shock.sum()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = [STRATEGY_COLORS[c] for c in shock_df.index]

# Return
ax = axes[0]
ax.bar(shock_df.index, shock_df["Annualized return"] * 100,
       color=colors, edgecolor="black", alpha=0.8)
ax.set_title("Annualized Return - Shock Regime", fontsize=14)
ax.set_ylabel("Return (%)", fontsize=15)
ax.tick_params(axis="both", labelsize=14)
ax.axhline(0, color="black", linewidth=0.8)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.tick_params(axis="x", rotation=15)

# Volatility
ax = axes[1]
ax.bar(shock_df.index, shock_df["Annualized volatility"] * 100,
       color=colors, edgecolor="black", alpha=0.8)
ax.set_title("Annualized Volatility - Shock Regime", fontsize=14)
ax.set_ylabel("Volatility (%)", fontsize=15)
ax.tick_params(axis="both", labelsize=14)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.tick_params(axis="x", rotation=15)

plt.suptitle("Shock Regime Analysis", fontsize=15)
finish_figure("08_shock_regime")


# ============================================================
# CUMULATIVE GROWTH WITH REGIME BACKGROUND
#
# Interpretation notes on the regimes:
# - Bull:  the Aggressive strategy benefits from its larger equity exposure.
#          Gold often shows a good Sharpe even in bull phases, since in growth
#          phases accompanied by inflation or macro uncertainty it rises with
#          relatively low volatility.
# - Bear:  gold plays a relative defensive role: it loses less or fluctuates
#          less, improving the return/volatility ratio without necessarily
#          generating positive alpha.
# - Shock: in systemic liquidity crises correlations converge towards 1 and
#          investors sell indiscriminately, safe-haven assets included, so gold
#          may fail to protect at moments of extreme stress, unlike in
#          ordinary bear markets.
# ============================================================

# The first 12 months (no rolling return) are classified as Bear by default
regime = regime.fillna("Bear")

fig, ax = plt.subplots(figsize=(13, 6))
for col in cum_all.columns:
    ax.plot(cum_all[col], label=col, color=STRATEGY_COLORS[col], linewidth=2)

# Coloured background for each contiguous regime period
current_regime = None
start_idx = None
for date, r in regime.items():
    if r != current_regime:
        if current_regime is not None:
            ax.axvspan(start_idx, date, color=REGIME_COLORS.get(current_regime, "grey"), alpha=0.2)
        current_regime = r
        start_idx = date
if current_regime is not None:
    ax.axvspan(start_idx, regime.index[-1],
               color=REGIME_COLORS.get(current_regime, "grey"), alpha=0.2)

ax.set_title("Cumulative Growth with Market Regimes (Green=Bull, Yellow=Bear, Red=Shock)",
             fontsize=15)
ax.set_xlabel("Year")
ax.tick_params(axis="both", labelsize=14)
ax.set_ylabel("Portfolio Value (EUR)", fontsize=15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.legend(loc="upper left", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.4)
finish_figure("09_cumulative_growth_regimes")


# ============================================================
# SUB-PERIOD ANALYSIS (robustness)
# 2012-2018: first sub-period (pre-COVID, low interest rates)
# 2019-2026: second sub-period (COVID, rate hikes, geopolitical tensions)
# ============================================================

SUBPERIODS = {
    "2012-2018": ("2012-01-01", "2018-12-31"),
    "2019-2026": ("2019-01-01", "2026-12-31"),
}

print("\n=== Sub-period Analysis ===")

for period_name, (start, end) in SUBPERIODS.items():
    mask = (strategies.index >= start) & (strategies.index <= end)
    sub = strategies[mask]

    if len(sub) < 12:
        print(f"\n{period_name}: insufficient data")
        continue

    sub_metrics = pd.DataFrame(
        {name: performance_metrics(sub[name]) for name in sub.columns}
    ).T

    print(f"\n--- {period_name} ({len(sub)} months) ---")
    print(sub_metrics[["Return", "Volatility", "Sharpe", "Max Drawdown"]].to_string())

    # Cumulative growth within the sub-period
    cum_sub = (1 + sub).cumprod() * INITIAL_VALUE
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in cum_sub.columns:
        ax.plot(cum_sub[col], label=col, color=STRATEGY_COLORS[col], linewidth=2)
    ax.set_title(f"Cumulative Growth - {period_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value (EUR)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    finish_figure(f"10_cumulative_growth_{period_name}")

    # Sharpe ratio by regime within the sub-period (regimes with < 3 months are skipped)
    sub_regime = regime[mask]
    sub_regime_sharpe = {}
    for r in ["Bull", "Bear", "Shock"]:
        r_mask = sub_regime == r
        if r_mask.sum() < 3:
            continue
        subset = sub[r_mask]
        ann_r = subset.mean() * 12
        ann_v = subset.std() * np.sqrt(12)
        sub_regime_sharpe[r] = (ann_r / ann_v).round(3)

    if sub_regime_sharpe:
        print(f"\nSharpe by regime - {period_name}:")
        print(pd.DataFrame(sub_regime_sharpe).to_string())


# ============================================================
# TRANSACTION COST SENSITIVITY ANALYSIS
# ============================================================

def sharpe_ratio(s):
    """Annualized Sharpe ratio from monthly returns."""
    return (s.mean() * 12 - RISK_FREE) / (s.std() * np.sqrt(12))


print("\n=== Transaction Cost Sensitivity (Sharpe ratio) ===")
print(f"{'TC':<10} {'Passive':>15} {'Protective':>18} {'Aggressive':>18} {'Gold Hedge':>12}")
print("-" * 77)
for tc_test in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]:
    r_pass, _, _, _ = simulate_portfolio(returns, w_passive, tc=tc_test)
    r_prot, _, _, _ = simulate_portfolio(returns, w_prot, tc=tc_test)
    r_aggr, _, _, _ = simulate_portfolio(returns, w_aggr, tc=tc_test)
    r_hedge, _, _, _ = simulate_portfolio(returns, w_hedge, tc=tc_test)
    print(f"{tc_test*100:.1f}%{'':<6} {sharpe_ratio(r_pass):>15.4f} {sharpe_ratio(r_prot):>18.4f} "
          f"{sharpe_ratio(r_aggr):>18.4f} {sharpe_ratio(r_hedge):>12.4f}")

print("\n=== Transaction Cost Sensitivity (annualized return) ===")
print(f"{'TC':<10} {'B&H':>12} {'Passive':>15} {'Protective':>18} {'Aggressive':>18} {'Gold Hedge':>12}")
print("-" * 90)

tc_values = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.07, 0.10, 0.15, 0.20]

# Buy & Hold never trades, so its return does not depend on the cost level
ann_bh = ret_bh.mean() * 12

ret_results = {"Buy & Hold": [], "Passive": [], "Protective": [],
               "Aggressive": [], "Gold Hedge": []}

for tc_test in tc_values:
    r_pass, _, _, _ = simulate_portfolio(returns, w_passive, tc=tc_test)
    r_prot, _, _, _ = simulate_portfolio(returns, w_prot, tc=tc_test)
    r_aggr, _, _, _ = simulate_portfolio(returns, w_aggr, tc=tc_test)
    r_hedge, _, _, _ = simulate_portfolio(returns, w_hedge, tc=tc_test)

    ret_results["Buy & Hold"].append(ann_bh * 100)
    ret_results["Passive"].append(r_pass.mean() * 12 * 100)
    ret_results["Protective"].append(r_prot.mean() * 12 * 100)
    ret_results["Aggressive"].append(r_aggr.mean() * 12 * 100)
    ret_results["Gold Hedge"].append(r_hedge.mean() * 12 * 100)

    print(f"{tc_test*100:.1f}%{'':<6} {ann_bh:>12.4f} {r_pass.mean()*12:>15.4f} "
          f"{r_prot.mean()*12:>18.4f} {r_aggr.mean()*12:>18.4f} {r_hedge.mean()*12:>12.4f}")

# Chart: annualized return as transaction costs increase
tc_labels = [f"{tc*100:.1f}%" for tc in tc_values]

plt.figure(figsize=(12, 5))
for name, color in STRATEGY_COLORS.items():
    plt.plot(tc_labels, ret_results[name], marker="o", label=name, color=color, linewidth=2)

plt.axvline(x="0.1%", color="gray", linestyle="--", linewidth=1, label="Assumed cost (0.1%)")
plt.title("Return Sensitivity to Transaction Costs", fontsize=15)
plt.xlabel("Transaction cost", fontsize=14)
plt.ylabel("Annualized return (%)", fontsize=14)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=13)
plt.grid(True, linestyle="--", alpha=0.4)
finish_figure("11_cost_sensitivity_return")
