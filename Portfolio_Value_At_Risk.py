import numpy as np
import openbb
import pandas as pd

# -----------------------------------------------------------------------------
# STEP 1: Define Portfolio Parameters & Fetch Alternative Asset Data
# -----------------------------------------------------------------------------
# Portfolio size: $10,000 retail account
PORTFOLIO_VALUE = 10000.00
CONFIDENCE_LEVEL = 0.95  # 95% certainty boundary

# Smaller, high-volatility alternative assets
assets = ["SOL-USD", "ADA-USD", "IWM"]  # Solana, Cardano, Russell 2000 (Small Caps)
weights = np.array([0.40, 0.30, 0.30])  # 40% SOL, 30% ADA, 30% Small Caps

print("📥 Fetching historical data via OpenBB...")
data_dict = {}

for asset in assets:
    try:
        # Fetching 2 years of daily data
        df = openbb.equity.price.historical(
            symbol=asset, start_date="2024-01-01", provider="yfinance"
        )
        # Handle structural column name variance natively
        close_col = "close" if "close" in df.columns else "Close"
        data_dict[asset] = df[close_col]
    except Exception:
        # Explicit fallback loop compatibility for direct SDK structures
        from openbb import obb

        df = obb.equity.price.historical(
            asset, start_date="2024-01-01", provider="yfinance"
        ).to_df()
        close_col = "close" if "close" in df.columns else "Close"
        data_dict[asset] = df[close_col]

# Combine to unified matrix, drop missing rows (handling weekend crypto/market alignment)
prices_df = pd.DataFrame(data_dict).dropna()

# -----------------------------------------------------------------------------
# STEP 2: Compute Portfolio Return Metrics
# -----------------------------------------------------------------------------
# Convert prices to daily percentage change returns
returns_df = prices_df.pct_change().dropna()

# Vector multiplication: Dot product of daily asset returns against structural weights
portfolio_returns = returns_df.dot(weights)

# -----------------------------------------------------------------------------
# STEP 3: Mathematical Historical VaR Cutoff
# -----------------------------------------------------------------------------
# Historical method takes actual past returns and finds the lower threshold percentile
# For a 95% confidence interval, we look at the worst 5% of trading days (1.0 - 0.95)
percentile_cutoff = (1.0 - CONFIDENCE_LEVEL) * 100
var_percentile = np.percentile(portfolio_returns, percentile_cutoff)

# Translate percentage loss to concrete dollar amounts
dollar_var = abs(var_percentile * PORTFOLIO_VALUE)

# Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
# This calculates the average loss on the days where our VaR threshold is breached
tail_losses = portfolio_returns[portfolio_returns <= var_percentile]
dollar_cvar = abs(tail_losses.mean() * PORTFOLIO_VALUE)

# -----------------------------------------------------------------------------
# STEP 4: Scannable Dashboard Diagnostics
# -----------------------------------------------------------------------------
print("\n" + "=" * 55)
print("📌 RETAIL QUANT RISK METRICS PORTFOLIO DASHBOARD")
print("=" * 55)
print(f"• Total Portfolio Value     : ${PORTFOLIO_VALUE:,.2f}")
print(f"• Target Confidence Horizon : {CONFIDENCE_LEVEL*100:.1f}% Daily")
print("-" * 55)
print(f"• Statistically Expected Max 1-Day Loss (VaR): ${dollar_var:.2f}")
print(f"  -> Interpretation: There is a 95% statistical certainty that this")
print(
    f"     account will not lose more than ${dollar_var:.2f} in a single day."
)
print("-" * 55)
print(f"• Tail Risk / Expected Shortfall (CVaR)      : ${dollar_cvar:.2f}")
print(f"  -> Interpretation: If the market experiences an extreme 5% anomaly")
print(f"     event, the average expected loss spikes to ${dollar_cvar:.2f}.")
print("=" * 55)
