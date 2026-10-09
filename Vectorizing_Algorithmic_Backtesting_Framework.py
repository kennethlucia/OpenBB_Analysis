import pandas as pd
import numpy as np

# Load our comprehensive historical dataset
df = pd.read_csv("generated/alternative_sentiment_backtest.csv")

# -----------------------------------------------------------------------------
# STEP 1: Compute Underlying Asset Daily Returns
# -----------------------------------------------------------------------------
# Asset_Daily_Return = (Price_Today / Price_Yesterday) - 1
df['Asset_Daily_Return'] = df['Asset_Price'].pct_change().fillna(0)

# -----------------------------------------------------------------------------
# STEP 2: Align the Strategy Execution Timeline (The Shift Rule)
# -----------------------------------------------------------------------------
# CRITICAL QUANT RULE: Avoid "Lookahead Bias". If we parse an RSS text feed today,
# we can only capture the price return on the FOLLOWING market day.
# We shift our positions matrix forward by exactly 1 day.
df['Strategy_Position'] = df['Numeric_Weight'].shift(1).fillna(0)

# -----------------------------------------------------------------------------
# STEP 3: Compute Cumulative Returns Matrix
# -----------------------------------------------------------------------------
# Strategy return = position direction (-1, 0, or 1) multiplied by asset's real return
df['Strategy_Daily_Return'] = df['Strategy_Position'] * df['Asset_Daily_Return']

# Compute the compound total equity curve performance over time
df['Cumulative_Asset_Return'] = (1 + df['Asset_Daily_Return']).cumprod() - 1
df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Daily_Return']).cumprod() - 1

# -----------------------------------------------------------------------------
# STEP 4: Generate Scannable Performance Scorecard
# -----------------------------------------------------------------------------
final_asset_perf = df['Cumulative_Asset_Return'].iloc[-1] * 100
final_strat_perf = df['Cumulative_Strategy_Return'].iloc[-1] * 100

# Calculate the hit rate (how often did our bullish/bearish directional bet guess correctly?)
correct_trades = np.sign(df['Strategy_Position'] * df['Asset_Daily_Return'])
win_rate = (correct_trades == 1).sum() / (correct_trades != 0).sum() * 100

print("\n" + "=" * 55)
print("📊 AUTOMATED STRATEGY BACKTEST RESULTS SCORECARD")
print("=" * 55)
print(f"• Total Testing Horizon      : {len(df)} Simulated Days")
print(f"• Baseline Buy & Hold Return : {final_asset_perf:.2f}%")
print(f"• Quant Sentiment Strategy   : {final_strat_perf:.2f}%")
print("-" * 55)
print(f"• Directional Prediction Hit Rate: {win_rate:.1f}%")
print("=" * 55)

if final_strat_perf > final_asset_perf:
    print("🚀 STRATEGY OUTPERFORMANCE: Alternative data captured alpha!")
else:
    print("⚠️ STRATEGY UNDERPERFORMANCE: Refine prompt weighting parameters.")
