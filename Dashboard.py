import feedparser
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import datetime
import os
import numpy as np
import openbb


# [ Step 1: Alternative Ingestion ] ──> Fetches RSS developer feeds & text logs.
#                 │
#                 ▼
# [ Step 2: Hugging Face NLP ]     ──> Evaluates text and prints strict BULLISH/BEARISH weights.
#                 │
#                 ▼
# [ Step 3: Data Appender ]        ──> Saves logs to 'alternative_sentiment_backtest.csv'.
#                 │
#                 ▼
# [ Step 4: Friction Backtest ]    ──> Deducts $2.00 flat fees and 0.10% slippage spreads.
#                 │
#                 ▼
# [ Step 5: Drawdown Analyzer ]    ──> Verifies your Maximum Drawdown stays below your personal pain limit.

TARGET_ASSET = "AAPL"
# 1. Fetch real historical price data via OpenBB to populate the column
print(f"📥 Fetching historical price parameters for {TARGET_ASSET}...")
try:
    price_df = openbb.equity.price.historical(
        symbol=TARGET_ASSET, start_date="2026-01-01", provider="yfinance"
    )
except AttributeError:
    from openbb import obb

    price_df = obb.equity.price.historical(
        TARGET_ASSET, start_date="2026-01-01", provider="yfinance"
    ).to_df()

# Handle column name updates cleanly
close_col = "close" if "close" in price_df.columns else "Close"
# -----------------------------------------------------------------------------
# STEP 1: Define Clean Live Data Stream Endpoints
# -----------------------------------------------------------------------------
# We leverage your verified, single-line Apple Financial RSS link
YAHOO_APPLE_FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US"

print("📡 Ingesting live asset streams from Yahoo Finance RSS...")
parsed_feed = feedparser.parse(YAHOO_APPLE_FEED)

articles = []
for entry in parsed_feed.entries[:5]:  # Limit to top 5 articles for execution speed
    articles.append({
        "title": entry.title,
        "summary": getattr(entry, "summary", "No baseline text description available.")
    })

print(f"✅ Ingestion Successful! Retreived {len(articles)} data entries.")

# -----------------------------------------------------------------------------
# STEP 2: Initialize a Native, Unrestricted Financial NLP Model
# -----------------------------------------------------------------------------
# FinScience/FinanceBERT is a natively structured Hugging Face repository.
# It contains all required tokenizer json files, eliminating ValueError crashes completely.
model_id = "ahmedrachid/FinancialBERT-Sentiment-Analysis"

print(f"\n🤖 Loading native financial text processor: {model_id}...")

# Load tokenizer and sequence classification layout natively
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# Sequence classification pipelines map raw news texts directly to market signals
financial_nlp = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)

# -----------------------------------------------------------------------------
# STEP 3: Execute Sentiment Matrix Classifications
# -----------------------------------------------------------------------------
print("\n🧠 Processing data streams through pipeline layers...")

processed_data = []

for article in articles:
    # Text classification models don't require hyper-complex prompting.
    # They read the headline text directly and return calculated label outputs.
    combined_text = f"{article['title']}. {article['summary']}"

    # Run model prediction
    # Truncation=True prevents token length errors if an summary text block is too long
    prediction = financial_nlp(combined_text, truncation=True)[0]

    # Map the model's custom classification tags directly to your dashboard variables
    model_label = prediction['label'].upper()

    # Translate model outputs into uniform trading indicators
    if "POSITIVE" in model_label or "BULLISH" in model_label:
        clean_sentiment = "BULLISH"
        weight = 1.0
    elif "NEGATIVE" in model_label or "BEARISH" in model_label:
        clean_sentiment = "BEARISH"
        weight = -1.0
    else:
        clean_sentiment = "NEUTRAL"
        weight = 0.0

    processed_data.append({
        "Title": article["title"][:60] + "...",  # Truncated title for cleaner display
        "Sentiment_Signal": clean_sentiment,
        "Numeric_Weight": weight
    })

# -----------------------------------------------------------------------------
# STEP 4: Render Your Dashboard Output Framework
# -----------------------------------------------------------------------------
sentiment_df = pd.DataFrame(processed_data)

print("\n" + "=" * 70)
print("📊 ALTERNATIVE DATA SENTIMENT PIPELINE RESULTS")
print("=" * 70)
print(sentiment_df.to_string(index=False))
print("=" * 70)

# Calculate your aggregate market directional edge
net_portfolio_bias = sentiment_df["Numeric_Weight"].mean()
print(f"\n💡 Aggregate Market Directional Signal: {net_portfolio_bias:.2f}")
if net_portfolio_bias > 0.1:
    print("🚀 Quant Action Recommendation: Search for LONG execution triggers.")
elif net_portfolio_bias < -0.1:
    print("📉 Quant Action Recommendation: Search for SHORT execution triggers.")
else:
    print("😐 Quant Action Recommendation: Market-Neutral position holding.")

processed_logs = []
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# Get the latest valid closing price from our OpenBB data array
latest_market_price = float(price_df[close_col].iloc[-1])

for item in processed_data:
    signal = item["Sentiment_Signal"]
    # Mapping qualitative labels directly into mathematical execution triggers
    weight = 1.0 if signal == "BULLISH" else (-1.0 if signal == "BEARISH" else 0.0)

    processed_logs.append({
        "Timestamp": current_time,
        "Source_Headline": item["Title"],
        "Sentiment_Signal": signal,
        "Numeric_Weight": weight,
        "Asset_Price": latest_market_price,  # 👈 THIS ADDS THE MISSING COLUMN KEY
    })

# Convert list into a structured dataframe
df = pd.DataFrame(processed_logs)

# Establish file paths
dir_path = os.path.dirname(os.path.realpath(__file__))
csv_filename = dir_path + ("\\generated\\alternative_sentiment_backtest.csv")

# Append natively if the file already exists, else create it fresh
# -------------------------------------------------------------------------------------
# 1. Check if the file is already initialized on your drive
# If the file DOES NOT exist, file_is_empty will be True, meaning we NEED a header.
# If the file DOES exist, file_is_empty will be False, meaning we SKIP the header.
file_is_empty = not os.path.exists(csv_filename)

# 2. Append data dynamically without text injection corruption
df.to_csv(
    csv_filename,
    mode='a',
    header=file_is_empty,  # 👈 This dynamically toggles True or False!
    index=False
)

print(f"✅ Data appended safely. Header injected: {file_is_empty}")

# if os.path.exists(csv_filename):
#     df.to_csv(csv_filename, mode='a', header=True, index=False)
#     print(f"📁 Successfully appended new pipeline logs into {csv_filename}")
# else:
#     df.to_csv(csv_filename, index=False)
#     print(f"✨ Created fresh master database file: {csv_filename}")

# Load your master historical tracking data matrix
df = pd.read_csv("generated/alternative_sentiment_backtest.csv")

# -----------------------------------------------------------------------------
# STEP 1: Core Framework Setup
# -----------------------------------------------------------------------------
df["Asset_Daily_Return"] = df["Asset_Price"].pct_change().fillna(0)
df["Strategy_Position"] = df["Numeric_Weight"].shift(1).fillna(0)

# Configure Retail Account Constraints
PORTFOLIO_START_VALUE = 10000.00
SLIPPAGE_PCT = 0.0010  # 0.10% slippage fee per trade execution
FIXED_FEE_DOLLAR = 2.00  # $2.00 flat broker commission or network gas
STOP_LOSS_THRESHOLD = -0.05  # -5.0% Stop-loss trigger level

# Track final simulation variables
portfolio_values = [PORTFOLIO_START_VALUE]
current_balance = PORTFOLIO_START_VALUE
stop_loss_counts = 0
last_position = 0.0

# -----------------------------------------------------------------------------
# STEP 2: Enhanced Trading Simulation Loop with Circuit Breakers
# -----------------------------------------------------------------------------
for i in range(1, len(df)):
    daily_asset_return = df.loc[i, "Asset_Daily_Return"]
    intended_position = df.loc[i, "Strategy_Position"]

    # Check if we are currently holding the asset (Long or Short)
    if intended_position != 0:
        # Calculate the direct impact of market movement on our specific bet direction
        realized_return = intended_position * daily_asset_return

        # RISK TRIGGER ENGINE: Did the asset breach our pain threshold today?
        if realized_return <= STOP_LOSS_THRESHOLD:
            # 🚨 TRIGGERED: Force an immediate exit from the market to cash (0)
            executed_position = 0.0
            stop_loss_counts += 1

            # The return is locked in at exactly our stop-loss threshold limit
            day_return = STOP_LOSS_THRESHOLD
        else:
            # Safe for today: Execute the model's intended position normally
            executed_position = intended_position
            day_return = realized_return
    else:
        executed_position = 0.0
        day_return = 0.0

    # 1. Grow or contract portfolio capital base
    current_balance = current_balance * (1 + day_return)

    # 2. Compute portfolio turnover mechanics to calculate friction fees accurately
    # If the position changed (including hitting our stop-loss), we pay broker friction
    position_change = np.abs(executed_position - last_position)

    variable_slippage = position_change * current_balance * SLIPPAGE_PCT
    fixed_broker_fee = FIXED_FEE_DOLLAR if position_change > 0 else 0.00

    # 3. Deduct transaction leaks from the account balance
    current_balance -= (variable_slippage + fixed_broker_fee)

    # Ensure account can't drop past absolute liquidation zero
    if current_balance <= 0:
        current_balance = 0

    portfolio_values.append(current_balance)
    last_position = executed_position

# Save results back to working layout arrays

df["Portfolio_With_StopLoss"] = portfolio_values

# -----------------------------------------------------------------------------
# STEP 3: Generate the Comparative Risk Metrics Scorecard
# -----------------------------------------------------------------------------
final_return_pct = ((df["Portfolio_With_StopLoss"].iloc[-1] - PORTFOLIO_START_VALUE) / PORTFOLIO_START_VALUE) * 100

# Compute Maximum Drawdown for our new stop-loss protected strategy
rolling_peak = df["Portfolio_With_StopLoss"].cummax()
drawdowns = (df["Portfolio_With_StopLoss"] - rolling_peak) / rolling_peak
max_drawdown_pct = drawdowns.min() * 100

print("\n" + "=" * 65)
print("🛡️ STOP-LOSS PROTECTED QUANT STRATEGY DASHBOARD")
print("=" * 65)
print(f"• Total Evaluation Timeline   : {len(df)} Simulated Days")
print(f"• Stop-Loss Triggers Hit       : {stop_loss_counts} Forced Market Exits")
print("-" * 65)
print(f"• Net Strategy Return (Post-Fee): {final_return_pct:.2f}%")
print(f"• Adjusted Maximum Drawdown    : {max_drawdown_pct:.2f}%")
print("=" * 65)

# What is Maximum Drawdown?Maximum Drawdown measures the absolute largest peak-to-trough drop in your portfolio's value before a new peak is achieved.
# The Formula: At any point in time t,
# your drawdown is:
#            Drawdown_{t} = Value}_{t}- Peak Value Up to Time t / Peak Value Up to Time t
#
#            Why Quants Care:
#            If your strategy makes a 50% return over a year, but suffers a -45% Max Drawdown along the way,
#            a retail account risks hitting a margin call, triggering account liquidation,
#            or causing the trader to panic and manually shut down the system right at the absolute bottom.
#            The Python script below builds natively on top of your post-fee portfolio values to track your rolling equity peaks,
#            map your daily drawdown percentages, and pinpoint your maximum historical portfolio stress.

# Load your historical post-fee performance matrix
#df = pd.read_csv("generated/alternative_sentiment_backtest.csv")

# Ensure your previous post-fee portfolio loop values are stored cleanly.
# (If running standalone, you can calculate this from your Portfolio_With_StopLoss column)
portfolio_series = df["Portfolio_With_StopLoss"]

# -----------------------------------------------------------------------------
# STEP 1: Calculate the Rolling Peak of the Portfolio Equity Curve
# -----------------------------------------------------------------------------
# .cummax() keeps track of the maximum value the portfolio has seen up to that day.
# If the portfolio drops, the rolling peak stays stuck at its historical highest point.
df["Rolling_Peak"] = portfolio_series.cummax()

# -----------------------------------------------------------------------------
# STEP 2: Compute Daily Drawdown Percentages
# -----------------------------------------------------------------------------
# Drawdown is the distance from today's value to the historical rolling peak
df["Daily_Drawdown"] = (portfolio_series - df["Rolling_Peak"]) / df["Rolling_Peak"]

# -----------------------------------------------------------------------------
# STEP 3: Extract Key Stress Metrics
# -----------------------------------------------------------------------------
# Max Drawdown is simply the minimum (most negative) point in our drawdown column
max_drawdown = df["Daily_Drawdown"].min()
max_drawdown_pct = max_drawdown * 100

# Locate the exact index locations where the worst distress occurred
idx_trough = df["Daily_Drawdown"].idxmin()
worst_trough_date = df.loc[idx_trough, "Timestamp"] if "Timestamp" in df.columns else f"Day {idx_trough}"

# Find the peak that preceded that specific worst trough
peak_value_before_crash = df.loc[idx_trough, "Rolling_Peak"]
trough_value = df.loc[idx_trough, "Portfolio_With_StopLoss"]

# -----------------------------------------------------------------------------
# STEP 4: Scannable Portfolio Stress Dashboard
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("📊 PORTFOLIO RISK & MAXIMUM DRAWDOWN DIAGNOSTIC")
print("=" * 65)
print(f"• Peak Value Before Crash  : ${peak_value_before_crash:,.2f}")
print(f"• Lowest Value at Trough   : ${trough_value:,.2f}")
print("-" * 65)
print(f"• MAXIMUM PORTFOLIO DRAWDOWN: {max_drawdown_pct:.2f}%")
print(f"• Date of Peak Distress    : {worst_trough_date}")
print("-" * 65)

# Quantitative interpretation guidelines for a novice analyst
print("💡 Quantitative Stress Evaluation:")
if abs(max_drawdown_pct) < 10:
    print("🟢 Conservative Risk Profile: Excellent volatility dampening.")
elif abs(max_drawdown_pct) < 25:
    print("🟡 Moderate Risk Profile: Standard for equity/alternative asset trading.")
else:
    print("🔴 High Risk Profile: Extreme downside exposure! Ensure your retail account\n"
          "   has sufficient cash buffers to avoid forced broker liquidations.")
print("=" * 65)

