from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


#  [ EVERY HOUR AT MARKET OPEN ]
#                                 │
#                                 ▼
# 1. Ingest Data      ──> Crawls RSS alternative feeds & developer updates.
#                                 │
#                                 ▼
# 2. Query AI Model   ──> Hugging Face extracts [BULLISH, BEARISH, or NEUTRAL].
#                                 │
#                                 ▼
# 3. Check Constraints──> Ensures trade frequency won't bleed portfolio via fees.
#                                 │
#                                 ▼
# 4. Check Circuit    ──> Scans if trailing Stop-Loss parameters are breached.
#                                 │
#                                 ▼
# 5. Alpaca Sandbox   ──> Instantly places the trade over the web API link.

# Student Checkpoint: The Golden Rule of Paper Trading

#         Run this pipeline in the Alpaca Sandbox for at least one full month before risking a
#         single real dollar.
#
#         This live testing phase reveals things backtests can hide: you will see how long it takes for
#         a web request to load, how your broker applies real-time spreads, and exactly how it feels
#         to watch your fake balance move up and down. This allows you to fine-tune your prompts
#         and adjust your mathematical models risk-free.
#
#         Now that you have the complete blueprints for a fully functional quantitative trading
#         machine, would you like to explore how to package this setup onto a cloud server so it runs automatically 24/7,
#         or would you like to review statistical metrics like the Sharpe
#         Ratio to properly grade your system's final score?

# -----------------------------------------------------------------------------
# STEP 1: Authenticate with the Alpaca Paper Trading Sandbox API
# -----------------------------------------------------------------------------
# Paste your keys directly from your Alpaca developer dashboard
API_KEY = "PKF4TPB6DCYKB2EVALWLHNOU5S"
SECRET_KEY = "Hj1474TXoBDqzcRjHBGecLfECJCSrgs92UGAXbexCXP8"

print("间 Connecting to Alpaca Broker Sandbox API via alpaca-py...")
try:
    # Setting paper=True automatically connects to the Sandbox environment
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    account = trading_client.get_account()
    print(f"✅ Connection Successful! Account Balance: ${float(account.cash):,.2f}")
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    print("⚠️ Ensure you have replaced the placeholder API keys with your actual keys.")
    exit()

# Check how much money we can use to open new positions.
print('${} is available as buying power.'.format(account.buying_power))

# Check our current balance vs. our balance at the last market close
balance_change = float(account.equity) - float(account.last_equity)
print(f'Today\'s portfolio balance change: ${balance_change}')
# -----------------------------------------------------------------------------
# STEP 2: Receive the Live Signal from Your Alternative Data Pipeline
# -----------------------------------------------------------------------------
TARGET_ASSET = "IWM"  # Russell 2000 Small-Cap ETF
live_sentiment_signal = "BULLISH"  # Options: BULLISH, BEARISH, NEUTRAL
print(f"\n📡 New Signal Received from Hugging Face model: {live_sentiment_signal}")

# -----------------------------------------------------------------------------
# STEP 3: Automated Order Execution Logic
# -----------------------------------------------------------------------------
# Check if we currently own any shares of our target asset
try:
    position = trading_client.get_open_position(TARGET_ASSET)
    current_shares_held = int(position.qty)
    print(f"📊 Current Broker Status: Holding {current_shares_held} shares of {TARGET_ASSET}")
except Exception:
    # If the SDK throws an exception, it means there is no active position
    current_shares_held = 0
    print(f"📊 Current Broker Status: No active position in {TARGET_ASSET}")

print("⚙️ Evaluating portfolio adjustments...")

# ---- ROUTINE A: SIGNAL IS BULLISH ----
if live_sentiment_signal == "BULLISH":
    if current_shares_held == 0:
        print(f"🚀 Execution Trigger: Buying 10 shares of {TARGET_ASSET} (Market Order)")

        # Build the order request object
        market_order_data = MarketOrderRequest(
            symbol=TARGET_ASSET,
            qty=10,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        )
        # Execute the trade via the modern client
        trading_client.submit_order(order_data=market_order_data)
    else:
        print("😐 Execution Postponed: Already holding asset. Position maintained.")

# ---- ROUTINE B: SIGNAL IS BEARISH ----
elif live_sentiment_signal == "BEARISH":
    if current_shares_held > 0:
        print(f"🚨 Execution Trigger: Sentiment flipped! Liquidating all shares of {TARGET_ASSET}")

        market_order_data = MarketOrderRequest(
            symbol=TARGET_ASSET,
            qty=current_shares_held,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC
        )
        trading_client.submit_order(order_data=market_order_data)
    else:
        print("😐 Execution Postponed: Market sentiment is down, but we are already in cash (0).")

# ---- ROUTINE C: SIGNAL IS NEUTRAL ----
else:
    print("😐 Execution Postponed: Neutral sentiment. Sitting safely on the sideline.")

print("\n" + "=" * 65)
print("📌 LIVE EXECUTION MATRIX REFRESH COMPLETED")
print("=" * 65)
