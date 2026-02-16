# zep_v2: Hierarchical Crypto Trading System

## Overview
zep_v2 is an automated cryptocurrency trading system designed to leverage a two-stage machine learning architecture. It aims to reduce market noise and improve trade execution by separating trend prediction from entry logic.

## Architecture

The system operates on a hierarchical model structure:

### Model A: Trend & Bias Predictor
- **Objective**: Determine the macro direction and potential price targets.
- **Input**: Higher timeframe data (1-hour or Daily OHLCV).
- **Output**: 
  - **Direction**: Bullish / Bearish / Neutral.
  - **Target**: Predicted price level (e.g., "Next candle close > $90,000").
- **Function**: Acts as a filter for Model B, ensuring trades are only taken in the direction of the dominant trend.

### Model B: Execution Engine
- **Objective**: Identify precise entry and exit points with favorable risk-reward ratios.
- **Input**: Lower timeframe data (15-minute OHLCV) combined with Model A's bias signal.
- **Output**: Buy / Sell / Hold signals.
- **Logic**: Executes trades when price action and technical indicators on the 15m timeframe align with Model A's prediction.

## Data Pipeline

This project uses a specific HuggingFace dataset for historical data.

- **Dataset ID**: `zongowo111/v2-crypto-ohlcv-data`
- **Supported Pairs**: 38 Major Pairs (e.g., BTCUSDT, ETHUSDT, SOLUSDT)
- **Timeframes**: 15m, 1h, 1d
- **Format**: Parquet files with columns: `open_time`, `open`, `high`, `low`, `close`, `volume`, etc.

### Data Loading
The project includes a dedicated loader in `data/loader.py` to seamlessly fetch and format data from HuggingFace.

```python
from data.loader import load_klines

# Load 1-hour data for Bitcoin
df_btc_1h = load_klines("BTCUSDT", "1h")

# Load 15-minute data for Ethereum
df_eth_15m = load_klines("ETHUSDT", "15m")
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/caizongxun/zep_v2.git
   cd zep_v2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development Roadmap

1. **Feature Engineering**: Implement technical indicators (RSI, MACD, Bollinger Bands) for both timeframes.
2. **Model A Training**: Train a classification/regression model on 1h/1d data to predict future trends.
3. **Model B Training**: Train a reinforcement learning or supervised model for 15m execution based on Model A's signals.
4. **Backtesting**: Validate the full pipeline using historical data.
5. **Live Execution**: Integrate with exchange APIs for automated trading.

## License
MIT
