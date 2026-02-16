import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.pipeline import TradingPipeline
from utils.features import add_technical_indicators

# Page Configuration
st.set_page_config(
    page_title="ZEP v2 Trading System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utilities ---
def plot_candlestick(df: pd.DataFrame, title: str, indicators: list = None):
    """
    Creates a Plotly candlestick chart with optional indicators.
    """
    if df.empty:
        return None
        
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['open_time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))

    # Add Indicators
    if indicators:
        if 'ema_50' in indicators and 'ema_50' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['open_time'], 
                y=df['ema_50'], 
                line=dict(color='orange', width=1), 
                name='EMA 50'
            ))
        if 'bb_high' in indicators and 'bb_high' in df.columns:
             fig.add_trace(go.Scatter(
                x=df['open_time'], 
                y=df['bb_high'], 
                line=dict(color='gray', width=1, dash='dot'), 
                name='BB High'
            ))
             fig.add_trace(go.Scatter(
                x=df['open_time'], 
                y=df['bb_low'], 
                line=dict(color='gray', width=1, dash='dot'), 
                name='BB Low'
            ))

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Price',
        height=600,
        template="plotly_dark"
    )
    return fig

# --- Main Application ---

st.title("ZEP v2: Hierarchical Trading System")

# Sidebar Controls
st.sidebar.header("Configuration")
symbol = st.sidebar.selectbox(
    "Select Symbol", 
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]
)

model_a_horizon = st.sidebar.slider("Model A Horizon (Hours)", 1, 24, 1)
risk_reward = st.sidebar.number_input("Risk/Reward Ratio", value=2.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Actions")

# Initialize Pipeline
if 'pipeline' not in st.session_state or st.session_state.symbol != symbol:
    st.session_state.pipeline = TradingPipeline(symbol)
    st.session_state.symbol = symbol
    st.sidebar.info(f"Initialized pipeline for {symbol}")

pipeline = st.session_state.pipeline
pipeline.model_b.risk_reward_ratio = risk_reward

# Training Action
if st.sidebar.button("Train Model A (1H Trend)"):
    with st.spinner("Loading data and training Model A..."):
        pipeline.run_training()
    st.success("Model A training complete!")

# Inference Action
if st.sidebar.button("Run Analysis & Generate Signal"):
    with st.spinner("Fetching data and running inference..."):
        signal = pipeline.run_inference()
        
        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model A: Trend Analysis (1H)")
            if not pipeline.df_1h.empty:
                # Re-calculate bias to show user
                latest_pred = pipeline.model_a.predict(pipeline.df_1h.tail(1))
                bias = latest_pred[0]
                trend_text = "BULLISH" if bias == 1 else "BEARISH"
                trend_color = "green" if bias == 1 else "red"
                st.markdown(f"### Predicted Trend: :{trend_color}[{trend_text}]")
                
                # Plot 1H Chart
                fig_1h = plot_candlestick(pipeline.df_1h.tail(100), f"{symbol} 1H Trend Context", ['ema_50'])
                st.plotly_chart(fig_1h, use_container_width=True)
            else:
                st.warning("1H Data not available. Train model first.")

        with col2:
            st.subheader("Model B: Execution (15m)")
            if not pipeline.df_15m.empty:
                st.write(f"**Action Signal:** {signal['signal'].upper()}")
                
                if signal['signal'] != 'hold':
                    st.success(f"Entry: {signal['entry_price']}")
                    st.error(f"Stop Loss: {signal['stop_loss']}")
                    st.info(f"Take Profit: {signal['take_profit']}")
                else:
                    st.info("Waiting for setup...")
                
                # Plot 15m Chart
                fig_15m = plot_candlestick(pipeline.df_15m.tail(100), f"{symbol} 15m Execution View", ['ema_50', 'bb_high', 'bb_low'])
                
                # Add Entry Line if signal exists
                if signal['signal'] != 'hold':
                    fig_15m.add_hline(y=signal['entry_price'], line_color="blue", annotation_text="Entry")
                    fig_15m.add_hline(y=signal['stop_loss'], line_color="red", annotation_text="SL")
                    fig_15m.add_hline(y=signal['take_profit'], line_color="green", annotation_text="TP")
                
                st.plotly_chart(fig_15m, use_container_width=True)
                
                # Show Indicator Values
                latest = pipeline.df_15m.iloc[-1]
                st.metric("RSI (14)", f"{latest['rsi']:.2f}")
                st.metric("MACD Diff", f"{latest['macd_diff']:.4f}")
            else:
                st.warning("15m Data not loaded.")

st.sidebar.markdown("---")
st.sidebar.markdown("v2.0.0 | Hierarchical Crypto Trading System")
