import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.pipeline import TradingPipeline
from core.backtest import BacktestEngine
from utils.features import add_technical_indicators

# Page Configuration
st.set_page_config(
    page_title="ZEP v2 虛擬貨幣交易系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utilities ---
def plot_candlestick(df: pd.DataFrame, title: str, indicators: list = None, trades: list = None):
    """
    Creates a Plotly candlestick chart with optional indicators and trade markers.
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
        name='價格'
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
                name='BB 上軌'
            ))
             fig.add_trace(go.Scatter(
                x=df['open_time'], 
                y=df['bb_low'], 
                line=dict(color='gray', width=1, dash='dot'), 
                name='BB 下軌'
            ))

    # Add Trades
    if trades is not None and not trades.empty:
        # Long Entries
        long_entries = trades[trades['type'] == 'OPEN_LONG']
        if not long_entries.empty:
             fig.add_trace(go.Scatter(
                x=long_entries['time'], y=long_entries['price'],
                mode='markers', marker=dict(symbol='triangle-up', size=10, color='blue'),
                name='做多進場'
            ))
        # Short Entries
        short_entries = trades[trades['type'] == 'OPEN_SHORT']
        if not short_entries.empty:
             fig.add_trace(go.Scatter(
                x=short_entries['time'], y=short_entries['price'],
                mode='markers', marker=dict(symbol='triangle-down', size=10, color='orange'),
                name='做空進場'
            ))
        # Exits (SL/TP)
        exits = trades[trades['type'].isin(['SL', 'TP'])]
        if not exits.empty:
             fig.add_trace(go.Scatter(
                x=exits['time'], y=[0]*len(exits), 
                mode='markers', marker=dict(symbol='x', size=8, color='red'),
                name='出場'
            ))

    fig.update_layout(
        title=title,
        xaxis_title='時間',
        yaxis_title='價格',
        height=600,
        template="plotly_dark"
    )
    return fig

# --- Main Application ---

st.title("ZEP v2: 分層式虛擬貨幣交易系統")

# Sidebar Controls
st.sidebar.header("系統配置")
symbol = st.sidebar.selectbox(
    "選擇交易對", 
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]
)

# Initialize Pipeline
if 'pipeline' not in st.session_state or st.session_state.symbol != symbol:
    st.session_state.pipeline = TradingPipeline(symbol)
    st.session_state.symbol = symbol
    st.sidebar.info(f"已初始化 {symbol} 管道")

pipeline = st.session_state.pipeline

# Tabs
tab1, tab2 = st.tabs(["即時訊號分析", "歷史回測系統"])

with tab1:
    st.header("即時市場分析")
    
    col_a, col_b = st.columns([1, 3])
    
    with col_a:
        st.subheader("控制面板")
        risk_reward = st.number_input("盈虧比 (Risk/Reward)", value=2.0, step=0.1)
        pipeline.model_b.risk_reward_ratio = risk_reward
        
        # Changed button text and behavior
        if st.button("訓練雙模型系統 (Model A & B)"):
            with st.spinner("正在執行雙階段訓練流程..."):
                pipeline.run_training()
            st.success("模型 A (趨勢) 與 模型 B (執行) 皆訓練完成！")

        if st.button("執行分析與生成訊號"):
            with st.spinner("正在獲取數據並進行推論..."):
                signal = pipeline.run_inference()
                st.session_state.last_signal = signal
    
    with col_b:
        if 'last_signal' in st.session_state:
            signal = st.session_state.last_signal
            
            # Trend Display
            if not pipeline.df_1h.empty:
                # Need to run predict again or cache it, run_inference prints it but we want to show it
                # For UI update we re-predict just the bias
                input_data = pipeline.df_1h.tail(100).copy()
                latest_pred = pipeline.model_a.predict(input_data)
                bias = latest_pred[-1]
                
                trend_text = "看漲 (BULLISH)" if bias == 1 else "看跌 (BEARISH)"
                trend_color = "green" if bias == 1 else "red"
                st.markdown(f"### 模型 A 趨勢預測: :{trend_color}[{trend_text}]")
            
            # Signal Display
            st.markdown("#### 模型 B 執行訊號 (15m)")
            
            sig_type = signal['signal'].upper()
            if sig_type == 'HOLD':
                st.info(f"當前動作: {sig_type} (等待機會)")
            else:
                st.success(f"當前動作: {sig_type}")
                m1, m2, m3 = st.columns(3)
                m1.metric("進場價格", f"{signal['entry_price']:.2f}")
                m2.metric("止損價格 (SL)", f"{signal['stop_loss']:.2f}")
                m3.metric("止盈價格 (TP)", f"{signal['take_profit']:.2f}")
            
            # Chart
            if not pipeline.df_15m.empty:
                fig_15m = plot_candlestick(pipeline.df_15m.tail(100), f"{symbol} 15m 執行視圖", ['ema_50', 'bb_high', 'bb_low'])
                
                # Add Lines
                if signal['signal'] != 'hold':
                    fig_15m.add_hline(y=signal['entry_price'], line_color="blue", annotation_text="Entry")
                    fig_15m.add_hline(y=signal['stop_loss'], line_color="red", annotation_text="SL")
                    fig_15m.add_hline(y=signal['take_profit'], line_color="green", annotation_text="TP")
                
                st.plotly_chart(fig_15m, use_container_width=True)

with tab2:
    st.header("歷史回測系統")
    
    bc1, bc2 = st.columns([1, 3])
    
    with bc1:
        st.subheader("回測參數")
        bt_days = st.number_input("回測天數", min_value=1, max_value=365, value=30)
        bt_leverage = st.number_input("槓桿倍數", min_value=1.0, max_value=20.0, value=1.0)
        bt_risk = st.slider("每筆交易倉位 (佔餘額 %)", 1, 100, 100)
        
        if st.button("開始回測"):
            engine = BacktestEngine(symbol, leverage=bt_leverage)
            if pipeline.df_1h.empty:
                 with st.spinner("尚未訓練模型，正在自動訓練..."):
                     pipeline.run_training()
            
            with st.spinner(f"正在回測過去 {bt_days} 天的數據..."):
                equity_curve, trades = engine.run(bt_days, pipeline.model_a, pipeline.model_b)
                st.session_state.bt_results = (equity_curve, trades)
    
    with bc2:
        if 'bt_results' in st.session_state:
            equity_curve, trades = st.session_state.bt_results
            
            if equity_curve is not None and not equity_curve.empty:
                # Metrics
                initial = equity_curve.iloc[0]['equity']
                final = equity_curve.iloc[-1]['equity']
                roi = ((final - initial) / initial) * 100
                drawdown = (equity_curve['equity'].cummax() - equity_curve['equity']).max()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("總回報率 (ROI)", f"{roi:.2f}%")
                m2.metric("最終權益", f"${final:.2f}")
                m3.metric("最大回撤", f"${drawdown:.2f}")
                
                # Equity Plot
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=equity_curve['time'], y=equity_curve['equity'], mode='lines', name='權益曲線'))
                fig_eq.update_layout(title="帳戶權益曲線", xaxis_title="時間", yaxis_title="金額 (USD)", template="plotly_dark")
                st.plotly_chart(fig_eq, use_container_width=True)
                
                # Trades Table
                st.subheader("交易紀錄")
                st.dataframe(trades)
            else:
                st.warning("無回測數據或期間內無交易。")

st.sidebar.markdown("---")
st.sidebar.markdown("v2.2.0 | 系統開發者: ZEP")
