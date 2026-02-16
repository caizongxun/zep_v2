import pandas as pd
import numpy as np
from models.model_b import ModelB
from models.model_a import ModelA
from data.loader import load_klines
from utils.features import add_technical_indicators

class BacktestEngine:
    def __init__(self, symbol: str, leverage: float = 1.0, risk_per_trade: float = 0.02):
        self.symbol = symbol
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade  # Percentage of account to risk per trade (not used in simple version, but good for sizing)
        self.initial_capital = 10000.0
        
    def run(self, days: int, model_a: ModelA, model_b: ModelB):
        """
        Run backtest.
        Strategy:
        1. Resample 15m data to get timestamps.
        2. Merge latest available 1H trend signal to each 15m candle (using merge_asof to avoid lookahead).
        3. Iterate through 15m candles to execute Model B logic.
        """
        # 1. Load Data
        df_1h = load_klines(self.symbol, "1h")
        df_15m = load_klines(self.symbol, "15m")
        
        if df_1h.empty or df_15m.empty:
            return None
            
        # Filter for the requested days
        end_time = df_15m['open_time'].max()
        start_time = end_time - pd.Timedelta(days=days)
        
        df_15m = df_15m[df_15m['open_time'] >= start_time].copy()
        df_1h = df_1h[df_1h['open_time'] >= start_time].copy()
        
        # 2. Pre-calculate Indicators & Signals
        # Important: We calculate indicators on the full history first to avoid cold-start issues, then slice.
        df_1h = add_technical_indicators(df_1h)
        df_15m = add_technical_indicators(df_15m)
        
        # 3. Generate Model A Trends for all 1H candles
        # Note: In a real backtest, we should walk-forward train. 
        # Here we assume Model A is pre-trained or using a simple rule for speed.
        # To strictly prevent lookahead, the prediction for Time T must use data up to T-1.
        # We use 'shift(1)' to ensure we are using closed candles for prediction input.
        trends = model_a.predict(df_1h) # This returns trend for the *next* candle based on *current* row
        
        # Align: Prediction made at 10:00 (using 09:00-10:00 data) applies to 10:00-11:00 period
        # df_1h['bias'] is the signal VALID for the timeframe starting at 'open_time'
        df_1h['bias'] = trends
        
        # 4. Merge 1H Trend into 15M Data
        # We use merge_asof with direction='backward' to find the latest closed 1H signal
        df_15m = df_15m.sort_values('open_time')
        df_1h = df_1h.sort_values('open_time')
        
        # We merge on close_time of 1h to open_time of 15m to ensure we only use CLOSED 1h data
        # Actually, simpler: The signal generated at 1H Open Time T is valid for 15m candles T, T+15, T+30, T+45
        merged_df = pd.merge_asof(
            df_15m, 
            df_1h[['open_time', 'bias']], 
            on='open_time', 
            direction='backward'
        )
        
        # 5. Iterative Execution (Simulating Loop)
        balance = self.initial_capital
        equity_curve = []
        trades = []
        
        position = None # {'type': 'long'/'short', 'entry': price, 'size': amount, 'sl': price, 'tp': price}
        
        for index, row in merged_df.iterrows():
            current_price = row['close']
            current_time = row['open_time']
            bias = row['bias'] # From Model A
            
            # --- Update Portfolio Value ---
            current_equity = balance
            if position:
                if position['type'] == 'long':
                    pnl = (current_price - position['entry']) * position['size']
                else:
                    pnl = (position['entry'] - current_price) * position['size']
                current_equity += pnl
            
            equity_curve.append({'time': current_time, 'equity': current_equity})
            
            # --- Check Exit Conditions (SL/TP) ---
            if position:
                hit_sl = (position['type'] == 'long' and row['low'] <= position['sl']) or \
                         (position['type'] == 'short' and row['high'] >= position['sl'])
                         
                hit_tp = (position['type'] == 'long' and row['high'] >= position['tp']) or \
                         (position['type'] == 'short' and row['low'] <= position['tp'])
                
                if hit_sl:
                    exit_price = position['sl']
                    pnl = (exit_price - position['entry']) * position['size'] if position['type'] == 'long' else (position['entry'] - exit_price) * position['size']
                    balance += pnl
                    trades.append({'time': current_time, 'type': 'SL', 'pnl': pnl})
                    position = None
                    continue # Trade closed, wait for next candle
                
                elif hit_tp:
                    exit_price = position['tp']
                    pnl = (exit_price - position['entry']) * position['size'] if position['type'] == 'long' else (position['entry'] - exit_price) * position['size']
                    balance += pnl
                    trades.append({'time': current_time, 'type': 'TP', 'pnl': pnl})
                    position = None
                    continue

            # --- Check Entry Conditions ---
            if position is None:
                # We pass a single-row DataFrame to Model B to mimic live behavior
                # In a loop this is slow, but accurate for logic check
                # Constructing a small DF for the function
                single_row_df = pd.DataFrame([row]) 
                
                # Check Signal
                signal_pkg = model_b.generate_signal(single_row_df, bias)
                
                if signal_pkg['signal'] == 'buy':
                    entry_price = current_price # Assuming close execution
                    # Position Sizing: (Balance * Risk%) / Distance to SL -> simplified to fixed leverage here
                    # Using leverage-based sizing:
                    position_size = (balance * self.leverage) / entry_price
                    
                    position = {
                        'type': 'long',
                        'entry': entry_price,
                        'size': position_size,
                        'sl': signal_pkg['stop_loss'],
                        'tp': signal_pkg['take_profit']
                    }
                    trades.append({'time': current_time, 'type': 'OPEN_LONG', 'price': entry_price})
                    
                elif signal_pkg['signal'] == 'sell':
                    entry_price = current_price
                    position_size = (balance * self.leverage) / entry_price
                    
                    position = {
                        'type': 'short',
                        'entry': entry_price,
                        'size': position_size,
                        'sl': signal_pkg['stop_loss'],
                        'tp': signal_pkg['take_profit']
                    }
                    trades.append({'time': current_time, 'type': 'OPEN_SHORT', 'price': entry_price})

        return pd.DataFrame(equity_curve), pd.DataFrame(trades)
