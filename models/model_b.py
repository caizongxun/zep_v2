import pandas as pd
import numpy as np
from utils.features import add_technical_indicators

class ModelB:
    """
    Model B: Execution Engine
    Executes trades on lower timeframe (15m) based on Model A's bias.
    """
    def __init__(self, risk_reward_ratio: float = 2.0):
        self.risk_reward_ratio = risk_reward_ratio

    def generate_signal(self, df_15m: pd.DataFrame, bias: int) -> dict:
        """
        Generates buy/sell/hold signal based on bias and 15m indicators.
        
        Args:
            df_15m: 15-minute timeframe DataFrame with indicators.
            bias: 1 (Bullish), 0 (Neutral/Bearish) from Model A.
            
        Returns:
            dict: {
                'signal': 'buy' | 'sell' | 'hold',
                'entry_price': float,
                'stop_loss': float,
                'take_profit': float
            }
        """
        if df_15m.empty:
            return {'signal': 'hold'}
        
        latest = df_15m.iloc[-1]
        
        # --- Bullish Strategy (Bias = 1) ---
        if bias == 1:
            # Condition 1: RSI not overbought (< 70)
            # Condition 2: Price > EMA 50 (Uptrend on 15m)
            # Condition 3: MACD Histogram is positive (Momentum)
            if latest['rsi'] < 70 and latest['close'] > latest['ema_50'] and latest['macd_diff'] > 0:
                entry_price = latest['close']
                atr = latest['atr']
                
                stop_loss = entry_price - (2 * atr)
                take_profit = entry_price + (2 * atr * self.risk_reward_ratio)
                
                return {
                    'signal': 'buy',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
        
        # --- Bearish Strategy (Bias = 0) ---
        elif bias == 0:
             # Condition 1: RSI not oversold (> 30)
             # Condition 2: Price < EMA 50 (Downtrend on 15m)
             # Condition 3: MACD Histogram is negative
            if latest['rsi'] > 30 and latest['close'] < latest['ema_50'] and latest['macd_diff'] < 0:
                entry_price = latest['close']
                atr = latest['atr']
                
                stop_loss = entry_price + (2 * atr)
                take_profit = entry_price - (2 * atr * self.risk_reward_ratio)
                
                return {
                    'signal': 'sell',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                
        return {'signal': 'hold'}

if __name__ == "__main__":
    # Test stub
    pass
