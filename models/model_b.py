import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils.features import add_technical_indicators

class ModelB:
    """
    Model B: Execution Model (Machine Learning)
    Learns to predict if a trade will be profitable given the current state and Model A's bias.
    """
    def __init__(self, risk_reward_ratio: float = 2.0):
        self.risk_reward_ratio = risk_reward_ratio
        # Model for Buying
        self.model_long = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        # Model for Selling (Optional, or can use one model with multiclass)
        # For simplicity, we'll focus on a single model that predicts "Success" given a direction
        self.model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Helper to select only valid numeric feature columns.
        """
        exclude_cols = [
            'open_time', 'close_time', 'ignore', 
            'future_close', 'target_return', 'target_direction',
            'trade_result', 'bias' # bias is a feature, but we might handle it explicitly
        ]
        # We WANT 'bias' as a feature
        features = [col for col in df.columns if col not in exclude_cols or col == 'bias']
        return df[features].select_dtypes(include=['number']).columns.tolist()

    def prepare_labels(self, df: pd.DataFrame, atr_multiplier: float = 2.0):
        """
        Create target labels: 1 if trade hits TP before SL, 0 otherwise.
        We simulate a trade at every candle to see the outcome.
        """
        df = df.copy()
        targets = []
        
        # This is computationally expensive in Python loops, optimizing with numpy is better
        # But for clarity/safety we stick to a simple loop or lookahead for now.
        # Window to look ahead (e.g., max 48 candles = 12 hours)
        lookahead = 48 
        
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        atrs = df['atr'].values
        biases = df['bias'].values if 'bias' in df.columns else np.zeros(len(df))
        
        for i in range(len(df) - lookahead):
            entry_price = closes[i]
            current_atr = atrs[i]
            bias = biases[i]
            
            if np.isnan(current_atr) or current_atr == 0:
                targets.append(0)
                continue

            # Define SL/TP based on bias
            if bias == 1: # Bullish Case
                tp = entry_price + (current_atr * self.risk_reward_ratio)
                sl = entry_price - (current_atr * 1.0) # Tighter SL for labeling? Or match strategy
            elif bias == 0: # Bearish Case (Model A output 0)
                # Wait, Model A 0 might mean Bearish or Neutral. 
                # Let's assume we train B to execute ONLY when A says 1 (for Long) or 0 (for Short)?
                # For simplicity, let's assume we only train Longs when Bias=1 and Shorts when Bias=0
                tp = entry_price - (current_atr * self.risk_reward_ratio)
                sl = entry_price + (current_atr * 1.0)
            else:
                targets.append(0)
                continue
                
            # Check outcome in window
            outcome = 0 # 0 = Fail/Neutral, 1 = Success
            
            # Future window
            window_highs = highs[i+1 : i+1+lookahead]
            window_lows = lows[i+1 : i+1+lookahead]
            
            for j in range(len(window_highs)):
                h = window_highs[j]
                l = window_lows[j]
                
                if bias == 1: # Long
                    if l <= sl: # Hit SL first
                        outcome = 0
                        break
                    if h >= tp: # Hit TP
                        outcome = 1
                        break
                else: # Short
                    if h >= sl: # Hit SL first
                        outcome = 0
                        break
                    if l <= tp: # Hit TP
                        outcome = 1
                        break
            
            targets.append(outcome)
            
        # Pad the end
        targets.extend([0] * lookahead)
        df['trade_result'] = targets
        return df

    def train(self, df_15m: pd.DataFrame, df_1h_bias: pd.DataFrame):
        """
        Train Model B.
        df_15m: 15m candle data
        df_1h_bias: DataFrame with ['open_time', 'bias'] from Model A
        """
        print("Preparing data for Model B...")
        
        # Merge Bias into 15m
        df = pd.merge_asof(
            df_15m.sort_values('open_time'),
            df_1h_bias.sort_values('open_time')[['open_time', 'bias']],
            on='open_time',
            direction='backward'
        )
        df = df.dropna()
        df = add_technical_indicators(df)
        
        # Create Labels
        df = self.prepare_labels(df)
        
        # Features
        features = self._get_feature_columns(df)
        X = df[features]
        y = df['trade_result']
        
        # Filter: Only train on data where there is a clear bias?
        # Or let the model learn that bias=neutral leads to no trade.
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        print("Training Model B (Execution)...")
        # Handle class imbalance (successful trades are rare)
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
        
        self.model.fit(X_train, y_train) #, scale_pos_weight=scale_pos_weight)
        
        y_pred = self.model.predict(X_test)
        print("Model B Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def generate_signal(self, df_15m: pd.DataFrame, bias: int) -> dict:
        """
        Predict if we should enter NOW.
        """
        if df_15m.empty:
             return {'signal': 'hold'}
             
        # Prepare single row for prediction
        df = df_15m.copy()
        df['bias'] = bias # Inject current bias
        
        # Features must match training
        features = self._get_feature_columns(df)
        X = df[features].tail(1)
        
        # Predict
        prediction = self.model.predict(X)[0]
        # probability = self.model.predict_proba(X)[0][1] # Can use threshold
        
        if prediction == 1:
            latest = df.iloc[-1]
            atr = latest['atr']
            entry_price = latest['close']
            
            if bias == 1:
                return {
                    'signal': 'buy',
                    'entry_price': entry_price,
                    'stop_loss': entry_price - atr,
                    'take_profit': entry_price + (atr * self.risk_reward_ratio)
                }
            else:
                 return {
                    'signal': 'sell',
                    'entry_price': entry_price,
                    'stop_loss': entry_price + atr,
                    'take_profit': entry_price - (atr * self.risk_reward_ratio)
                }
        
        return {'signal': 'hold'}
