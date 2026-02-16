import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils.features import add_technical_indicators, add_target_model_a

class ModelA:
    """
    Model A: Trend Prediction
    Predicts the direction of the next candle (1 for Up, 0 for Down).
    """
    def __init__(self):
        self.model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        
    def prepare_data(self, df: pd.DataFrame):
        """
        Prepares data for training/inference.
        Uses ONLY closed candles logic.
        """
        # Ensure we are working on a copy
        df = df.copy()
        
        # Calculate indicators
        df = add_technical_indicators(df)
        
        # Create Target: Direction of NEXT candle
        # We shift(-1) so that row T has the label of T+1's return
        df = add_target_model_a(df)
        
        # Select features
        features = [col for col in df.columns if col not in ['open_time', 'close_time', 'future_close', 'target_return', 'target_direction']]
        X = df[features]
        y = df['target_direction']
        
        return X, y
        
    def train(self, df: pd.DataFrame):
        """
        Trains the model.
        """
        X, y = self.prepare_data(df)
        # Using time-series split (no shuffle)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        print("Training Model A...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print("Model A Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        
    def predict(self, df: pd.DataFrame):
        """
        Predicts future direction based on input data.
        CRITICAL: Input 'df' should contain CLOSED candles.
        If the last row is an open/incomplete candle, it should be removed before calling this,
        OR the caller accepts that the prediction will fluctuate.
        
        For production safety, we assume input is history up to time T.
        We predict T+1.
        """
        df = add_technical_indicators(df)
        features = [col for col in df.columns if col not in ['open_time', 'close_time', 'future_close', 'target_return', 'target_direction']]
        
        # Ensure we have the exact same features as training
        # (In a real app, save/load feature names to ensure consistency)
        # Here we just re-select based on the column logic
        X = df[features]
        
        return self.model.predict(X)
