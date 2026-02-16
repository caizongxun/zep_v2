import pandas as pd
from data.loader import load_klines
from utils.features import add_technical_indicators
from models.model_a import ModelA
from models.model_b import ModelB

class TradingPipeline:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model_a = ModelA()
        self.model_b = ModelB()
        self.df_1h = pd.DataFrame()
        self.df_15m = pd.DataFrame()

    def run_training(self):
        """
        Train Model A on historical 1h data.
        """
        print(f"Loading 1h data for {self.symbol}...")
        self.df_1h = load_klines(self.symbol, "1h")
        if self.df_1h.empty:
            print("Failed to load 1h data.")
            return

        print("Training Model A...")
        self.model_a.train(self.df_1h)
        print("Model A training complete.")

    def run_inference(self):
        """
        Run the pipeline:
        1. Predict trend using Model A (1h).
        2. Generate signal using Model B (15m) based on trend.
        """
        # --- Step 1: Model A Prediction ---
        if self.df_1h.empty:
             self.df_1h = load_klines(self.symbol, "1h")
             
        # Predict trend for the *next* candle
        # FIX: We need enough history to calculate indicators (e.g. 50+ candles)
        # Passing only 1 row causes IndexError in TA library
        history_window = 100 
        input_data = self.df_1h.tail(history_window).copy()
        
        if len(input_data) < 50:
            print("Not enough 1h data for inference.")
            return {'signal': 'hold', 'reason': 'insufficient_data'}

        # Get predictions for the chunk
        # The model's predict() calls add_technical_indicators internally on this chunk
        predictions = self.model_a.predict(input_data)
        
        # We only care about the prediction for the very last closed candle
        bias = predictions[-1] 
        
        bias_str = "BULLISH" if bias == 1 else "BEARISH"
        print(f"Model A Prediction (1h): {bias_str}")

        # --- Step 2: Model B Execution ---
        print(f"Loading 15m data for {self.symbol}...")
        self.df_15m = load_klines(self.symbol, "15m")
        if self.df_15m.empty:
            print("Failed to load 15m data.")
            return {'signal': 'hold', 'reason': 'no_data'}

        # Add indicators to 15m data for Model B
        # We also need enough history here
        self.df_15m = add_technical_indicators(self.df_15m)
        
        # Generate signal
        signal = self.model_b.generate_signal(self.df_15m, bias)
        
        print(f"Model B Signal (15m): {signal}")
        return signal

if __name__ == "__main__":
    # Example Usage
    pipeline = TradingPipeline("BTCUSDT")
    pipeline.run_training()
    pipeline.run_inference()
