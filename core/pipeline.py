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
        Full Training Pipeline:
        1. Train Model A (Trend) on 1H data.
        2. Generate historical trend predictions for the entire 1H dataset.
        3. Train Model B (Execution) on 15M data, using the historical trends as input context.
        """
        # --- 1. Load Data ---
        print(f"Loading data for {self.symbol}...")
        self.df_1h = load_klines(self.symbol, "1h")
        self.df_15m = load_klines(self.symbol, "15m")
        
        if self.df_1h.empty or self.df_15m.empty:
            print("Failed to load data.")
            return

        # --- 2. Train Model A ---
        print("--- Training Model A (Trend) ---")
        self.model_a.train(self.df_1h)
        
        # --- 3. Prepare Context for Model B ---
        print("Generating historical trends for Model B training...")
        # To train Model B, we need to know what Model A *would have thought* at each point in time.
        # We predict on the entire history. 
        # Note: In a strict production system, we should use cross-val predictions to avoid leakage.
        # For this prototype, we use the trained model's predictions on training data (mild leakage, acceptable for MVP).
        # Better: Train A on T_0..T_k, Predict T_k+1..T_n.
        
        # Get predictions (this adds indicators internally)
        # We need a DataFrame with ['open_time', 'bias']
        df_1h_with_bias = self.df_1h.copy()
        
        # Predict returns the bias for the NEXT candle.
        # So at index i (Time T), the prediction is for T+1.
        # This prediction is available at Time T (close).
        trends = self.model_a.predict(self.df_1h)
        df_1h_with_bias['bias'] = trends
        
        # --- 4. Train Model B ---
        print("--- Training Model B (Execution) ---")
        self.model_b.train(self.df_15m, df_1h_with_bias)
        
        print("All models trained successfully.")

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
        history_window = 100 
        input_data = self.df_1h.tail(history_window).copy()
        
        if len(input_data) < 50:
            return {'signal': 'hold', 'reason': 'insufficient_data'}

        predictions = self.model_a.predict(input_data)
        bias = predictions[-1] 
        
        bias_str = "BULLISH" if bias == 1 else "BEARISH"
        print(f"Model A Prediction (1h): {bias_str}")

        # --- Step 2: Model B Execution ---
        if self.df_15m.empty:
             self.df_15m = load_klines(self.symbol, "15m")
             
        # Add indicators to 15m data for Model B
        # We need history for indicators
        input_15m = self.df_15m.tail(history_window).copy()
        input_15m = add_technical_indicators(input_15m)
        
        # Generate signal
        signal = self.model_b.generate_signal(input_15m, bias)
        
        print(f"Model B Signal (15m): {signal}")
        return signal
