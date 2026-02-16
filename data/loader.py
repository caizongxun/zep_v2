from huggingface_hub import hf_hub_download
import pandas as pd
import os

def load_klines(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load cryptocurrency candle data from HuggingFace dataset.
    
    Args:
        symbol: Trading pair, e.g., 'BTCUSDT', 'ETHUSDT'
        timeframe: '15m', '1h', '1d'
        
    Returns:
        pd.DataFrame: DataFrame with columns [open_time, open, high, low, close, volume, ...]
    """
    repo_id = "zongowo111/v2-crypto-ohlcv-data"
    base = symbol.replace("USDT", "")
    filename = f"{base}_{timeframe}.parquet"
    path_in_repo = f"klines/{symbol}/{filename}"

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset"
        )
        return pd.read_parquet(local_path)
    except Exception as e:
        print(f"Error loading data for {symbol} {timeframe}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test loading
    try:
        df = load_klines("BTCUSDT", "1h")
        if not df.empty:
            print(f"Successfully loaded BTCUSDT 1h data with {len(df)} rows.")
            print(df.head())
        else:
            print("Failed to load data.")
    except Exception as e:
        print(f"An error occurred: {e}")
