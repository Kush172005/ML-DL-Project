"""
Download ETTh1 dataset for the Hybrid Temporal Forecaster project.
ETTh1 = Electricity Transformer Temperature (hourly), commonly used benchmark.
"""

import pandas as pd
from pathlib import Path
import urllib.request

def download_etth1():
    """Download ETTh1.csv from public source."""
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # ETTh1 from the ETDataset repository
    url = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv'
    output_path = data_dir / 'ETTh1.csv'
    
    print(f"Downloading ETTh1 from {url}...")
    urllib.request.urlretrieve(url, output_path)
    
    # Load and verify
    df = pd.read_csv(output_path)
    print(f"Downloaded ETTh1: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"Saved to {output_path}")
    
    return output_path

if __name__ == '__main__':
    download_etth1()
    print("\nData download complete!")
