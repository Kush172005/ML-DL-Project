import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data, target, seq_len=48, horizon=24):
        self.data = torch.FloatTensor(data)
        self.target = torch.FloatTensor(target)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.target[idx + self.seq_len : idx + self.seq_len + self.horizon]
        return x, y

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    # Basic cleaning
    df = df.ffill()
    
    # Target and Features
    target_col = 'OT'
    feature_cols = [c for c in df.columns if c not in ['date', target_col]]
    
    # Normalization (Standard Research Practice)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[feature_cols + [target_col]] = scaler.fit_transform(df[feature_cols + [target_col]])
    
    return df[feature_cols].values, df[target_col].values, len(feature_cols)
