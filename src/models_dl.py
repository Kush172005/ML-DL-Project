"""
Deep Learning models for multi-step time series forecasting.
LSTM/GRU encoder with linear decoder for horizon-step predictions.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series with sliding windows.
    """
    
    def __init__(self, data, window_size=20, horizon=5, target_col_idx=0):
        """
        Args:
            data: Array of shape (n_timesteps, n_features)
            window_size: Length of input sequence
            horizon: Number of steps to predict
            target_col_idx: Index of target column in data
        """
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.target_col_idx = target_col_idx
        
        # Create valid indices
        self.indices = []
        for i in range(len(data) - window_size - horizon + 1):
            self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_size
        
        # Input sequence
        X = self.data[start_idx:end_idx]
        
        # Target: next horizon steps of target column
        y = self.data[end_idx:end_idx + self.horizon, self.target_col_idx]
        
        return torch.FloatTensor(X), torch.FloatTensor(y)

class LSTMForecaster(nn.Module):
    """
    LSTM-based multi-step forecaster.
    Architecture: LSTM encoder -> last hidden state -> Linear decoder
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, horizon=5, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.decoder = nn.Linear(hidden_size, horizon)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
        
        Returns:
            predictions: (batch_size, horizon)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_size)
        
        # Decode to horizon predictions
        predictions = self.decoder(last_hidden)
        
        return predictions

class GRUForecaster(nn.Module):
    """
    GRU-based multi-step forecaster (alternative to LSTM).
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, horizon=5, dropout=0.2):
        super(GRUForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.decoder = nn.Linear(hidden_size, horizon)
    
    def forward(self, x):
        gru_out, hidden = self.gru(x)
        last_hidden = hidden[-1]
        predictions = self.decoder(last_hidden)
        return predictions

class DLForecasterTrainer:
    """
    Trainer class for deep learning forecasters with early stopping.
    """
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, epochs=100, patience=10, verbose=True):
        """
        Train with early stopping.
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Print progress
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        if verbose:
            print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
        
        return self
    
    def predict(self, data_loader):
        """Generate predictions."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for X_batch, _ in data_loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch)
                predictions.append(pred.cpu().numpy())
        
        return np.vstack(predictions)
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {path}")
        return self

def create_dl_forecaster(model_type='lstm', input_size=1, hidden_size=64, 
                        num_layers=2, horizon=5, dropout=0.2):
    """
    Factory function to create DL forecaster.
    
    Args:
        model_type: 'lstm' or 'gru'
        input_size: Number of input features
        hidden_size: Hidden layer size
        num_layers: Number of recurrent layers
        horizon: Forecast horizon
        dropout: Dropout rate
    """
    if model_type == 'lstm':
        return LSTMForecaster(input_size, hidden_size, num_layers, horizon, dropout)
    elif model_type == 'gru':
        return GRUForecaster(input_size, hidden_size, num_layers, horizon, dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

if __name__ == '__main__':
    # Test DL models
    print("Testing LSTM forecaster...")
    
    # Create synthetic data
    np.random.seed(42)
    data = np.cumsum(np.random.randn(1000, 3), axis=0) * 0.1
    
    # Create dataset
    dataset = TimeSeriesDataset(data, window_size=20, horizon=5, target_col_idx=0)
    print(f"Dataset size: {len(dataset)}")
    
    # Create model
    model = create_dl_forecaster(model_type='lstm', input_size=3, hidden_size=32, 
                                 num_layers=2, horizon=5)
    print(f"Model created: {model}")
    
    # Test forward pass
    X_sample, y_sample = dataset[0]
    X_sample = X_sample.unsqueeze(0)  # Add batch dimension
    pred = model(X_sample)
    print(f"Prediction shape: {pred.shape}")
    
    print("\nDL models ready!")
