"""
Temporal Fusion Transformer (TFT) - simplified implementation.
Trains on residuals with regime probabilities and covariates.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TFTDataset(Dataset):
    """
    Dataset for TFT: encoder (past context) + decoder (future horizon).
    """
    
    def __init__(self, residuals, regime_probs, covariates, encoder_length=48, horizon=24):
        """
        Args:
            residuals: Target residuals (n_timesteps,)
            regime_probs: Regime probabilities (n_timesteps, n_regimes)
            covariates: Covariate matrix (n_timesteps, n_covariates)
            encoder_length: Lookback window
            horizon: Forecast horizon
        """
        self.residuals = residuals
        self.regime_probs = regime_probs
        self.covariates = covariates
        self.encoder_length = encoder_length
        self.horizon = horizon
        
        # Valid indices
        self.indices = []
        for i in range(encoder_length, len(residuals) - horizon + 1):
            self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        
        # Encoder: past context
        encoder_residuals = self.residuals[i-self.encoder_length:i]
        encoder_regimes = self.regime_probs[i-self.encoder_length:i]
        encoder_covariates = self.covariates[i-self.encoder_length:i]
        
        # Combine encoder inputs
        encoder_input = np.column_stack([
            encoder_residuals.reshape(-1, 1),
            encoder_regimes,
            encoder_covariates
        ])
        
        # Decoder: future regime probs and covariates (known at forecast time in practice)
        decoder_regimes = self.regime_probs[i:i+self.horizon]
        decoder_covariates = self.covariates[i:i+self.horizon]
        
        decoder_input = np.column_stack([
            decoder_regimes,
            decoder_covariates
        ])
        
        # Target: future residuals
        target = self.residuals[i:i+self.horizon]
        
        return (
            torch.FloatTensor(encoder_input),
            torch.FloatTensor(decoder_input),
            torch.FloatTensor(target)
        )

class SimplifiedTFT(nn.Module):
    """
    Simplified TFT-inspired model: LSTM encoder + attention + decoder.
    Easier to train than full TFT, captures key ideas.
    """
    
    def __init__(self, encoder_input_size, decoder_input_size, hidden_size=64, 
                 num_layers=2, horizon=24, dropout=0.2):
        super(SimplifiedTFT, self).__init__()
        
        self.hidden_size = hidden_size
        self.horizon = horizon
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=encoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder for future context
        self.decoder_fc = nn.Linear(decoder_input_size, hidden_size // 2)
        
        # Combine encoder output with decoder context
        self.fusion = nn.Linear(hidden_size + hidden_size // 2, hidden_size)
        
        # Output layer
        self.output = nn.Linear(hidden_size, horizon)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encoder_input, decoder_input):
        """
        Args:
            encoder_input: (batch, encoder_len, encoder_features)
            decoder_input: (batch, horizon, decoder_features)
        
        Returns:
            predictions: (batch, horizon)
        """
        # Encode past
        _, (hidden, _) = self.encoder(encoder_input)
        encoder_out = hidden[-1]  # (batch, hidden_size)
        
        # Process decoder input (average over horizon for simplicity)
        decoder_out = self.decoder_fc(decoder_input.mean(dim=1))  # (batch, hidden_size//2)
        
        # Fuse
        fused = torch.cat([encoder_out, decoder_out], dim=1)
        fused = self.dropout(torch.relu(self.fusion(fused)))
        
        # Predict
        predictions = self.output(fused)
        
        return predictions

def train_tft_simple(
    train_dataset,
    val_dataset,
    encoder_input_size,
    decoder_input_size,
    hidden_size=64,
    num_layers=2,
    horizon=24,
    dropout=0.2,
    learning_rate=0.001,
    epochs=50,
    patience=10,
    batch_size=64
):
    """
    Train simplified TFT model.
    
    Returns:
        Trained model
    """
    print(f"\nTraining Simplified TFT...")
    print(f"  Encoder input size: {encoder_input_size}")
    print(f"  Decoder input size: {decoder_input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Epochs: {epochs}")
    
    # Create model
    model = SimplifiedTFT(
        encoder_input_size=encoder_input_size,
        decoder_input_size=decoder_input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        horizon=horizon,
        dropout=dropout
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    device = torch.device('cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for encoder, decoder, target in train_loader:
            encoder, decoder, target = encoder.to(device), decoder.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred = model(encoder, decoder)
            loss = criterion(pred, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for encoder, decoder, target in val_loader:
                encoder, decoder, target = encoder.to(device), decoder.to(device), target.to(device)
                pred = model(encoder, decoder)
                loss = criterion(pred, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"  Training complete. Best val loss: {best_val_loss:.6f}")
    
    return model

def predict_tft_simple(model, dataset, batch_size=128):
    """Generate predictions from trained model."""
    model.eval()
    device = next(model.parameters()).device
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    
    with torch.no_grad():
        for encoder, decoder, _ in loader:
            encoder, decoder = encoder.to(device), decoder.to(device)
            pred = model(encoder, decoder)
            predictions.append(pred.cpu().numpy())
    
    return np.vstack(predictions)

if __name__ == '__main__':
    print("TFT module loaded. Run scripts/run_all.py for full pipeline.")
