"""
Phase 2: TemporalHybridNet — BiLSTM encoder with multi-head self-attention and
gated residual connection for residual forecasting in the hybrid pipeline.

Replaces Phase 1's SimplifiedTFT (vanilla LSTM + mean-pool) with a more
principled architecture while keeping the same public API so run_all.py
and ablation.py work without changes.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR


class TFTDataset(Dataset):
    """
    Sliding-window dataset for the hybrid residual model.

    Each sample contains:
      encoder — past [residual | regime_probs | covariates] window
      decoder — future [regime_probs | covariates] window (known at forecast time)
      target  — future residual values to predict
    """

    def __init__(self, residuals, regime_probs, covariates, encoder_length=48, horizon=24):
        self.residuals = residuals
        self.regime_probs = regime_probs
        self.covariates = covariates
        self.encoder_length = encoder_length
        self.horizon = horizon

        self.indices = list(range(encoder_length, len(residuals) - horizon + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        enc_resid = self.residuals[i - self.encoder_length:i].reshape(-1, 1)
        enc_regime = self.regime_probs[i - self.encoder_length:i]
        enc_cov = self.covariates[i - self.encoder_length:i]
        encoder_input = np.concatenate([enc_resid, enc_regime, enc_cov], axis=1)

        dec_regime = self.regime_probs[i:i + self.horizon]
        dec_cov = self.covariates[i:i + self.horizon]
        decoder_input = np.concatenate([dec_regime, dec_cov], axis=1)

        target = self.residuals[i:i + self.horizon]

        return (
            torch.FloatTensor(encoder_input),
            torch.FloatTensor(decoder_input),
            torch.FloatTensor(target),
        )


class GatedResidualBlock(nn.Module):
    """
    Gated residual connection: output = LayerNorm(x + gate * transformed_x).

    The gate is a learned sigmoid scalar per feature, giving the network
    explicit control over how much of the transformed signal to add back.
    This prevents gradient degradation and lets lower layers remain stable
    even as the attention block is trained.
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        g = torch.sigmoid(self.gate(residual))
        out = self.dropout(torch.relu(self.fc(x)))
        return self.norm(residual + g * out)


class TemporalHybridNet(nn.Module):
    """
    Phase 2 architecture for residual forecasting.

    Encoder path:
      1. BiLSTM (2 layers, hidden=64) — processes past context bidirectionally.
         Bidirectional is valid here because at inference time the full encoder
         window is already observed; we are not doing online step-ahead prediction.
      2. Multi-Head Self-Attention (2 heads) over BiLSTM output sequence —
         captures non-local temporal dependencies the recurrence cannot model
         efficiently (e.g., same-hour-of-day patterns 24 steps apart).
      3. Gated Residual connection around attention — stabilises gradients and
         lets the LSTM output pass through unchanged when attention adds noise.

    Decoder path:
      Linear projection over the decoder sequence, then mean-pool to a single
      context vector. More principled than Phase 1's plain mean-pool because
      the projection learns which decoder features matter before pooling.

    Fusion:
      Concat encoder context + decoder context → MLP → linear to horizon outputs.

    Weight initialisation:
      FC layers: Xavier uniform. LSTM forget-gate bias set to 1 to discourage
      early forgetting of long-range context.
    """

    def __init__(self, encoder_input_size, decoder_input_size,
                 hidden_size=64, num_layers=2, horizon=24,
                 dropout=0.2, n_heads=2, bidirectional=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.horizon = horizon
        self.bidirectional = bidirectional
        directions = 2 if bidirectional else 1
        lstm_out = hidden_size * directions

        self.encoder = nn.LSTM(
            input_size=encoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Project BiLSTM outputs to d_model for attention
        self.encoder_proj = nn.Linear(lstm_out, hidden_size)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.gated_residual = GatedResidualBlock(hidden_size, dropout=dropout)

        self.decoder_proj = nn.Linear(decoder_input_size, hidden_size // 2)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.encoder.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Set forget-gate bias to 1 (bias is [input, forget, cell, output])
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)

        for module in [self.encoder_proj, self.decoder_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        for module in self.fusion:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, encoder_input, decoder_input):
        """
        encoder_input : (batch, encoder_len, encoder_features)
        decoder_input : (batch, horizon, decoder_features)
        returns       : (batch, horizon) residual predictions
        """
        lstm_out, _ = self.encoder(encoder_input)   # (B, T, lstm_out)
        proj = torch.relu(self.encoder_proj(lstm_out))  # (B, T, hidden)

        attn_out, _ = self.attention(proj, proj, proj)  # (B, T, hidden)
        enc_ctx = self.gated_residual(attn_out, proj)   # (B, T, hidden)
        enc_ctx = enc_ctx[:, -1, :]                     # last step context

        dec_proj = torch.relu(self.decoder_proj(decoder_input))  # (B, H, hidden//2)
        dec_ctx = dec_proj.mean(dim=1)                           # (B, hidden//2)

        fused = torch.cat([enc_ctx, dec_ctx], dim=1)
        return self.fusion(fused)


# Keep old name as an alias so any external import of SimplifiedTFT still works
SimplifiedTFT = TemporalHybridNet


def _warmup_cosine_schedule(warmup_steps, total_steps):
    """Linear warm-up then cosine decay to zero."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


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
    batch_size=64,
    weight_decay=1e-4,
    n_heads=2,
    bidirectional=True,
):
    """
    Train TemporalHybridNet.

    Uses AdamW (weight decay = L2 regularisation decoupled from adaptive
    gradient scaling) and a warm-up + cosine LR schedule. Returns the trained
    model AND training/validation loss history for learning curve plots.
    """
    print("\nTraining TemporalHybridNet (Phase 2)...")
    print(f"  encoder_input_size={encoder_input_size}, decoder_input_size={decoder_input_size}")
    print(f"  hidden={hidden_size}, layers={num_layers}, heads={n_heads}, bidirectional={bidirectional}")

    model = TemporalHybridNet(
        encoder_input_size=encoder_input_size,
        decoder_input_size=decoder_input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        horizon=horizon,
        dropout=dropout,
        n_heads=n_heads,
        bidirectional=bidirectional,
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    device = torch.device('cpu')
    model = model.to(device)

    # AdamW decouples weight decay from the adaptive gradient update (unlike Adam + L2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    total_steps = epochs * len(train_loader)
    warmup_steps = min(10 * len(train_loader), total_steps // 10)
    scheduler = LambdaLR(optimizer, _warmup_cosine_schedule(warmup_steps, total_steps))

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train = 0.0
        for enc, dec, tgt in train_loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
            optimizer.zero_grad()
            pred = model(enc, dec)
            loss = criterion(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_train += loss.item()
        epoch_train /= len(train_loader)

        model.eval()
        epoch_val = 0.0
        with torch.no_grad():
            for enc, dec, tgt in val_loader:
                enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
                epoch_val += criterion(model(enc, dec), tgt).item()
        epoch_val /= len(val_loader)

        train_losses.append(epoch_train)
        val_losses.append(epoch_val)

        if (epoch + 1) % 5 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1}/{epochs}  train={epoch_train:.6f}  val={epoch_val:.6f}  lr={lr_now:.6f}")

        if epoch_val < best_val_loss:
            best_val_loss = epoch_val
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    print(f"  Best val loss: {best_val_loss:.6f}")

    # Attach loss history to model so run_all.py can retrieve it
    model.train_losses = train_losses
    model.val_losses = val_losses

    return model


def predict_tft_simple(model, dataset, batch_size=128):
    """Batched inference; returns (n_samples, horizon) array."""
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for enc, dec, _ in loader:
            enc, dec = enc.to(device), dec.to(device)
            preds.append(model(enc, dec).cpu().numpy())
    return np.vstack(preds)
