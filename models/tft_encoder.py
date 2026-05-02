import torch
import torch.nn as nn

class TFTFeatureEncoder(nn.Module):
    """
    Deep Learning Component: Temporal Fusion Transformer (Encoder-only).
    Purpose: Extracts complex temporal dependencies and produces a latent 
    context vector that captures the 'state' of the energy system.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super(TFTFeatureEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        # Multi-Head Attention to capture global dependencies
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=4, batch_first=True)
        
        # Gated Residual Connection (GRN) for stable feature extraction
        self.feature_proj = nn.Linear(hidden_size*2, hidden_size)
        
    def forward(self, x):
        # x shape: [Batch, Sequence, Features]
        lstm_out, _ = self.lstm(x)
        
        # Self-attention over the sequence
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last time step as the global temporal feature
        latent_vector = torch.relu(self.feature_proj(attn_out[:, -1, :]))
        
        return latent_vector # Latent feature passed to ML component
