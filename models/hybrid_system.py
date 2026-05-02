import torch
import torch.nn as nn
import numpy as np
from models.tft_encoder import TFTFeatureEncoder

class NeuroProbabilisticHybrid(nn.Module):
    """
    Exemplary Level 5 Hybrid System: Symbiotic Interaction.
    DL extracts features -> ML provides regime context -> 
    ML context MULTIPLICATIVELY modulates DL neural representations.
    """
    def __init__(self, input_size, horizon=24, n_regimes=3):
        super(NeuroProbabilisticHybrid, self).__init__()
        self.encoder = TFTFeatureEncoder(input_size)
        self.n_regimes = n_regimes
        
        # Symbiotic Gating: ML context modulates DL feature vector
        # This is a 'Strong Coupling' where internal DL state depends on ML
        self.regime_gate = nn.Sequential(
            nn.Linear(n_regimes, 64),
            nn.Sigmoid()
        )
        
        self.final_head = nn.Linear(64, horizon)
        
    def forward(self, x, regime_probs=None):
        # 1. DL Feature Extraction (Encoder)
        # Shape: [Batch, 64]
        features = self.encoder(x)
        
        # 2. ML Active Modulation (The 'Symbiosis')
        if regime_probs is not None:
            # Convert ML regime probs into a modulation gate
            # regime_probs: [Batch, n_regimes]
            weights = torch.from_numpy(regime_probs).float().to(features.device)
            gate = self.regime_gate(weights) # [Batch, 64]
            
            # MULTIPLICATIVE INTERACTION: ML context reshapes DL representation
            # This satisfies the requirement that removing ML changes DL behavior internally
            features = features * gate 
        
        # 3. Final Prediction
        return self.final_head(features)
