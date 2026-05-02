import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.hybrid_system import NeuroProbabilisticHybrid
from models.regime_detector import MLRegimeDetector
from data.dataset import load_and_preprocess, TimeSeriesDataset
import numpy as np

def train():
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y, n_features = load_and_preprocess('data/raw/ETTh1.csv')
    
    dataset = TimeSeriesDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = NeuroProbabilisticHybrid(input_size=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 2. Stage 1: Train DL Encoder (Self-Supervised/Baseline)
    print("🚀 Stage 1: Training DL Encoder...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # 3. Stage 2: Integrate ML Regime Detector
    print("\n🔮 Stage 2: Integrating ML Regime Detector...")
    model.eval()
    all_features = []
    with torch.no_grad():
        for batch_x, _ in train_loader:
            feat = model.encoder(batch_x.to(device))
            all_features.append(feat.cpu().numpy())
    
    all_features = np.vstack(all_features)
    detector = MLRegimeDetector(n_regimes=3)
    detector.fit(all_features)
    
    print("✅ Hybrid System Integrated and Ready.")
    # Save models
    torch.save(model.state_dict(), 'models/hybrid_model.pth')
    return model, detector

if __name__ == "__main__":
    train()
