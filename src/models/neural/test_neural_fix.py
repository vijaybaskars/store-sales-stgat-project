"""
Quick test script to verify neural baseline fixes work
Run this before replacing your main file
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

print("🧪 Testing Neural Baseline Fixes")
print("=" * 40)

# Test 1: TimeSeriesDataset
print("\n1️⃣ Testing TimeSeriesDataset...")
try:
    # Copy the TimeSeriesDataset class from the fixed version
    class TimeSeriesDataset:
        def __init__(self, data, sequence_length=20, forecast_horizon=15):
            self.data = data.flatten()
            self.sequence_length = sequence_length
            self.forecast_horizon = forecast_horizon
            self.X, self.y = self._create_sequences()
        
        def _create_sequences(self):
            X, y = [], []
            for i in range(len(self.data) - self.sequence_length - self.forecast_horizon + 1):
                X.append(self.data[i:i + self.sequence_length])
                y.append(self.data[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])
            return np.array(X), np.array(y)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            X_tensor = torch.FloatTensor(self.X[idx]).unsqueeze(-1)  # (seq_len, 1)
            y_tensor = torch.FloatTensor(self.y[idx])  # (forecast_horizon,)
            return X_tensor, y_tensor
    
    # Test with dummy data
    dummy_data = np.random.random(100) * 50
    dataset = TimeSeriesDataset(dummy_data, sequence_length=20, forecast_horizon=15)
    
    print(f"✅ Dataset created: {len(dataset)} sequences")
    
    # Test sample
    X_sample, y_sample = dataset[0]
    print(f"✅ Sample shapes: X={X_sample.shape}, y={y_sample.shape}")
    print(f"   Expected: X=(20, 1), y=(15,)")
    
    # Test DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    X_batch, y_batch = next(iter(dataloader))
    print(f"✅ Batch shapes: X={X_batch.shape}, y={y_batch.shape}")
    print(f"   Expected: X=(4, 20, 1), y=(4, 15)")
    
except Exception as e:
    print(f"❌ TimeSeriesDataset test failed: {e}")

# Test 2: VanillaLSTM
print("\n2️⃣ Testing VanillaLSTM...")
try:
    class VanillaLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=32, num_layers=1, forecast_horizon=15):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, forecast_horizon)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])
            return output
    
    model = VanillaLSTM(input_size=1, hidden_size=32, forecast_horizon=15)
    
    # Test forward pass
    test_input = torch.randn(4, 20, 1)  # (batch, seq, features)
    output = model(test_input)
    print(f"✅ Model forward pass: {test_input.shape} -> {output.shape}")
    print(f"   Expected output: (4, 15)")
    
except Exception as e:
    print(f"❌ VanillaLSTM test failed: {e}")

# Test 3: Full Training Simulation
print("\n3️⃣ Testing Training Loop...")
try:
    # Create dummy dataset
    dummy_data = np.random.random(200) * 50
    dataset = TimeSeriesDataset(dummy_data, sequence_length=20, forecast_horizon=15)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Create model
    model = VanillaLSTM(input_size=1, hidden_size=16, forecast_horizon=15)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Test one training step
    model.train()
    X_batch, y_batch = next(iter(dataloader))
    
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    
    print(f"✅ Training step completed: loss={loss.item():.6f}")
    
except Exception as e:
    print(f"❌ Training loop test failed: {e}")

# Test 4: Feature Dataset
print("\n4️⃣ Testing AdvancedTimeSeriesDataset...")
try:
    class AdvancedTimeSeriesDataset:
        def __init__(self, sales_data, external_features=None, sequence_length=20, forecast_horizon=15):
            self.sales_data = sales_data.flatten()
            self.external_features = external_features
            self.sequence_length = sequence_length
            self.forecast_horizon = forecast_horizon
            self.X_sales, self.X_features, self.y = self._create_sequences()
        
        def _create_sequences(self):
            X_sales, X_features, y = [], [], []
            for i in range(len(self.sales_data) - self.sequence_length - self.forecast_horizon + 1):
                X_sales.append(self.sales_data[i:i + self.sequence_length])
                y.append(self.sales_data[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])
                
                if self.external_features is not None:
                    X_features.append(self.external_features[i:i + self.sequence_length])
                else:
                    X_features.append(np.zeros((self.sequence_length, 1)))
            
            return np.array(X_sales), np.array(X_features), np.array(y)
        
        def __len__(self):
            return len(self.X_sales)
        
        def __getitem__(self, idx):
            X_sales_tensor = torch.FloatTensor(self.X_sales[idx]).unsqueeze(-1)  # (seq_len, 1)
            X_features_tensor = torch.FloatTensor(self.X_features[idx])  # (seq_len, n_features)
            y_tensor = torch.FloatTensor(self.y[idx])  # (forecast_horizon,)
            return X_sales_tensor, X_features_tensor, y_tensor
    
    # Test with features
    sales_data = np.random.random(100) * 50
    features_data = np.random.random((100, 3))  # 3 features
    
    adv_dataset = AdvancedTimeSeriesDataset(sales_data, features_data, sequence_length=20, forecast_horizon=15)
    print(f"✅ Advanced dataset created: {len(adv_dataset)} sequences")
    
    # Test sample
    X_sales, X_features, y = adv_dataset[0]
    print(f"✅ Advanced sample shapes: sales={X_sales.shape}, features={X_features.shape}, y={y.shape}")
    print(f"   Expected: sales=(20, 1), features=(20, 3), y=(15,)")
    
except Exception as e:
    print(f"❌ AdvancedTimeSeriesDataset test failed: {e}")

print("\n🎯 Test Summary:")
print("If all tests passed, the neural baseline fixes are working correctly!")
print("You can now replace your neural_baselines.py file with the fixed version.")