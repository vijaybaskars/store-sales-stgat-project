import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_log_error
import warnings
warnings.filterwarnings('ignore')

class OptimizedLSTMModel(nn.Module):
    """
    Optimized LSTM architecture for Phase 3.5
    Enhanced with dropout, batch normalization, and improved architecture
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(OptimizedLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        print(f"🧠 Creating OptimizedLSTMModel:")
        print(f"   Input size: {input_size}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Num layers: {num_layers}")
        print(f"   Dropout: {dropout}")
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Output layers with progressive size reduction
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # Input normalization (reshape for batch norm)
        x_reshaped = x.view(-1, features)
        x_normalized = self.input_bn(x_reshaped)
        x = x_normalized.view(batch_size, seq_len, features)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Progressive dense layers with batch norm and dropout
        out = self.dropout(last_output)
        
        # First layer
        out = self.fc1(out)
        if out.size(0) > 1:  # Only apply batch norm if batch size > 1
            out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        
        # Second layer
        out = self.fc2(out)
        if out.size(0) > 1:
            out = self.bn2(out)
        out = self.leaky_relu(out)
        
        # Final output
        out = self.fc3(out)
        
        return out

class Phase35LSTMTrainer:
    """
    Enhanced LSTM trainer integrating with EvaluationCaseManager
    """
    
    def __init__(self, case_manager, device='cpu'):
        self.case_manager = case_manager
        self.device = device
        self.models = {}
        self.training_history = {}
        
        print(f"🚀 Phase35LSTMTrainer initialized")
        print(f"   Device: {device}")
        print(f"   EvaluationCaseManager: {'✅ Connected' if case_manager else '❌ None'}")
    
    def create_sequences(self, data, feature_columns, sequence_length=21, target_col='sales'):
        """Create sequences with enhanced preprocessing"""
        sequences = []
        targets = []
        
        # Sort by date to ensure proper sequence order
        data_sorted = data.sort_values('date')
        features = data_sorted[feature_columns].values
        target_values = data_sorted[target_col].values
        
        # Create overlapping sequences
        for i in range(sequence_length, len(data_sorted)):
            feature_sequence = features[i-sequence_length:i]
            sequences.append(feature_sequence)
            targets.append(target_values[i])
        
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        print(f"   Created {len(sequences)} sequences of length {sequence_length}")
        print(f"   Feature dimensions: {sequences.shape}")
        print(f"   Target range: {targets.min():.4f} to {targets.max():.4f}")
        
        return sequences, targets
    
    def train_model(self, train_data, feature_columns, store_id, family, 
                   epochs=100, batch_size=32, learning_rate=0.001):
        """Train optimized LSTM model with enhanced procedures"""
        print(f"\n🚀 TRAINING OPTIMIZED MODEL: Store {store_id}, {family}")
        print("=" * 60)
        
        # Create sequences
        sequences, targets = self.create_sequences(train_data, feature_columns)
        
        if len(sequences) < 50:
            print(f"❌ Insufficient data: {len(sequences)} sequences (minimum: 50)")
            return None
        
        # Convert to tensors
        X = torch.FloatTensor(sequences).to(self.device)
        y = torch.FloatTensor(targets).to(self.device)
        
        print(f"   Training data shape: {X.shape}")
        print(f"   Target data shape: {y.shape}")
        
        # Create model
        model = OptimizedLSTMModel(
            input_size=len(feature_columns),
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        # Enhanced optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        patience = 15
        training_losses = []
        
        print(f"   Starting training: {epochs} epochs, batch size {batch_size}")
        
        model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(loss)
            
            training_losses.append(loss.item())
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
            
            # Progress reporting
            if epoch % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch+1:3d}: Loss = {loss.item():.6f}, LR = {current_lr:.2e}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Store model and training info
        model_key = f"{store_id}_{family}"
        self.models[model_key] = {
            'model': model,
            'feature_columns': feature_columns,
            'best_loss': best_loss,
            'final_epoch': epoch + 1,
            'training_losses': training_losses
        }
        
        print(f"✅ Training complete:")
        print(f"   Best loss: {best_loss:.6f}")
        print(f"   Final epoch: {epoch + 1}")
        print(f"   Model saved as: {model_key}")
        
        return model
    
    def evaluate_model(self, model, test_data, feature_columns, scaler, sequence_length=21):
        """Evaluate model performance with proper inverse transformation"""
        print(f"📊 Evaluating model performance...")
        
        model.eval()
        
        # Create sequences for prediction
        sequences, actuals = self.create_sequences(test_data, feature_columns, sequence_length)
        
        if len(sequences) == 0:
            print(f"❌ No sequences available for evaluation")
            return {'rmsle': float('inf'), 'predictions': [], 'actuals': [], 'n_predictions': 0}
        
        # Convert to tensor and predict
        X = torch.FloatTensor(sequences).to(self.device)
        
        with torch.no_grad():
            predictions = model(X)
            predictions = predictions.cpu().numpy().flatten()
        
        print(f"   Generated {len(predictions)} predictions")
        print(f"   Prediction range: {predictions.min():.4f} to {predictions.max():.4f}")
        print(f"   Actual range: {actuals.min():.4f} to {actuals.max():.4f}")
        
        # Inverse log1p transformation for RMSLE calculation
        predictions_exp = np.expm1(predictions)
        actuals_exp = np.expm1(actuals)
        
        # Ensure non-negative values
        predictions_exp = np.maximum(predictions_exp, 0)
        actuals_exp = np.maximum(actuals_exp, 0)
        
        print(f"   After inverse transform:")
        print(f"   Prediction range: {predictions_exp.min():.2f} to {predictions_exp.max():.2f}")
        print(f"   Actual range: {actuals_exp.min():.2f} to {actuals_exp.max():.2f}")
        
        # Calculate RMSLE
        try:
            rmsle = np.sqrt(mean_squared_log_error(actuals_exp, predictions_exp))
            print(f"   ✅ RMSLE calculated: {rmsle:.4f}")
        except ValueError as e:
            print(f"   ❌ RMSLE calculation failed: {e}")
            rmsle = float('inf')
        
        return {
            'rmsle': rmsle,
            'predictions': predictions_exp,
            'actuals': actuals_exp,
            'n_predictions': len(predictions)
        }