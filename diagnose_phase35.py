# EMERGENCY FIX: Replace your Phase 3.5 optimization with this simplified approach

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler  # Use MinMaxScaler instead of StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Clean up paths
sys.path = [p for p in sys.path if 'StoreSalesSTGAT' not in p]
sys.path.insert(0, 'src')

from data.evaluation_cases import EvaluationCaseManager

class SimplifiedLSTM(nn.Module):
    """
    Simplified LSTM - much simpler than the complex OptimizedLSTMModel
    """
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(SimplifiedLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0  # No dropout for simplicity
        )
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        prediction = self.linear(lstm_out[:, -1, :])  # Use last output
        return prediction

class SimplifiedTrainer:
    """
    Simplified trainer focusing on core functionality
    """
    def __init__(self, case_manager):
        self.case_manager = case_manager
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def create_simple_features(self, data):
        """
        Create MINIMAL features - only essential ones
        """
        data = data.sort_values('date').copy()
        
        # Just basic lag features - NO complex rolling stats
        data['sales_lag_1'] = data['sales'].shift(1)
        data['sales_lag_7'] = data['sales'].shift(7)
        data['sales_lag_14'] = data['sales'].shift(14)
        
        # Simple day of week (no cyclical encoding)
        data['dayofweek'] = data['date'].dt.dayofweek
        
        # Fill NaN and select features
        data = data.fillna(method='ffill').fillna(0)
        
        feature_cols = ['sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'dayofweek']
        return data, feature_cols
    
    def create_sequences(self, data, feature_cols, seq_len=14):  # Shorter sequence
        """
        Create sequences with minimal features
        """
        # Apply log1p transformation
        data_transformed = data.copy()
        data_transformed['sales'] = np.log1p(data['sales'])
        
        # Simple MinMaxScaler instead of StandardScaler
        scaler = MinMaxScaler()
        
        # Scale features only (not target)
        features = data_transformed[feature_cols].values
        features_scaled = scaler.fit_transform(features)
        
        targets = data_transformed['sales'].values
        
        # Create sequences
        sequences, sequence_targets = [], []
        for i in range(seq_len, len(data_transformed)):
            seq = features_scaled[i-seq_len:i]
            target = targets[i]
            sequences.append(seq)
            sequence_targets.append(target)
        
        return np.array(sequences), np.array(sequence_targets), scaler
    
    def train_simple_model(self, train_data, store_id, family):
        """
        Train simplified model
        """
        print(f"🚀 SIMPLIFIED TRAINING: Store {store_id}, {family}")
        
        # Create simple features
        train_processed, feature_cols = self.create_simple_features(train_data)
        
        print(f"   Features: {len(feature_cols)} (simplified)")
        
        # Create sequences
        sequences, targets, scaler = self.create_sequences(train_processed, feature_cols)
        
        if len(sequences) < 30:
            print(f"   ❌ Insufficient data: {len(sequences)} sequences")
            return None, None, None
        
        print(f"   Sequences: {sequences.shape}")
        
        # Convert to tensors
        X = torch.FloatTensor(sequences).to(self.device)
        y = torch.FloatTensor(targets).to(self.device)
        
        # Simple model
        model = SimplifiedLSTM(input_size=len(feature_cols), hidden_size=32, num_layers=1).to(self.device)
        
        # Simple training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        model.train()
        initial_loss = None
        for epoch in range(50):  # Fewer epochs
            predictions = model(X)
            loss = criterion(predictions.squeeze(), y)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Loss = {loss.item():.6f}")
        
        print(f"   Training complete: {initial_loss:.6f} → {loss.item():.6f}")
        
        return model, scaler, feature_cols
    
    def evaluate_simple_model(self, model, scaler, feature_cols, test_data):
        """
        Evaluate simplified model
        """
        model.eval()
        
        # Process test data
        test_processed, _ = self.create_simple_features(test_data)
        test_sequences, test_targets, _ = self.create_sequences(test_processed, feature_cols)
        
        if len(test_sequences) == 0:
            return {'rmsle': float('inf')}
        
        # Predict
        X_test = torch.FloatTensor(test_sequences).to(self.device)
        
        with torch.no_grad():
            predictions = model(X_test).squeeze().cpu().numpy()
        
        # Inverse transform (log1p → expm1)
        pred_original = np.expm1(predictions)
        actual_original = np.expm1(test_targets)
        
        # Ensure positive values
        pred_original = np.maximum(pred_original, 0)
        actual_original = np.maximum(actual_original, 0)
        
        print(f"   Prediction range: {pred_original.min():.2f} to {pred_original.max():.2f}")
        print(f"   Actual range: {actual_original.min():.2f} to {actual_original.max():.2f}")
        
        # Calculate RMSLE
        try:
            rmsle = np.sqrt(mean_squared_log_error(actual_original, pred_original))
            print(f"   ✅ RMSLE: {rmsle:.4f}")
            return {'rmsle': rmsle, 'predictions': pred_original, 'actuals': actual_original}
        except Exception as e:
            print(f"   ❌ RMSLE calculation failed: {e}")
            return {'rmsle': float('inf')}

# EMERGENCY OPTIMIZATION RUNNER
def emergency_phase35_optimization():
    """
    Run emergency simplified optimization
    """
    print("🚨 EMERGENCY PHASE 3.5 OPTIMIZATION")
    print("=" * 50)
    print("Using SIMPLIFIED approach to diagnose issues")
    
    # Load components
    case_manager = EvaluationCaseManager()
    evaluation_cases = case_manager.get_cases_list()
    trainer = SimplifiedTrainer(case_manager)
    
    # Load sales data
    sales_data = pd.read_csv('data/raw/train.csv')
    sales_data['date'] = pd.to_datetime(sales_data['date'])
    
    # Test on first 3 cases only
    test_cases = evaluation_cases[:3]
    results = []
    
    for i, case in enumerate(test_cases, 1):
        store_id = case['store_nbr']
        family = case['family']
        
        print(f"\n🔥 EMERGENCY CASE {i}/3: Store {store_id}, {family}")
        print("-" * 40)
        
        try:
            # Get data
            case_info = {'store_nbr': store_id, 'family': family}
            train_data, test_data = case_manager.get_case_data(sales_data, case_info)
            
            # Train simplified model
            model, scaler, feature_cols = trainer.train_simple_model(train_data, store_id, family)
            
            if model is None:
                print("   ❌ Training failed")
                continue
            
            # Evaluate
            evaluation = trainer.evaluate_simple_model(model, scaler, feature_cols, test_data)
            
            rmsle = evaluation['rmsle']
            
            # Compare to baselines
            traditional_baseline = 0.4755
            improvement = (traditional_baseline - rmsle) / traditional_baseline * 100
            
            results.append({
                'store_id': store_id,
                'family': family,
                'rmsle': rmsle,
                'improvement': improvement,
                'beats_traditional': rmsle < traditional_baseline
            })
            
            status = "✅ GOOD" if rmsle < traditional_baseline else "⚠️ NEEDS WORK"
            print(f"   Result: {rmsle:.4f} ({improvement:+.1f}% vs traditional) {status}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Summary
    if results:
        avg_rmsle = np.mean([r['rmsle'] for r in results])
        cases_beat_traditional = sum(r['beats_traditional'] for r in results)
        
        print(f"\n📊 EMERGENCY RESULTS SUMMARY:")
        print(f"   Cases tested: {len(results)}")
        print(f"   Average RMSLE: {avg_rmsle:.4f}")
        print(f"   Cases beating traditional: {cases_beat_traditional}/{len(results)}")
        
        if avg_rmsle < 0.6:
            print("✅ SIMPLIFIED APPROACH WORKING - Build from here!")
        else:
            print("❌ FUNDAMENTAL ISSUE - Need deeper investigation")
    
    return results

if __name__ == "__main__":
    emergency_results = emergency_phase35_optimization()