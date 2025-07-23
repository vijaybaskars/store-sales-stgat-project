"""
WORKING Neural Baseline Models - Fixed Tensor Dimensions
This is a complete working replacement for the neural_baselines.py file
All tensor dimension issues have been systematically resolved
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

@dataclass
class NeuralModelResults:
    """Container for neural model results"""
    model_name: str
    store_nbr: int
    family: str
    train_rmsle: float
    test_rmsle: float
    test_mae: float
    test_mape: float
    predictions: List[float]
    actuals: List[float]
    model_params: Dict
    fit_time: float
    training_history: Dict

class TimeSeriesDataset(Dataset):
    """FIXED: PyTorch Dataset for time series forecasting"""
    
    def __init__(self, data: np.ndarray, sequence_length: int = 30, forecast_horizon: int = 15):
        self.data = data.flatten()  # Ensure 1D
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.X, self.y = self._create_sequences()
    
    def _create_sequences(self):
        X, y = [], []
        for i in range(len(self.data) - self.sequence_length - self.forecast_horizon + 1):
            # X: (sequence_length,) -> will be reshaped to (sequence_length, 1) in __getitem__
            X.append(self.data[i:i + self.sequence_length])
            # y: (forecast_horizon,)
            y.append(self.data[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])
        return np.array(X), np.array(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # FIXED: Proper tensor shapes for LSTM
        X_tensor = torch.FloatTensor(self.X[idx]).unsqueeze(-1)  # (seq_len, 1)
        y_tensor = torch.FloatTensor(self.y[idx])  # (forecast_horizon,)
        return X_tensor, y_tensor

class AdvancedTimeSeriesDataset(Dataset):
    """FIXED: Enhanced dataset with external features"""
    
    def __init__(self, sales_data: np.ndarray, 
                 external_features: Optional[np.ndarray] = None,
                 sequence_length: int = 30, 
                 forecast_horizon: int = 15):
        self.sales_data = sales_data.flatten()  # Ensure 1D
        self.external_features = external_features
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.X_sales, self.X_features, self.y = self._create_sequences()
    
    def _create_sequences(self):
        X_sales, X_features, y = [], [], []
        
        for i in range(len(self.sales_data) - self.sequence_length - self.forecast_horizon + 1):
            # Sales sequences
            X_sales.append(self.sales_data[i:i + self.sequence_length])
            y.append(self.sales_data[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])
            
            # External features
            if self.external_features is not None:
                X_features.append(self.external_features[i:i + self.sequence_length])
            else:
                # Create dummy features
                X_features.append(np.zeros((self.sequence_length, 1)))
        
        return np.array(X_sales), np.array(X_features), np.array(y)
    
    def __len__(self):
        return len(self.X_sales)
    
    def __getitem__(self, idx):
        # FIXED: Proper tensor shapes
        X_sales_tensor = torch.FloatTensor(self.X_sales[idx]).unsqueeze(-1)  # (seq_len, 1)
        X_features_tensor = torch.FloatTensor(self.X_features[idx])  # (seq_len, n_features)
        y_tensor = torch.FloatTensor(self.y[idx])  # (forecast_horizon,)
        return X_sales_tensor, X_features_tensor, y_tensor

class VanillaLSTM(nn.Module):
    """FIXED: Basic LSTM for time series forecasting"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2, 
                 forecast_horizon: int = 15):
        super(VanillaLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last output: (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]
        # Generate forecast: (batch_size, forecast_horizon)
        output = self.fc(last_output)
        return output

class BidirectionalLSTM(nn.Module):
    """FIXED: Bidirectional LSTM"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2, 
                 forecast_horizon: int = 15):
        super(BidirectionalLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Bidirectional LSTM outputs hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        output = self.fc(last_output)
        return output

class GRUModel(nn.Module):
    """FIXED: GRU model"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2, 
                 forecast_horizon: int = 15):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.fc(last_output)
        return output

class LSTMWithFeatures(nn.Module):
    """FIXED: LSTM with external features"""
    
    def __init__(self, sales_input_size: int = 1, feature_input_size: int = 1,
                 hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, forecast_horizon: int = 15):
        super(LSTMWithFeatures, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # Sales LSTM
        self.sales_lstm = nn.LSTM(
            input_size=sales_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Features LSTM
        self.feature_lstm = nn.LSTM(
            input_size=feature_input_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )
    
    def forward(self, sales_x, feature_x):
        # Process sales data
        sales_out, _ = self.sales_lstm(sales_x)
        sales_features = sales_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Process external features
        feature_out, _ = self.feature_lstm(feature_x)
        ext_features = feature_out[:, -1, :]  # (batch_size, hidden_size // 2)
        
        # Combine features
        combined = torch.cat([sales_features, ext_features], dim=1)
        output = self.fusion(combined)
        return output

class NeuralBaselines:
    """FIXED: Neural baseline models implementation"""
    
    def __init__(self, evaluation_case_manager=None, device='auto'):
        if evaluation_case_manager is None:
            import sys
            sys.path.insert(0, 'src')
            from data import EvaluationCaseManager
            self.case_manager = EvaluationCaseManager()
        else:
            self.case_manager = evaluation_case_manager
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Neural Baselines initialized on device: {self.device}")
        
        self.results = {}
        self.sales_data = None
        self._load_sales_data()
        
        # Simplified model configurations
        self.model_configs = {
            'vanilla_lstm': {
                'hidden_size': 32,
                'num_layers': 1,
                'dropout': 0.1,
                'sequence_length': 20
            },
            'bidirectional_lstm': {
                'hidden_size': 32,
                'num_layers': 1,
                'dropout': 0.1,
                'sequence_length': 20
            },
            'gru': {
                'hidden_size': 32,
                'num_layers': 1,
                'dropout': 0.1,
                'sequence_length': 20
            },
            'lstm_with_features': {
                'hidden_size': 32,
                'num_layers': 1,
                'dropout': 0.1,
                'sequence_length': 20
            }
        }
    
    def _load_sales_data(self):
        """Load sales data"""
        try:
            possible_paths = [
                'data/raw/train.csv',
                '../data/raw/train.csv',
                '../../data/raw/train.csv',
                './data/raw/train.csv'
            ]
            
            for path in possible_paths:
                try:
                    self.sales_data = pd.read_csv(path)
                    self.sales_data['date'] = pd.to_datetime(self.sales_data['date'])
                    print(f"✅ Sales data loaded from: {path}")
                    return
                except FileNotFoundError:
                    continue
            
            print("❌ Could not find train.csv")
            self.sales_data = None
                    
        except Exception as e:
            print(f"❌ Could not load sales data: {e}")
            self.sales_data = None
    
    def prepare_case_data(self, store_nbr: int, family: str, add_features: bool = False) -> Tuple:
        """FIXED: Prepare data with proper validation"""
        
        import sys
        sys.path.insert(0, 'src')
        from data import get_case_train_test_data
        
        if self.sales_data is None:
            raise ValueError("Sales data not available")
        
        # Get train/test data
        train_data, test_data = get_case_train_test_data(self.sales_data, store_nbr, family)
        
        if train_data is None or test_data is None or len(train_data) == 0 or len(test_data) == 0:
            raise ValueError(f"No data available for store {store_nbr}, family {family}")
        
        # Extract sales values and ensure they're 1D
        train_sales = train_data['sales'].values.flatten()
        test_sales = test_data['sales'].values.flatten()
        
        print(f"    📊 Raw data shapes: train={train_sales.shape}, test={test_sales.shape}")
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_sales_scaled = scaler.fit_transform(train_sales.reshape(-1, 1)).flatten()
        test_sales_scaled = scaler.transform(test_sales.reshape(-1, 1)).flatten()
        
        print(f"    📊 Scaled data shapes: train={train_sales_scaled.shape}, test={test_sales_scaled.shape}")
        
        # Prepare features if requested
        features_train, features_test = None, None
        if add_features:
            features_train, features_test = self._prepare_features(train_data, test_data)
            print(f"    📊 Feature shapes: train={features_train.shape}, test={features_test.shape}")
        
        return train_sales_scaled, test_sales_scaled, scaler, features_train, features_test
    
    def _prepare_features(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple:
        """FIXED: Prepare external features"""
        
        def create_features(data):
            features = []
            
            # Day of week (0-6)
            features.append(data['date'].dt.dayofweek.values)
            
            # Month (1-12) 
            features.append(data['date'].dt.month.values)
            
            # Rolling mean (7 days)
            rolling_mean = data['sales'].rolling(7, min_periods=1).mean()
            features.append(rolling_mean.values)
            
            # Lag feature (1 day)
            lag_1 = data['sales'].shift(1).fillna(data['sales'].mean())
            features.append(lag_1.values)
            
            return np.column_stack(features)
        
        features_train = create_features(train_data)
        features_test = create_features(test_data)
        
        # Normalize features
        feature_scaler = StandardScaler()
        features_train = feature_scaler.fit_transform(features_train)
        features_test = feature_scaler.transform(features_test)
        
        return features_train, features_test
    
    def calculate_metrics(self, y_true: np.array, y_pred: np.array) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        
        # Ensure positive values for RMSLE
        y_true_pos = np.maximum(y_true, 0)
        y_pred_pos = np.maximum(y_pred, 0)
        
        # RMSLE
        rmsle = np.sqrt(np.mean((np.log1p(y_pred_pos) - np.log1p(y_true_pos))**2))
        
        # MAE
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE
        mape_values = []
        for true, pred in zip(y_true, y_pred):
            if abs(true) > 1e-6:
                mape_values.append(abs((true - pred) / true))
        mape = np.mean(mape_values) * 100 if mape_values else float('inf')
        
        return {'rmsle': rmsle, 'mae': mae, 'mape': mape}
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, epochs: int = 50) -> Dict:
        """FIXED: Train neural network model"""
        
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_data in train_loader:
                try:
                    if len(batch_data) == 2:  # Simple models
                        X, y = batch_data
                        X, y = X.to(self.device), y.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = model(X)
                        loss = criterion(outputs, y)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        train_batches += 1
                        
                    else:  # Feature-enhanced models
                        X_sales, X_features, y = batch_data
                        X_sales = X_sales.to(self.device)
                        X_features = X_features.to(self.device)
                        y = y.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = model(X_sales, X_features)
                        loss = criterion(outputs, y)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        train_batches += 1
                
                except Exception as e:
                    print(f"      Training batch error: {e}")
                    continue
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_data in val_loader:
                    try:
                        if len(batch_data) == 2:
                            X, y = batch_data
                            X, y = X.to(self.device), y.to(self.device)
                            outputs = model(X)
                            val_loss += criterion(outputs, y).item()
                            val_batches += 1
                        else:
                            X_sales, X_features, y = batch_data
                            X_sales = X_sales.to(self.device)
                            X_features = X_features.to(self.device)
                            y = y.to(self.device)
                            outputs = model(X_sales, X_features)
                            val_loss += criterion(outputs, y).item()
                            val_batches += 1
                    
                    except Exception as e:
                        print(f"      Validation batch error: {e}")
                        continue
            
            # Calculate average losses
            if train_batches > 0:
                train_loss /= train_batches
            if val_batches > 0:
                val_loss /= val_batches
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses)
        }
    
    def evaluate_case(self, store_nbr: int, family: str, forecast_horizon: int = 15, fast_mode: bool = False, production_mode: bool = True) -> Dict[str, NeuralModelResults]:
        """
        Evaluate neural models for a single case
        
        Args:
            production_mode: If True, use only top 2 research-validated models (Phase 6 optimization)
            fast_mode: If True, use reduced epochs and faster training
        """
        
        print(f"\n🧠 Neural Evaluation: Store {store_nbr} - {family}")
        if production_mode:
            print("   ⚡ Production mode: Using top 2 research-validated models")
        if fast_mode:
            print("   ⚡ Fast mode enabled - using reduced epochs and simplified training")
        
        case_results = {}
        
        # Models to test based on mode
        if production_mode:
            # Research-validated top 2 performers: bidirectional_lstm (#1), vanilla_lstm (#2)
            models_to_test = {
                'bidirectional_lstm': (BidirectionalLSTM, False),  # Research #1 performer
                'vanilla_lstm': (VanillaLSTM, False)                # Research #2 performer
            }
            print(f"   🎯 Training top models: {list(models_to_test.keys())}")
        elif fast_mode:
            # Fast mode: only simplest model
            models_to_test = {
                'vanilla_lstm': (VanillaLSTM, False)  # Only test the simplest/fastest model
            }
        else:
            # Research mode: all models
            models_to_test = {
                'vanilla_lstm': (VanillaLSTM, False),
                'bidirectional_lstm': (BidirectionalLSTM, False),
                'gru': (GRUModel, False),
                'lstm_with_features': (LSTMWithFeatures, True)
            }
        
        for model_name, (model_class, use_features) in models_to_test.items():
            print(f"  🚀 Training {model_name}...")
            start_time = time.time()
            
            try:
                # Prepare data
                train_sales, test_sales, scaler, features_train, features_test = self.prepare_case_data(
                    store_nbr, family, add_features=use_features
                )
                
                config = self.model_configs[model_name]
                sequence_length = config['sequence_length']
                
                # Check data sufficiency
                if len(train_sales) < sequence_length + forecast_horizon + 10:
                    print(f"    ⚠️  Insufficient training data: {len(train_sales)}")
                    continue
                
                # Create datasets
                if use_features:
                    train_dataset = AdvancedTimeSeriesDataset(
                        train_sales, features_train, sequence_length, forecast_horizon
                    )
                else:
                    train_dataset = TimeSeriesDataset(
                        train_sales, sequence_length, forecast_horizon
                    )
                
                if len(train_dataset) == 0:
                    print(f"    ⚠️  No sequences created")
                    continue
                
                # Create validation split
                val_size = max(1, len(train_dataset) // 5)
                train_size = len(train_dataset) - val_size
                train_subset, val_subset = torch.utils.data.random_split(
                    train_dataset, [train_size, val_size]
                )
                
                print(f"    📊 Dataset: train={len(train_subset)}, val={len(val_subset)}")
                
                # Create data loaders with smaller batch size
                train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)
                
                # Initialize model
                if use_features:
                    model = model_class(
                        sales_input_size=1,
                        feature_input_size=features_train.shape[1],
                        hidden_size=config['hidden_size'],
                        num_layers=config['num_layers'],
                        dropout=config['dropout'],
                        forecast_horizon=forecast_horizon
                    )
                else:
                    model = model_class(
                        input_size=1,
                        hidden_size=config['hidden_size'],
                        num_layers=config['num_layers'],
                        dropout=config['dropout'],
                        forecast_horizon=forecast_horizon
                    )
                
                # Train model (use fewer epochs in fast mode)
                epochs = 10 if fast_mode else 30
                training_history = self.train_model(model, train_loader, val_loader, epochs=epochs)
                
                # Generate predictions
                model.eval()
                with torch.no_grad():
                    # Use last sequence for prediction
                    if use_features:
                        last_sales = torch.FloatTensor(
                            train_sales[-sequence_length:].reshape(1, sequence_length, 1)
                        ).to(self.device)
                        last_features = torch.FloatTensor(
                            features_train[-sequence_length:].reshape(1, sequence_length, -1)
                        ).to(self.device)
                        predictions = model(last_sales, last_features).cpu().numpy().flatten()
                    else:
                        last_sequence = torch.FloatTensor(
                            train_sales[-sequence_length:].reshape(1, sequence_length, 1)
                        ).to(self.device)
                        predictions = model(last_sequence).cpu().numpy().flatten()
                
                # Denormalize predictions
                predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                
                # Get actual test values
                test_actual = test_sales[:forecast_horizon] if len(test_sales) >= forecast_horizon else test_sales
                test_actual_rescaled = scaler.inverse_transform(test_actual.reshape(-1, 1)).flatten()
                
                # Adjust prediction length
                min_length = min(len(predictions_rescaled), len(test_actual_rescaled))
                predictions_final = predictions_rescaled[:min_length]
                actuals_final = test_actual_rescaled[:min_length]
                
                # Calculate metrics
                test_metrics = self.calculate_metrics(actuals_final, predictions_final)
                train_metrics = {'rmsle': training_history['best_val_loss'], 'mae': 0, 'mape': 0}
                
                fit_time = time.time() - start_time
                
                case_results[model_name] = NeuralModelResults(
                    model_name=model_name,
                    store_nbr=store_nbr,
                    family=family,
                    train_rmsle=train_metrics['rmsle'],
                    test_rmsle=test_metrics['rmsle'],
                    test_mae=test_metrics['mae'],
                    test_mape=test_metrics['mape'],
                    predictions=predictions_final.tolist(),
                    actuals=actuals_final.tolist(),
                    model_params=config,
                    fit_time=fit_time,
                    training_history=training_history
                )
                
                print(f"    ✅ {model_name} - RMSLE: {test_metrics['rmsle']:.4f} ({fit_time:.1f}s)")
                
            except Exception as e:
                print(f"    ❌ {model_name} failed: {str(e)}")
                continue
        
        return case_results
    
    def evaluate_all_cases(self, evaluation_cases: List[Dict]) -> Dict:
        """Evaluate all neural models on all evaluation cases"""
        
        print("🧠 Starting Neural Baseline Evaluation")
        print(f"📊 Evaluating {len(evaluation_cases)} cases")
        print("=" * 60)
        
        all_results = {}
        
        for i, case in enumerate(evaluation_cases, 1):
            store_nbr = case['store_nbr']
            family = case['family']
            case_key = f"store_{store_nbr}_family_{family}"
            
            print(f"\n[{i}/{len(evaluation_cases)}] Case: {case_key}")
            
            try:
                case_results = self.evaluate_case(store_nbr, family)
                if case_results:  # Only add if we have results
                    all_results[case_key] = case_results
                    print(f"    ✅ Completed successfully")
                else:
                    print(f"    ⚠️  No models succeeded")
                
            except Exception as e:
                print(f"    ❌ Failed: {str(e)}")
                continue
        
        # Calculate summary statistics
        model_names = set()
        for case_results in all_results.values():
            model_names.update(case_results.keys())
        
        summary_stats = {}
        for model_name in model_names:
            rmsle_scores = []
            mae_scores = []
            fit_times = []
            
            for case_results in all_results.values():
                if model_name in case_results:
                    rmsle_scores.append(case_results[model_name].test_rmsle)
                    mae_scores.append(case_results[model_name].test_mae)
                    fit_times.append(case_results[model_name].fit_time)
            
            if rmsle_scores:  # Only create stats if we have data
                summary_stats[model_name] = {
                    'mean_rmsle': np.mean(rmsle_scores),
                    'std_rmsle': np.std(rmsle_scores),
                    'mean_mae': np.mean(mae_scores),
                    'std_mae': np.std(mae_scores),
                    'mean_fit_time': np.mean(fit_times),
                    'count': len(rmsle_scores)
                }
        
        print("\n" + "=" * 60)
        print("🧠 NEURAL BASELINE RESULTS")
        print("=" * 60)
        
        for model_name, stats in summary_stats.items():
            print(f"{model_name:20s} | RMSLE: {stats['mean_rmsle']:.4f} ± {stats['std_rmsle']:.4f} | Cases: {stats['count']}")
        
        return {
            'detailed_results': all_results,
            'summary_statistics': summary_stats,
            'evaluation_metadata': {
                'test_split_date': '2017-07-01',
                'forecast_horizon': 15,
                'total_cases': len(evaluation_cases),
                'successful_cases': len(all_results),
                'device_used': str(self.device)
            }
        }
    
    def save_results(self, results: Dict, filepath: str):
        """Save results to JSON file"""
        
        # Convert to serializable format
        serializable_results = {}
        
        for case_key, case_results in results['detailed_results'].items():
            serializable_results[case_key] = {}
            for model_name, model_result in case_results.items():
                serializable_results[case_key][model_name] = {
                    'model_name': model_result.model_name,
                    'store_nbr': model_result.store_nbr,
                    'family': model_result.family,
                    'train_rmsle': model_result.train_rmsle,
                    'test_rmsle': model_result.test_rmsle,
                    'test_mae': model_result.test_mae,
                    'test_mape': model_result.test_mape,
                    'predictions': model_result.predictions,
                    'actuals': model_result.actuals,
                    'model_params': model_result.model_params,
                    'fit_time': model_result.fit_time,
                    'training_history': model_result.training_history
                }
        
        final_results = {
            'detailed_results': serializable_results,
            'summary_statistics': results['summary_statistics'],
            'evaluation_metadata': results['evaluation_metadata']
        }
        
        with open(filepath, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Neural results saved to: {filepath}")