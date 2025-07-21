import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class Phase35DataProcessor:
    """
    Advanced data preprocessing for Phase 3.5 optimization
    Implements log1p transformation, cyclical encoding, and enhanced features
    """
    
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.feature_stats = {}
    
    def apply_log1p_transformation(self, sales_data):
        """
        Apply log1p transformation to sales data
        Critical for handling skewed sales distributions
        """
        print("🔄 Applying log1p transformation...")
        
        # Create copy to avoid modifying original
        data = sales_data.copy()
        
        # Store original sales for evaluation
        data['sales_original'] = data['sales']
        
        # Apply log1p transformation (handles zeros naturally)
        data['sales_log1p'] = np.log1p(data['sales'])
        
        # Verify transformation
        print(f"   Original sales range: {data['sales'].min():.2f} to {data['sales'].max():.2f}")
        print(f"   Log1p sales range: {data['sales_log1p'].min():.4f} to {data['sales_log1p'].max():.4f}")
        
        # Replace sales column with transformed version
        data['sales'] = data['sales_log1p']
        
        return data
    
    def create_cyclical_features(self, data):
        """
        Create cyclical encodings for temporal features
        Essential for capturing weekly/monthly patterns
        """
        print("🔄 Creating cyclical temporal features...")
        
        # Ensure date column is datetime
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            
            # Day of week (0-6)
            data['dayofweek'] = data['date'].dt.dayofweek
            data['dayofweek_sin'] = np.sin(2 * np.pi * data['dayofweek'] / 7)
            data['dayofweek_cos'] = np.cos(2 * np.pi * data['dayofweek'] / 7)
            
            # Day of month (1-31)
            data['dayofmonth'] = data['date'].dt.day
            data['dayofmonth_sin'] = np.sin(2 * np.pi * data['dayofmonth'] / 31)
            data['dayofmonth_cos'] = np.cos(2 * np.pi * data['dayofmonth'] / 31)
            
            # Month (1-12)
            data['month'] = data['date'].dt.month
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            
            # Quarter (1-4)
            data['quarter'] = data['date'].dt.quarter
            data['quarter_sin'] = np.sin(2 * np.pi * data['quarter'] / 4)
            data['quarter_cos'] = np.cos(2 * np.pi * data['quarter'] / 4)
            
            print(f"   ✅ Created 8 cyclical features")
        
        return data
    
    def create_enhanced_lag_features(self, data, store_id, family):
        """
        Create enhanced lag and rolling features
        """
        print(f"🔄 Creating enhanced features for Store {store_id}, {family}...")
        
        # Filter to specific store-family
        store_family_data = data[
            (data['store_nbr'] == store_id) & 
            (data['family'] == family)
        ].sort_values('date').copy()
        
        if len(store_family_data) < 30:
            return store_family_data
        
        # Lag features (multiple lags)
        lag_periods = [1, 2, 3, 7, 14, 28]
        for lag in lag_periods:
            store_family_data[f'sales_lag_{lag}'] = store_family_data['sales'].shift(lag)
        
        # Rolling statistics (multiple windows)
        windows = [3, 7, 14, 28]
        for window in windows:
            store_family_data[f'sales_mean_{window}'] = store_family_data['sales'].rolling(window, min_periods=1).mean()
            store_family_data[f'sales_std_{window}'] = store_family_data['sales'].rolling(window, min_periods=1).std()
            store_family_data[f'sales_min_{window}'] = store_family_data['sales'].rolling(window, min_periods=1).min()
            store_family_data[f'sales_max_{window}'] = store_family_data['sales'].rolling(window, min_periods=1).max()
        
        # Trend features
        store_family_data['sales_trend_7'] = store_family_data['sales'].rolling(7, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
        )
        
        # Volatility features
        store_family_data['sales_volatility_7'] = store_family_data['sales'].rolling(7, min_periods=2).std()
        store_family_data['sales_volatility_14'] = store_family_data['sales'].rolling(14, min_periods=2).std()
        
        # Fill NaN values
        store_family_data = store_family_data.fillna(method='forward').fillna(0)
        
        print(f"   ✅ Created {len([c for c in store_family_data.columns if c.startswith('sales_')])} sales features")
        
        return store_family_data
    
    def apply_standardization(self, train_data, test_data, feature_columns):
        """
        Apply StandardScaler to feature columns
        Critical for neural network convergence
        """
        print("🔄 Applying StandardScaler to features...")
        
        # Initialize scaler
        scaler = StandardScaler()
        
        # Fit on training data only
        train_scaled = train_data.copy()
        train_scaled[feature_columns] = scaler.fit_transform(train_data[feature_columns])
        
        # Transform test data
        test_scaled = test_data.copy()
        test_scaled[feature_columns] = scaler.transform(test_data[feature_columns])
        
        # Store scaler for future use
        self.scalers['features'] = scaler
        
        print(f"   ✅ Standardized {len(feature_columns)} features")
        print(f"   Feature means: {scaler.mean_[:5].round(4)}")  # Show first 5
        print(f"   Feature stds: {scaler.scale_[:5].round(4)}")  # Show first 5
        
        return train_scaled, test_scaled
    
    def create_interaction_features(self, data):
        """
        Create interaction features between temporal and sales features
        """
        print("🔄 Creating interaction features...")
        
        # Sales * cyclical interactions
        if 'sales_lag_7' in data.columns and 'dayofweek_sin' in data.columns:
            data['sales_lag7_dayofweek'] = data['sales_lag_7'] * data['dayofweek_sin']
            data['sales_mean7_month'] = data['sales_mean_7'] * data['month_sin']
        
        # Volatility * trend interactions
        if 'sales_volatility_7' in data.columns and 'sales_trend_7' in data.columns:
            data['volatility_trend_interaction'] = data['sales_volatility_7'] * data['sales_trend_7']
        
        print(f"   ✅ Created interaction features")
        
        return data
    
    def get_feature_columns(self, data):
        """
        Identify all feature columns for modeling
        """
        feature_cols = [col for col in data.columns if any([
            col.startswith('sales_lag_'),
            col.startswith('sales_mean_'),
            col.startswith('sales_std_'),
            col.startswith('sales_min_'),
            col.startswith('sales_max_'),
            col.startswith('sales_trend_'),
            col.startswith('sales_volatility_'),
            col.endswith('_sin'),
            col.endswith('_cos'),
            col.endswith('_interaction'),
            col == 'onpromotion',
        ])]
        
        return feature_cols
    
    def process_evaluation_case(self, train_data, test_data, store_id, family):
        """
        Complete preprocessing pipeline for a single evaluation case
        """
        print(f"\n🔧 PROCESSING CASE: Store {store_id}, {family}")
        print("=" * 50)
        
        # Step 1: Apply log1p transformation
        train_log = self.apply_log1p_transformation(train_data)
        test_log = self.apply_log1p_transformation(test_data)
        
        # Step 2: Create cyclical features
        train_cyclical = self.create_cyclical_features(train_log)
        test_cyclical = self.create_cyclical_features(test_log)
        
        # Step 3: Create enhanced features for this specific case
        train_enhanced = self.create_enhanced_lag_features(train_cyclical, store_id, family)
        test_enhanced = self.create_enhanced_lag_features(test_cyclical, store_id, family)
        
        # Step 4: Create interaction features
        train_interactions = self.create_interaction_features(train_enhanced)
        test_interactions = self.create_interaction_features(test_enhanced)
        
        # Step 5: Get feature columns
        feature_columns = self.get_feature_columns(train_interactions)
        print(f"📊 Total features: {len(feature_columns)}")
        
        # Step 6: Apply standardization
        train_final, test_final = self.apply_standardization(
            train_interactions, test_interactions, feature_columns
        )
        
        # Verification
        print(f"✅ PREPROCESSING COMPLETE:")
        print(f"   Train shape: {train_final.shape}")
        print(f"   Test shape: {test_final.shape}")
        print(f"   Features: {len(feature_columns)}")
        
        return train_final, test_final, feature_columns

# Usage example
processor = Phase35DataProcessor()