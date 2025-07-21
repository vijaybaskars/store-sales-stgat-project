import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class Phase35DataProcessor:
    """
    Advanced data preprocessing for Phase 3.5 optimization
    Integrates with existing EvaluationCaseManager infrastructure
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_stats = {}
        print("🔧 Phase35DataProcessor initialized")
    
    def apply_log1p_transformation(self, sales_data):
        """Apply log1p transformation to sales data"""
        print("🔄 Applying log1p transformation...")
        
        data = sales_data.copy()
        data['sales_original'] = data['sales']
        data['sales_log1p'] = np.log1p(data['sales'])
        data['sales'] = data['sales_log1p']
        
        print(f"   Sales range: {data['sales_original'].min():.2f} to {data['sales_original'].max():.2f}")
        print(f"   Log1p range: {data['sales'].min():.4f} to {data['sales'].max():.4f}")
        
        return data
    
    def create_cyclical_features(self, data):
        """Create cyclical encodings for temporal features"""
        print("🔄 Creating cyclical temporal features...")
        
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            
            # Day of week cyclical encoding
            data['dayofweek'] = data['date'].dt.dayofweek
            data['dayofweek_sin'] = np.sin(2 * np.pi * data['dayofweek'] / 7)
            data['dayofweek_cos'] = np.cos(2 * np.pi * data['dayofweek'] / 7)
            
            # Month cyclical encoding
            data['month'] = data['date'].dt.month
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            
            # Day of month cyclical encoding
            data['dayofmonth'] = data['date'].dt.day
            data['dayofmonth_sin'] = np.sin(2 * np.pi * data['dayofmonth'] / 31)
            data['dayofmonth_cos'] = np.cos(2 * np.pi * data['dayofmonth'] / 31)
            
            print(f"   ✅ Created 6 cyclical features")
        
        return data
    
    def create_enhanced_lag_features(self, data, store_id, family):
        """Create enhanced lag and rolling features"""
        print(f"🔄 Creating enhanced features for Store {store_id}, {family}...")
        
        # Filter and sort data
        store_family_data = data[
            (data['store_nbr'] == store_id) & 
            (data['family'] == family)
        ].sort_values('date').copy()
        
        if len(store_family_data) < 30:
            print(f"   ⚠️ Limited data: {len(store_family_data)} records")
            return store_family_data
        
        # Enhanced lag features
        lag_periods = [1, 2, 3, 7, 14, 21, 28]
        for lag in lag_periods:
            store_family_data[f'sales_lag_{lag}'] = store_family_data['sales'].shift(lag)
        
        # Rolling statistics with multiple windows
        windows = [3, 7, 14, 21, 28]
        for window in windows:
            store_family_data[f'sales_mean_{window}'] = store_family_data['sales'].rolling(window, min_periods=1).mean()
            store_family_data[f'sales_std_{window}'] = store_family_data['sales'].rolling(window, min_periods=1).std()
            store_family_data[f'sales_min_{window}'] = store_family_data['sales'].rolling(window, min_periods=1).min()
            store_family_data[f'sales_max_{window}'] = store_family_data['sales'].rolling(window, min_periods=1).max()
            
            # Additional statistical features
            store_family_data[f'sales_q25_{window}'] = store_family_data['sales'].rolling(window, min_periods=1).quantile(0.25)
            store_family_data[f'sales_q75_{window}'] = store_family_data['sales'].rolling(window, min_periods=1).quantile(0.75)
        
        # Trend and volatility features
        store_family_data['sales_trend_7'] = store_family_data['sales'].rolling(7, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
        )
        store_family_data['sales_trend_14'] = store_family_data['sales'].rolling(14, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
        )
        
        # Volatility features
        store_family_data['sales_volatility_7'] = store_family_data['sales'].rolling(7, min_periods=2).std()
        store_family_data['sales_volatility_14'] = store_family_data['sales'].rolling(14, min_periods=2).std()
        
        # Fill NaN values (using updated pandas method)
        store_family_data = store_family_data.fillna(method='ffill').fillna(0)
        
        feature_count = len([c for c in store_family_data.columns if c.startswith('sales_')])
        print(f"   ✅ Created {feature_count} enhanced features")
        
        return store_family_data
    
    def apply_standardization(self, train_data, test_data, feature_columns):
        """Apply StandardScaler to feature columns"""
        print("🔄 Applying StandardScaler to features...")
        
        scaler = StandardScaler()
        
        # Fit on training data only
        train_scaled = train_data.copy()
        train_scaled[feature_columns] = scaler.fit_transform(train_data[feature_columns])
        
        # Transform test data
        test_scaled = test_data.copy()
        test_scaled[feature_columns] = scaler.transform(test_data[feature_columns])
        
        # Store scaler
        self.scalers['features'] = scaler
        
        print(f"   ✅ Standardized {len(feature_columns)} features")
        print(f"   Feature means (first 5): {scaler.mean_[:5].round(4)}")
        print(f"   Feature stds (first 5): {scaler.scale_[:5].round(4)}")
        
        return train_scaled, test_scaled, scaler
    
    def get_feature_columns(self, data):
        """Identify all feature columns for modeling"""
        feature_cols = [col for col in data.columns if any([
            col.startswith('sales_lag_'),
            col.startswith('sales_mean_'),
            col.startswith('sales_std_'),
            col.startswith('sales_min_'),
            col.startswith('sales_max_'),
            col.startswith('sales_q25_'),
            col.startswith('sales_q75_'),
            col.startswith('sales_trend_'),
            col.startswith('sales_volatility_'),
            col.endswith('_sin'),
            col.endswith('_cos'),
            col == 'onpromotion',
        ])]
        
        return feature_cols
    
    def process_evaluation_case(self, case_manager, store_id, family):
        """
        Complete preprocessing pipeline for a single evaluation case
        Uses existing EvaluationCaseManager infrastructure
        """
        print(f"\n🔧 PROCESSING CASE: Store {store_id}, {family}")
        print("=" * 50)
        
        # Load sales data and use existing infrastructure to get case data
        # First, we need the full sales dataset
        import pandas as pd
        import os
        
        # Try to load sales data
        possible_paths = [
            'data/raw/train.csv',
            '../data/raw/train.csv', 
            '../../data/raw/train.csv',
            '../../../data/raw/train.csv'
        ]
        
        sales_data = None
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    sales_data = pd.read_csv(path)
                    sales_data['date'] = pd.to_datetime(sales_data['date'])
                    print(f"   ✅ Loaded sales data from: {path}")
                    break
            except:
                continue
        
        if sales_data is None:
            raise ValueError("Could not find sales data file (train.csv)")
        
        # Use the case manager's get_case_data method with the sales data
        case_info = {'store_nbr': store_id, 'family': family}
        train_data, test_data = case_manager.get_case_data(sales_data, case_info)
        
        if train_data is None or test_data is None:
            raise ValueError(f"No data available for Store {store_id}, {family}")
        
        print(f"   Raw train data: {len(train_data)} records")
        print(f"   Raw test data: {len(test_data)} records")
        
        # Step 1: Apply log1p transformation
        train_log = self.apply_log1p_transformation(train_data)
        test_log = self.apply_log1p_transformation(test_data)
        
        # Step 2: Create cyclical features
        train_cyclical = self.create_cyclical_features(train_log)
        test_cyclical = self.create_cyclical_features(test_log)
        
        # Step 3: Create enhanced features
        train_enhanced = self.create_enhanced_lag_features(train_cyclical, store_id, family)
        test_enhanced = self.create_enhanced_lag_features(test_cyclical, store_id, family)
        
        # Step 4: Get feature columns
        feature_columns = self.get_feature_columns(train_enhanced)
        print(f"📊 Total features identified: {len(feature_columns)}")
        
        # Step 5: Apply standardization
        train_final, test_final, scaler = self.apply_standardization(
            train_enhanced, test_enhanced, feature_columns
        )
        
        print(f"✅ PREPROCESSING COMPLETE:")
        print(f"   Final train shape: {train_final.shape}")
        print(f"   Final test shape: {test_final.shape}")
        print(f"   Feature count: {len(feature_columns)}")
        
        return train_final, test_final, feature_columns, scaler