# Phase 3.5: Neural Baseline Optimization - New Chat Handoff

## 🎯 OBJECTIVE
Optimize Phase 3 neural baselines with **minimal architecture changes** but **maximum performance impact** to achieve 15-25% improvement over traditional baselines (target RMSLE < 0.40).

## 📊 CURRENT STATUS (Phase 3 Completion)
- **Branch**: phase3.5-optimization  
- **Environment**: store_sales_stgat (conda)
- **Traditional Baseline**: 0.4755 RMSLE (ARIMA)
- **Current Neural Best**: 0.5466 RMSLE (LSTM+Features)
- **Performance Gap**: -15% degradation (NEEDS FIXING)

## 🚨 IDENTIFIED PROBLEMS
1. **Suboptimal hyperparameters**: hidden_size=32, num_layers=1, epochs=30
2. **Poor preprocessing**: MinMaxScaler, basic features, short sequences (20)
3. **Insufficient training**: Early stopping, no optimization
4. **Architecture too simple**: Needs more capacity

## 🎯 PHASE 3.5 HIGH-IMPACT TARGETS

### Priority 1: Data Preprocessing (HIGHEST IMPACT)
- ✅ **log1p transformation** for sales data (handles skewness)
- ✅ **StandardScaler** instead of MinMaxScaler  
- ✅ **Cyclical encoding** for temporal features (sin/cos)
- ✅ **Enhanced lag features** (1, 7, 14, 30 days)
- ✅ **Rolling statistics** (mean, std, min, max for 7/14/30 days)

### Priority 2: Hyperparameter Optimization (HIGH IMPACT)
- ✅ **Optuna optimization** for systematic tuning
- ✅ **Longer sequences** (45-60 timesteps vs 20)
- ✅ **Larger architectures** (hidden_size: 128-256 vs 32)
- ✅ **More layers** (2-3 vs 1)
- ✅ **Longer training** (100-200 epochs vs 30)

### Priority 3: Training Improvements (MEDIUM IMPACT)
- ✅ **Learning rate scheduling** (CosineAnnealingLR)
- ✅ **Gradient clipping** (prevent exploding gradients)
- ✅ **Multiple seeds** (ensemble of random initializations)
- ✅ **Better early stopping** (patience=20 vs 10)

## 📋 IMPLEMENTATION STRATEGY

### Approach: Minimal Code Changes, Maximum Impact
1. **Keep existing infrastructure** (EvaluationCaseManager, etc.)
2. **Modify only preprocessing and model configs**
3. **Add Optuna for systematic optimization**
4. **Focus on 2-3 best performing architectures**

### Expected Timeline
- **Data preprocessing improvements**: 1 hour
- **Hyperparameter optimization setup**: 1 hour  
- **Optimization run**: 2-3 hours (automated)
- **Evaluation and analysis**: 1 hour
- **Total**: 5-6 hours for complete Phase 3.5

## 🎯 SUCCESS CRITERIA

### Minimum Success (Phase 4 Readiness)
- **Neural Best RMSLE**: < 0.4300 (10% improvement over traditional)
- **Statistical Significance**: p < 0.05
- **Consistency**: At least 7/10 cases beat traditional baseline

### Target Success (Strong Foundation for STGAT)
- **Neural Best RMSLE**: < 0.3800 (20% improvement over traditional)  
- **Statistical Significance**: p < 0.01
- **Consistency**: At least 8/10 cases beat traditional baseline

### Stretch Success (Excellent STGAT Setup)
- **Neural Best RMSLE**: < 0.3300 (30% improvement over traditional)
- **All neural models**: Beat traditional baseline
- **Ensemble approach**: Further improvements

## 📁 KEY FILES FOR PHASE 3.5

### Existing Infrastructure (DO NOT CHANGE)
- src/data/evaluation_cases.py - Evaluation framework
- results/evaluation_cases.json - 10 test cases
- results/traditional_baseline_results.json - Traditional baselines

### Files to Modify/Create
- src/models/neural/neural_baselines_optimized.py - Enhanced models
- notebooks/03.5_neural_optimization.ipynb - Optimization notebook
- results/phase3.5/ - Optimization results

## 🔧 TECHNICAL SPECIFICATIONS

### Environment Setup
```bash
conda activate store_sales_stgat
pip install optuna
pip install mlflow
Model Configuration Targets
pythonoptimized_configs = {
    'lstm_enhanced': {
        'hidden_size': [128, 256, 512],
        'num_layers': [2, 3, 4],
        'dropout': [0.2, 0.3, 0.4],
        'sequence_length': [45, 60, 90],
        'learning_rate': [0.0001, 0.0005, 0.001],
        'batch_size': [8, 16, 32]
    }
}
Data Processing Improvements
pythondef enhanced_preprocessing():
    # 1. Log transformation
    sales_log = np.log1p(sales_data)
    
    # 2. Cyclical encoding
    time_features = create_cyclical_features(dates)
    
    # 3. Advanced lag features
    lag_features = create_lag_features(sales_log, lags=[1,7,14,30])
    
    # 4. Rolling statistics
    rolling_features = create_rolling_features(sales_log, windows=[7,14,30])
🚀 NEW CHAT SESSION PROMPT
Use this prompt when starting Phase 3.5 in new chat:

I'm continuing a Store Sales STGAT forecasting project with a critical optimization phase.
CONTEXT:

✅ Phase 1-2: Completed with exceptional results
✅ Phase 3: Neural baselines implemented but underperforming
🎯 Phase 3.5: OPTIMIZATION NEEDED - Neural models performing 15% worse than traditional

CURRENT STATUS:

Branch: phase3.5-optimization
Environment: store_sales_stgat (conda)
Traditional baseline: 0.4755 RMSLE (ARIMA)
Neural baseline: 0.5466 RMSLE (needs improvement)

PHASE 3.5 OBJECTIVE:
Optimize neural baselines with minimal code changes but maximum impact to achieve 15-25% improvement over traditional baselines (target < 0.40 RMSLE).
HIGH-IMPACT OPTIMIZATIONS NEEDED:

log1p transformation + StandardScaler
Cyclical encoding for temporal features
Enhanced lag/rolling features
Optuna hyperparameter optimization
Larger architectures + longer training

SUCCESS TARGET: Neural RMSLE < 0.38 (20% improvement) to set up strong STGAT foundation.
Please help me implement Phase 3.5 optimization with focus on maximum impact improvements.

