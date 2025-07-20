# Phase 2: Traditional Baseline Models

## Objectives
Implement traditional time series forecasting models to establish baseline performance for STGAT comparison.

## Models to Implement
1. **ARIMA** (AutoARIMA with seasonal components)
2. **Exponential Smoothing** (Holt-Winters)
3. **Simple Baselines** (Moving Average, Seasonal Naive)

## Evaluation Framework
- **Cases**: 10 high-quality cases from Phase 1
- **Metric**: RMSLE (Root Mean Squared Logarithmic Error)
- **Split**: 2017-07-01 (established in Phase 1)
- **Horizon**: 15-day forecasting

## Expected Deliverables
- `notebooks/02_traditional_baselines.ipynb`
- `src/models/traditional.py`
- `results/baseline_performance.json`
- Performance comparison framework

## Success Criteria
- All 10 evaluation cases successfully modeled
- RMSLE baselines established for each case
- Statistical significance testing framework
- Production-ready evaluation pipeline

---
*Planning document for Phase 2 implementation*
