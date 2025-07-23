# Phase 6: Store Sales Forecasting Dashboard

## Overview

Phase 6 implements a working prototype with **FastAPI backend** and **Streamlit dashboard** that showcases the **Pattern-Based Selection methodology** (Phase 3.6 champion with 0.4190 RMSLE).

## Key Features

✅ **Interactive Dashboard**: Streamlit-based web interface
✅ **REST API**: Flask backend with comprehensive endpoints
✅ **Pattern-Based Selection**: Adaptive routing between Neural and Traditional models  
✅ **Real-time Predictions**: Generate forecasts for evaluation cases
✅ **Performance Visualization**: Charts comparing vs Phase 2/3 baselines
✅ **Single-Command Startup**: Easy deployment with one script

## Architecture

```
Phase 6 Architecture:
├── Streamlit Dashboard (Port 8501)
│   ├── Interactive case selection
│   ├── Pattern analysis visualization  
│   ├── Prediction charts
│   └── Performance comparison
├── Flask API Backend (Port 8000)
│   ├── /health - Health check
│   ├── /cases - List evaluation cases
│   ├── /analysis/{store}/{family} - Pattern analysis
│   ├── /predict/{store}/{family} - Generate predictions
│   └── /dashboard/summary - Summary stats
└── Data Integration
    ├── Existing PatternBasedSelector
    ├── 10 high-quality evaluation cases
    └── Phase 3.6 champion methodology
```

## Quick Start

### 1. Test Components
```bash
python test_phase6.py
```

### 2. Start Dashboard
```bash
python run_phase6.py
```

The script will:
- ✅ Validate environment and dependencies
- 🚀 Start Flask API backend (port 8000)
- 🎨 Start Streamlit dashboard (port 8501)
- 🌐 Open browser automatically
- ⏳ Run until Ctrl+C

### 3. Access Interfaces

- **Main Dashboard**: http://127.0.0.1:8501
- **API Backend**: http://127.0.0.1:8000
- **API Endpoints**: http://127.0.0.1:8000/ (JSON info)

## Usage Guide

### Dashboard Interface

1. **Select Store-Family**: Choose from 10 high-quality evaluation cases
2. **Set Forecast Horizon**: 1-30 days (default: 15)
3. **View Results**:
   - Prediction vs actual sales chart
   - Pattern analysis (CV-based routing decision)
   - Performance metrics (RMSLE, MAE, MAPE)
   - Baseline comparison
4. **Explore Details**:
   - Pattern analysis dashboard with gauges and charts
   - Raw time series characteristics
   - Model selection confidence

### API Endpoints

```bash
# Health check
curl http://127.0.0.1:8000/health

# List evaluation cases  
curl http://127.0.0.1:8000/cases

# Pattern analysis
curl http://127.0.0.1:8000/analysis/49/PET_SUPPLIES

# Generate prediction
curl -X POST http://127.0.0.1:8000/predict/49/PET_SUPPLIES \
  -H "Content-Type: application/json" \
  -d '{"forecast_horizon": 15}'

# Dashboard summary
curl http://127.0.0.1:8000/dashboard/summary
```

## Implementation Details

### Pattern-Based Selection Logic

```python
# CV-based routing (threshold = 1.5)
if coefficient_variation < 1.5:
    route_to = "NEURAL"  # LSTM variants
else:
    route_to = "TRADITIONAL"  # ARIMA, Exp Smoothing
```

### Performance Baselines

- **Traditional Baseline** (Phase 2): 0.4755 RMSLE
- **Neural Baseline** (Phase 3): 0.5466 RMSLE  
- **Phase 6 Champion**: 0.4190 RMSLE (+11.9% vs Traditional, +23.3% vs Neural)

### Evaluation Cases

10 high-quality store-family combinations with quality scores 98.1-99.2:
- Store 49 - PET SUPPLIES
- Store 45 - HARDWARE  
- Store 47 - BOOKS
- Store 3 - CLEANING
- And 6 more...

## File Structure

```
src/serving/
├── config.py           # Configuration management
├── models.py           # Pydantic response models  
├── data_service.py     # Data loading & ML integration
├── flask_api.py        # Flask REST API
└── dashboard.py        # Streamlit dashboard

run_phase6.py          # Single-command launcher
test_phase6.py         # Component testing
PHASE6_README.md       # This documentation
```

## Dependencies

Uses existing environment dependencies:
- ✅ **Flask**: Web API backend
- ✅ **Streamlit**: Interactive dashboard
- ✅ **Plotly**: Visualization charts  
- ✅ **Pandas/Numpy**: Data processing
- ✅ **Pydantic**: API models
- ✅ **PyTorch**: Neural models
- ✅ **Statsmodels**: Traditional models
- ✅ **PMDarima**: ARIMA optimization

No additional installations required!

## Troubleshooting

### Port Already in Use
```bash
# Check port usage
lsof -i :8000
lsof -i :8501

# Kill processes if needed
kill -9 <PID>
```

### Import Errors
```bash
# Test components individually
python test_phase6.py

# Check Python path
python -c "import sys; print(sys.path)"
```

### Data Loading Issues
```bash
# Verify data files exist
ls -la data/raw/train.csv
ls -la results/evaluation_cases.json

# Test data loading
python -c "
import sys; sys.path.insert(0, 'src')
from serving.data_service import data_service
print(len(data_service.get_evaluation_cases()))
"
```

## Development Notes

- **Flask vs FastAPI**: Switched to Flask to use existing dependencies
- **Model Integration**: Leverages existing Pattern-Based Selection implementation
- **Performance**: Lightweight design for demonstration purposes
- **Extensibility**: Modular architecture allows easy enhancements

## Next Steps

Future enhancements could include:
- 📊 Additional visualization charts
- 🔄 Real-time model retraining
- 📈 Extended evaluation metrics
- 🚀 Production deployment configuration
- 📱 Mobile-responsive design

---

**Phase 6 Status**: ✅ Complete - Ready for demonstration
**Champion Result**: 0.4190 RMSLE (Best performance across all phases)