# Store Sales Forecasting with STGAT - Setup Instructions

This document provides comprehensive setup instructions for the Store Sales Forecasting project, supporting both Linux and macOS environments.

## Quick Start

### Option 1: Production Setup (Traditional Models Only)
```bash
# Clone the repository
git clone https://github.com/your-username/store-sales-stgat-project.git
cd store-sales-stgat-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements-core.txt

# Run the application
python run_phase6.py
```

### Option 2: Full Development Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-username/store-sales-stgat-project.git
cd store-sales-stgat-project

# Create conda environment
conda env create -f environment.yml
conda activate store-sales-stgat

# Run the application
python run_phase6.py
```

## Detailed Setup Guide

### Environment Options

The project provides multiple ways to manage dependencies:

1. **requirements-core.txt**: Essential dependencies for traditional models and API
2. **requirements-neural.txt**: Neural model dependencies (PyTorch, etc.)
3. **requirements-dev.txt**: Development tools (Jupyter, testing, etc.)
4. **environment.yml**: Complete conda environment specification
5. **requirements.txt**: Main requirements file with installation options

### Prerequisites

- **Python**: 3.10 or higher (3.12 recommended)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: At least 8GB RAM (16GB recommended for neural models)
- **Storage**: At least 5GB free space

### Installation Methods

#### Method 1: Using Conda (Recommended for Development)

```bash
# Install Miniconda or Anaconda if not already installed
# https://docs.conda.io/en/latest/miniconda.html

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate store-sales-stgat

# Verify installation
python -c "import pandas, numpy, sklearn; print('Core packages installed successfully')"

# Test neural models (optional)
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

#### Method 2: Using pip with Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies based on your needs:

# Option A: Core only (production)
pip install -r requirements-core.txt

# Option B: Core + Neural models
pip install -r requirements-core.txt -r requirements-neural.txt

# Option C: Full development setup
pip install -r requirements-core.txt -r requirements-neural.txt -r requirements-dev.txt
```

#### Method 3: Docker Setup (Future Enhancement)

```dockerfile
# Dockerfile would be created for containerized deployment
# This ensures consistent environments across different systems
```

### Neural Models Configuration

Neural models are **disabled by default** due to stability issues in production environments.

To enable neural models:

1. **Install neural dependencies**:
   ```bash
   pip install -r requirements-neural.txt
   ```

2. **Enable in configuration**:
   Edit `src/serving/config.py`:
   ```python
   # Change line 36 from:
   self.enable_neural_models = False
   # to:
   self.enable_neural_models = True
   ```

3. **Verify neural model functionality**:
   ```bash
   python test_neural_config.py
   ```

### Platform-Specific Notes

#### Linux Servers
- Use the conda environment for best compatibility
- For production servers, consider using only `requirements-core.txt`
- Ensure you have build tools installed: `sudo apt-get install build-essential`

#### macOS
- The project was originally developed on macOS
- Both conda and pip methods work well
- For M1/M2 Macs, ensure you're using the appropriate PyTorch builds

#### Windows
- Use conda environment for best results
- Some packages may require Visual C++ build tools
- Consider using WSL2 for Linux-like environment

### GPU Support (Optional)

If you have an NVIDIA GPU and want to use CUDA:

1. **Install CUDA-enabled PyTorch**:
   ```bash
   # Replace the PyTorch installation in requirements-neural.txt with:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Verify GPU availability**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   ```

### Running the Application

#### Start the Complete Application
```bash
python run_phase6.py
```

This starts both:
- **FastAPI backend** on http://127.0.0.1:8000
- **Streamlit dashboard** on http://127.0.0.1:8501

#### Start Components Separately

**API only**:
```bash
python -m uvicorn src.serving.flask_api:app --host 127.0.0.1 --port 8000
```

**Dashboard only** (requires API to be running):
```bash
streamlit run src/serving/dashboard.py --server.port 8501
```

### Development Setup

If you're planning to develop or modify the project:

1. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Set up pre-commit hooks** (optional):
   ```bash
   pre-commit install
   ```

3. **Run tests**:
   ```bash
   pytest tests/
   ```

4. **Start Jupyter Lab** for notebook development:
   ```bash
   jupyter lab
   ```

### Troubleshooting

#### Common Issues

1. **Import errors**: Ensure you've activated the correct environment
2. **Neural model crashes**: Keep neural models disabled for production
3. **Port conflicts**: Change ports in `src/serving/config.py`
4. **Memory issues**: Reduce batch sizes or use fewer models

#### Debugging Neural Models

If neural models are enabled but crashing:

```bash
# Run the neural model debugger
python debug_neural_models.py

# Check the debug log
cat neural_debug.log
```

#### Environment Conflicts

If you're experiencing package conflicts:

```bash
# Remove existing environment
conda env remove -n store-sales-stgat

# Recreate from scratch
conda env create -f environment.yml
```

### Data Requirements

The project expects the following data files in `data/raw/`:
- `train.csv`: Training data
- `test.csv`: Test data (optional)
- `stores.csv`: Store information
- `oil.csv`: Oil price data
- `holidays_events.csv`: Holiday information
- `transactions.csv`: Transaction data

Download these from the Kaggle Store Sales Competition.

### Performance Optimization

For better performance:

1. **Use SSD storage** for faster data loading
2. **Increase RAM** for larger datasets
3. **Use conda environment** for optimized packages
4. **Disable neural models** in production unless necessary

### Support

If you encounter issues:

1. Check this setup guide
2. Review the project's README.md
3. Check existing GitHub issues
4. Create a new issue with detailed error information

---

**Last Updated**: January 2025
**Python Version**: 3.12
**Tested Platforms**: macOS (Intel/M1), Ubuntu 20.04+, Windows 10+