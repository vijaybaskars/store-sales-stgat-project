"""
Configuration management for Phase 6 serving layer
Handles path resolution and environment settings
"""

import os
from pathlib import Path
from typing import Optional


class Phase6Config:
    """Configuration for Phase 6 FastAPI + Streamlit serving"""
    
    def __init__(self):
        # Auto-detect project root
        self.project_root = self._find_project_root()
        
        # API configuration
        self.api_host = "127.0.0.1"
        self.api_port = 8000
        
        # Streamlit configuration  
        self.streamlit_host = "127.0.0.1"
        self.streamlit_port = 8501
        
        # Data paths
        self.evaluation_cases_path = self.project_root / "results" / "evaluation_cases.json"
        self.pattern_results_path = self.project_root / "results" / "pattern_selection" / "adaptive_results.json"
        self.data_path = self.project_root / "data" / "raw" / "train.csv"
        
        # Model configuration
        self.pattern_threshold = 1.5  # CV threshold for model routing
        self.forecast_horizon = 15  # Default forecast horizon (can be overridden by API requests)
        
        # Emergency neural model bypass (set to False to disable neural models)
        self.enable_neural_models = False  # Disabled for production - neural models unstable
        
        # PRODUCTION NOTE: With neural models disabled, ALL cases will route to traditional models
        # This means CV < 1.5 cases will still use the pattern-based selection framework,
        # but will be served by the best traditional models instead of neural models
        
        # Debug configuration for neural models (kept for future debugging)
        self.neural_debug_mode = False  # Disabled for production
        self.neural_timeout_seconds = 300  # 5 minute timeout to prevent hanging
        self.neural_fallback_on_error = True  # Fall back to traditional if neural crashes
        
        # Performance baselines (from Phase 2 & 3)
        self.traditional_baseline = 0.4755
        self.neural_baseline = 0.5466
        
    def _find_project_root(self) -> Path:
        """Find the project root directory"""
        current = Path.cwd()
        
        # Look for key project files
        key_files = ["requirements.txt", "results", "src", "data"]
        
        for _ in range(5):  # Avoid infinite loop
            if all((current / file).exists() for file in key_files):
                return current
            current = current.parent
            
        # Fallback to current directory
        return Path.cwd()
    
    @property
    def api_url(self) -> str:
        """Get the FastAPI base URL"""
        return f"http://{self.api_host}:{self.api_port}"
    
    @property
    def streamlit_url(self) -> str:
        """Get the Streamlit base URL"""
        return f"http://{self.streamlit_host}:{self.streamlit_port}"
    
    def validate_paths(self) -> dict:
        """Validate that all required paths exist"""
        validation_results = {}
        
        paths_to_check = {
            "project_root": self.project_root,
            "evaluation_cases": self.evaluation_cases_path,
            "data_file": self.data_path,
            "results_dir": self.project_root / "results"
        }
        
        for name, path in paths_to_check.items():
            validation_results[name] = {
                "path": str(path),
                "exists": path.exists(),
                "is_file": path.is_file() if path.exists() else False,
                "is_dir": path.is_dir() if path.exists() else False
            }
            
        return validation_results


# Global configuration instance
config = Phase6Config()