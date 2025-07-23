#!/usr/bin/env python3
"""
Phase 6 Startup Script
Single-command launcher for FastAPI backend + Streamlit dashboard
"""

import sys
import subprocess
import time
import signal
import platform
from pathlib import Path
from typing import List
import webbrowser
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Apply numpy compatibility fix early
try:
    import numpy_compat
    logger.info("✅ NumPy compatibility patch applied")
except ImportError:
    logger.warning("⚠️  NumPy compatibility patch not found")

from src.serving.config import config


class Phase6Launcher:
    """Launcher for Phase 6 FastAPI + Streamlit stack"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.project_root = project_root
        
    def validate_environment(self) -> bool:
        """Validate environment and dependencies"""
        logger.info("🔍 Validating environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            logger.error(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            return False
        
        logger.info(f"✅ Python {python_version.major}.{python_version.minor} OK")
        
        # Check required directories
        required_dirs = ["src", "results", "data"]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                logger.error(f"❌ Required directory missing: {dir_path}")
                return False
        
        logger.info("✅ Required directories found")
        
        # Check critical files
        critical_files = [
            "src/serving/api.py",
            "src/serving/dashboard.py", 
            "src/serving/config.py",
            "requirements.txt"
        ]
        
        for file_path in critical_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                logger.error(f"❌ Critical file missing: {full_path}")
                return False
        
        logger.info("✅ Critical files found")
        
        # Test imports
        try:
            import flask
            import streamlit
            import plotly
            import pandas
            import numpy
            logger.info("✅ Key dependencies available")
        except ImportError as e:
            logger.error(f"❌ Missing dependency: {e}")
            logger.error("Please install dependencies: pip install -r requirements.txt")
            return False
        
        return True
    
    def check_ports(self) -> bool:
        """Check if required ports are available"""
        import socket
        
        ports = [config.api_port, config.streamlit_port]
        
        for port in ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(('127.0.0.1', port))
                logger.info(f"✅ Port {port} available")
            except OSError:
                logger.error(f"❌ Port {port} already in use")
                return False
        
        return True
    
    def start_flask_api(self) -> subprocess.Popen:
        """Start Flask API backend"""
        logger.info("🚀 Starting Flask backend...")
        
        api_script = self.project_root / "src" / "serving" / "flask_api.py"
        
        # Run Flask directly
        cmd = [
            sys.executable,
            str(api_script)
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Start thread to monitor output
        threading.Thread(
            target=self._monitor_process,
            args=(process, "Flask"),
            daemon=True
        ).start()
        
        return process
    
    def start_streamlit(self) -> subprocess.Popen:
        """Start Streamlit dashboard"""
        logger.info("🎨 Starting Streamlit dashboard...")
        
        dashboard_script = self.project_root / "src" / "serving" / "dashboard.py"
        
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_script),
            "--server.port", str(config.streamlit_port),
            "--server.address", config.streamlit_host,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Start thread to monitor output
        threading.Thread(
            target=self._monitor_process,
            args=(process, "Streamlit"),
            daemon=True
        ).start()
        
        return process
    
    def _monitor_process(self, process: subprocess.Popen, name: str):
        """Monitor process output"""
        for line in iter(process.stdout.readline, ''):
            if line.strip():
                logger.info(f"[{name}] {line.strip()}")
    
    def wait_for_services(self) -> bool:
        """Wait for services to be ready"""
        import requests
        
        logger.info("⏳ Waiting for services to start...")
        
        # Wait for Flask API
        for attempt in range(30):  # 30 second timeout
            try:
                response = requests.get(f"{config.api_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Flask API is ready")
                    break
            except requests.RequestException:
                pass
            
            time.sleep(1)
        else:
            logger.error("❌ Flask API failed to start within 30 seconds")
            return False
        
        # Wait for Streamlit (just check if port is listening)
        import socket
        for attempt in range(30):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex((config.streamlit_host, config.streamlit_port))
                    if result == 0:
                        logger.info("✅ Streamlit is ready")
                        break
            except:
                pass
            
            time.sleep(1)
        else:
            logger.error("❌ Streamlit failed to start within 30 seconds")
            return False
        
        return True
    
    def open_browser(self):
        """Open browser tabs for both services"""
        logger.info("🌐 Opening browser...")
        
        # Small delay to ensure services are fully ready
        time.sleep(2)
        
        try:
            # Open Streamlit dashboard (main interface)
            webbrowser.open(config.streamlit_url)
            logger.info(f"📊 Dashboard: {config.streamlit_url}")
            
            # Optionally open API root
            time.sleep(1)
            webbrowser.open(f"{config.api_url}")
            logger.info(f"📚 API Root: {config.api_url}")
            
        except Exception as e:
            logger.warning(f"Failed to open browser: {e}")
            logger.info(f"Manual URLs:")
            logger.info(f"  Dashboard: {config.streamlit_url}")
            logger.info(f"  API Root:  {config.api_url}")
    
    def run(self):
        """Main run method"""
        logger.info("🏪 Starting Phase 6: Store Sales Forecasting Dashboard")
        logger.info("=" * 60)
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Pre-flight checks
            if not self.validate_environment():
                logger.error("❌ Environment validation failed")
                return False
            
            if not self.check_ports():
                logger.error("❌ Port check failed")
                return False
            
            # Start services
            api_process = self.start_flask_api()
            self.processes.append(api_process)
            
            streamlit_process = self.start_streamlit()
            self.processes.append(streamlit_process)
            
            # Wait for services
            if not self.wait_for_services():
                logger.error("❌ Services failed to start")
                self.cleanup()
                return False
            
            # Open browser
            self.open_browser()
            
            # Success message
            logger.info("=" * 60)
            logger.info("🎉 Phase 6 Dashboard Successfully Started!")
            logger.info("=" * 60)
            logger.info(f"📊 Streamlit Dashboard: {config.streamlit_url}")
            logger.info(f"🔧 Flask Backend:       {config.api_url}")
            logger.info(f"📚 API Endpoints:       {config.api_url}")
            logger.info("=" * 60)
            logger.info("Press Ctrl+C to stop all services")
            logger.info("=" * 60)
            
            # Keep running
            try:
                while True:
                    # Check if processes are still running
                    for process in self.processes:
                        if process.poll() is not None:
                            logger.error(f"Process died with return code {process.returncode}")
                            self.cleanup()
                            return False
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("🛑 Shutdown requested by user")
            
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return False
        
        finally:
            self.cleanup()
        
        return True
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up processes"""
        logger.info("🧹 Cleaning up processes...")
        
        for process in self.processes:
            if process.poll() is None:
                logger.info(f"Terminating process {process.pid}")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing process {process.pid}")
                    process.kill()
                    process.wait()
        
        self.processes.clear()
        logger.info("✅ Cleanup completed")


def main():
    """Main entry point"""
    launcher = Phase6Launcher()
    success = launcher.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()