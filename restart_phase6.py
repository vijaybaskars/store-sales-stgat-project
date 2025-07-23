#!/usr/bin/env python3
"""
Quick restart script for Phase 6 when the API crashes
Cleans up any hanging processes and restarts the system
"""

import subprocess
import signal
import time
import sys
import os

def cleanup_processes():
    """Clean up any hanging Phase 6 processes"""
    print("🧹 Cleaning up existing processes...")
    
    # Kill processes on ports 8000 and 8501
    for port in [8000, 8501]:
        try:
            result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        print(f"  Killing process {pid} on port {port}")
                        os.kill(int(pid), signal.SIGTERM)
                        time.sleep(1)
        except Exception as e:
            print(f"  Note: {e}")
    
    # Kill any python processes running Phase 6 scripts
    try:
        result = subprocess.run(['pgrep', '-f', 'phase6'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"  Killing Phase 6 process {pid}")
                    os.kill(int(pid), signal.SIGTERM)
                    time.sleep(1)
    except Exception:
        pass
    
    print("✅ Cleanup completed")

def restart_system():
    """Restart the Phase 6 system"""
    print("🚀 Restarting Phase 6 system...")
    
    # Change to project directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Start the system
    try:
        subprocess.run([sys.executable, 'run_phase6.py'])
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested")
    except Exception as e:
        print(f"❌ Failed to start: {e}")

def main():
    print("🔄 Phase 6 Restart Utility")
    print("=" * 40)
    
    cleanup_processes()
    time.sleep(2)  # Give processes time to die
    restart_system()

if __name__ == "__main__":
    main()