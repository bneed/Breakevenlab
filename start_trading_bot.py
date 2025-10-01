#!/usr/bin/env python3
"""
Startup script for the Trading Bot
Launches the Streamlit application with proper configuration
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'yfinance',
        'plotly',
        'sklearn',  # scikit-learn imports as sklearn
        'ta'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_environment():
    """Set up the environment for the trading bot"""
    # Set working directory to app folder
    app_dir = Path(__file__).parent / "app"
    os.chdir(app_dir)
    
    # Set environment variables
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    
    print(f"Working directory: {os.getcwd()}")
    print("Environment configured")

def start_streamlit():
    """Start the Streamlit application"""
    print("Starting Trading Bot...")
    print("Open your browser to: http://localhost:8501")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\nTrading Bot stopped by user")
    except Exception as e:
        print(f"Error starting Trading Bot: {e}")

def main():
    """Main startup function"""
    print("Trading Bot Startup")
    print("=" * 30)
    
    # Check requirements
    if not check_requirements():
        print("\nPlease install missing packages and try again")
        sys.exit(1)
    
    print("All requirements satisfied")
    
    # Setup environment
    setup_environment()
    
    # Start the application
    start_streamlit()

if __name__ == "__main__":
    main()
