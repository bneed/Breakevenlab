#!/usr/bin/env python3
"""
TradeScrubber - Smart Stock & Options Screener with Timing
Main application launcher with NiceGUI (fallback to Streamlit)
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.utils import load_env, setup_logging
from core.data import DataManager
from core.indicators import add_indicators
from core.signals import compute_signals
from core.ranker import rank_universe
from core.options import OptionsAnalyzer
from core.ml import MLPredictor
from core.backtest import BacktestEngine

# Configuration
USE_NICEGUI = True  # Toggle between NiceGUI and Streamlit

def main():
    """Main application entry point"""
    # Load environment and setup logging
    load_env()
    logger = setup_logging()
    
    logger.info("Starting TradeScrubber application...")
    
    if USE_NICEGUI:
        try:
            from nicegui import ui, app
            run_nicegui_app()
        except ImportError:
            logger.warning("NiceGUI not available, falling back to Streamlit")
            run_streamlit_app()
    else:
        run_streamlit_app()

def run_nicegui_app():
    """Run the NiceGUI application"""
    from nicegui import ui, app
    from ui.nicegui_interface import create_nicegui_interface
    
    # Create the main interface
    create_nicegui_interface()
    
    # Run the app
    ui.run(
        title="TradeScrubber",
        port=8080,
        show=True,
        dark=True
    )

def run_streamlit_app():
    """Run the Streamlit application (fallback)"""
    import streamlit as st
    from ui.streamlit_interface import create_streamlit_interface
    
    st.set_page_config(
        page_title="TradeScrubber",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    create_streamlit_interface()

if __name__ == "__main__":
    main()
