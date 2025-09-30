#!/usr/bin/env python3
"""
Test script to verify TradeScrubber installation
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all core imports"""
    print("Testing imports...")
    
    try:
        from core.utils import load_env, setup_logging
        print("✓ core.utils imported successfully")
    except ImportError as e:
        print(f"✗ core.utils import failed: {e}")
        return False
    
    try:
        from core.data import get_prices, get_options_chain
        print("✓ core.data imported successfully")
    except ImportError as e:
        print(f"✗ core.data import failed: {e}")
        return False
    
    try:
        from core.indicators import add_indicators
        print("✓ core.indicators imported successfully")
    except ImportError as e:
        print(f"✗ core.indicators import failed: {e}")
        return False
    
    try:
        from core.signals import compute_signals
        print("✓ core.signals imported successfully")
    except ImportError as e:
        print(f"✗ core.signals import failed: {e}")
        return False
    
    try:
        from core.ranker import rank_universe
        print("✓ core.ranker imported successfully")
    except ImportError as e:
        print(f"✗ core.ranker import failed: {e}")
        return False
    
    try:
        from core.options import analyze_options_chain
        print("✓ core.options imported successfully")
    except ImportError as e:
        print(f"✗ core.options import failed: {e}")
        return False
    
    try:
        from core.ml import train_ml_models, predict_for_today
        print("✓ core.ml imported successfully")
    except ImportError as e:
        print(f"✗ core.ml import failed: {e}")
        return False
    
    try:
        from core.backtest import backtest_strategy
        print("✓ core.backtest imported successfully")
    except ImportError as e:
        print(f"✗ core.backtest import failed: {e}")
        return False
    
    return True

def test_dependencies():
    """Test required dependencies"""
    print("\nTesting dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scipy', 'scikit-learn',
        'yfinance', 'requests', 'ta', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def test_ui_imports():
    """Test UI imports"""
    print("\nTesting UI imports...")
    
    # Test NiceGUI
    try:
        from nicegui import ui
        print("✓ NiceGUI available")
        nicegui_available = True
    except ImportError:
        print("✗ NiceGUI not available (optional)")
        nicegui_available = False
    
    # Test Streamlit
    try:
        import streamlit as st
        print("✓ Streamlit available")
        streamlit_available = True
    except ImportError:
        print("✗ Streamlit not available (optional)")
        streamlit_available = False
    
    if not nicegui_available and not streamlit_available:
        print("⚠️  No UI framework available. Install NiceGUI or Streamlit.")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from core.utils import load_env, setup_logging
        from core.data import get_prices
        from core.indicators import add_indicators
        from core.signals import compute_signals
        
        # Setup logging
        logger = setup_logging()
        
        # Test data fetching
        print("Testing data fetching...")
        prices = get_prices(['SPY'], '1d', 30)
        
        if not prices or 'SPY' not in prices or prices['SPY'].empty:
            print("✗ Data fetching failed")
            return False
        
        print("✓ Data fetching successful")
        
        # Test indicators
        print("Testing indicators...")
        df = prices['SPY']
        df_with_indicators = add_indicators(df)
        
        if df_with_indicators.empty:
            print("✗ Indicators calculation failed")
            return False
        
        print("✓ Indicators calculation successful")
        
        # Test signals
        print("Testing signals...")
        df_with_signals = compute_signals(df_with_indicators)
        
        if df_with_signals.empty:
            print("✗ Signals calculation failed")
            return False
        
        print("✓ Signals calculation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("TradeScrubber Installation Test")
    print("=" * 40)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test dependencies
    if not test_dependencies():
        all_passed = False
    
    # Test UI imports
    if not test_ui_imports():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! TradeScrubber is ready to use.")
        print("\nTo start the application, run:")
        print("python app.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nTo install missing dependencies, run:")
        print("pip install -r requirements.txt")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
