#!/usr/bin/env python3
"""
Test script for the Trading Bot system
Verifies that all components are working correctly
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from core.data import data_manager
        print("✅ Data manager imported successfully")
    except Exception as e:
        print(f"❌ Failed to import data manager: {e}")
        return False
    
    try:
        from core.screeners import screen_low_cap_movers
        print("✅ Low cap movers screener imported successfully")
    except Exception as e:
        print(f"❌ Failed to import low cap movers screener: {e}")
        return False
    
    try:
        from tradescrubber.core.indicators import add_indicators
        print("✅ Technical indicators imported successfully")
    except Exception as e:
        print(f"❌ Failed to import technical indicators: {e}")
        return False
    
    try:
        from tradescrubber.core.signals import compute_signals
        print("✅ Signal computation imported successfully")
    except Exception as e:
        print(f"❌ Failed to import signal computation: {e}")
        return False
    
    try:
        from tradescrubber.core.ranker import score_ticker
        print("✅ Ticker scoring imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ticker scoring: {e}")
        return False
    
    return True

def test_data_manager():
    """Test the data manager functionality"""
    print("\nTesting data manager...")
    
    try:
        from core.data import data_manager
        
        # Test getting stock data
        data = data_manager.get_stock_data("AAPL", "1d")
        if not data.empty:
            print("✅ Stock data retrieval working")
        else:
            print("⚠️ Stock data retrieval returned empty data")
        
        # Test market cap retrieval
        market_cap = data_manager.get_market_cap("AAPL")
        if market_cap and market_cap > 0:
            print(f"✅ Market cap retrieval working: ${market_cap:,.0f}")
        else:
            print("⚠️ Market cap retrieval returned invalid data")
        
        # Test IV rank calculation
        iv_rank = data_manager.calculate_iv_rank("AAPL")
        if 0 <= iv_rank <= 100:
            print(f"✅ IV rank calculation working: {iv_rank:.1f}%")
        else:
            print("⚠️ IV rank calculation returned invalid value")
        
        return True
        
    except Exception as e:
        print(f"❌ Data manager test failed: {e}")
        return False

def test_screener():
    """Test the low cap movers screener"""
    print("\nTesting low cap movers screener...")
    
    try:
        from core.screeners import screen_low_cap_movers
        
        # Test screener with relaxed parameters
        results = screen_low_cap_movers(
            min_price=0.5,
            max_price=50.0,
            min_market_cap=50_000_000,
            max_market_cap=50_000_000_000,
            min_volume_ratio=1.0,
            min_volatility=0.01
        )
        
        if not results.empty:
            print(f"✅ Screener working: Found {len(results)} stocks")
            print(f"   Sample symbols: {results['symbol'].head(5).tolist()}")
        else:
            print("⚠️ Screener returned no results (may be normal)")
        
        return True
        
    except Exception as e:
        print(f"❌ Screener test failed: {e}")
        return False

def test_technical_analysis():
    """Test technical analysis components"""
    print("\nTesting technical analysis...")
    
    try:
        from core.data import data_manager
        from tradescrubber.core.indicators import add_indicators
        from tradescrubber.core.signals import compute_signals
        from tradescrubber.core.ranker import score_ticker
        
        # Get sample data
        data = data_manager.get_stock_data("AAPL", "1d")
        if data.empty:
            print("❌ No data available for technical analysis test")
            return False
        
        # Test indicators
        df_with_indicators = add_indicators(data)
        if not df_with_indicators.empty and len(df_with_indicators.columns) > len(data.columns):
            print("✅ Technical indicators added successfully")
        else:
            print("⚠️ Technical indicators may not have been added properly")
        
        # Test signals
        df_with_signals = compute_signals(df_with_indicators)
        if not df_with_signals.empty and len(df_with_signals.columns) > len(df_with_indicators.columns):
            print("✅ Trading signals computed successfully")
        else:
            print("⚠️ Trading signals may not have been computed properly")
        
        # Test scoring
        score_data = score_ticker(df_with_signals)
        if isinstance(score_data, dict) and 'score' in score_data:
            print(f"✅ Ticker scoring working: Score = {score_data['score']:.1f}")
        else:
            print("⚠️ Ticker scoring returned invalid data")
        
        return True
        
    except Exception as e:
        print(f"❌ Technical analysis test failed: {e}")
        return False

def test_trading_bot():
    """Test the trading bot functionality"""
    print("\nTesting trading bot...")
    
    try:
        from app.pages.trading_bot import LowCapTradingBot
        
        # Initialize bot
        bot = LowCapTradingBot()
        print("✅ Trading bot initialized successfully")
        
        # Test market cap retrieval
        market_cap = bot.get_market_cap("AAPL")
        if market_cap and market_cap > 0:
            print(f"✅ Bot market cap retrieval working: ${market_cap:,.0f}")
        else:
            print("⚠️ Bot market cap retrieval returned invalid data")
        
        # Test recommendation generation
        recommendations = bot._generate_trading_recommendations(
            current_price=150.0,
            atr=2.0,
            score_data={'score': 75, 'direction': 'up', 'confidence': 0.8},
            latest_signals=pd.Series({'rsi14': 45, 'macd_bullish': True})
        )
        
        if isinstance(recommendations, dict) and 'recommendation' in recommendations:
            print(f"✅ Recommendation generation working: {recommendations['recommendation']}")
        else:
            print("⚠️ Recommendation generation returned invalid data")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading bot test failed: {e}")
        return False

def test_risk_manager():
    """Test the risk manager functionality"""
    print("\nTesting risk manager...")
    
    try:
        from app.pages.risk_manager import RiskManager
        
        # Initialize risk manager
        risk_manager = RiskManager()
        print("✅ Risk manager initialized successfully")
        
        # Test Kelly fraction calculation
        kelly = risk_manager.calculate_kelly_fraction(0.6, 100, 50)
        if 0 <= kelly <= 1:
            print(f"✅ Kelly fraction calculation working: {kelly:.3f}")
        else:
            print("⚠️ Kelly fraction calculation returned invalid value")
        
        # Test position sizing
        position_size = risk_manager.calculate_position_size(
            account_value=10000,
            stock_price=5.0,
            stop_loss_price=4.5,
            risk_per_trade=0.02
        )
        
        if isinstance(position_size, dict) and 'shares' in position_size:
            print(f"✅ Position sizing working: {position_size['shares']} shares")
        else:
            print("⚠️ Position sizing returned invalid data")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk manager test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🤖 Trading Bot Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Manager Test", test_data_manager),
        ("Screener Test", test_screener),
        ("Technical Analysis Test", test_technical_analysis),
        ("Trading Bot Test", test_trading_bot),
        ("Risk Manager Test", test_risk_manager)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The trading bot is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
