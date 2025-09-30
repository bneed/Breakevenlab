"""
Streamlit interface for TradeScrubber
Fallback UI implementation when NiceGUI is not available
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from ..core.data import get_prices, get_options_chain, get_market_status
from ..core.indicators import add_indicators
from ..core.signals import compute_signals
from ..core.ranker import rank_universe, filter_by_strategy
from ..core.options import analyze_options_chain
from ..core.ml import train_ml_models, predict_for_today, get_ml_model_info
from ..core.backtest import backtest_strategy, compare_strategies
from ..core.utils import load_watchlist, load_strategy_presets, get_cache_stats

logger = logging.getLogger(__name__)

def create_streamlit_interface():
    """Create the Streamlit interface"""
    
    # Page config
    st.set_page_config(
        page_title="TradeScrubber",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("TradeScrubber")
    st.subtitle("Smart Stock & Options Screener with Timing")
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        # Data source
        data_source = st.selectbox(
            "Data Source",
            ["yfinance", "polygon", "alpaca"],
            index=0
        )
        
        # Watchlist selection
        watchlist_type = st.selectbox(
            "Watchlist",
            ["default", "custom"],
            index=0
        )
        
        if watchlist_type == "custom":
            custom_tickers = st.text_input(
                "Custom Tickers (comma-separated)",
                placeholder="AAPL,MSFT,GOOGL"
            )
            tickers = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()] if custom_tickers else []
        else:
            tickers = load_watchlist('default')
        
        # Strategy selection
        strategy_presets = load_strategy_presets()
        selected_strategy = st.selectbox(
            "Strategy Preset",
            list(strategy_presets.keys()),
            index=0
        )
        
        # Filters
        st.header("Filters")
        min_score = st.slider("Minimum Score", 0, 100, 50)
        max_results = st.slider("Maximum Results", 1, 100, 20)
        
        # Action buttons
        if st.button("🔄 Refresh Data", type="primary"):
            st.session_state.refresh_data = True
        
        if st.button("🔍 Scan Market"):
            st.session_state.scan_market = True
    
    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💡 Ideas", "📈 Charts", "🔍 Screener", "📞 Options", "🤖 ML", "📊 Backtest"
    ])
    
    # Initialize session state
    if 'current_data' not in st.session_state:
        st.session_state.current_data = {}
    if 'current_rankings' not in st.session_state:
        st.session_state.current_rankings = pd.DataFrame()
    if 'ml_predictions' not in st.session_state:
        st.session_state.ml_predictions = {}
    
    # Ideas Tab
    with tab1:
        st.header("Top Trading Ideas")
        
        # Market status
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                status = get_market_status()
                st.metric("Market Status", status.get('market_state', 'Unknown'))
            except:
                st.metric("Market Status", "Unknown")
        
        with col2:
            try:
                status = get_market_status()
                st.metric("Market Open", "Yes" if status.get('is_market_open', False) else "No")
            except:
                st.metric("Market Open", "Unknown")
        
        with col3:
            try:
                cache_stats = get_cache_stats()
                st.metric("Cache Files", cache_stats.get('total_files', 0))
            except:
                st.metric("Cache Files", 0)
        
        # Refresh data if requested
        if st.session_state.get('refresh_data', False):
            with st.spinner("Fetching market data..."):
                try:
                    st.session_state.current_data = get_prices(tickers, '1d', 365)
                    
                    # Add indicators and signals
                    for ticker, df in st.session_state.current_data.items():
                        if not df.empty:
                            df_with_indicators = add_indicators(df)
                            df_with_signals = compute_signals(df_with_indicators)
                            st.session_state.current_data[ticker] = df_with_signals
                    
                    # Get ML predictions
                    st.session_state.ml_predictions = predict_for_today(st.session_state.current_data)
                    
                    st.success("Data refreshed successfully!")
                    st.session_state.refresh_data = False
                except Exception as e:
                    st.error(f"Error refreshing data: {e}")
                    st.session_state.refresh_data = False
        
        # Scan market if requested
        if st.session_state.get('scan_market', False):
            if not st.session_state.current_data:
                st.warning("Please refresh data first")
            else:
                with st.spinner("Scanning market..."):
                    try:
                        st.session_state.current_rankings = rank_universe(
                            st.session_state.current_data,
                            st.session_state.ml_predictions,
                            selected_strategy
                        )
                        
                        # Apply filters
                        filtered_rankings = filter_by_strategy(
                            st.session_state.current_rankings,
                            selected_strategy,
                            min_score,
                            max_results
                        )
                        
                        st.session_state.current_rankings = filtered_rankings
                        st.success(f"Found {len(filtered_rankings)} trading ideas!")
                        st.session_state.scan_market = False
                    except Exception as e:
                        st.error(f"Error scanning market: {e}")
                        st.session_state.scan_market = False
        
        # Display results
        if not st.session_state.current_rankings.empty:
            st.dataframe(
                st.session_state.current_rankings[
                    ['ticker', 'direction', 'score', 'confidence', 'price', 'entry_zone', 'stop_loss', 'target']
                ],
                use_container_width=True
            )
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📈 View Chart"):
                    st.info("Chart functionality coming soon")
            with col2:
                if st.button("📞 Analyze Options"):
                    st.info("Options analysis coming soon")
            with col3:
                if st.button("🔔 Add Alert"):
                    st.info("Alert functionality coming soon")
        else:
            st.info("Click 'Scan Market' to find trading ideas")
    
    # Charts Tab
    with tab2:
        st.header("Price Charts")
        
        # Ticker selection
        if st.session_state.current_data:
            selected_ticker = st.selectbox(
                "Select Ticker",
                list(st.session_state.current_data.keys()),
                index=0
            )
            
            if selected_ticker in st.session_state.current_data:
                df = st.session_state.current_data[selected_ticker]
                
                if not df.empty:
                    # Display basic info
                    latest = df.iloc[-1]
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Price", f"${latest['close']:.2f}")
                    with col2:
                        change = latest['close'] - df.iloc[-2]['close'] if len(df) > 1 else 0
                        st.metric("Change", f"${change:.2f}")
                    with col3:
                        change_pct = (change / df.iloc[-2]['close']) * 100 if len(df) > 1 else 0
                        st.metric("Change %", f"{change_pct:.2f}%")
                    with col4:
                        st.metric("Volume", f"{latest['volume']:,}")
                    
                    # Display recent data
                    st.subheader("Recent Data")
                    recent_data = df.tail(10)[['close', 'volume', 'rsi14', 'macd', 'sma20', 'sma50']]
                    st.dataframe(recent_data, use_container_width=True)
                    
                    # Simple indicators
                    st.subheader("Technical Indicators")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("RSI (14)", f"{latest.get('rsi14', 0):.1f}")
                    with col2:
                        st.metric("MACD", f"{latest.get('macd', 0):.3f}")
                    with col3:
                        st.metric("SMA 50", f"${latest.get('sma50', 0):.2f}")
                else:
                    st.warning("No data available for selected ticker")
            else:
                st.warning("Selected ticker not available")
        else:
            st.info("Please refresh data first")
    
    # Screener Tab
    with tab3:
        st.header("Market Screener")
        
        if st.session_state.current_data:
            # Create screener data
            screener_data = []
            for ticker, df in st.session_state.current_data.items():
                if not df.empty:
                    latest = df.iloc[-1]
                    screener_data.append({
                        'ticker': ticker,
                        'price': latest['close'],
                        'change': ((latest['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100) if len(df) > 1 else 0,
                        'volume': latest['volume'],
                        'rsi': latest.get('rsi14', 0),
                        'macd': latest.get('macd', 0),
                        'sma50': latest.get('sma50', 0),
                        'sma200': latest.get('sma200', 0),
                        'signals': ', '.join([k for k, v in latest.items() if isinstance(v, bool) and v])
                    })
            
            if screener_data:
                screener_df = pd.DataFrame(screener_data)
                
                # Filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    price_min = st.number_input("Min Price", value=0.0, min_value=0.0)
                    price_max = st.number_input("Max Price", value=1000.0, min_value=0.0)
                
                with col2:
                    rsi_min = st.number_input("Min RSI", value=0, min_value=0, max_value=100)
                    rsi_max = st.number_input("Max RSI", value=100, min_value=0, max_value=100)
                
                with col3:
                    min_volume = st.number_input("Min Volume", value=1000000, min_value=0)
                
                # Apply filters
                filtered_df = screener_df[
                    (screener_df['price'] >= price_min) &
                    (screener_df['price'] <= price_max) &
                    (screener_df['rsi'] >= rsi_min) &
                    (screener_df['rsi'] <= rsi_max) &
                    (screener_df['volume'] >= min_volume)
                ]
                
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.info("No data available for screener")
        else:
            st.info("Please refresh data first")
    
    # Options Tab
    with tab4:
        st.header("Options Analysis")
        
        if st.session_state.current_data:
            selected_ticker = st.selectbox(
                "Select Ticker for Options",
                list(st.session_state.current_data.keys()),
                index=0,
                key="options_ticker"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                direction = st.selectbox("Direction", ["long", "short"], index=0)
            with col2:
                target_delta = st.slider("Target Delta", 0.1, 0.5, 0.3, 0.1)
            
            if st.button("Analyze Options"):
                try:
                    current_price = st.session_state.current_data[selected_ticker].iloc[-1]['close']
                    options_data = get_options_chain(selected_ticker, 30)
                    
                    if options_data and 'calls' in options_data and not options_data['calls'].empty:
                        analysis = analyze_options_chain(
                            selected_ticker,
                            current_price,
                            options_data,
                            direction,
                            target_delta
                        )
                        
                        # Display options recommendations
                        st.subheader("Options Recommendations")
                        
                        for option_type in ['calls', 'puts']:
                            if option_type in analysis and analysis[option_type]:
                                st.write(f"**{option_type.title()}**")
                                options_df = pd.DataFrame(analysis[option_type])
                                st.dataframe(options_df, use_container_width=True)
                    else:
                        st.warning("No options data available")
                except Exception as e:
                    st.error(f"Error analyzing options: {e}")
        else:
            st.info("Please refresh data first")
    
    # ML Tab
    with tab5:
        st.header("Machine Learning")
        
        # Model info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Information")
            try:
                ml_info = get_ml_model_info()
                st.write(f"**Trained:** {ml_info.get('is_trained', False)}")
                st.write(f"**Features:** {ml_info.get('feature_count', 0)}")
                st.write(f"**Last Training:** {ml_info.get('last_training_date', 'Never')}")
            except Exception as e:
                st.error(f"Error loading ML info: {e}")
        
        with col2:
            st.subheader("Training Controls")
            lookback_days = st.number_input("Lookback Days", value=365, min_value=30, max_value=1095)
            horizon_days = st.number_input("Prediction Horizon", value=3, min_value=1, max_value=30)
            
            if st.button("Train Models"):
                if not st.session_state.current_data:
                    st.warning("Please refresh data first")
                else:
                    with st.spinner("Training ML models..."):
                        try:
                            result = train_ml_models(
                                st.session_state.current_data,
                                lookback_days=lookback_days,
                                horizon_days=horizon_days
                            )
                            
                            if 'error' in result:
                                st.error(f"Training failed: {result['error']}")
                            else:
                                st.success("Models trained successfully!")
                                st.json(result)
                        except Exception as e:
                            st.error(f"Error training models: {e}")
        
        # ML Predictions
        st.subheader("ML Predictions")
        if st.session_state.ml_predictions:
            predictions_data = []
            for ticker, pred in st.session_state.ml_predictions.items():
                predictions_data.append({
                    'ticker': ticker,
                    'expected_move': f"{pred.get('expected_move', 0):.2%}",
                    'up_prob': f"{pred.get('up_prob', 0.5):.2%}",
                    'confidence': f"{pred.get('confidence', 0):.2%}",
                    'direction': pred.get('direction', 'neutral')
                })
            
            predictions_df = pd.DataFrame(predictions_data)
            st.dataframe(predictions_df, use_container_width=True)
        else:
            st.info("No ML predictions available")
    
    # Backtest Tab
    with tab6:
        st.header("Backtesting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Backtest Parameters")
            strategy = st.selectbox("Strategy", ["default", "reversal", "breakout", "trend"], index=0)
            
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
            end_date = st.date_input("End Date", value=datetime.now())
            
            position_size = st.slider("Position Size %", 1, 100, 10)
            max_positions = st.slider("Max Positions", 1, 20, 5)
        
        with col2:
            st.subheader("Run Backtest")
            if st.button("Run Backtest", type="primary"):
                if not st.session_state.current_data:
                    st.warning("Please refresh data first")
                else:
                    with st.spinner("Running backtest..."):
                        try:
                            result = backtest_strategy(
                                st.session_state.current_data,
                                strategy,
                                start_date=start_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d'),
                                position_size=position_size / 100,
                                max_positions=max_positions
                            )
                            
                            if 'error' in result:
                                st.error(f"Backtest failed: {result['error']}")
                            else:
                                st.success("Backtest completed!")
                                
                                # Display results
                                metrics = result.get('metrics', {})
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
                                with col2:
                                    st.metric("CAGR", f"{metrics.get('cagr', 0):.2%}")
                                with col3:
                                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                                with col4:
                                    st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
                                
                                # Detailed metrics
                                st.subheader("Detailed Results")
                                metrics_df = pd.DataFrame([
                                    {"Metric": "Win Rate", "Value": f"{metrics.get('win_rate', 0):.2%}"},
                                    {"Metric": "Total Trades", "Value": f"{metrics.get('total_trades', 0)}"},
                                    {"Metric": "Avg Win", "Value": f"{metrics.get('avg_win', 0):.2%}"},
                                    {"Metric": "Avg Loss", "Value": f"{metrics.get('avg_loss', 0):.2%}"},
                                    {"Metric": "Profit Factor", "Value": f"{metrics.get('profit_factor', 0):.2f}"}
                                ])
                                st.dataframe(metrics_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error running backtest: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("**TradeScrubber** - Smart Stock & Options Screener with Timing")
    st.markdown("Built with Python, Streamlit, and machine learning")
