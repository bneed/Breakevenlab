"""
Real-time Market Scanner for Trading Bot
Continuously scans the market for trading opportunities and sends alerts
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import queue
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px

from core.screeners import screen_low_cap_movers
from core.data import data_manager
from tradescrubber.core.indicators import add_indicators
from tradescrubber.core.signals import compute_signals
from tradescrubber.core.ranker import score_ticker

class MarketScanner:
    """Real-time market scanner for trading opportunities"""
    
    def __init__(self):
        self.scan_interval = 300  # 5 minutes
        self.is_scanning = False
        self.scan_results = []
        self.alerts = []
        self.last_scan_time = None
        
        # Alert thresholds
        self.min_score_threshold = 70
        self.min_volume_ratio = 2.0
        self.min_volatility = 0.03
        
        # Price and market cap filters
        self.min_price = 1.0
        self.max_price = 10.0
        self.min_market_cap = 100_000_000
        self.max_market_cap = 10_000_000_000
    
    def start_scanning(self):
        """Start the background scanning process"""
        self.is_scanning = True
        st.success("Market scanner started! Scanning every 5 minutes.")
    
    def stop_scanning(self):
        """Stop the background scanning process"""
        self.is_scanning = False
        st.info("Market scanner stopped.")
    
    def scan_market(self) -> List[Dict]:
        """Perform a single market scan"""
        try:
            # Run the low cap movers screener
            results = screen_low_cap_movers(
                min_price=self.min_price,
                max_price=self.max_price,
                min_market_cap=self.min_market_cap,
                max_market_cap=self.max_market_cap,
                min_volume_ratio=self.min_volume_ratio,
                min_volatility=self.min_volatility
            )
            
            if results.empty:
                return []
            
            # Add technical analysis and scoring
            enhanced_results = []
            for _, row in results.iterrows():
                try:
                    # Get detailed data for technical analysis
                    data = data_manager.get_stock_data(row['symbol'], "1d")
                    if data.empty:
                        continue
                    
                    # Add indicators and signals
                    df_with_indicators = add_indicators(data)
                    df_with_signals = compute_signals(df_with_indicators)
                    
                    # Score the stock
                    score_data = score_ticker(df_with_signals)
                    
                    # Create enhanced result
                    enhanced_result = {
                        'symbol': row['symbol'],
                        'price': row['price'],
                        'market_cap_billions': row['market_cap_billions'],
                        'volume_ratio': row['volume_ratio'],
                        'volatility': row['volatility'],
                        'price_change_1d': row['price_change_1d'],
                        'price_change_5d': row['price_change_5d'],
                        'score': score_data.get('score', 0),
                        'direction': score_data.get('direction', 'neutral'),
                        'confidence': score_data.get('confidence', 0),
                        'reasons': ', '.join(score_data.get('reasons', [])),
                        'rsi': df_with_signals['rsi14'].iloc[-1] if 'rsi14' in df_with_signals.columns else 50,
                        'macd_signal': df_with_signals['macd_bullish'].iloc[-1] if 'macd_bullish' in df_with_signals.columns else False,
                        'volume_spike': df_with_signals['volume_spike'].iloc[-1] if 'volume_spike' in df_with_signals.columns else False,
                        'scan_time': datetime.now()
                    }
                    
                    enhanced_results.append(enhanced_result)
                    
                except Exception as e:
                    st.error(f"Error processing {row['symbol']}: {str(e)}")
                    continue
            
            # Sort by score
            enhanced_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Check for new alerts
            self._check_for_alerts(enhanced_results)
            
            return enhanced_results
            
        except Exception as e:
            st.error(f"Error scanning market: {str(e)}")
            return []
    
    def _check_for_alerts(self, results: List[Dict]):
        """Check for new trading alerts"""
        for result in results:
            if result['score'] >= self.min_score_threshold:
                # Check if this is a new alert
                alert_key = f"{result['symbol']}_{result['scan_time'].strftime('%Y%m%d_%H%M')}"
                
                if not any(alert['key'] == alert_key for alert in self.alerts):
                    alert = {
                        'key': alert_key,
                        'symbol': result['symbol'],
                        'price': result['price'],
                        'score': result['score'],
                        'direction': result['direction'],
                        'reasons': result['reasons'],
                        'timestamp': result['scan_time'],
                        'dismissed': False
                    }
                    self.alerts.append(alert)
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts within specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert['timestamp'] > cutoff_time and not alert['dismissed']]
    
    def dismiss_alert(self, alert_key: str):
        """Dismiss a specific alert"""
        for alert in self.alerts:
            if alert['key'] == alert_key:
                alert['dismissed'] = True
                break
    
    def clear_all_alerts(self):
        """Clear all alerts"""
        self.alerts = []

def show_market_scanner():
    """Display the market scanner interface"""
    
    st.title("📡 Market Scanner")
    st.markdown("**Real-time scanning for low cap trading opportunities**")
    
    # Initialize scanner in session state
    if 'market_scanner' not in st.session_state:
        st.session_state.market_scanner = MarketScanner()
    
    scanner = st.session_state.market_scanner
    
    # Sidebar for scanner settings
    with st.sidebar:
        st.header("⚙️ Scanner Settings")
        
        # Scan interval
        scan_interval = st.selectbox(
            "Scan Interval",
            [300, 600, 900, 1800],  # 5min, 10min, 15min, 30min
            index=0,
            format_func=lambda x: f"{x//60} minutes"
        )
        scanner.scan_interval = scan_interval
        
        # Alert thresholds
        st.subheader("Alert Thresholds")
        scanner.min_score_threshold = st.slider(
            "Min Score Threshold",
            min_value=50,
            max_value=100,
            value=70,
            step=5
        )
        
        scanner.min_volume_ratio = st.slider(
            "Min Volume Ratio",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1
        )
        
        scanner.min_volatility = st.slider(
            "Min Volatility",
            min_value=0.01,
            max_value=0.10,
            value=0.03,
            step=0.01
        )
        
        # Price and market cap filters
        st.subheader("Price & Market Cap Filters")
        scanner.min_price = st.number_input(
            "Min Price ($)",
            min_value=0.01,
            value=1.0,
            step=0.01
        )
        
        scanner.max_price = st.number_input(
            "Max Price ($)",
            min_value=0.01,
            value=10.0,
            step=0.01
        )
        
        scanner.min_market_cap = st.number_input(
            "Min Market Cap (B)",
            min_value=0.1,
            value=0.1,
            step=0.1
        ) * 1_000_000_000
        
        scanner.max_market_cap = st.number_input(
            "Max Market Cap (B)",
            min_value=0.1,
            value=10.0,
            step=0.1
        ) * 1_000_000_000
        
        # Scanner controls
        st.subheader("Scanner Controls")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start Scanner", type="primary"):
                scanner.start_scanning()
        
        with col2:
            if st.button("⏹️ Stop Scanner"):
                scanner.stop_scanning()
        
        # Manual scan
        if st.button("🔍 Manual Scan", use_container_width=True):
            with st.spinner("Scanning market..."):
                results = scanner.scan_market()
                scanner.scan_results = results
                scanner.last_scan_time = datetime.now()
                st.success(f"Scan completed! Found {len(results)} opportunities.")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Results", "🚨 Alerts", "📈 Charts", "📋 History"])
    
    with tab1:
        st.subheader("📊 Live Market Scan Results")
        
        # Scanner status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if scanner.is_scanning:
                st.success("🟢 Scanner Active")
            else:
                st.info("🔴 Scanner Inactive")
        
        with col2:
            if scanner.last_scan_time:
                st.metric("Last Scan", scanner.last_scan_time.strftime("%H:%M:%S"))
            else:
                st.metric("Last Scan", "Never")
        
        with col3:
            st.metric("Total Alerts", len(scanner.alerts))
        
        # Display results
        if scanner.scan_results:
            results_df = pd.DataFrame(scanner.scan_results)
            
            # Filter by score
            filtered_results = results_df[results_df['score'] >= scanner.min_score_threshold]
            
            if not filtered_results.empty:
                st.markdown(f"**Found {len(filtered_results)} high-scoring opportunities**")
                
                # Display top results
                for idx, (_, result) in enumerate(filtered_results.head(10).iterrows(), 1):
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                        
                        with col1:
                            st.markdown(f"**#{idx} {result['symbol']}**")
                            st.markdown(f"${result['price']:.2f}")
                            st.markdown(f"Score: {result['score']:.0f}")
                        
                        with col2:
                            st.markdown(f"**{result['direction'].upper()}**")
                            st.markdown(f"Vol: {result['volume_ratio']:.1f}x")
                            st.markdown(f"Volatility: {result['volatility']:.1%}")
                        
                        with col3:
                            st.markdown(f"1D: {result['price_change_1d']:+.1f}%")
                            st.markdown(f"5D: {result['price_change_5d']:+.1f}%")
                            st.markdown(f"MCap: ${result['market_cap_billions']:.1f}B")
                        
                        with col4:
                            st.markdown(f"RSI: {result['rsi']:.0f}")
                            st.markdown(f"MACD: {'✅' if result['macd_signal'] else '❌'}")
                            st.markdown(f"Vol Spike: {'✅' if result['volume_spike'] else '❌'}")
                        
                        # Reasons
                        if result['reasons']:
                            st.markdown(f"*{result['reasons']}*")
                        
                        st.markdown("---")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg Score", f"{filtered_results['score'].mean():.0f}")
                
                with col2:
                    st.metric("Avg Volume", f"{filtered_results['volume_ratio'].mean():.1f}x")
                
                with col3:
                    st.metric("Avg Volatility", f"{filtered_results['volatility'].mean():.1%}")
                
                with col4:
                    up_signals = len(filtered_results[filtered_results['direction'] == 'up'])
                    st.metric("Up Signals", up_signals)
            
            else:
                st.warning("No high-scoring opportunities found in current scan.")
        
        else:
            st.info("No scan results available. Click 'Manual Scan' to get started.")
    
    with tab2:
        st.subheader("🚨 Trading Alerts")
        
        # Alert controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔔 Refresh Alerts"):
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All Alerts"):
                scanner.clear_all_alerts()
                st.rerun()
        
        # Display alerts
        recent_alerts = scanner.get_recent_alerts(hours=24)
        
        if recent_alerts:
            st.markdown(f"**{len(recent_alerts)} recent alerts**")
            
            for alert in recent_alerts:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**{alert['symbol']}** - ${alert['price']:.2f}")
                        st.markdown(f"Score: {alert['score']:.0f} | {alert['direction'].upper()}")
                    
                    with col2:
                        st.markdown(f"**{alert['timestamp'].strftime('%H:%M:%S')}**")
                        st.markdown(f"*{alert['reasons']}*")
                    
                    with col3:
                        if alert['score'] >= 80:
                            st.error("🚨 HIGH PRIORITY")
                        elif alert['score'] >= 70:
                            st.warning("⚠️ MEDIUM PRIORITY")
                        else:
                            st.info("ℹ️ LOW PRIORITY")
                    
                    with col4:
                        if st.button("❌", key=f"dismiss_{alert['key']}"):
                            scanner.dismiss_alert(alert['key'])
                            st.rerun()
                    
                    st.markdown("---")
        
        else:
            st.info("No recent alerts. The scanner will notify you when opportunities arise.")
    
    with tab3:
        st.subheader("📈 Market Analysis Charts")
        
        if scanner.scan_results:
            results_df = pd.DataFrame(scanner.scan_results)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution
                fig = px.histogram(
                    results_df,
                    x='score',
                    nbins=20,
                    title="Score Distribution",
                    labels={'score': 'Trading Score', 'count': 'Number of Stocks'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Volume vs Volatility scatter
                fig = px.scatter(
                    results_df,
                    x='volume_ratio',
                    y='volatility',
                    size='score',
                    color='direction',
                    hover_data=['symbol', 'price'],
                    title="Volume vs Volatility",
                    labels={'volume_ratio': 'Volume Ratio', 'volatility': 'Volatility'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Price change analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # 1-day price changes
                fig = px.bar(
                    results_df.head(20),
                    x='symbol',
                    y='price_change_1d',
                    title="Top 20 - 1 Day Price Changes",
                    labels={'price_change_1d': '1D Change (%)', 'symbol': 'Symbol'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 5-day price changes
                fig = px.bar(
                    results_df.head(20),
                    x='symbol',
                    y='price_change_5d',
                    title="Top 20 - 5 Day Price Changes",
                    labels={'price_change_5d': '5D Change (%)', 'symbol': 'Symbol'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No scan data available for charts. Run a scan first.")
    
    with tab4:
        st.subheader("📋 Scan History")
        
        if scanner.scan_results:
            # Create history table
            history_data = []
            for result in scanner.scan_results:
                history_data.append({
                    'Symbol': result['symbol'],
                    'Price': f"${result['price']:.2f}",
                    'Score': result['score'],
                    'Direction': result['direction'].upper(),
                    'Volume Ratio': f"{result['volume_ratio']:.1f}x",
                    'Volatility': f"{result['volatility']:.1%}",
                    '1D Change': f"{result['price_change_1d']:+.1f}%",
                    '5D Change': f"{result['price_change_5d']:+.1f}%",
                    'Scan Time': result['scan_time'].strftime('%H:%M:%S')
                })
            
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
            
            # Download history
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Scan History (CSV)",
                data=csv,
                file_name=f"market_scan_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        else:
            st.info("No scan history available. Run a scan to see results here.")

if __name__ == "__main__":
    show_market_scanner()
