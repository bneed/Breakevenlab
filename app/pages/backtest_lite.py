"""
Backtest Lite - Tier 1 Feature
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.backtest import run_backtest
from core.auth import is_pro_feature, show_pro_upgrade_prompt

def show_backtest_lite():
    """Display the backtest lite page"""
    
    st.title("📉 Backtest Lite")
    st.markdown("Backtest trading strategies with daily bars")
    
    # Check if user has pro access
    if not is_pro_feature():
        show_pro_upgrade_prompt("Backtest Lite")
        return
    
    # Sidebar for strategy parameters
    with st.sidebar:
        st.header("Strategy Parameters")
        
        # Symbol selection
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter the stock symbol to backtest")
        
        # Strategy selection
        strategy = st.selectbox(
            "Strategy",
            ["RSI Mean Reversion", "MACD Crossover", "Covered Call", "Custom"]
        )
        
        # Date range
        st.subheader("Backtest Period")
        start_date = st.date_input(
            "Start Date", 
            value=datetime.now() - timedelta(days=365),
            help="Start date for the backtest"
        )
        
        end_date = st.date_input(
            "End Date", 
            value=datetime.now(),
            help="End date for the backtest"
        )
        
        # Strategy-specific parameters
        if strategy == "RSI Mean Reversion":
            st.subheader("RSI Parameters")
            rsi_oversold = st.slider("RSI Oversold Level", 10, 40, 30)
            rsi_overbought = st.slider("RSI Overbought Level", 60, 90, 70)
            position_size = st.slider("Position Size", 0.1, 1.0, 1.0, 0.1)
            
        elif strategy == "MACD Crossover":
            st.subheader("MACD Parameters")
            position_size = st.slider("Position Size", 0.1, 1.0, 1.0, 0.1)
            
        elif strategy == "Covered Call":
            st.subheader("Covered Call Parameters")
            strike_delta = st.slider("Strike Delta", 0.1, 0.5, 0.3, 0.1)
            roll_dte = st.slider("Roll Days to Expiration", 1, 14, 7)
            position_size = st.slider("Position Size", 0.1, 1.0, 1.0, 0.1)
        
        # Risk management
        st.subheader("Risk Management")
        initial_capital = st.number_input(
            "Initial Capital ($)", 
            min_value=1000, 
            max_value=1000000, 
            value=10000
        )
        
        # Run backtest button
        if st.button("🚀 Run Backtest", type="primary"):
            st.rerun()
    
    # Main content area
    if st.button("Run Backtest", type="primary"):
        
        # Validate inputs
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return
        
        if initial_capital <= 0:
            st.error("Initial capital must be positive")
            return
        
        # Prepare strategy parameters
        strategy_params = {}
        
        if strategy == "RSI Mean Reversion":
            strategy_params = {
                'rsi_oversold': rsi_oversold,
                'rsi_overbought': rsi_overbought,
                'position_size': position_size
            }
            strategy_name = 'rsi'
            
        elif strategy == "MACD Crossover":
            strategy_params = {
                'position_size': position_size
            }
            strategy_name = 'macd'
            
        elif strategy == "Covered Call":
            strategy_params = {
                'strike_delta': strike_delta,
                'roll_dte': roll_dte,
                'position_size': position_size
            }
            strategy_name = 'covered_call'
        
        else:  # Custom strategy
            st.info("Custom strategy feature coming soon!")
            return
        
        # Run the backtest
        with st.spinner("Running backtest..."):
            try:
                results, metrics = run_backtest(
                    symbol=symbol,
                    strategy=strategy_name,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    **strategy_params
                )
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
                return
        
        if results.empty:
            st.error("No data available for the selected period")
            return
        
        # Display results
        st.subheader("Backtest Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"${metrics.get('total_return', 0):,.2f}")
        
        with col2:
            st.metric("CAGR", f"{metrics.get('cagr', 0):.1f}%")
        
        with col3:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1f}%")
        
        with col4:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        
        with col2:
            st.metric("Total Trades", f"{metrics.get('total_trades', 0)}")
        
        with col3:
            st.metric("Winning Trades", f"{metrics.get('winning_trades', 0)}")
        
        with col4:
            st.metric("Avg Trade", f"${metrics.get('avg_trade', 0):.2f}")
        
        # Equity curve chart
        st.subheader("Equity Curve")
        
        if 'cumulative_pnl' in results.columns:
            fig = go.Figure()
            
            # Add equity curve
            fig.add_trace(go.Scatter(
                x=results.index,
                y=results['cumulative_pnl'],
                mode='lines',
                name='Equity Curve',
                line=dict(color='blue', width=2)
            ))
            
            # Add buy and sell signals
            if 'trade_pnl' in results.columns:
                buy_signals = results[results['trade_pnl'] > 0]
                sell_signals = results[results['trade_pnl'] < 0]
                
                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['cumulative_pnl'],
                        mode='markers',
                        name='Winning Trades',
                        marker=dict(color='green', size=8, symbol='triangle-up')
                    ))
                
                if not sell_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['cumulative_pnl'],
                        mode='markers',
                        name='Losing Trades',
                        marker=dict(color='red', size=8, symbol='triangle-down')
                    ))
            
            fig.update_layout(
                title=f"{symbol} {strategy} Backtest Results",
                xaxis_title="Date",
                yaxis_title="Cumulative P/L ($)",
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown chart
        st.subheader("Drawdown Analysis")
        
        if 'cumulative_pnl' in results.columns:
            # Calculate drawdown
            peak = results['cumulative_pnl'].expanding().max()
            drawdown = (results['cumulative_pnl'] - peak) / peak * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=2),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title="Drawdown Over Time",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade analysis
        st.subheader("Trade Analysis")
        
        if 'trade_pnl' in results.columns:
            trades = results[results['trade_pnl'] != 0]['trade_pnl']
            
            if not trades.empty:
                # Trade distribution
                fig = px.histogram(
                    trades, 
                    nbins=20,
                    title="Trade P/L Distribution",
                    labels={'value': 'Trade P/L ($)', 'count': 'Number of Trades'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade statistics
                trade_stats = {
                    'Metric': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate', 'Avg Win', 'Avg Loss', 'Best Trade', 'Worst Trade'],
                    'Value': [
                        len(trades),
                        len(trades[trades > 0]),
                        len(trades[trades < 0]),
                        f"{len(trades[trades > 0]) / len(trades) * 100:.1f}%",
                        f"${trades[trades > 0].mean():.2f}" if len(trades[trades > 0]) > 0 else "$0.00",
                        f"${trades[trades < 0].mean():.2f}" if len(trades[trades < 0]) > 0 else "$0.00",
                        f"${trades.max():.2f}",
                        f"${trades.min():.2f}"
                    ]
                }
                
                trade_stats_df = pd.DataFrame(trade_stats)
                st.dataframe(trade_stats_df, use_container_width=True)
        
        # Strategy performance comparison
        st.subheader("Strategy Performance")
        
        # Create performance comparison table
        performance_data = {
            'Metric': ['Total Return', 'CAGR', 'Max Drawdown', 'Sharpe Ratio', 'Win Rate', 'Total Trades'],
            'Value': [
                f"${metrics.get('total_return', 0):,.2f}",
                f"{metrics.get('cagr', 0):.1f}%",
                f"{metrics.get('max_drawdown', 0):.1f}%",
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                f"{metrics.get('win_rate', 0):.1f}%",
                f"{metrics.get('total_trades', 0)}"
            ]
        }
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # Download results
        st.subheader("Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download backtest data
            csv = results.to_csv()
            st.download_button(
                label="Download Backtest Data (CSV)",
                data=csv,
                file_name=f"{symbol}_{strategy_name}_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download performance metrics
            metrics_csv = performance_df.to_csv(index=False)
            st.download_button(
                label="Download Performance Metrics (CSV)",
                data=metrics_csv,
                file_name=f"{symbol}_{strategy_name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Strategy explanations
    st.subheader("Strategy Explanations")
    
    if strategy == "RSI Mean Reversion":
        st.info("""
        **RSI Mean Reversion Strategy:**
        - Buys when RSI falls below the oversold level (typically 30)
        - Sells when RSI rises above the overbought level (typically 70)
        - Based on the assumption that extreme RSI values tend to revert to the mean
        - Works best in ranging markets
        """)
    
    elif strategy == "MACD Crossover":
        st.info("""
        **MACD Crossover Strategy:**
        - Buys when MACD line crosses above the signal line
        - Sells when MACD line crosses below the signal line
        - Uses momentum to identify trend changes
        - Works best in trending markets
        """)
    
    elif strategy == "Covered Call":
        st.info("""
        **Covered Call Strategy:**
        - Buys stock and sells call options against it
        - Generates income from option premiums
        - Limits upside potential but provides downside protection
        - Works best in sideways or slightly bullish markets
        """)
    
    # Backtesting tips
    st.subheader("Backtesting Tips")
    
    tips = [
        "Use at least 1 year of data for meaningful results",
        "Consider transaction costs and slippage in real trading",
        "Backtest results don't guarantee future performance",
        "Test strategies across different market conditions",
        "Use proper risk management and position sizing",
        "Consider the impact of market regime changes"
    ]
    
    for i, tip in enumerate(tips, 1):
        st.write(f"{i}. {tip}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>⚠️ <strong>Disclaimer:</strong> This backtesting tool is for educational purposes only. Not investment advice.</p>
        <p>Past performance does not guarantee future results. Always do your own research.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_backtest_lite()
