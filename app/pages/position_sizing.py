"""
Position Sizing & Risk Management - Tier 0 Free Tool
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.data import get_stock_price

def show_position_sizing():
    """Display the position sizing calculator page"""
    
    st.title("⚖️ Position Sizing & Risk Management")
    st.markdown("Calculate optimal position sizes using various risk management methods")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Account & Risk Parameters")
        
        # Account information
        account_size = st.number_input(
            "Account Size ($)", 
            min_value=1000, 
            max_value=10000000, 
            value=10000,
            help="Total account value"
        )
        
        risk_per_trade = st.slider(
            "Risk per Trade (%)", 
            min_value=0.1, 
            max_value=10.0, 
            value=2.0, 
            step=0.1,
            help="Percentage of account to risk per trade"
        )
        
        # Trade parameters
        st.subheader("Trade Parameters")
        
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter the stock symbol")
        
        # Get current stock price
        current_price = get_stock_price(symbol)
        if current_price > 0:
            st.success(f"Current {symbol} Price: ${current_price:.2f}")
        else:
            st.error(f"Could not fetch price for {symbol}")
            current_price = 100  # Default fallback
        
        entry_price = st.number_input(
            "Entry Price ($)", 
            min_value=0.01, 
            value=current_price,
            help="Your planned entry price"
        )
        
        stop_loss = st.number_input(
            "Stop Loss ($)", 
            min_value=0.01, 
            value=current_price * 0.95,
            help="Your stop loss price"
        )
        
        # Kelly Criterion parameters
        st.subheader("Kelly Criterion Parameters")
        
        win_rate = st.slider(
            "Win Rate (%)", 
            min_value=10, 
            max_value=90, 
            value=60, 
            step=1,
            help="Your historical win rate"
        )
        
        avg_win = st.number_input(
            "Average Win ($)", 
            min_value=0.01, 
            value=200.0,
            help="Average profit per winning trade"
        )
        
        avg_loss = st.number_input(
            "Average Loss ($)", 
            min_value=0.01, 
            value=100.0,
            help="Average loss per losing trade"
        )
    
    # Main content area
    if st.button("Calculate Position Size", type="primary"):
        
        # Calculate risk per trade in dollars
        risk_amount = account_size * (risk_per_trade / 100)
        
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)
        stop_percentage = (stop_distance / entry_price) * 100
        
        # Method 1: Fixed Fractional Position Sizing
        if stop_distance > 0:
            fixed_fractional_shares = int(risk_amount / stop_distance)
            fixed_fractional_value = fixed_fractional_shares * entry_price
        else:
            fixed_fractional_shares = 0
            fixed_fractional_value = 0
        
        # Method 2: Fixed Dollar Amount
        fixed_dollar_shares = int(risk_amount / entry_price)
        fixed_dollar_value = fixed_dollar_shares * entry_price
        
        # Method 3: Kelly Criterion
        if avg_loss > 0:
            kelly_fraction = (win_rate / 100) - ((1 - win_rate / 100) / (avg_win / avg_loss))
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            kelly_shares = int((account_size * kelly_fraction) / entry_price)
            kelly_value = kelly_shares * entry_price
        else:
            kelly_fraction = 0
            kelly_shares = 0
            kelly_value = 0
        
        # Method 4: Volatility-based sizing (simplified)
        # Assume 2% daily volatility
        daily_volatility = 0.02
        volatility_shares = int((account_size * 0.01) / (entry_price * daily_volatility))
        volatility_value = volatility_shares * entry_price
        
        # Display results
        st.subheader("Position Sizing Results")
        
        # Create results DataFrame
        results_data = [
            {
                "Method": "Fixed Fractional",
                "Shares": fixed_fractional_shares,
                "Position Value": f"${fixed_fractional_value:,.2f}",
                "Risk Amount": f"${risk_amount:,.2f}",
                "Risk %": f"{risk_per_trade:.1f}%",
                "Description": "Risk-based sizing using stop loss"
            },
            {
                "Method": "Fixed Dollar",
                "Shares": fixed_dollar_shares,
                "Position Value": f"${fixed_dollar_value:,.2f}",
                "Risk Amount": f"${risk_amount:,.2f}",
                "Risk %": f"{risk_per_trade:.1f}%",
                "Description": "Fixed dollar amount per trade"
            },
            {
                "Method": "Kelly Criterion",
                "Shares": kelly_shares,
                "Position Value": f"${kelly_value:,.2f}",
                "Risk Amount": f"${kelly_value * 0.1:,.2f}",
                "Risk %": f"{kelly_fraction * 100:.1f}%",
                "Description": "Optimal sizing based on win rate and R:R"
            },
            {
                "Method": "Volatility-based",
                "Shares": volatility_shares,
                "Position Value": f"${volatility_value:,.2f}",
                "Risk Amount": f"${volatility_value * 0.01:,.2f}",
                "Risk %": "1.0%",
                "Description": "Sizing based on daily volatility"
            }
        ]
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        
        # Risk of Ruin Calculator
        st.subheader("Risk of Ruin Analysis")
        
        # Calculate risk of ruin for different scenarios
        scenarios = []
        for risk_pct in [0.5, 1.0, 2.0, 3.0, 5.0]:
            risk_amt = account_size * (risk_pct / 100)
            if stop_distance > 0:
                shares = int(risk_amt / stop_distance)
                position_value = shares * entry_price
                
                # Simplified risk of ruin calculation
                # ROR = ((1 - p) / p)^(account_size / risk_per_trade)
                p = win_rate / 100
                if p > 0 and p < 1:
                    ror = ((1 - p) / p) ** (account_size / risk_amt)
                    ror = min(ror, 1.0)  # Cap at 100%
                else:
                    ror = 0
                
                scenarios.append({
                    "Risk %": f"{risk_pct:.1f}%",
                    "Position Value": f"${position_value:,.2f}",
                    "Risk of Ruin": f"{ror * 100:.1f}%",
                    "Shares": shares
                })
        
        scenarios_df = pd.DataFrame(scenarios)
        st.dataframe(scenarios_df, use_container_width=True)
        
        # Risk of Ruin Chart
        st.subheader("Risk of Ruin Chart")
        
        risk_percentages = np.arange(0.5, 6.0, 0.1)
        ror_values = []
        
        for risk_pct in risk_percentages:
            risk_amt = account_size * (risk_pct / 100)
            if stop_distance > 0:
                p = win_rate / 100
                if p > 0 and p < 1:
                    ror = ((1 - p) / p) ** (account_size / risk_amt)
                    ror = min(ror, 1.0)
                else:
                    ror = 0
                ror_values.append(ror * 100)
            else:
                ror_values.append(0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=risk_percentages,
            y=ror_values,
            mode='lines',
            name='Risk of Ruin',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Risk of Ruin vs Risk per Trade",
            xaxis_title="Risk per Trade (%)",
            yaxis_title="Risk of Ruin (%)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Position sizing recommendations
        st.subheader("Recommendations")
        
        # Find the recommended position size
        recommended_shares = fixed_fractional_shares
        recommended_method = "Fixed Fractional"
        
        if kelly_shares > 0 and kelly_shares < fixed_fractional_shares:
            recommended_shares = kelly_shares
            recommended_method = "Kelly Criterion"
        
        st.info(f"""
        **Recommended Position Size:** {recommended_shares} shares ({recommended_method})
        
        **Position Value:** ${recommended_shares * entry_price:,.2f}
        
        **Risk Amount:** ${risk_amount:,.2f} ({risk_per_trade:.1f}% of account)
        
        **Stop Loss:** ${stop_loss:.2f} ({stop_percentage:.1f}% from entry)
        """)
        
        # Risk management tips
        st.subheader("Risk Management Tips")
        
        tips = [
            "Never risk more than 2-3% of your account on a single trade",
            "Use stop losses to limit downside risk",
            "Diversify across different stocks and sectors",
            "Review and adjust position sizes based on market conditions",
            "Consider using trailing stops for profitable positions",
            "Keep a trading journal to track your win rate and R:R ratio"
        ]
        
        for i, tip in enumerate(tips, 1):
            st.write(f"{i}. {tip}")
        
        # Download results
        st.subheader("Download Results")
        
        # Create comprehensive results DataFrame
        comprehensive_results = pd.DataFrame({
            'Method': [r['Method'] for r in results_data],
            'Shares': [r['Shares'] for r in results_data],
            'Position_Value': [r['Position Value'].replace('$', '').replace(',', '') for r in results_data],
            'Risk_Amount': [r['Risk Amount'].replace('$', '').replace(',', '') for r in results_data],
            'Risk_Percentage': [r['Risk %'] for r in results_data],
            'Description': [r['Description'] for r in results_data]
        })
        
        csv = comprehensive_results.to_csv(index=False)
        st.download_button(
            label="Download Position Sizing Results (CSV)",
            data=csv,
            file_name=f"{symbol}_position_sizing.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>⚠️ <strong>Disclaimer:</strong> This calculator is for educational purposes only. Not investment advice.</p>
        <p>Position sizing should be based on your risk tolerance and trading experience.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_position_sizing()
