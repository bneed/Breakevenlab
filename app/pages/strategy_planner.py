"""
Strategy Planner - Tier 2 Feature
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

from core.options import OptionsPricer
from core.data import get_stock_price
from core.auth import is_pro_feature, show_pro_upgrade_prompt

def show_strategy_planner():
    """Display the strategy planner page"""
    
    st.title("🎯 Strategy Planner")
    st.markdown("Plan and simulate advanced options strategies")
    
    # Check if user has pro access
    if not is_pro_feature():
        show_pro_upgrade_prompt("Strategy Planner")
        return
    
    # Sidebar for strategy selection
    with st.sidebar:
        st.header("Strategy Selection")
        
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Wheel Strategy", "Covered Call Designer", "Iron Condor", "Butterfly Spread", "Custom Strategy"]
        )
        
        # Common parameters
        st.subheader("Common Parameters")
        
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter the underlying stock symbol")
        
        # Get current stock price
        current_price = get_stock_price(symbol)
        if current_price > 0:
            st.success(f"Current {symbol} Price: ${current_price:.2f}")
        else:
            st.error(f"Could not fetch price for {symbol}")
            current_price = 100  # Default fallback
        
        account_size = st.number_input(
            "Account Size ($)", 
            min_value=1000, 
            max_value=1000000, 
            value=10000
        )
        
        # Strategy-specific parameters
        if strategy_type == "Wheel Strategy":
            st.subheader("Wheel Strategy Parameters")
            
            strike_delta = st.slider("Put Strike Delta", 0.1, 0.5, 0.3, 0.1)
            call_delta = st.slider("Call Strike Delta", 0.1, 0.5, 0.3, 0.1)
            dte = st.slider("Days to Expiration", 7, 45, 30)
            cycles_per_year = st.slider("Cycles per Year", 4, 12, 6)
            
        elif strategy_type == "Covered Call Designer":
            st.subheader("Covered Call Parameters")
            
            strike_delta = st.slider("Strike Delta", 0.1, 0.5, 0.3, 0.1)
            dte = st.slider("Days to Expiration", 7, 45, 30)
            roll_dte = st.slider("Roll at DTE", 1, 14, 7)
            position_size = st.slider("Position Size", 0.1, 1.0, 1.0, 0.1)
            
        elif strategy_type == "Iron Condor":
            st.subheader("Iron Condor Parameters")
            
            short_put_delta = st.slider("Short Put Delta", 0.1, 0.3, 0.2, 0.05)
            short_call_delta = st.slider("Short Call Delta", 0.1, 0.3, 0.2, 0.05)
            wing_width = st.slider("Wing Width", 5, 20, 10)
            dte = st.slider("Days to Expiration", 7, 45, 30)
            
        elif strategy_type == "Butterfly Spread":
            st.subheader("Butterfly Parameters")
            
            center_strike = st.number_input("Center Strike", value=current_price)
            wing_width = st.slider("Wing Width", 5, 20, 10)
            dte = st.slider("Days to Expiration", 7, 45, 30)
    
    # Main content area
    if st.button("Plan Strategy", type="primary"):
        
        if strategy_type == "Wheel Strategy":
            show_wheel_strategy_planner(symbol, current_price, account_size, strike_delta, call_delta, dte, cycles_per_year)
            
        elif strategy_type == "Covered Call Designer":
            show_covered_call_designer(symbol, current_price, account_size, strike_delta, dte, roll_dte, position_size)
            
        elif strategy_type == "Iron Condor":
            show_iron_condor_planner(symbol, current_price, account_size, short_put_delta, short_call_delta, wing_width, dte)
            
        elif strategy_type == "Butterfly Spread":
            show_butterfly_planner(symbol, current_price, account_size, center_strike, wing_width, dte)
            
        else:  # Custom Strategy
            st.info("Custom strategy planner coming soon!")

def show_wheel_strategy_planner(symbol, current_price, account_size, strike_delta, call_delta, dte, cycles_per_year):
    """Display wheel strategy planner"""
    
    st.subheader("Wheel Strategy Analysis")
    
    # Calculate strikes
    put_strike = current_price * (1 - strike_delta)
    call_strike = current_price * (1 + call_delta)
    
    # Estimate premiums (simplified)
    put_premium = current_price * 0.02  # 2% of stock price
    call_premium = current_price * 0.015  # 1.5% of stock price
    
    # Calculate position size
    shares = int(account_size / current_price)
    position_value = shares * current_price
    
    # Calculate expected returns
    put_income = put_premium * shares * 100
    call_income = call_premium * shares * 100
    total_income = put_income + call_income
    
    # Annual projections
    annual_income = total_income * cycles_per_year
    annual_return = (annual_income / account_size) * 100
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Put Strike", f"${put_strike:.2f}")
    
    with col2:
        st.metric("Call Strike", f"${call_strike:.2f}")
    
    with col3:
        st.metric("Shares", f"{shares:,}")
    
    with col4:
        st.metric("Position Value", f"${position_value:,.2f}")
    
    # Income projections
    st.subheader("Income Projections")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Put Premium", f"${put_premium:.2f}")
    
    with col2:
        st.metric("Call Premium", f"${call_premium:.2f}")
    
    with col3:
        st.metric("Total Premium", f"${total_income:,.2f}")
    
    # Annual projections
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Annual Income", f"${annual_income:,.2f}")
    
    with col2:
        st.metric("Annual Return", f"{annual_return:.1f}%")
    
    with col3:
        st.metric("Cycles per Year", f"{cycles_per_year}")
    
    # P/L analysis
    st.subheader("P/L Analysis")
    
    # Create price range for analysis
    price_range = np.arange(current_price * 0.7, current_price * 1.3, current_price * 0.01)
    
    # Calculate P/L for different scenarios
    pnl_data = []
    
    for price in price_range:
        # Put scenario (if assigned)
        if price <= put_strike:
            put_pnl = (put_strike - price) * shares * 100 + put_income
        else:
            put_pnl = put_income
        
        # Call scenario (if assigned)
        if price >= call_strike:
            call_pnl = (call_strike - price) * shares * 100 + call_income
        else:
            call_pnl = call_income
        
        # Total P/L
        total_pnl = put_pnl + call_pnl
        
        pnl_data.append({
            'price': price,
            'put_pnl': put_pnl,
            'call_pnl': call_pnl,
            'total_pnl': total_pnl
        })
    
    pnl_df = pd.DataFrame(pnl_data)
    
    # Create P/L chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pnl_df['price'],
        y=pnl_df['total_pnl'],
        mode='lines',
        name='Total P/L',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_vline(x=current_price, line_dash="dot", line_color="green", annotation_text="Current Price")
    fig.add_vline(x=put_strike, line_dash="dot", line_color="red", annotation_text="Put Strike")
    fig.add_vline(x=call_strike, line_dash="dot", line_color="red", annotation_text="Call Strike")
    
    fig.update_layout(
        title=f"{symbol} Wheel Strategy P/L Analysis",
        xaxis_title="Stock Price ($)",
        yaxis_title="P/L ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk analysis
    st.subheader("Risk Analysis")
    
    # Calculate max loss scenarios
    max_loss_put = (put_strike - current_price * 0.5) * shares * 100 - put_income
    max_loss_call = (current_price * 1.5 - call_strike) * shares * 100 - call_income
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Max Loss (Put)", f"${max_loss_put:,.2f}")
    
    with col2:
        st.metric("Max Loss (Call)", f"${max_loss_call:,.2f}")
    
    # Assignment probability (simplified)
    put_assignment_prob = 0.3  # 30% chance
    call_assignment_prob = 0.2  # 20% chance
    
    st.write(f"**Estimated Assignment Probabilities:**")
    st.write(f"- Put Assignment: {put_assignment_prob * 100:.0f}%")
    st.write(f"- Call Assignment: {call_assignment_prob * 100:.0f}%")

def show_covered_call_designer(symbol, current_price, account_size, strike_delta, dte, roll_dte, position_size):
    """Display covered call designer"""
    
    st.subheader("Covered Call Strategy Designer")
    
    # Calculate strike
    strike = current_price * (1 + strike_delta)
    
    # Estimate premium
    premium = current_price * 0.02  # 2% of stock price
    
    # Calculate position
    shares = int((account_size * position_size) / current_price)
    position_value = shares * current_price
    
    # Calculate returns
    premium_income = premium * shares * 100
    annual_return = (premium_income * 12 / dte) / account_size * 100
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Strike Price", f"${strike:.2f}")
    
    with col2:
        st.metric("Premium", f"${premium:.2f}")
    
    with col3:
        st.metric("Shares", f"{shares:,}")
    
    with col4:
        st.metric("Position Value", f"${position_value:,.2f}")
    
    # Income analysis
    st.subheader("Income Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Premium Income", f"${premium_income:,.2f}")
    
    with col2:
        st.metric("Annual Return", f"{annual_return:.1f}%")
    
    with col3:
        st.metric("Days to Expiration", f"{dte}")
    
    # P/L analysis
    st.subheader("P/L Analysis")
    
    # Create price range
    price_range = np.arange(current_price * 0.8, current_price * 1.2, current_price * 0.01)
    
    pnl_data = []
    for price in price_range:
        if price <= strike:
            # Not assigned
            pnl = (price - current_price) * shares * 100 + premium_income
        else:
            # Assigned
            pnl = (strike - current_price) * shares * 100 + premium_income
        
        pnl_data.append({
            'price': price,
            'pnl': pnl
        })
    
    pnl_df = pd.DataFrame(pnl_data)
    
    # Create P/L chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pnl_df['price'],
        y=pnl_df['pnl'],
        mode='lines',
        name='P/L',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_vline(x=current_price, line_dash="dot", line_color="green", annotation_text="Current Price")
    fig.add_vline(x=strike, line_dash="dot", line_color="red", annotation_text="Strike")
    
    fig.update_layout(
        title=f"{symbol} Covered Call P/L Analysis",
        xaxis_title="Stock Price ($)",
        yaxis_title="P/L ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_iron_condor_planner(symbol, current_price, account_size, short_put_delta, short_call_delta, wing_width, dte):
    """Display iron condor planner"""
    
    st.subheader("Iron Condor Strategy Planner")
    
    # Calculate strikes
    short_put_strike = current_price * (1 - short_put_delta)
    long_put_strike = short_put_strike - wing_width
    short_call_strike = current_price * (1 + short_call_delta)
    long_call_strike = short_call_strike + wing_width
    
    # Estimate premiums
    short_put_premium = current_price * 0.015
    long_put_premium = current_price * 0.005
    short_call_premium = current_price * 0.015
    long_call_premium = current_price * 0.005
    
    # Calculate net credit
    net_credit = (short_put_premium + short_call_premium) - (long_put_premium + long_call_premium)
    
    # Calculate max profit and loss
    max_profit = net_credit * 100
    max_loss = (wing_width - net_credit) * 100
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Short Put Strike", f"${short_put_strike:.2f}")
    
    with col2:
        st.metric("Long Put Strike", f"${long_put_strike:.2f}")
    
    with col3:
        st.metric("Short Call Strike", f"${short_call_strike:.2f}")
    
    with col4:
        st.metric("Long Call Strike", f"${long_call_strike:.2f}")
    
    # Strategy metrics
    st.subheader("Strategy Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Net Credit", f"${net_credit:.2f}")
    
    with col2:
        st.metric("Max Profit", f"${max_profit:,.2f}")
    
    with col3:
        st.metric("Max Loss", f"${max_loss:,.2f}")
    
    # P/L analysis
    st.subheader("P/L Analysis")
    
    # Create price range
    price_range = np.arange(long_put_strike - 5, long_call_strike + 5, 1)
    
    pnl_data = []
    for price in price_range:
        # Calculate P/L at expiration
        if price <= long_put_strike:
            pnl = max_loss
        elif price <= short_put_strike:
            pnl = max_loss - (price - long_put_strike) * 100
        elif price <= short_call_strike:
            pnl = max_profit
        elif price <= long_call_strike:
            pnl = max_profit - (price - short_call_strike) * 100
        else:
            pnl = max_loss
        
        pnl_data.append({
            'price': price,
            'pnl': pnl
        })
    
    pnl_df = pd.DataFrame(pnl_data)
    
    # Create P/L chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pnl_df['price'],
        y=pnl_df['pnl'],
        mode='lines',
        name='P/L',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_vline(x=current_price, line_dash="dot", line_color="green", annotation_text="Current Price")
    fig.add_vline(x=short_put_strike, line_dash="dot", line_color="red", annotation_text="Short Put")
    fig.add_vline(x=short_call_strike, line_dash="dot", line_color="red", annotation_text="Short Call")
    
    fig.update_layout(
        title=f"{symbol} Iron Condor P/L Analysis",
        xaxis_title="Stock Price ($)",
        yaxis_title="P/L ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_butterfly_planner(symbol, current_price, account_size, center_strike, wing_width, dte):
    """Display butterfly spread planner"""
    
    st.subheader("Butterfly Spread Planner")
    
    # Calculate strikes
    lower_strike = center_strike - wing_width
    upper_strike = center_strike + wing_width
    
    # Estimate premiums
    lower_premium = current_price * 0.01
    center_premium = current_price * 0.02
    upper_premium = current_price * 0.01
    
    # Calculate net debit
    net_debit = (center_premium * 2) - (lower_premium + upper_premium)
    
    # Calculate max profit and loss
    max_profit = (wing_width - net_debit) * 100
    max_loss = net_debit * 100
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Lower Strike", f"${lower_strike:.2f}")
    
    with col2:
        st.metric("Center Strike", f"${center_strike:.2f}")
    
    with col3:
        st.metric("Upper Strike", f"${upper_strike:.2f}")
    
    # Strategy metrics
    st.subheader("Strategy Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Net Debit", f"${net_debit:.2f}")
    
    with col2:
        st.metric("Max Profit", f"${max_profit:,.2f}")
    
    with col3:
        st.metric("Max Loss", f"${max_loss:,.2f}")
    
    # P/L analysis
    st.subheader("P/L Analysis")
    
    # Create price range
    price_range = np.arange(lower_strike - 5, upper_strike + 5, 1)
    
    pnl_data = []
    for price in price_range:
        # Calculate P/L at expiration
        if price <= lower_strike:
            pnl = -max_loss
        elif price <= center_strike:
            pnl = -max_loss + (price - lower_strike) * 100
        elif price <= upper_strike:
            pnl = max_profit - (price - center_strike) * 100
        else:
            pnl = -max_loss
        
        pnl_data.append({
            'price': price,
            'pnl': pnl
        })
    
    pnl_df = pd.DataFrame(pnl_data)
    
    # Create P/L chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pnl_df['price'],
        y=pnl_df['pnl'],
        mode='lines',
        name='P/L',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_vline(x=current_price, line_dash="dot", line_color="green", annotation_text="Current Price")
    fig.add_vline(x=center_strike, line_dash="dot", line_color="red", annotation_text="Center Strike")
    
    fig.update_layout(
        title=f"{symbol} Butterfly Spread P/L Analysis",
        xaxis_title="Stock Price ($)",
        yaxis_title="P/L ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    show_strategy_planner()
