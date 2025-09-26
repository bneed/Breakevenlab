"""
Portfolio Dashboard - Tier 2 Feature
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

def show_portfolio_dashboard():
    """Display the portfolio dashboard page"""
    
    st.title("💼 Portfolio Dashboard")
    st.markdown("Monitor your options portfolio risk and exposure")
    
    # Check if user has pro access
    if not is_pro_feature():
        show_pro_upgrade_prompt("Portfolio Dashboard")
        return
    
    # Sidebar for portfolio input
    with st.sidebar:
        st.header("Portfolio Input")
        
        input_method = st.radio(
            "Input Method",
            ["Manual Entry", "CSV Upload", "Sample Portfolio"]
        )
        
        if input_method == "Manual Entry":
            st.subheader("Add Position")
            
            symbol = st.text_input("Symbol", value="AAPL")
            option_type = st.selectbox("Option Type", ["call", "put", "stock"])
            quantity = st.number_input("Quantity", value=1)
            strike = st.number_input("Strike Price", value=100.0)
            entry_price = st.number_input("Entry Price", value=5.0)
            expiration = st.date_input("Expiration", value=datetime.now() + timedelta(days=30))
            
            if st.button("Add Position"):
                st.success("Position added!")
        
        elif input_method == "CSV Upload":
            uploaded_file = st.file_uploader(
                "Upload Portfolio CSV",
                type=['csv'],
                help="Upload a CSV file with your portfolio positions"
            )
            
            if uploaded_file:
                st.success("File uploaded successfully!")
        
        else:  # Sample Portfolio
            st.info("Using sample portfolio for demonstration")
    
    # Main content area
    if st.button("Analyze Portfolio", type="primary"):
        
        # Create sample portfolio data
        portfolio_data = create_sample_portfolio()
        
        # Display portfolio overview
        st.subheader("Portfolio Overview")
        
        # Calculate portfolio metrics
        total_value = portfolio_data['position_value'].sum()
        total_delta = portfolio_data['delta'].sum()
        total_gamma = portfolio_data['gamma'].sum()
        total_theta = portfolio_data['theta'].sum()
        total_vega = portfolio_data['vega'].sum()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Value", f"${total_value:,.2f}")
        
        with col2:
            st.metric("Total Delta", f"{total_delta:.2f}")
        
        with col3:
            st.metric("Total Gamma", f"{total_gamma:.4f}")
        
        with col4:
            st.metric("Total Theta", f"{total_theta:.2f}")
        
        with col5:
            st.metric("Total Vega", f"{total_vega:.2f}")
        
        # Portfolio positions table
        st.subheader("Portfolio Positions")
        
        # Display the portfolio table
        st.dataframe(
            portfolio_data[['symbol', 'type', 'quantity', 'strike', 'entry_price', 'position_value', 'delta', 'gamma', 'theta', 'vega']],
            use_container_width=True
        )
        
        # Risk analysis
        st.subheader("Risk Analysis")
        
        # Delta exposure by symbol
        delta_by_symbol = portfolio_data.groupby('symbol')['delta'].sum().reset_index()
        
        fig = px.bar(
            delta_by_symbol, 
            x='symbol', 
            y='delta',
            title="Delta Exposure by Symbol",
            labels={'delta': 'Delta', 'symbol': 'Symbol'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Greeks distribution
        st.subheader("Greeks Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Delta distribution
            fig = px.histogram(
                portfolio_data, 
                x='delta', 
                nbins=20,
                title="Delta Distribution",
                labels={'delta': 'Delta', 'count': 'Number of Positions'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Theta distribution
            fig = px.histogram(
                portfolio_data, 
                x='theta', 
                nbins=20,
                title="Theta Distribution",
                labels={'theta': 'Theta', 'count': 'Number of Positions'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Scenario analysis
        st.subheader("Scenario Analysis")
        
        # Price shock scenarios
        scenarios = [-10, -5, 0, 5, 10]  # Percentage changes
        scenario_results = []
        
        for scenario in scenarios:
            pnl_change = total_delta * (scenario / 100) * total_value
            scenario_results.append({
                'Scenario': f"{scenario:+d}%",
                'PnL Change': pnl_change,
                'New Portfolio Value': total_value + pnl_change
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        
        # Display scenario table
        st.dataframe(scenario_df, use_container_width=True)
        
        # Scenario chart
        fig = px.bar(
            scenario_df, 
            x='Scenario', 
            y='PnL Change',
            title="Portfolio P/L Under Different Scenarios",
            labels={'PnL Change': 'P/L Change ($)', 'Scenario': 'Price Change'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Time decay analysis
        st.subheader("Time Decay Analysis")
        
        # Calculate time decay over next 30 days
        days_range = range(1, 31)
        time_decay_data = []
        
        for days in days_range:
            daily_theta = total_theta
            cumulative_decay = daily_theta * days
            time_decay_data.append({
                'Days': days,
                'Daily Theta': daily_theta,
                'Cumulative Decay': cumulative_decay
            })
        
        time_decay_df = pd.DataFrame(time_decay_data)
        
        # Time decay chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_decay_df['Days'],
            y=time_decay_df['Cumulative Decay'],
            mode='lines',
            name='Cumulative Time Decay',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Time Decay Over Time",
            xaxis_title="Days",
            yaxis_title="Cumulative Time Decay ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volatility analysis
        st.subheader("Volatility Analysis")
        
        # Calculate vega exposure
        vega_by_symbol = portfolio_data.groupby('symbol')['vega'].sum().reset_index()
        
        fig = px.bar(
            vega_by_symbol, 
            x='symbol', 
            y='vega',
            title="Vega Exposure by Symbol",
            labels={'vega': 'Vega', 'symbol': 'Symbol'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio heatmap
        st.subheader("Portfolio Heatmap")
        
        # Create a heatmap of Greeks by symbol and type
        heatmap_data = portfolio_data.pivot_table(
            values=['delta', 'gamma', 'theta', 'vega'], 
            index='symbol', 
            columns='type', 
            aggfunc='sum',
            fill_value=0
        )
        
        # Flatten the multi-level columns
        heatmap_data.columns = [f"{col[1]}_{col[0]}" for col in heatmap_data.columns]
        
        # Create heatmap
        fig = px.imshow(
            heatmap_data.T,
            title="Portfolio Greeks Heatmap",
            labels=dict(x="Symbol", y="Greek", color="Value"),
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk metrics
        st.subheader("Risk Metrics")
        
        # Calculate various risk metrics
        portfolio_beta = 1.0  # Simplified
        portfolio_volatility = 0.2  # Simplified
        
        # Value at Risk (simplified)
        var_95 = total_value * 0.05  # 5% VaR
        var_99 = total_value * 0.01  # 1% VaR
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
        
        with col2:
            st.metric("Portfolio Volatility", f"{portfolio_volatility:.1%}")
        
        with col3:
            st.metric("VaR (95%)", f"${var_95:,.2f}")
        
        with col4:
            st.metric("VaR (99%)", f"${var_99:,.2f}")
        
        # Download options
        st.subheader("Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download portfolio data
            csv = portfolio_data.to_csv(index=False)
            st.download_button(
                label="Download Portfolio Data (CSV)",
                data=csv,
                file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download scenario analysis
            scenario_csv = scenario_df.to_csv(index=False)
            st.download_button(
                label="Download Scenario Analysis (CSV)",
                data=scenario_csv,
                file_name=f"scenario_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Portfolio management tips
    st.subheader("Portfolio Management Tips")
    
    tips = [
        "Monitor your total delta exposure regularly",
        "Keep theta negative for income strategies",
        "Diversify across different symbols and strategies",
        "Use stop losses and position sizing",
        "Rebalance your portfolio periodically",
        "Consider correlation between positions",
        "Monitor volatility exposure (vega)",
        "Review and adjust based on market conditions"
    ]
    
    for i, tip in enumerate(tips, 1):
        st.write(f"{i}. {tip}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>⚠️ <strong>Disclaimer:</strong> This portfolio dashboard is for educational purposes only. Not investment advice.</p>
        <p>Portfolio analysis should be used as one tool in your risk management process.</p>
    </div>
    """, unsafe_allow_html=True)

def create_sample_portfolio():
    """Create a sample portfolio for demonstration"""
    
    sample_positions = [
        {
            'symbol': 'AAPL',
            'type': 'call',
            'quantity': 10,
            'strike': 150.0,
            'entry_price': 5.0,
            'position_value': 5000.0,
            'delta': 0.6,
            'gamma': 0.02,
            'theta': -0.5,
            'vega': 0.3
        },
        {
            'symbol': 'AAPL',
            'type': 'put',
            'quantity': -5,
            'strike': 145.0,
            'entry_price': 3.0,
            'position_value': -1500.0,
            'delta': 0.3,
            'gamma': 0.01,
            'theta': 0.2,
            'vega': 0.15
        },
        {
            'symbol': 'MSFT',
            'type': 'call',
            'quantity': 20,
            'strike': 300.0,
            'entry_price': 8.0,
            'position_value': 16000.0,
            'delta': 0.7,
            'gamma': 0.025,
            'theta': -0.8,
            'vega': 0.4
        },
        {
            'symbol': 'GOOGL',
            'type': 'put',
            'quantity': 15,
            'strike': 2500.0,
            'entry_price': 50.0,
            'position_value': 75000.0,
            'delta': -0.4,
            'gamma': 0.015,
            'theta': -1.2,
            'vega': 0.6
        },
        {
            'symbol': 'TSLA',
            'type': 'stock',
            'quantity': 100,
            'strike': 0.0,
            'entry_price': 200.0,
            'position_value': 20000.0,
            'delta': 100.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0
        }
    ]
    
    return pd.DataFrame(sample_positions)

if __name__ == "__main__":
    show_portfolio_dashboard()
