"""
Greeks Viewer - Tier 0 Free Tool
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
from core.data import get_stock_price, get_options_chain

def show_greeks_viewer():
    """Display the Greeks viewer page"""
    
    st.title("📊 Options Greeks Viewer")
    st.markdown("Analyze Delta, Gamma, Theta, Vega, and Rho for options strategies")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Options Parameters")
        
        # Stock symbol input
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter the underlying stock symbol")
        
        # Get current stock price
        current_price = get_stock_price(symbol)
        if current_price > 0:
            st.success(f"Current {symbol} Price: ${current_price:.2f}")
        else:
            st.error(f"Could not fetch price for {symbol}")
            current_price = 100  # Default fallback
        
        # Option parameters
        option_type = st.selectbox("Option Type", ["call", "put"])
        
        strike_price = st.number_input(
            "Strike Price ($)", 
            min_value=0.01, 
            value=current_price,
            help="Option strike price"
        )
        
        days_to_expiration = st.number_input(
            "Days to Expiration", 
            min_value=1, 
            max_value=365, 
            value=30,
            help="Days until option expiration"
        )
        
        implied_volatility = st.slider(
            "Implied Volatility (%)", 
            min_value=5.0, 
            max_value=200.0, 
            value=30.0, 
            step=1.0,
            help="Implied volatility percentage"
        )
        
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)", 
            min_value=0.0, 
            max_value=10.0, 
            value=5.0, 
            step=0.1,
            help="Risk-free interest rate"
        )
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        
        price_range_pct = st.slider(
            "Price Range (%)", 
            min_value=10, 
            max_value=100, 
            value=50, 
            step=5,
            help="Percentage range around current price for analysis"
        )
        
        time_range_days = st.slider(
            "Time Range (Days)", 
            min_value=1, 
            max_value=90, 
            value=30, 
            step=1,
            help="Time range for analysis"
        )
    
    # Main content area
    if st.button("Calculate Greeks", type="primary"):
        
        # Convert parameters
        time_to_exp = days_to_expiration / 365.0
        volatility = implied_volatility / 100.0
        risk_free = risk_free_rate / 100.0
        
        # Calculate current Greeks
        current_greeks = OptionsPricer.calculate_greeks(
            current_price, strike_price, time_to_exp, risk_free, volatility, option_type
        )
        
        # Display current Greeks
        st.subheader("Current Greeks")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Delta", f"{current_greeks['delta']:.4f}")
        with col2:
            st.metric("Gamma", f"{current_greeks['gamma']:.6f}")
        with col3:
            st.metric("Theta", f"{current_greeks['theta']:.4f}")
        with col4:
            st.metric("Vega", f"{current_greeks['vega']:.4f}")
        with col5:
            st.metric("Rho", f"{current_greeks['rho']:.4f}")
        
        # Greeks sensitivity analysis
        st.subheader("Greeks Sensitivity Analysis")
        
        # Price sensitivity
        price_min = current_price * (1 - price_range_pct / 100)
        price_max = current_price * (1 + price_range_pct / 100)
        price_range = np.linspace(price_min, price_max, 100)
        
        # Calculate Greeks across price range
        greeks_data = []
        for price in price_range:
            greeks = OptionsPricer.calculate_greeks(
                price, strike_price, time_to_exp, risk_free, volatility, option_type
            )
            greeks_data.append({
                'price': price,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'rho': greeks['rho']
            })
        
        greeks_df = pd.DataFrame(greeks_data)
        
        # Create subplots for each Greek
        fig = go.Figure()
        
        # Add traces for each Greek
        fig.add_trace(go.Scatter(
            x=greeks_df['price'],
            y=greeks_df['delta'],
            mode='lines',
            name='Delta',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=greeks_df['price'],
            y=greeks_df['gamma'],
            mode='lines',
            name='Gamma',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=greeks_df['price'],
            y=greeks_df['theta'],
            mode='lines',
            name='Theta',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=greeks_df['price'],
            y=greeks_df['vega'],
            mode='lines',
            name='Vega',
            line=dict(color='orange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=greeks_df['price'],
            y=greeks_df['rho'],
            mode='lines',
            name='Rho',
            line=dict(color='purple', width=2)
        ))
        
        # Add current price line
        fig.add_vline(x=current_price, line_dash="dot", line_color="black", annotation_text="Current Price")
        
        fig.update_layout(
            title=f"{symbol} {option_type.title()} Greeks vs Stock Price",
            xaxis_title="Stock Price ($)",
            yaxis_title="Greek Value",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Time decay analysis
        st.subheader("Time Decay Analysis")
        
        # Calculate Greeks across time
        time_range = np.linspace(1, time_range_days, 100)
        time_greeks_data = []
        
        for days in time_range:
            tte = days / 365.0
            greeks = OptionsPricer.calculate_greeks(
                current_price, strike_price, tte, risk_free, volatility, option_type
            )
            time_greeks_data.append({
                'days_to_exp': days,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'rho': greeks['rho']
            })
        
        time_greeks_df = pd.DataFrame(time_greeks_data)
        
        # Create time decay chart
        fig_time = go.Figure()
        
        fig_time.add_trace(go.Scatter(
            x=time_greeks_df['days_to_exp'],
            y=time_greeks_df['delta'],
            mode='lines',
            name='Delta',
            line=dict(color='blue', width=2)
        ))
        
        fig_time.add_trace(go.Scatter(
            x=time_greeks_df['days_to_exp'],
            y=time_greeks_df['gamma'],
            mode='lines',
            name='Gamma',
            line=dict(color='red', width=2)
        ))
        
        fig_time.add_trace(go.Scatter(
            x=time_greeks_df['days_to_exp'],
            y=time_greeks_df['theta'],
            mode='lines',
            name='Theta',
            line=dict(color='green', width=2)
        ))
        
        fig_time.add_trace(go.Scatter(
            x=time_greeks_df['days_to_exp'],
            y=time_greeks_df['vega'],
            mode='lines',
            name='Vega',
            line=dict(color='orange', width=2)
        ))
        
        fig_time.add_trace(go.Scatter(
            x=time_greeks_df['days_to_exp'],
            y=time_greeks_df['rho'],
            mode='lines',
            name='Rho',
            line=dict(color='purple', width=2)
        ))
        
        fig_time.update_layout(
            title=f"{symbol} {option_type.title()} Greeks vs Time to Expiration",
            xaxis_title="Days to Expiration",
            yaxis_title="Greek Value",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Greeks explanation
        st.subheader("Greeks Explanation")
        
        greeks_explanation = {
            "Delta": "Measures the rate of change of the option price with respect to the underlying asset price. For calls, delta is positive (0 to 1); for puts, delta is negative (-1 to 0).",
            "Gamma": "Measures the rate of change of delta with respect to the underlying asset price. Gamma is highest for at-the-money options and decreases as options move in or out of the money.",
            "Theta": "Measures the rate of change of the option price with respect to time. Theta is typically negative, representing time decay. Options lose value as time passes.",
            "Vega": "Measures the rate of change of the option price with respect to implied volatility. Vega is positive for both calls and puts, meaning options gain value when volatility increases.",
            "Rho": "Measures the rate of change of the option price with respect to the risk-free interest rate. Rho is positive for calls and negative for puts."
        }
        
        for greek, explanation in greeks_explanation.items():
            st.write(f"**{greek}:** {explanation}")
        
        # Greeks table
        st.subheader("Detailed Greeks Table")
        
        # Create a table with current and projected Greeks
        greeks_table_data = []
        
        # Current Greeks
        greeks_table_data.append({
            "Scenario": "Current",
            "Stock Price": f"${current_price:.2f}",
            "Days to Exp": f"{days_to_expiration}",
            "Delta": f"{current_greeks['delta']:.4f}",
            "Gamma": f"{current_greeks['gamma']:.6f}",
            "Theta": f"{current_greeks['theta']:.4f}",
            "Vega": f"{current_greeks['vega']:.4f}",
            "Rho": f"{current_greeks['rho']:.4f}"
        })
        
        # Projected scenarios
        scenarios = [
            ("+10% Price", current_price * 1.1, days_to_expiration),
            ("-10% Price", current_price * 0.9, days_to_expiration),
            ("+7 Days", current_price, days_to_expiration + 7),
            ("-7 Days", current_price, max(1, days_to_expiration - 7))
        ]
        
        for scenario_name, price, days in scenarios:
            tte = days / 365.0
            greeks = OptionsPricer.calculate_greeks(
                price, strike_price, tte, risk_free, volatility, option_type
            )
            
            greeks_table_data.append({
                "Scenario": scenario_name,
                "Stock Price": f"${price:.2f}",
                "Days to Exp": f"{days}",
                "Delta": f"{greeks['delta']:.4f}",
                "Gamma": f"{greeks['gamma']:.6f}",
                "Theta": f"{greeks['theta']:.4f}",
                "Vega": f"{greeks['vega']:.4f}",
                "Rho": f"{greeks['rho']:.4f}"
            })
        
        greeks_table_df = pd.DataFrame(greeks_table_data)
        st.dataframe(greeks_table_df, use_container_width=True)
        
        # Download options
        st.subheader("Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download Greeks data
            csv = greeks_df.to_csv(index=False)
            st.download_button(
                label="Download Greeks Data (CSV)",
                data=csv,
                file_name=f"{symbol}_greeks_analysis.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download time decay data
            time_csv = time_greeks_df.to_csv(index=False)
            st.download_button(
                label="Download Time Decay Data (CSV)",
                data=time_csv,
                file_name=f"{symbol}_time_decay.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>⚠️ <strong>Disclaimer:</strong> This calculator is for educational purposes only. Not investment advice.</p>
        <p>Greeks are theoretical values and may not match actual market behavior.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_greeks_viewer()
