"""
Break-even Calculator - Tier 0 Free Tool
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

from core.options import OptionsPricer, OptionsStrategy, create_strategy_from_input
from core.data import get_stock_price, get_options_chain

def show_breakeven_calculator():
    """Display the break-even calculator page"""
    
    st.title("📈 Options Break-even Calculator")
    st.markdown("Calculate P/L, Greeks, and break-even points for multi-leg options strategies")
    
    # Sidebar for strategy input
    with st.sidebar:
        st.header("Strategy Setup")
        
        # Stock symbol input
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter the underlying stock symbol")
        
        # Get current stock price
        try:
            current_price = get_stock_price(symbol)
            if current_price > 0:
                st.success(f"Current {symbol} Price: ${current_price:.2f}")
            else:
                st.warning(f"Could not fetch price for {symbol}, using default")
                current_price = 100  # Default fallback
        except Exception as e:
            st.error(f"Error fetching price for {symbol}: {str(e)}")
            current_price = 100  # Default fallback
        
        # Strategy type selection
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Custom", "Long Call", "Long Put", "Covered Call", "Cash Secured Put", "Straddle", "Strangle"]
        )
        
        # Number of legs
        num_legs = st.number_input("Number of Legs", min_value=1, max_value=4, value=1)
        
        # Leg inputs
        legs = []
        for i in range(num_legs):
            st.subheader(f"Leg {i+1}")
            
            col1, col2 = st.columns(2)
            with col1:
                option_type = st.selectbox(f"Type {i+1}", ["call", "put"], key=f"type_{i}")
                quantity = st.number_input(f"Quantity {i+1}", min_value=1, max_value=100, value=1, key=f"qty_{i}")
            with col2:
                strike = st.number_input(f"Strike {i+1}", min_value=0.01, value=current_price, key=f"strike_{i}")
                entry_price = st.number_input(f"Entry Price {i+1}", min_value=0.01, value=5.0, key=f"price_{i}")
            
            action = st.selectbox(f"Action {i+1}", ["buy", "sell"], key=f"action_{i}")
            expiration = st.date_input(f"Expiration {i+1}", value=datetime.now() + timedelta(days=30), key=f"exp_{i}")
            
            legs.append({
                'type': option_type,
                'quantity': quantity,
                'strike': strike,
                'entry_price': entry_price,
                'action': action,
                'expiration': expiration
            })
    
    # Main content area
    if st.button("Calculate Strategy", type="primary"):
        if not legs:
            st.error("Please add at least one leg to the strategy")
            return
        
        # Create strategy
        try:
            strategy = create_strategy_from_input(legs)
        except Exception as e:
            st.error(f"Error creating strategy: {str(e)}")
            return
        
        # Calculate time to expiration (use first leg's expiration)
        try:
            time_to_exp = (legs[0]['expiration'] - datetime.now()).days / 365.0
            
            if time_to_exp <= 0:
                st.error("Expiration date must be in the future")
                return
        except Exception as e:
            st.error(f"Error calculating time to expiration: {str(e)}")
            return
        
        # Display strategy summary
        st.subheader("Strategy Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Stock Price", f"${current_price:.2f}")
        with col2:
            st.metric("Days to Expiration", f"{time_to_exp * 365:.0f}")
        with col3:
            st.metric("Strategy Legs", f"{len(legs)}")
        
        # Calculate and display current P/L
        current_pnl = strategy.calculate_current_pnl(current_price, time_to_exp)
        st.metric("Current P/L", f"${current_pnl:.2f}")
        
        # Calculate Greeks
        greeks = strategy.calculate_total_greeks(current_price, time_to_exp)
        
        st.subheader("Portfolio Greeks")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Delta", f"{greeks['delta']:.2f}")
        with col2:
            st.metric("Gamma", f"{greeks['gamma']:.4f}")
        with col3:
            st.metric("Theta", f"{greeks['theta']:.2f}")
        with col4:
            st.metric("Vega", f"{greeks['vega']:.2f}")
        with col5:
            st.metric("Rho", f"{greeks['rho']:.2f}")
        
        # P/L Chart
        st.subheader("P/L Analysis")
        
        # Create price range for chart
        price_range = np.arange(current_price * 0.5, current_price * 1.5, current_price * 0.01)
        
        # Calculate P/L at expiration
        pnl_at_exp = [strategy.calculate_pnl_at_expiration(price) for price in price_range]
        
        # Calculate current P/L
        pnl_current = [strategy.calculate_current_pnl(price, time_to_exp) for price in price_range]
        
        # Create the chart
        fig = go.Figure()
        
        # Add P/L at expiration line
        fig.add_trace(go.Scatter(
            x=price_range,
            y=pnl_at_exp,
            mode='lines',
            name='P/L at Expiration',
            line=dict(color='blue', width=2)
        ))
        
        # Add current P/L line
        fig.add_trace(go.Scatter(
            x=price_range,
            y=pnl_current,
            mode='lines',
            name='Current P/L',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add break-even lines
        fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Break-even")
        fig.add_vline(x=current_price, line_dash="dot", line_color="green", annotation_text="Current Price")
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Options Strategy P/L Analysis",
            xaxis_title="Stock Price ($)",
            yaxis_title="P/L ($)",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Break-even analysis
        st.subheader("Break-even Analysis")
        
        # Find break-even points
        break_even_points = strategy.find_break_even_points(
            min_price=current_price * 0.5,
            max_price=current_price * 1.5
        )
        
        if break_even_points:
            st.write("**Break-even Points:**")
            for i, be_point in enumerate(break_even_points[:5]):  # Show up to 5 points
                st.write(f"Break-even {i+1}: ${be_point:.2f}")
        else:
            st.write("No break-even points found in the analyzed range")
        
        # Strategy details table
        st.subheader("Strategy Details")
        
        strategy_df = pd.DataFrame(legs)
        strategy_df['Total Cost'] = strategy_df['quantity'] * strategy_df['entry_price'] * 100
        strategy_df['Action'] = strategy_df['action'].str.title()
        strategy_df['Type'] = strategy_df['type'].str.title()
        
        # Display the table
        st.dataframe(
            strategy_df[['Type', 'Action', 'Quantity', 'Strike', 'Entry Price', 'Total Cost']],
            use_container_width=True
        )
        
        # Download options
        st.subheader("Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download P/L data as CSV
            pnl_data = pd.DataFrame({
                'Stock_Price': price_range,
                'PL_At_Expiration': pnl_at_exp,
                'Current_PL': pnl_current
            })
            
            csv = pnl_data.to_csv(index=False)
            st.download_button(
                label="Download P/L Data (CSV)",
                data=csv,
                file_name=f"{symbol}_options_pl_analysis.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download strategy details
            strategy_csv = strategy_df.to_csv(index=False)
            st.download_button(
                label="Download Strategy Details (CSV)",
                data=strategy_csv,
                file_name=f"{symbol}_strategy_details.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>⚠️ <strong>Disclaimer:</strong> This calculator is for educational purposes only. Not investment advice.</p>
        <p>Options trading involves significant risk and may not be suitable for all investors.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_breakeven_calculator()
