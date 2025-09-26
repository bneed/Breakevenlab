"""
IV Rank Screener - Tier 0 Free Tool
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

from core.screeners import screen_high_iv_rank, screen_earnings_soon, screen_unusual_volume
from core.data import get_stock_price

def show_ivr_screener():
    """Display the IV Rank screener page"""
    
    st.title("🔍 IV Rank Screener")
    st.markdown("Screen stocks with high implied volatility rank and unusual activity")
    
    # Sidebar for screening parameters
    with st.sidebar:
        st.header("Screening Parameters")
        
        # IV Rank filters
        st.subheader("IV Rank Filters")
        min_ivr = st.slider(
            "Minimum IV Rank (%)", 
            min_value=0, 
            max_value=100, 
            value=50, 
            step=5,
            help="Minimum IV Rank percentage"
        )
        
        # Price filters
        st.subheader("Price Filters")
        min_price = st.number_input(
            "Minimum Price ($)", 
            min_value=0.01, 
            value=5.0,
            help="Minimum stock price"
        )
        
        max_price = st.number_input(
            "Maximum Price ($)", 
            min_value=0.01, 
            value=1000.0,
            help="Maximum stock price"
        )
        
        # Volume filters
        st.subheader("Volume Filters")
        min_volume_ratio = st.slider(
            "Minimum Volume Ratio", 
            min_value=1.0, 
            max_value=10.0, 
            value=1.0, 
            step=0.1,
            help="Minimum volume vs average"
        )
        
        # Earnings filters
        st.subheader("Earnings Filters")
        days_ahead = st.slider(
            "Earnings Days Ahead", 
            min_value=1, 
            max_value=30, 
            value=7, 
            step=1,
            help="Days ahead to look for earnings"
        )
        
        # Refresh button
        if st.button("🔄 Refresh Data", type="primary"):
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["High IV Rank", "Earnings Soon", "Unusual Volume"])
    
    with tab1:
        st.subheader("High IV Rank Stocks")
        st.markdown(f"Showing stocks with IV Rank ≥ {min_ivr}%")
        
        # Run the screener
        with st.spinner("Screening stocks..."):
            ivr_results = screen_high_iv_rank(
                min_ivr=min_ivr,
                max_price=max_price,
                min_price=min_price
            )
        
        if not ivr_results.empty:
            # Display results
            st.dataframe(
                ivr_results[['symbol', 'price', 'iv_rank', 'volume_ratio']],
                use_container_width=True
            )
            
            # Create IV Rank distribution chart
            fig = px.histogram(
                ivr_results, 
                x='iv_rank', 
                nbins=20,
                title="IV Rank Distribution",
                labels={'iv_rank': 'IV Rank (%)', 'count': 'Number of Stocks'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 10 IV Rank stocks
            st.subheader("Top 10 IV Rank Stocks")
            top_10 = ivr_results.head(10)
            
            # Create bar chart
            fig = px.bar(
                top_10, 
                x='symbol', 
                y='iv_rank',
                title="Top 10 IV Rank Stocks",
                labels={'iv_rank': 'IV Rank (%)', 'symbol': 'Symbol'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            csv = ivr_results.to_csv(index=False)
            st.download_button(
                label="Download IV Rank Results (CSV)",
                data=csv,
                file_name=f"iv_rank_screener_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("No stocks found matching the criteria. Try adjusting the filters.")
    
    with tab2:
        st.subheader("Earnings Coming Soon")
        st.markdown(f"Showing stocks with earnings in the next {days_ahead} days")
        
        # Run the earnings screener
        with st.spinner("Screening earnings..."):
            earnings_results = screen_earnings_soon(days_ahead=days_ahead)
        
        if not earnings_results.empty:
            # Display results
            st.dataframe(
                earnings_results[['symbol', 'company', 'earnings_date', 'days_to_earnings', 'current_price', 'iv_rank']],
                use_container_width=True
            )
            
            # Create earnings timeline
            fig = px.scatter(
                earnings_results, 
                x='days_to_earnings', 
                y='iv_rank',
                size='current_price',
                hover_data=['symbol', 'company'],
                title="Earnings Timeline vs IV Rank",
                labels={'days_to_earnings': 'Days to Earnings', 'iv_rank': 'IV Rank (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Earnings by day
            earnings_by_day = earnings_results.groupby('days_to_earnings').size().reset_index(name='count')
            
            fig = px.bar(
                earnings_by_day, 
                x='days_to_earnings', 
                y='count',
                title="Earnings by Day",
                labels={'days_to_earnings': 'Days to Earnings', 'count': 'Number of Companies'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            csv = earnings_results.to_csv(index=False)
            st.download_button(
                label="Download Earnings Results (CSV)",
                data=csv,
                file_name=f"earnings_screener_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("No earnings found in the specified timeframe.")
    
    with tab3:
        st.subheader("Unusual Volume Stocks")
        st.markdown(f"Showing stocks with volume ≥ {min_volume_ratio}x average")
        
        # Run the unusual volume screener
        with st.spinner("Screening unusual volume..."):
            volume_results = screen_unusual_volume(
                min_volume_multiplier=min_volume_ratio,
                min_price=min_price
            )
        
        if not volume_results.empty:
            # Display results
            st.dataframe(
                volume_results[['symbol', 'price', 'volume_ratio', 'iv_rank']],
                use_container_width=True
            )
            
            # Create volume ratio chart
            fig = px.scatter(
                volume_results, 
                x='volume_ratio', 
                y='iv_rank',
                size='price',
                hover_data=['symbol'],
                title="Volume Ratio vs IV Rank",
                labels={'volume_ratio': 'Volume Ratio', 'iv_rank': 'IV Rank (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 10 volume stocks
            st.subheader("Top 10 Unusual Volume Stocks")
            top_10_volume = volume_results.head(10)
            
            # Create bar chart
            fig = px.bar(
                top_10_volume, 
                x='symbol', 
                y='volume_ratio',
                title="Top 10 Unusual Volume Stocks",
                labels={'volume_ratio': 'Volume Ratio', 'symbol': 'Symbol'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            csv = volume_results.to_csv(index=False)
            st.download_button(
                label="Download Volume Results (CSV)",
                data=csv,
                file_name=f"unusual_volume_screener_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("No stocks found with unusual volume. Try adjusting the filters.")
    
    # Summary statistics
    st.subheader("Screening Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High IV Rank Stocks", len(ivr_results) if not ivr_results.empty else 0)
    
    with col2:
        st.metric("Earnings Soon", len(earnings_results) if not earnings_results.empty else 0)
    
    with col3:
        st.metric("Unusual Volume", len(volume_results) if not volume_results.empty else 0)
    
    # Screening tips
    st.subheader("Screening Tips")
    
    tips = [
        "High IV Rank indicates options are expensive relative to historical volatility",
        "Stocks with earnings coming up often have elevated IV Rank",
        "Unusual volume can indicate institutional interest or news",
        "Combine multiple screens for better results",
        "Always do your own research before trading",
        "Consider the overall market context when screening"
    ]
    
    for i, tip in enumerate(tips, 1):
        st.write(f"{i}. {tip}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>⚠️ <strong>Disclaimer:</strong> This screener is for educational purposes only. Not investment advice.</p>
        <p>Screen results are based on historical data and may not reflect current market conditions.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_ivr_screener()
