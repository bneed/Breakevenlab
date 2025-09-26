import streamlit as st
import sys
import os

# Add the parent directory to the path so we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.auth import check_user_subscription, get_user_tier
from core.data import get_top_ivr_tickers

# Page configuration
st.set_page_config(
    page_title="Break-even Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .pro-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .free-badge {
        background-color: #4ecdc4;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Break-even Lab</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional-grade trading and finance tools</p>', unsafe_allow_html=True)
    
    # Sidebar for user info and navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=Break-even+Lab", width=200)
        
        # User authentication status
        user_tier = get_user_tier()
        if user_tier == "free":
            st.markdown('<span class="free-badge">FREE TIER</span>', unsafe_allow_html=True)
        elif user_tier == "pro":
            st.markdown('<span class="pro-badge">PRO TIER</span>', unsafe_allow_html=True)
        elif user_tier == "founder":
            st.markdown('<span class="pro-badge">FOUNDER</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("## 🧭 Navigation")
        page = st.selectbox(
            "Choose a tool:",
            [
                "🏠 Home",
                "📈 Break-even Calculator",
                "⚖️ Position Sizing",
                "📊 Greeks Viewer",
                "🔍 IV Rank Screener",
                "📰 Earnings Sentiment",
                "📉 Backtest Lite",
                "🎯 Strategy Planner",
                "💼 Portfolio Dashboard"
            ]
        )
        
        st.markdown("---")
        
        # Live data widget
        st.markdown("## 📊 Top IVR Tickers Today")
        try:
            top_tickers = get_top_ivr_tickers(limit=5)
            for ticker, ivr in top_tickers:
                st.write(f"**{ticker}**: {ivr:.1f}% IVR")
        except:
            st.write("Data loading...")
    
    # Main content based on selection
    if page == "🏠 Home":
        show_home_page()
    elif page == "📈 Break-even Calculator":
        from pages.breakeven_calculator import show_breakeven_calculator
        show_breakeven_calculator()
    elif page == "⚖️ Position Sizing":
        from pages.position_sizing import show_position_sizing
        show_position_sizing()
    elif page == "📊 Greeks Viewer":
        from pages.greeks_viewer import show_greeks_viewer
        show_greeks_viewer()
    elif page == "🔍 IV Rank Screener":
        from pages.ivr_screener import show_ivr_screener
        show_ivr_screener()
    elif page == "📰 Earnings Sentiment":
        from pages.earnings_sentiment import show_earnings_sentiment
        show_earnings_sentiment()
    elif page == "📉 Backtest Lite":
        from pages.backtest_lite import show_backtest_lite
        show_backtest_lite()
    elif page == "🎯 Strategy Planner":
        from pages.strategy_planner import show_strategy_planner
        show_strategy_planner()
    elif page == "💼 Portfolio Dashboard":
        from pages.portfolio_dashboard import show_portfolio_dashboard
        show_portfolio_dashboard()

def show_home_page():
    """Display the home page with feature overview"""
    
    # Hero section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Break-even Calculator</h3>
            <p>Multi-leg options P/L analysis with Greeks</p>
            <span class="free-badge">FREE</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚖️ Position Sizing</h3>
            <p>Kelly fraction, risk management tools</p>
            <span class="free-badge">FREE</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Greeks Snapshot</h3>
            <p>Delta, Gamma, Theta, Vega, Rho analysis</p>
            <span class="free-badge">FREE</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Pro features section
    st.markdown("---")
    st.markdown("## 🚀 Pro Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📰 Earnings Sentiment</h3>
            <p>AI-powered transcript analysis</p>
            <span class="pro-badge">PRO</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📉 Backtest Lite</h3>
            <p>Strategy backtesting with daily bars</p>
            <span class="pro-badge">PRO</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Strategy Planner</h3>
            <p>Wheel strategy simulation</p>
            <span class="pro-badge">PRO</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Pricing section
    st.markdown("---")
    st.markdown("## 💰 Pricing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🆓 Free Tier
        - Break-even Calculator
        - Position Sizing
        - Greeks Viewer
        - Basic IVR Screener
        - 1 Alert per day
        """)
        
    with col2:
        st.markdown("""
        ### ⭐ Pro Tier
        **$12-19/month**
        - All Free features
        - Unlimited screeners/alerts
        - Backtest Lite
        - Earnings Sentiment
        - Strategy Planner
        - Portfolio Dashboard
        """)
        
    with col3:
        st.markdown("""
        ### 🏆 Founders Plan
        **$5/month** (Early Adopter)
        - All Pro features
        - Lifetime access
        - Priority support
        - Beta features
        """)
    
    # CTA buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start with Free Tools", use_container_width=True):
            st.switch_page("pages/breakeven_calculator.py")
    
    with col2:
        if st.button("⭐ Upgrade to Pro", use_container_width=True):
            st.info("Pro upgrade coming soon!")
    
    with col3:
        if st.button("📧 Get Daily Digest", use_container_width=True):
            st.info("Email signup coming soon!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. Not investment advice.</p>
        <p>© 2024 Break-even Lab. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
