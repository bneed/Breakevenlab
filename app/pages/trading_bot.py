"""
Trading Bot - Low Cap Stock Screener & Trading Recommendations
Focused on $1-10 stocks with high volume and volatility for Robinhood trading
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import yfinance as yf
import requests
from typing import Dict, List, Tuple, Optional

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.data import data_manager
from core.screeners import StockScreener
from tradescrubber.core.indicators import add_indicators
from tradescrubber.core.signals import compute_signals
from tradescrubber.core.ranker import score_ticker

class LowCapTradingBot:
    """Trading bot specialized for low to mid cap stocks with high volatility"""
    
    def __init__(self):
        self.data_manager = data_manager
        self.screener = StockScreener()
        
        # Low cap stock universe - Curated list of truly active, liquid stocks
        self.low_cap_tickers = [
            # Biotech/Pharma (Active & Liquid)
            'OCGN', 'BNTX', 'NVAX', 'INO', 'VXRT', 'CODX', 'AXSM', 'SRPT',
            'FOLD', 'ARCT', 'INO', 'VXRT', 'CODX', 'AXSM', 'SRPT', 'FOLD',
            'ABBV', 'GILD', 'AMGN', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'BNTX',
            
            # Tech/Small Cap (Active & Liquid)
            'PLTR', 'SOFI', 'AFRM', 'UPST', 'LC', 'HOOD', 'COIN', 'RBLX', 'DOCU', 'SNOW',
            'ZM', 'UBER', 'LYFT', 'SNAP', 'PINS', 'ROKU', 'SPOT',
            'CRWD', 'OKTA', 'ZM', 'DOCU', 'SNOW', 'PLTR', 'RBLX', 'COIN', 'HOOD',
            'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'ADI', 'MRVL', 'KLAC',
            
            # Energy/Resources (Active & Liquid)
            'FCEL', 'PLUG', 'BLDP', 'BE', 'RUN', 'SPWR', 'CSIQ', 'JKS', 'ENPH',
            'NEE', 'SEDG', 'FSLR', 'ICLN', 'PBW', 'QCLN', 'TAN', 'SMOG', 'ACES', 'ERTH',
            'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'F', 'GM',
            
            # Financial Services (Active & Liquid)
            'SOFI', 'AFRM', 'UPST', 'LC', 'HOOD', 'COIN', 'PYPL', 'V', 'MA',
            'AXP', 'SYF', 'COF', 'FITB', 'HBAN', 'KEY', 'RF', 'STI', 'USB',
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP',
            
            # Healthcare (Active & Liquid)
            'OCGN', 'BNTX', 'NVAX', 'INO', 'VXRT', 'CODX', 'AXSM', 'SRPT',
            'FOLD', 'ARCT', 'INO', 'VXRT', 'CODX', 'AXSM', 'SRPT', 'FOLD',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
            
            # Retail/Consumer (Active & Liquid)
            'GME', 'AMC', 'BB', 'NOK', 'BBBY', 'KOSS', 'SNDL', 'TLRY',
            'CGC', 'ACB', 'CRON', 'OGI', 'VFF', 'GTBIF', 'TCNNF', 'CURLF',
            'WMT', 'TGT', 'COST', 'HD', 'LOW', 'MCD', 'SBUX', 'NKE', 'DIS',
            
            # Crypto/Blockchain (Active & Liquid)
            'COIN', 'HOOD', 'PYPL', 'V', 'MA', 'AXP', 'SYF', 'COF',
            'MSTR', 'RIOT', 'MARA', 'HUT', 'BITF', 'CAN', 'ARB', 'HIVE', 'BTBT', 'EBON',
            
            # EV/Transportation (Active & Liquid)
            'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'F', 'GM',
            'WKHS', 'HYLN', 'QS', 'BLNK', 'CHPT', 'EVGO', 'LEV',
            'F', 'GM', 'FORD', 'TM', 'HMC', 'NIO', 'XPEV', 'LI', 'RIVN',
            
            # Meme Stocks (Active & Liquid)
            'GME', 'AMC', 'BB', 'NOK', 'BBBY', 'KOSS', 'SNDL', 'TLRY',
            'CGC', 'ACB', 'CRON', 'OGI', 'VFF', 'GTBIF', 'TCNNF', 'CURLF',
            'SPCE', 'WKHS', 'CLOV', 'CLNE', 'BBIG', 'ATER',
            
            # Growth Stocks (Active & Liquid)
            'PLTR', 'SOFI', 'AFRM', 'UPST', 'LC', 'HOOD', 'COIN', 'RBLX', 'DOCU', 'SNOW',
            'ZM', 'UBER', 'LYFT', 'SNAP', 'PINS', 'ROKU', 'SPOT',
            
            # Additional Active Low-Cap Stocks
            'SPCE', 'CLOV', 'CLNE', 'BBIG', 'ATER', 'BGFV', 'BB', 'NOK', 'BBBY', 'KOSS', 'SNDL', 'TLRY', 'CGC', 'ACB',
            'CRON', 'OGI', 'VFF', 'GTBIF', 'TCNNF', 'CURLF', 'SPCE', 'CLOV',
            'CLNE', 'BBIG', 'ATER', 'BGFV',
            
            # More Low-Cap Stocks
            'SNDL', 'TLRY', 'CGC', 'ACB', 'CRON', 'OGI', 'VFF', 'GTBIF', 'TCNNF', 'CURLF',
            'SPCE', 'WKHS', 'CLOV', 'CLNE', 'BBIG', 'ATER',
            'BGFV', 'BB', 'NOK', 'BBBY', 'KOSS', 'SNDL', 'TLRY',
            'CGC', 'ACB', 'CRON', 'OGI', 'VFF', 'GTBIF', 'TCNNF', 'CURLF',
        ]
    
    def get_market_cap(self, symbol: str) -> Optional[float]:
        """Get market cap for a symbol"""
        return self.data_manager.get_market_cap(symbol)
    
    def screen_low_cap_stocks(self, 
                            min_price: float = 1.0, 
                            max_price: float = 10.0,
                            min_market_cap: float = 100_000_000,  # 100M
                            max_market_cap: float = 10_000_000_000,  # 10B
                            min_volume_ratio: float = 1.5,
                            min_volatility: float = 0.02) -> pd.DataFrame:
        """
        Screen for low to mid cap stocks with high volume and volatility
        
        Args:
            min_price: Minimum stock price
            max_price: Maximum stock price
            min_market_cap: Minimum market cap
            max_market_cap: Maximum market cap
            min_volume_ratio: Minimum volume vs average
            min_volatility: Minimum daily volatility
            
        Returns:
            DataFrame with screening results
        """
        results = []
        
        # Progress bar
        progress_bar = st.progress(0)
        total_tickers = len(self.low_cap_tickers)
        
        for i, ticker in enumerate(self.low_cap_tickers):
            try:
                # Update progress
                progress_bar.progress((i + 1) / total_tickers)
                
                # Get stock data
                data = self.data_manager.get_stock_data(ticker, "1d")
                if data.empty or len(data) < 30:
                    continue
                
                current_price = data['Close'].iloc[-1]
                
                # Check price filters
                if current_price < min_price or current_price > max_price:
                    continue
                
                # Get market cap
                market_cap = self.get_market_cap(ticker)
                if market_cap is None or market_cap < min_market_cap or market_cap > max_market_cap:
                    continue
                
                # Calculate volume metrics
                volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].mean()
                volume_ratio = volume / avg_volume if avg_volume > 0 else 0
                
                if volume_ratio < min_volume_ratio:
                    continue
                
                # Calculate volatility (20-day)
                returns = data['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized
                
                if volatility < min_volatility:
                    continue
                
                # Calculate technical indicators
                df_with_indicators = add_indicators(data)
                df_with_signals = compute_signals(df_with_indicators)
                
                # Score the stock
                score_data = score_ticker(df_with_signals)
                
                # Calculate additional metrics
                price_change_1d = (current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
                price_change_5d = (current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6] * 100 if len(data) >= 6 else 0
                
                # Calculate ATR for stop loss
                atr = df_with_indicators['atr14'].iloc[-1] if 'atr14' in df_with_indicators.columns else 0
                
                # Generate trading recommendations
                recommendations = self._generate_trading_recommendations(
                    current_price, atr, score_data, df_with_signals.iloc[-1]
                )
                
                results.append({
                    'symbol': ticker,
                    'price': current_price,
                    'market_cap': market_cap,
                    'market_cap_billions': market_cap / 1_000_000_000,
                    'volume': volume,
                    'avg_volume': avg_volume,
                    'volume_ratio': volume_ratio,
                    'volatility': volatility,
                    'price_change_1d': price_change_1d,
                    'price_change_5d': price_change_5d,
                    'score': score_data.get('score', 0),
                    'direction': score_data.get('direction', 'neutral'),
                    'confidence': score_data.get('confidence', 0),
                    'reasons': ', '.join(score_data.get('reasons', [])),
                    'buy_price': recommendations['buy_price'],
                    'sell_price': recommendations['sell_price'],
                    'stop_loss': recommendations['stop_loss'],
                    'risk_reward': recommendations['risk_reward'],
                    'recommendation': recommendations['recommendation'],
                    'rsi': df_with_signals['rsi14'].iloc[-1] if 'rsi14' in df_with_signals.columns else 50,
                    'macd_signal': df_with_signals['macd_bullish'].iloc[-1] if 'macd_bullish' in df_with_signals.columns else False,
                    'volume_spike': df_with_signals['volume_spike'].iloc[-1] if 'volume_spike' in df_with_signals.columns else False,
                })
                
            except Exception as e:
                st.error(f"Error processing {ticker}: {str(e)}")
                continue
        
        progress_bar.empty()
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('score', ascending=False)
        
        return df
    
    def _generate_trading_recommendations(self, 
                                        current_price: float, 
                                        atr: float, 
                                        score_data: Dict, 
                                        latest_signals: pd.Series) -> Dict:
        """Generate specific trading recommendations"""
        
        score = score_data.get('score', 0)
        direction = score_data.get('direction', 'neutral')
        confidence = score_data.get('confidence', 0)
        
        # Base recommendations on score and direction
        if score >= 80 and direction == 'up':
            recommendation = "STRONG BUY"
            buy_price = current_price * 0.98  # 2% below current for entry
            sell_price = current_price * 1.15  # 15% target
            stop_loss = current_price * 0.92  # 8% stop loss
        elif score >= 70 and direction == 'up':
            recommendation = "BUY"
            buy_price = current_price * 0.99  # 1% below current
            sell_price = current_price * 1.10  # 10% target
            stop_loss = current_price * 0.94  # 6% stop loss
        elif score >= 60 and direction == 'up':
            recommendation = "WEAK BUY"
            buy_price = current_price * 0.995  # 0.5% below current
            sell_price = current_price * 1.08  # 8% target
            stop_loss = current_price * 0.95  # 5% stop loss
        elif score >= 70 and direction == 'down':
            recommendation = "SELL"
            buy_price = None
            sell_price = current_price * 0.90  # 10% down target
            stop_loss = current_price * 1.06  # 6% up stop loss
        else:
            recommendation = "HOLD"
            buy_price = None
            sell_price = None
            stop_loss = None
        
        # Use ATR for more sophisticated stop loss if available
        if atr > 0 and stop_loss is not None:
            if direction == 'up':
                stop_loss = current_price - (atr * 2)
            else:
                stop_loss = current_price + (atr * 2)
        
        # Calculate risk/reward ratio
        risk_reward = None
        if buy_price and sell_price and stop_loss:
            potential_profit = sell_price - buy_price
            potential_loss = buy_price - stop_loss
            if potential_loss > 0:
                risk_reward = potential_profit / potential_loss
        
        return {
            'recommendation': recommendation,
            'buy_price': round(buy_price, 2) if buy_price else None,
            'sell_price': round(sell_price, 2) if sell_price else None,
            'stop_loss': round(stop_loss, 2) if stop_loss else None,
            'risk_reward': round(risk_reward, 2) if risk_reward else None
        }
    
    def get_top_recommendations(self, df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
        """Get top trading recommendations"""
        if df.empty:
            return df
        
        # Filter for actionable recommendations (if recommendation column exists)
        if 'recommendation' in df.columns:
            actionable = df[df['recommendation'].isin(['STRONG BUY', 'BUY', 'WEAK BUY'])]
            if not actionable.empty:
                return actionable.head(limit)
        
        # Fallback: return top stocks by score or just top N
        if 'score' in df.columns:
            return df.head(limit)
        else:
            return df.head(limit)

def show_trading_bot():
    """Display the trading bot page"""
    
    st.title("🤖 Low Cap Trading Bot")
    st.markdown("**Automated screener for $1-10 stocks with high volume and volatility**")
    st.markdown("*Perfect for Robinhood trading - finds the movers before they move*")
    
    # Initialize bot
    if 'trading_bot' not in st.session_state:
        st.session_state.trading_bot = LowCapTradingBot()
    
    bot = st.session_state.trading_bot
    
    # Sidebar for screening parameters
    with st.sidebar:
        st.header("🎯 Screening Parameters")
        
        # Price filters
        st.subheader("Price Range")
        min_price = st.number_input(
            "Minimum Price ($)", 
            min_value=0.01, 
            value=1.0,
            step=0.01,
            help="Minimum stock price"
        )
        
        max_price = st.number_input(
            "Maximum Price ($)", 
            min_value=0.01, 
            value=10.0,
            step=0.01,
            help="Maximum stock price"
        )
        
        # Market cap filters
        st.subheader("Market Cap Range")
        min_market_cap = st.number_input(
            "Min Market Cap (B)", 
            min_value=0.01, 
            value=0.1,
            step=0.01,
            help="Minimum market cap in billions"
        ) * 1_000_000_000
        
        max_market_cap = st.number_input(
            "Max Market Cap (B)", 
            min_value=0.1, 
            value=2.0,
            step=0.1,
            help="Maximum market cap in billions"
        ) * 1_000_000_000
        
        # Volume and volatility filters
        st.subheader("Activity Filters")
        min_volume_ratio = st.slider(
            "Minimum Volume Ratio", 
            min_value=1.0, 
            max_value=10.0, 
            value=2.0, 
            step=0.1,
            help="Volume vs 30-day average"
        )
        
        min_volatility = st.slider(
            "Minimum Volatility", 
            min_value=0.01, 
            max_value=0.50, 
            value=0.15, 
            step=0.01,
            help="Minimum daily volatility"
        )
        
        # Score filter
        st.subheader("Quality Filter")
        min_score = st.slider(
            "Minimum Score", 
            min_value=0.0, 
            max_value=10.0, 
            value=1.0, 
            step=0.1,
            help="Minimum trading score"
        )
        
        # Refresh button
        if st.button("🔄 Scan Market", type="primary"):
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Top Picks", "📈 All Results", "📋 Watchlist", "📚 Strategy Guide"])
    
    with tab1:
        st.subheader("🎯 Top Trading Recommendations")
        st.markdown("**Best opportunities for Robinhood trading**")
        
        # Run the screener
        with st.spinner("Scanning low cap stocks..."):
            from core.screeners import screen_low_cap_movers
            results = screen_low_cap_movers(
                min_price=min_price,
                max_price=max_price,
                min_market_cap=min_market_cap,
                max_market_cap=max_market_cap,
                min_volume_ratio=min_volume_ratio,
                min_volatility=min_volatility
            )
            
            # Debug: Show what columns we have
            if not results.empty:
                st.write(f"Debug: Found {len(results)} stocks")
                st.write(f"Debug: Columns: {list(results.columns)}")
                
                # Force add missing columns if they don't exist
                if 'score' not in results.columns:
                    st.write("Debug: Adding missing score column")
                    results['score'] = (
                        results['volume_ratio'] * 0.4 +
                        results['volatility'] * 10 * 0.3 +
                        abs(results['price_change_1d']) * 0.2 +
                        abs(results['price_change_5d']) * 0.1
                    )
                
                if 'recommendation' not in results.columns:
                    st.write("Debug: Adding missing recommendation column")
                    results['recommendation'] = 'HOLD'
                    results.loc[results['score'] >= 3.0, 'recommendation'] = 'STRONG BUY'
                    results.loc[(results['score'] >= 2.0) & (results['score'] < 3.0), 'recommendation'] = 'BUY'
                    results.loc[(results['score'] >= 1.5) & (results['score'] < 2.0), 'recommendation'] = 'WEAK BUY'
                    results.loc[(results['score'] >= 1.0) & (results['score'] < 1.5), 'recommendation'] = 'HOLD'
                    results.loc[results['score'] < 1.0, 'recommendation'] = 'WEAK SELL'
                
                if 'buy_price' not in results.columns:
                    st.write("Debug: Adding missing price target columns")
                    results['buy_price'] = results['price'] * 0.98
                    results['sell_price'] = results['price'] * 1.15
                    results['stop_loss'] = results['price'] * 0.90
                
                st.write(f"Debug: After adding columns - Columns: {list(results.columns)}")
            else:
                st.write("Debug: No results found")
        
        # Filter by minimum score (if score column exists)
        if not results.empty and 'score' in results.columns:
            results = results[results['score'] >= min_score]
        
        if not results.empty:
            # Debug: Check if we have the required columns
            st.write(f"Debug: Before get_top_recommendations - Columns: {list(results.columns)}")
            
            # Get top recommendations
            top_picks = bot.get_top_recommendations(results, limit=10)
            
            # Display top picks
            for idx, (_, stock) in enumerate(top_picks.iterrows(), 1):
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                    
                    with col1:
                        st.markdown(f"**#{idx} {stock['symbol']}**")
                        st.markdown(f"${stock['price']:.2f}")
                        st.markdown(f"Score: {stock['score']:.0f}")
                    
                    with col2:
                        st.markdown(f"**{stock['recommendation']}**")
                        if stock['buy_price']:
                            st.markdown(f"Buy: ${stock['buy_price']:.2f}")
                        if stock['sell_price']:
                            st.markdown(f"Sell: ${stock['sell_price']:.2f}")
                    
                    with col3:
                        st.markdown(f"Vol: {stock['volume_ratio']:.1f}x")
                        st.markdown(f"Volatility: {stock['volatility']:.1%}")
                        st.markdown(f"1D: {stock['price_change_1d']:+.1f}%")
                    
                    with col4:
                        if stock['stop_loss']:
                            st.markdown(f"Stop: ${stock['stop_loss']:.2f}")
                        if stock['risk_reward']:
                            st.markdown(f"R/R: {stock['risk_reward']:.1f}")
                        st.markdown(f"MCap: ${stock['market_cap_billions']:.1f}B")
                    
                    # Reasons
                    if stock['reasons']:
                        st.markdown(f"*{stock['reasons']}*")
                    
                    st.markdown("---")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Picks", len(top_picks))
            
            with col2:
                strong_buys = len(top_picks[top_picks['recommendation'] == 'STRONG BUY'])
                st.metric("Strong Buys", strong_buys)
            
            with col3:
                avg_score = top_picks['score'].mean()
                st.metric("Avg Score", f"{avg_score:.0f}")
            
            with col4:
                avg_vol = top_picks['volume_ratio'].mean()
                st.metric("Avg Volume", f"{avg_vol:.1f}x")
        
        else:
            st.warning("No stocks found matching your criteria. Try adjusting the filters.")
    
    with tab2:
        st.subheader("📈 All Screening Results")
        st.markdown("**Complete list of screened stocks**")
        
        if not results.empty:
            # Display results table
            display_cols = [
                'symbol', 'price', 'market_cap_billions', 'volume_ratio', 
                'volatility', 'score', 'recommendation', 'buy_price', 
                'sell_price', 'price_change_1d', 'price_change_5d'
            ]
            
            st.dataframe(
                results[display_cols],
                use_container_width=True,
                column_config={
                    'price': st.column_config.NumberColumn('Price', format='$%.2f'),
                    'market_cap_billions': st.column_config.NumberColumn('Market Cap (B)', format='$%.1fB'),
                    'volume_ratio': st.column_config.NumberColumn('Volume Ratio', format='%.1fx'),
                    'volatility': st.column_config.NumberColumn('Volatility', format='%.1%'),
                    'score': st.column_config.NumberColumn('Score', format='%.0f'),
                    'buy_price': st.column_config.NumberColumn('Buy Price', format='$%.2f'),
                    'sell_price': st.column_config.NumberColumn('Sell Price', format='$%.2f'),
                    'price_change_1d': st.column_config.NumberColumn('1D Change', format='%+.1f%%'),
                    'price_change_5d': st.column_config.NumberColumn('5D Change', format='%+.1f%%'),
                }
            )
            
            # Download results
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"trading_bot_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution
                fig = px.histogram(
                    results, 
                    x='score', 
                    nbins=20,
                    title="Score Distribution",
                    labels={'score': 'Trading Score', 'count': 'Number of Stocks'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Volume vs Volatility
                fig = px.scatter(
                    results, 
                    x='volume_ratio', 
                    y='volatility',
                    size='score',
                    color='recommendation',
                    hover_data=['symbol', 'price'],
                    title="Volume vs Volatility",
                    labels={'volume_ratio': 'Volume Ratio', 'volatility': 'Volatility'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("No stocks found matching your criteria.")
    
    with tab3:
        st.subheader("📋 Watchlist Management")
        st.markdown("**Track your favorite picks**")
        
        # Initialize watchlist in session state
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = []
        
        # Add to watchlist
        if not results.empty:
            st.markdown("**Add to Watchlist:**")
            selected_symbols = st.multiselect(
                "Select stocks to add to watchlist:",
                options=results['symbol'].tolist(),
                default=[]
            )
            
            if st.button("➕ Add Selected"):
                for symbol in selected_symbols:
                    if symbol not in st.session_state.watchlist:
                        st.session_state.watchlist.append(symbol)
                        st.success(f"Added {symbol} to watchlist")
        
        # Display watchlist
        if st.session_state.watchlist:
            st.markdown("**Your Watchlist:**")
            
            for symbol in st.session_state.watchlist:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(symbol)
                
                with col2:
                    if st.button(f"📊", key=f"chart_{symbol}"):
                        # Show chart for this symbol
                        st.session_state[f"show_chart_{symbol}"] = True
                
                with col3:
                    if st.button(f"❌", key=f"remove_{symbol}"):
                        st.session_state.watchlist.remove(symbol)
                        st.rerun()
            
            # Remove all button
            if st.button("🗑️ Clear Watchlist"):
                st.session_state.watchlist = []
                st.rerun()
        
        else:
            st.info("Your watchlist is empty. Add some stocks from the screening results!")
    
    with tab4:
        st.subheader("📚 Trading Strategy Guide")
        st.markdown("**How to use this bot effectively**")
        
        # Strategy sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 **Screening Strategy**")
            st.markdown("""
            **Target Stocks:**
            - Price: $1-10 (your sweet spot)
            - Market Cap: $100M - $10B (low to mid cap)
            - Volume: 1.5x+ average (institutional interest)
            - Volatility: 2%+ daily (high movement potential)
            
            **Why These Work:**
            - Low cap = less analyst coverage = more opportunity
            - High volume = liquidity for entry/exit
            - High volatility = bigger moves = more profit potential
            """)
            
            st.markdown("### 📊 **Reading the Scores**")
            st.markdown("""
            **Score Breakdown:**
            - 80-100: STRONG BUY (rare, high conviction)
            - 70-79: BUY (good opportunity)
            - 60-69: WEAK BUY (watch closely)
            - Below 60: HOLD/AVOID
            
            **Key Indicators:**
            - RSI: Oversold (<30) or Overbought (>70)
            - MACD: Bullish crossover signals
            - Volume: Spikes indicate interest
            - Moving Averages: Trend direction
            """)
        
        with col2:
            st.markdown("### 💰 **Risk Management**")
            st.markdown("""
            **Position Sizing:**
            - Never risk more than 2% of account per trade
            - Use stop losses religiously
            - Take profits at targets
            
            **Entry Strategy:**
            - Buy 2% below current price for better entry
            - Scale in if price drops further
            - Don't chase if it gaps up
            
            **Exit Strategy:**
            - Take 50% profit at first target
            - Let winners run with trailing stops
            - Cut losses quickly at stop loss
            """)
            
            st.markdown("### 🚀 **Robinhood Tips**")
            st.markdown("""
            **Best Practices:**
            - Use limit orders, not market orders
            - Set alerts for price targets
            - Trade during market hours for best fills
            - Keep some cash for opportunities
            
            **Common Mistakes:**
            - FOMO buying after big moves
            - Not using stop losses
            - Overtrading (too many positions)
            - Ignoring market conditions
            """)
        
        # Quick reference
        st.markdown("### ⚡ **Quick Reference**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Strong Buy Signals:**")
            st.markdown("""
            - Score > 80
            - RSI < 30 (oversold)
            - Volume spike
            - MACD bullish
            - Price above SMA50
            """)
        
        with col2:
            st.markdown("**Sell Signals:**")
            st.markdown("""
            - Score < 60
            - RSI > 70 (overbought)
            - Volume drying up
            - MACD bearish
            - Price below SMA50
            """)
        
        with col3:
            st.markdown("**Risk Levels:**")
            st.markdown("""
            - **Low Risk:** Score > 80, strong trends
            - **Medium Risk:** Score 60-80, mixed signals
            - **High Risk:** Score < 60, weak signals
            - **Avoid:** No clear direction
            """)

if __name__ == "__main__":
    show_trading_bot()
