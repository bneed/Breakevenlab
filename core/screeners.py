"""
Screening tools for Break-even Lab
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st
from .data import data_manager

class StockScreener:
    """Stock screening functionality"""
    
    def __init__(self):
        self.data_manager = data_manager
    
    def screen_high_iv_rank(self, min_ivr: float = 50.0, max_price: float = 1000.0, 
                           min_price: float = 5.0, sectors: List[str] = None) -> pd.DataFrame:
        """
        Screen stocks with high IV Rank
        
        Args:
            min_ivr: Minimum IV Rank percentage
            max_price: Maximum stock price
            min_price: Minimum stock price
            sectors: List of sectors to filter by
        
        Returns:
            DataFrame with screening results
        """
        # Popular tickers to screen
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC',
            'SPY', 'QQQ', 'IWM', 'VIX', 'TLT', 'GLD', 'SLV', 'BTC-USD', 'ETH-USD',
            'CRM', 'ADBE', 'ORCL', 'CSCO', 'IBM', 'UBER', 'LYFT', 'SQ', 'PYPL', 'ZM',
            'DOCU', 'SNOW', 'PLTR', 'RBLX', 'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'LC'
        ]
        
        results = []
        
        for ticker in tickers:
            try:
                # Get current price
                data = self.data_manager.get_stock_data(ticker, "1d")
                if data.empty:
                    continue
                
                current_price = data['Close'].iloc[-1]
                
                # Check price filters
                if current_price < min_price or current_price > max_price:
                    continue
                
                # Calculate IV Rank
                ivr = self.data_manager.calculate_iv_rank(ticker)
                
                if ivr >= min_ivr:
                    # Get additional data
                    volume = data['Volume'].iloc[-1]
                    avg_volume = data['Volume'].mean()
                    
                    results.append({
                        'symbol': ticker,
                        'price': current_price,
                        'iv_rank': ivr,
                        'volume': volume,
                        'avg_volume': avg_volume,
                        'volume_ratio': volume / avg_volume if avg_volume > 0 else 0
                    })
                    
            except Exception as e:
                st.error(f"Error screening {ticker}: {str(e)}")
                continue
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('iv_rank', ascending=False)
        
        return df
    
    def screen_earnings_soon(self, days_ahead: int = 7) -> pd.DataFrame:
        """
        Screen stocks with earnings coming up
        
        Args:
            days_ahead: Number of days ahead to look for earnings
        
        Returns:
            DataFrame with earnings data
        """
        earnings_data = self.data_manager.get_earnings_calendar(days_ahead)
        
        if earnings_data.empty:
            return pd.DataFrame()
        
        # Add additional screening criteria
        results = []
        for _, row in earnings_data.iterrows():
            symbol = row['symbol']
            
            try:
                # Get current price and IV Rank
                data = self.data_manager.get_stock_data(symbol, "1d")
                if data.empty:
                    continue
                
                current_price = data['Close'].iloc[-1]
                ivr = self.data_manager.calculate_iv_rank(symbol)
                
                # Calculate days to earnings
                days_to_earnings = (row['earnings_date'] - pd.Timestamp.now()).days
                
                results.append({
                    'symbol': symbol,
                    'company': row['company'],
                    'earnings_date': row['earnings_date'],
                    'days_to_earnings': days_to_earnings,
                    'current_price': current_price,
                    'iv_rank': ivr
                })
                
            except Exception as e:
                st.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('days_to_earnings')
        
        return df
    
    def screen_unusual_volume(self, min_volume_multiplier: float = 2.0, 
                            min_price: float = 5.0) -> pd.DataFrame:
        """
        Screen stocks with unusual volume
        
        Args:
            min_volume_multiplier: Minimum volume multiplier vs average
            min_price: Minimum stock price
        
        Returns:
            DataFrame with unusual volume data
        """
        unusual_volume_data = self.data_manager.get_unusual_volume(min_volume_multiplier)
        
        if unusual_volume_data.empty:
            return pd.DataFrame()
        
        # Filter by minimum price
        filtered_data = unusual_volume_data[unusual_volume_data['price'] >= min_price]
        
        # Add IV Rank for additional context
        results = []
        for _, row in filtered_data.iterrows():
            symbol = row['symbol']
            
            try:
                ivr = self.data_manager.calculate_iv_rank(symbol)
                
                results.append({
                    'symbol': symbol,
                    'price': row['price'],
                    'current_volume': row['current_volume'],
                    'avg_volume': row['avg_volume'],
                    'volume_ratio': row['volume_ratio'],
                    'iv_rank': ivr
                })
                
            except Exception as e:
                st.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('volume_ratio', ascending=False)
        
        return df
    
    def screen_high_theta(self, min_theta: float = 0.1, max_dte: int = 30) -> pd.DataFrame:
        """
        Screen options with high theta (time decay)
        
        Args:
            min_theta: Minimum theta value
            max_dte: Maximum days to expiration
        
        Returns:
            DataFrame with high theta options
        """
        # This is a simplified version - in production you'd use a proper options API
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        results = []
        
        for ticker in tickers:
            try:
                # Get options data
                options_data = self.data_manager.get_options_data(ticker)
                if not options_data:
                    continue
                
                current_price = self.data_manager.get_stock_data(ticker, "1d")['Close'].iloc[-1]
                
                # Process calls and puts
                for option_type in ['calls', 'puts']:
                    if option_type in options_data:
                        options_df = options_data[option_type]
                        
                        for _, option in options_df.iterrows():
                            try:
                                # Calculate theoretical Greeks
                                from .options import OptionsPricer
                                
                                strike = option['strike']
                                bid = option['bid']
                                ask = option['ask']
                                mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                                
                                if mid_price <= 0:
                                    continue
                                
                                # Estimate time to expiration (simplified)
                                dte = 7  # Assume 7 days for now
                                if dte > max_dte:
                                    continue
                                
                                # Calculate Greeks
                                greeks = OptionsPricer.calculate_greeks(
                                    current_price, strike, dte/365, 0.05, 0.3, option_type[:-1]
                                )
                                
                                if greeks['theta'] >= min_theta:
                                    results.append({
                                        'symbol': ticker,
                                        'option_type': option_type[:-1],
                                        'strike': strike,
                                        'price': mid_price,
                                        'dte': dte,
                                        'theta': greeks['theta'],
                                        'delta': greeks['delta'],
                                        'gamma': greeks['gamma'],
                                        'vega': greeks['vega']
                                    })
                                    
                            except Exception as e:
                                continue
                                
            except Exception as e:
                st.error(f"Error screening {ticker}: {str(e)}")
                continue
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('theta', ascending=False)
        
        return df
    
    def screen_low_cap_movers(self, 
                            min_price: float = 20.0, 
                            max_price: float = 50.0,
                            min_market_cap: float = 1_000_000_000,
                            max_market_cap: float = 50_000_000_000,
                            min_volume_ratio: float = 0.5,
                            min_volatility: float = 0.01) -> pd.DataFrame:
        """
        Screen for low cap stocks with high movement potential
        
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
        # Focused list of known active mid-cap stocks
        low_cap_tickers = [
            # Tech/Fintech (Known Active)
            'SOFI', 'PLTR', 'HOOD', 'COIN', 'RBLX', 'DOCU', 'SNOW', 'ZM', 'UBER', 'LYFT',
            'SNAP', 'PINS', 'ROKU', 'SPOT', 'CRWD', 'OKTA', 'AFRM', 'UPST', 'LC',
            
            # EV/Transportation (Known Active)
            'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'F', 'GM', 'WKHS', 'CHPT', 'EVGO',
            
            # Energy/Clean Energy (Known Active)
            'FCEL', 'PLUG', 'RUN', 'SPWR', 'ENPH', 'NEE', 'SEDG', 'FSLR',
            
            # Healthcare/Biotech (Known Active)
            'OCGN', 'BNTX', 'NVAX', 'INO', 'VXRT', 'CODX', 'AXSM', 'SRPT', 'FOLD', 'ARCT',
            'ABBV', 'GILD', 'AMGN', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA',
            
            # Financial Services (Known Active)
            'PYPL', 'V', 'MA', 'AXP', 'SYF', 'COF', 'FITB', 'HBAN', 'KEY', 'RF',
            
            # Retail/Consumer (Known Active)
            'GME', 'AMC', 'BB', 'NOK', 'BBBY', 'KOSS', 'SNDL', 'TLRY', 'CGC', 'ACB',
            'CRON', 'OGI', 'VFF', 'GTBIF', 'TCNNF', 'CURLF', 'SPCE', 'CLOV', 'CLNE',
            
            # Crypto/Blockchain (Known Active)
            'MSTR', 'RIOT', 'MARA', 'HUT', 'BITF', 'CAN', 'ARB', 'HIVE', 'BTBT', 'EBON',
            
            # Growth/Cloud (Known Active)
            'ZM', 'DOCU', 'SNOW', 'PLTR', 'RBLX', 'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST',
        ]
        
        results = []
        
        for ticker in low_cap_tickers:
            try:
                # Get current price
                data = self.data_manager.get_stock_data(ticker, "1mo")  # Get 1 month of data
                if data.empty or len(data) < 5:  # Reduced requirement to 5 days
                    # Skip stocks with insufficient data
                    continue
                
                current_price = data['Close'].iloc[-1]
                
                # Check price filters
                if current_price < min_price or current_price > max_price:
                    continue
                
                # Get market cap
                market_cap = self.data_manager.get_market_cap(ticker)
                if market_cap is None or market_cap <= 0 or market_cap < min_market_cap or market_cap > max_market_cap:
                    # Skip stocks with invalid or out-of-range market cap
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
                
                # Calculate additional metrics
                price_change_1d = (current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
                price_change_5d = (current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6] * 100 if len(data) >= 6 else 0
                
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
                    'iv_rank': self.data_manager.calculate_iv_rank(ticker)
                })
                
            except Exception as e:
                # Only show errors for debugging, don't spam the UI
                # st.error(f"Error screening {ticker}: {str(e)}")
                continue
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('volume_ratio', ascending=False)
        
        return df

# Global screener instance
screener = StockScreener()

def screen_high_iv_rank(min_ivr: float = 50.0, max_price: float = 1000.0, 
                       min_price: float = 5.0, sectors: List[str] = None) -> pd.DataFrame:
    """Screen stocks with high IV Rank"""
    return screener.screen_high_iv_rank(min_ivr, max_price, min_price, sectors)

def screen_earnings_soon(days_ahead: int = 7) -> pd.DataFrame:
    """Screen stocks with earnings coming up"""
    return screener.screen_earnings_soon(days_ahead)

def screen_unusual_volume(min_volume_multiplier: float = 2.0, 
                        min_price: float = 5.0) -> pd.DataFrame:
    """Screen stocks with unusual volume"""
    return screener.screen_unusual_volume(min_volume_multiplier, min_price)

def screen_high_theta(min_theta: float = 0.1, max_dte: int = 30) -> pd.DataFrame:
    """Screen options with high theta"""
    return screener.screen_high_theta(min_theta, max_dte)

def screen_low_cap_movers(min_price: float = 20.0, 
                        max_price: float = 50.0,
                        min_market_cap: float = 1_000_000_000,
                        max_market_cap: float = 50_000_000_000,
                        min_volume_ratio: float = 0.5,
                        min_volatility: float = 0.01) -> pd.DataFrame:
    """Screen for low cap stocks with high movement potential"""
    return screener.screen_low_cap_movers(
        min_price, max_price, min_market_cap, 
        max_market_cap, min_volume_ratio, min_volatility
    )