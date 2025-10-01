"""
Data fetching and management for Break-even Lab
Supports multiple data sources: yfinance, Alpha Vantage, Finnhub
"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from typing import List, Tuple, Optional, Dict, Any
import streamlit as st
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class DataManager:
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.cache_duration = 300  # 5 minutes cache
        
    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get stock data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if data.empty:
                st.warning(f"No data available for {symbol}")
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_options_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get options chain data"""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                return {}
            
            # Get the nearest expiration
            nearest_exp = expirations[0]
            options_chain = ticker.option_chain(nearest_exp)
            
            return {
                'calls': options_chain.calls,
                'puts': options_chain.puts,
                'expiration': nearest_exp
            }
        except Exception as e:
            st.error(f"Error fetching options data for {symbol}: {str(e)}")
            return {}
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock information including market cap"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'employees': info.get('fullTimeEmployees', 0),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', ''),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'beta': info.get('beta', 1.0),
                'dividend_yield': info.get('dividendYield', 0),
                'ex_dividend_date': info.get('exDividendDate', None),
                'dividend_rate': info.get('dividendRate', 0),
                'payout_ratio': info.get('payoutRatio', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                '50_day_avg': info.get('fiftyDayAverage', 0),
                '200_day_avg': info.get('twoHundredDayAverage', 0),
                'avg_volume': info.get('averageVolume', 0),
                'avg_volume_10days': info.get('averageVolume10days', 0),
                'avg_volume_3months': info.get('averageVolume3months', 0),
                'max_supply': info.get('maxSupply', 0),
                'total_supply': info.get('totalSupply', 0),
                'circulating_supply': info.get('circulatingSupply', 0),
            }
        except Exception as e:
            st.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return {}
    
    def get_market_cap(self, symbol: str) -> Optional[float]:
        """Get market cap for a symbol"""
        try:
            info = self.get_stock_info(symbol)
            return info.get('market_cap', 0) if info else None
        except:
            return None
    
    def calculate_iv_rank(self, symbol: str) -> float:
        """Calculate IV Rank for a symbol"""
        try:
            # Get 1 year of data
            data = self.get_stock_data(symbol, "1y")
            
            if data.empty:
                return 0.0
            
            # Calculate 30-day rolling volatility
            data['returns'] = data['Close'].pct_change()
            data['volatility'] = data['returns'].rolling(window=30).std() * np.sqrt(252)
            
            # Get current volatility and 52-week range
            current_vol = data['volatility'].iloc[-1]
            min_vol = data['volatility'].min()
            max_vol = data['volatility'].max()
            
            if max_vol == min_vol:
                return 50.0  # Default to middle if no variation
            
            iv_rank = ((current_vol - min_vol) / (max_vol - min_vol)) * 100
            return round(iv_rank, 2)
            
        except Exception as e:
            st.error(f"Error calculating IV Rank for {symbol}: {str(e)}")
            return 0.0
    
    def get_earnings_calendar(self, days_ahead: int = 7) -> pd.DataFrame:
        """Get upcoming earnings calendar"""
        try:
            # Use yfinance to get earnings dates
            # This is a simplified version - in production you'd use a proper earnings API
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
            earnings_data = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if 'earningsDate' in info and info['earningsDate']:
                        earnings_date = pd.to_datetime(info['earningsDate'], unit='s')
                        if earnings_date >= datetime.now():
                            earnings_data.append({
                                'symbol': symbol,
                                'earnings_date': earnings_date,
                                'company': info.get('longName', symbol)
                            })
                except:
                    continue
            
            df = pd.DataFrame(earnings_data)
            if not df.empty:
                df = df.sort_values('earnings_date')
                df = df.head(20)  # Limit to top 20
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching earnings calendar: {str(e)}")
            return pd.DataFrame()
    
    def get_unusual_volume(self, min_volume_multiplier: float = 2.0) -> pd.DataFrame:
        """Get stocks with unusual volume"""
        try:
            # This is a simplified version - in production you'd use a proper volume API
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC']
            unusual_volume = []
            
            for symbol in symbols:
                try:
                    data = self.get_stock_data(symbol, "5d")
                    if len(data) >= 2:
                        current_volume = data['Volume'].iloc[-1]
                        avg_volume = data['Volume'].mean()
                        
                        if current_volume > avg_volume * min_volume_multiplier:
                            unusual_volume.append({
                                'symbol': symbol,
                                'current_volume': current_volume,
                                'avg_volume': avg_volume,
                                'volume_ratio': current_volume / avg_volume,
                                'price': data['Close'].iloc[-1]
                            })
                except:
                    continue
            
            df = pd.DataFrame(unusual_volume)
            if not df.empty:
                df = df.sort_values('volume_ratio', ascending=False)
                df = df.head(20)
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching unusual volume data: {str(e)}")
            return pd.DataFrame()

# Global data manager instance
data_manager = DataManager()

def get_top_ivr_tickers(limit: int = 10) -> List[Tuple[str, float]]:
    """Get top IVR tickers for the sidebar widget"""
    try:
        # Cache the result for 5 minutes
        cache_key = f"top_ivr_{limit}"
        try:
            cached_data = st.session_state[cache_key]
            data, timestamp = cached_data
            if time.time() - timestamp < 300:  # 5 minutes
                return data
        except (AttributeError, KeyError):
            pass
        
        # Popular tickers to check
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC', 
                  'SPY', 'QQQ', 'IWM', 'VIX', 'TLT', 'GLD', 'SLV', 'BTC-USD', 'ETH-USD']
        
        ivr_data = []
        for ticker in tickers:
            ivr = data_manager.calculate_iv_rank(ticker)
            if ivr > 0:
                ivr_data.append((ticker, ivr))
        
        # Sort by IVR and return top results
        ivr_data.sort(key=lambda x: x[1], reverse=True)
        result = ivr_data[:limit]
        
        # Cache the result
        st.session_state[cache_key] = (result, time.time())
        
        return result
        
    except Exception as e:
        st.error(f"Error getting top IVR tickers: {str(e)}")
        return []

def get_stock_price(symbol: str) -> float:
    """Get current stock price"""
    try:
        data = data_manager.get_stock_data(symbol, "1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return 0.0
    except:
        return 0.0

def get_options_chain(symbol: str) -> Dict[str, pd.DataFrame]:
    """Get options chain for a symbol"""
    return data_manager.get_options_data(symbol)

def get_earnings_calendar(days_ahead: int = 7) -> pd.DataFrame:
    """Get upcoming earnings calendar"""
    return data_manager.get_earnings_calendar(days_ahead)

def get_unusual_volume(min_volume_multiplier: float = 2.0) -> pd.DataFrame:
    """Get stocks with unusual volume"""
    return data_manager.get_unusual_volume(min_volume_multiplier)
