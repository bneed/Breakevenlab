"""
Data ingestion layer for TradeScrubber
Supports multiple data sources: yfinance, Polygon.io, Alpaca
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from .utils import disk_cache, get_config, get_cache_ttl, CACHE_DIR

logger = logging.getLogger(__name__)

class DataManager:
    """Main data manager for fetching and caching market data"""
    
    def __init__(self):
        self.data_source = get_config("DATA_SOURCE", "yfinance")
        self.cache_ttl = get_cache_ttl()
        self.polygon_key = get_config("POLYGON_API_KEY")
        self.alpaca_key_id = get_config("ALPACA_API_KEY_ID")
        self.alpaca_secret = get_config("ALPHA_API_SECRET_KEY")
        self.tradier_key = get_config("TRADIER_API_KEY")
        
        # Initialize data source
        self._init_data_source()
    
    def _init_data_source(self):
        """Initialize the configured data source"""
        if self.data_source == "yfinance":
            try:
                import yfinance as yf
                self.yf = yf
                logger.info("Initialized yfinance data source")
            except ImportError:
                logger.error("yfinance not installed. Install with: pip install yfinance")
                raise
        
        elif self.data_source == "polygon":
            if not self.polygon_key:
                logger.warning("Polygon API key not found, falling back to yfinance")
                self.data_source = "yfinance"
                self._init_data_source()
                return
            
            try:
                import requests
                self.polygon_session = requests.Session()
                self.polygon_base_url = "https://api.polygon.io"
                logger.info("Initialized Polygon.io data source")
            except ImportError:
                logger.error("requests not installed. Install with: pip install requests")
                raise
        
        elif self.data_source == "alpaca":
            if not self.alpaca_key_id or not self.alpaca_secret:
                logger.warning("Alpaca API credentials not found, falling back to yfinance")
                self.data_source = "yfinance"
                self._init_data_source()
                return
            
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame
                
                self.alpaca_client = StockHistoricalDataClient(
                    self.alpaca_key_id, 
                    self.alpaca_secret
                )
                self.alpaca_request = StockBarsRequest
                self.alpaca_timeframe = TimeFrame
                logger.info("Initialized Alpaca data source")
            except ImportError:
                logger.error("alpaca-py not installed. Install with: pip install alpaca-py")
                raise
    
    @disk_cache("prices", ttl_minutes=15)
    def get_prices(
        self, 
        tickers: List[str], 
        interval: str = "1d", 
        lookback_days: int = 365,
        source: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get price data for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            interval: Data interval (1d, 1h, 5m, 1m)
            lookback_days: Number of days to look back
            source: Override data source
            
        Returns:
            Dictionary mapping ticker to DataFrame with OHLCV data
        """
        source = source or self.data_source
        
        if source == "yfinance":
            return self._get_prices_yfinance(tickers, interval, lookback_days)
        elif source == "polygon":
            return self._get_prices_polygon(tickers, interval, lookback_days)
        elif source == "alpaca":
            return self._get_prices_alpaca(tickers, interval, lookback_days)
        else:
            raise ValueError(f"Unsupported data source: {source}")
    
    def _get_prices_yfinance(self, tickers: List[str], interval: str, lookback_days: int) -> Dict[str, pd.DataFrame]:
        """Get prices using yfinance"""
        try:
            # Convert interval to yfinance format
            yf_interval_map = {
                "1d": "1d",
                "1h": "1h", 
                "5m": "5m",
                "1m": "1m"
            }
            yf_interval = yf_interval_map.get(interval, "1d")
            
            # Calculate period
            if lookback_days <= 5:
                period = "5d"
            elif lookback_days <= 30:
                period = "1mo"
            elif lookback_days <= 90:
                period = "3mo"
            elif lookback_days <= 365:
                period = "1y"
            else:
                period = "2y"
            
            # Fetch data
            data = self.yf.download(
                tickers, 
                period=period,
                interval=yf_interval,
                group_by='ticker',
                progress=False
            )
            
            # Process data
            result = {}
            if len(tickers) == 1:
                # Single ticker
                df = data.copy()
                df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]
                result[tickers[0]] = self._standardize_dataframe(df)
            else:
                # Multiple tickers
                for ticker in tickers:
                    if ticker in data.columns.get_level_values(0):
                        df = data[ticker].copy()
                        result[ticker] = self._standardize_dataframe(df)
                    else:
                        logger.warning(f"No data found for {ticker}")
                        result[ticker] = pd.DataFrame()
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching yfinance data: {e}")
            return {ticker: pd.DataFrame() for ticker in tickers}
    
    def _get_prices_polygon(self, tickers: List[str], interval: str, lookback_days: int) -> Dict[str, pd.DataFrame]:
        """Get prices using Polygon.io"""
        # Implementation for Polygon.io
        logger.warning("Polygon.io implementation not yet available, using yfinance")
        return self._get_prices_yfinance(tickers, interval, lookback_days)
    
    def _get_prices_alpaca(self, tickers: List[str], interval: str, lookback_days: int) -> Dict[str, pd.DataFrame]:
        """Get prices using Alpaca"""
        # Implementation for Alpaca
        logger.warning("Alpaca implementation not yet available, using yfinance")
        return self._get_prices_yfinance(tickers, interval, lookback_days)
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame columns and format"""
        if df.empty:
            return df
        
        # Standardize column names
        column_map = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        df = df.rename(columns=column_map)
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0  # Default volume
                else:
                    logger.warning(f"Missing required column: {col}")
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_options_chain(self, ticker: str, dte_pref: int = 30) -> Dict[str, Any]:
        """
        Get options chain for a ticker
        
        Args:
            ticker: Stock ticker symbol
            dte_pref: Preferred days to expiration
            
        Returns:
            Dictionary with calls, puts, and expiration info
        """
        if self.data_source == "yfinance":
            return self._get_options_yfinance(ticker, dte_pref)
        elif self.tradier_key:
            return self._get_options_tradier(ticker, dte_pref)
        else:
            logger.warning("No options data source available")
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiration": None}
    
    def _get_options_yfinance(self, ticker: str, dte_pref: int) -> Dict[str, Any]:
        """Get options using yfinance"""
        try:
            stock = self.yf.Ticker(ticker)
            expirations = stock.options
            
            if not expirations:
                logger.warning(f"No options data available for {ticker}")
                return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiration": None}
            
            # Find closest expiration to preferred DTE
            today = datetime.now().date()
            best_exp = None
            min_diff = float('inf')
            
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                diff = abs((exp_date - today).days - dte_pref)
                if diff < min_diff:
                    min_diff = diff
                    best_exp = exp_str
            
            if not best_exp:
                best_exp = expirations[0]
            
            # Get options chain
            chain = stock.option_chain(best_exp)
            
            return {
                "calls": chain.calls,
                "puts": chain.puts,
                "expiration": best_exp
            }
            
        except Exception as e:
            logger.error(f"Error fetching options for {ticker}: {e}")
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiration": None}
    
    def _get_options_tradier(self, ticker: str, dte_pref: int) -> Dict[str, Any]:
        """Get options using Tradier API"""
        # Implementation for Tradier API
        logger.warning("Tradier API implementation not yet available")
        return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiration": None}
    
    def get_intraday_data(self, ticker: str, days: int = 5) -> pd.DataFrame:
        """Get intraday data for a ticker"""
        try:
            if self.data_source == "yfinance":
                stock = self.yf.Ticker(ticker)
                data = stock.history(period=f"{days}d", interval="5m")
                return self._standardize_dataframe(data)
            else:
                logger.warning("Intraday data only available with yfinance")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching intraday data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        try:
            if self.data_source == "yfinance":
                # Use SPY as market proxy
                spy = self.yf.Ticker("SPY")
                info = spy.info
                
                return {
                    "market_state": info.get("marketState", "UNKNOWN"),
                    "currency": info.get("currency", "USD"),
                    "timezone": info.get("timezone", "EST"),
                    "regular_market_price": info.get("regularMarketPrice", 0),
                    "is_market_open": info.get("marketState") == "REGULAR"
                }
            else:
                return {
                    "market_state": "UNKNOWN",
                    "currency": "USD", 
                    "timezone": "EST",
                    "regular_market_price": 0,
                    "is_market_open": False
                }
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
            return {
                "market_state": "UNKNOWN",
                "currency": "USD",
                "timezone": "EST", 
                "regular_market_price": 0,
                "is_market_open": False
            }

# Global data manager instance
data_manager = DataManager()

# Convenience functions
def get_prices(tickers: List[str], interval: str = "1d", lookback_days: int = 365) -> Dict[str, pd.DataFrame]:
    """Get price data for multiple tickers"""
    return data_manager.get_prices(tickers, interval, lookback_days)

def get_options_chain(ticker: str, dte_pref: int = 30) -> Dict[str, Any]:
    """Get options chain for a ticker"""
    return data_manager.get_options_chain(ticker, dte_pref)

def get_intraday_data(ticker: str, days: int = 5) -> pd.DataFrame:
    """Get intraday data for a ticker"""
    return data_manager.get_intraday_data(ticker, days)

def get_market_status() -> Dict[str, Any]:
    """Get current market status"""
    return data_manager.get_market_status()
