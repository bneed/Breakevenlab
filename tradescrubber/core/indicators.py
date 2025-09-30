"""
Technical indicators for TradeScrubber
Implements SMA, EMA, RSI, MACD, VWAP, ATR, ORB, Volume Spike, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def add_indicators(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    """
    Add technical indicators to a DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        intraday: Whether this is intraday data (affects VWAP calculation)
        
    Returns:
        DataFrame with added indicator columns
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Missing required column: {col}")
            return df
    
    # Price-based indicators
    df = _add_sma_indicators(df)
    df = _add_ema_indicators(df)
    df = _add_rsi_indicator(df)
    df = _add_macd_indicator(df)
    df = _add_atr_indicator(df)
    df = _add_bollinger_bands(df)
    
    # Volume-based indicators
    df = _add_volume_indicators(df)
    
    # Intraday-specific indicators
    if intraday:
        df = _add_vwap_indicator(df)
        df = _add_orb_indicators(df)
    
    # Price position indicators
    df = _add_price_position_indicators(df)
    
    # Volatility indicators
    df = _add_volatility_indicators(df)
    
    return df

def _add_sma_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add Simple Moving Average indicators"""
    periods = [20, 50, 200]
    
    for period in periods:
        df[f'sma{period}'] = df['close'].rolling(window=period).mean()
    
    return df

def _add_ema_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add Exponential Moving Average indicators"""
    periods = [8, 21]
    
    for period in periods:
        df[f'ema{period}'] = df['close'].ewm(span=period).mean()
    
    return df

def _add_rsi_indicator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index indicator"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    df[f'rsi{period}'] = 100 - (100 / (1 + rs))
    
    return df

def _add_macd_indicator(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD indicator"""
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    return df

def _add_atr_indicator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range indicator"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df[f'atr{period}'] = true_range.rolling(window=period).mean()
    
    return df

def _add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
    """Add Bollinger Bands indicator"""
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    
    df['bb_upper'] = sma + (std * std_dev)
    df['bb_middle'] = sma
    df['bb_lower'] = sma - (std * std_dev)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df

def _add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based indicators"""
    # Volume moving averages
    df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_sma_30'] = df['volume'].rolling(window=30).mean()
    
    # Relative volume (current volume vs 30-day average)
    df['rel_vol_30d'] = df['volume'] / df['volume_sma_30']
    
    # Volume spike detection
    df['vol_spike'] = df['rel_vol_30d'] > 1.5
    
    # On-Balance Volume (OBV)
    df['obv'] = (df['volume'] * np.where(df['close'] > df['close'].shift(1), 1, 
                                        np.where(df['close'] < df['close'].shift(1), -1, 0))).cumsum()
    
    # Volume Rate of Change
    df['volume_roc'] = df['volume'].pct_change(periods=10) * 100
    
    return df

def _add_vwap_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Add Volume Weighted Average Price indicator (intraday only)"""
    if 'volume' not in df.columns:
        return df
    
    # Calculate typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate VWAP
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    # VWAP bands (1 and 2 standard deviations)
    vwap_std = ((typical_price - df['vwap']) ** 2 * df['volume']).cumsum() / df['volume'].cumsum()
    vwap_std = np.sqrt(vwap_std)
    
    df['vwap_upper_1'] = df['vwap'] + vwap_std
    df['vwap_lower_1'] = df['vwap'] - vwap_std
    df['vwap_upper_2'] = df['vwap'] + (2 * vwap_std)
    df['vwap_lower_2'] = df['vwap'] - (2 * vwap_std)
    
    return df

def _add_orb_indicators(df: pd.DataFrame, orb_minutes: int = 30) -> pd.DataFrame:
    """Add Opening Range Breakout indicators"""
    if df.index.tz is None:
        # Assume market hours if no timezone
        df.index = df.index.tz_localize('US/Eastern')
    
    # Group by date to find opening range
    df['date'] = df.index.date
    df['time'] = df.index.time
    
    # Find opening range high and low for each day
    orb_data = df.groupby('date').apply(
        lambda x: x[x['time'] <= pd.Timestamp(f'09:{orb_minutes:02d}:00').time()]
    )
    
    orb_high = orb_data.groupby('date')['high'].max()
    orb_low = orb_data.groupby('date')['low'].min()
    
    # Map to original dataframe
    df['orb_high'] = df['date'].map(orb_high)
    df['orb_low'] = df['date'].map(orb_low)
    
    # Breakout signals
    df['orb_breakout_up'] = df['close'] > df['orb_high']
    df['orb_breakout_down'] = df['close'] < df['orb_low']
    
    # Clean up temporary columns
    df = df.drop(['date', 'time'], axis=1)
    
    return df

def _add_price_position_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add price position relative to moving averages"""
    # Price above/below moving averages
    if 'sma20' in df.columns:
        df['price_above_sma20'] = df['close'] > df['sma20']
    if 'sma50' in df.columns:
        df['price_above_sma50'] = df['close'] > df['sma50']
    if 'sma200' in df.columns:
        df['price_above_sma200'] = df['close'] > df['sma200']
    
    # Price distance from moving averages (as percentage)
    if 'sma20' in df.columns:
        df['price_dist_sma20_pct'] = ((df['close'] - df['sma20']) / df['sma20']) * 100
    if 'sma50' in df.columns:
        df['price_dist_sma50_pct'] = ((df['close'] - df['sma50']) / df['sma50']) * 100
    if 'sma200' in df.columns:
        df['price_dist_sma200_pct'] = ((df['close'] - df['sma200']) / df['sma200']) * 100
    
    # Moving average crossovers
    if 'sma20' in df.columns and 'sma50' in df.columns:
        df['sma20_above_sma50'] = df['sma20'] > df['sma50']
    if 'sma50' in df.columns and 'sma200' in df.columns:
        df['sma50_above_sma200'] = df['sma50'] > df['sma200']
    
    return df

def _add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility-based indicators"""
    # Price volatility (standard deviation of returns)
    df['returns'] = df['close'].pct_change()
    df['volatility_10d'] = df['returns'].rolling(window=10).std() * np.sqrt(252)  # Annualized
    df['volatility_30d'] = df['returns'].rolling(window=30).std() * np.sqrt(252)
    
    # ATR as percentage of price
    if 'atr14' in df.columns:
        df['atr_pct'] = (df['atr14'] / df['close']) * 100
    
    # Price range (high-low as percentage of close)
    df['price_range_pct'] = ((df['high'] - df['low']) / df['close']) * 100
    
    # Gap detection (overnight gap)
    df['gap_pct'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * 100
    
    return df

def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate support and resistance levels"""
    # Rolling high and low
    df['resistance'] = df['high'].rolling(window=window).max()
    df['support'] = df['low'].rolling(window=window).min()
    
    # Distance from support/resistance
    df['dist_from_resistance_pct'] = ((df['resistance'] - df['close']) / df['close']) * 100
    df['dist_from_support_pct'] = ((df['close'] - df['support']) / df['close']) * 100
    
    return df

def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum indicators"""
    # Rate of Change
    for period in [5, 10, 20]:
        df[f'roc_{period}d'] = df['close'].pct_change(periods=period) * 100
    
    # Momentum (current close vs N periods ago)
    for period in [5, 10, 20]:
        df[f'momentum_{period}d'] = df['close'] - df['close'].shift(period)
    
    # Stochastic Oscillator
    lowest_low = df['low'].rolling(window=14).min()
    highest_high = df['high'].rolling(window=14).max()
    df['stoch_k'] = ((df['close'] - lowest_low) / (highest_high - lowest_low)) * 100
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    return df

def get_indicator_summary(df: pd.DataFrame) -> Dict[str, any]:
    """Get a summary of current indicator values"""
    if df.empty:
        return {}
    
    latest = df.iloc[-1]
    summary = {}
    
    # Moving averages
    for col in ['sma20', 'sma50', 'sma200', 'ema8', 'ema21']:
        if col in df.columns:
            summary[col] = latest[col]
    
    # RSI
    if 'rsi14' in df.columns:
        summary['rsi'] = latest['rsi14']
        summary['rsi_signal'] = 'oversold' if latest['rsi14'] < 30 else 'overbought' if latest['rsi14'] > 70 else 'neutral'
    
    # MACD
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        summary['macd'] = latest['macd']
        summary['macd_signal'] = latest['macd_signal']
        summary['macd_bullish'] = latest['macd'] > latest['macd_signal']
    
    # Volume
    if 'rel_vol_30d' in df.columns:
        summary['relative_volume'] = latest['rel_vol_30d']
        summary['volume_spike'] = latest.get('vol_spike', False)
    
    # Price position
    summary['price_above_sma200'] = latest.get('price_above_sma200', False)
    summary['price_above_sma50'] = latest.get('price_above_sma50', False)
    
    return summary
