"""
Signal detection and rule engine for TradeScrubber
Implements various trading signals and signal combinations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trading signals for a DataFrame with indicators
    
    Args:
        df: DataFrame with OHLCV data and indicators
        
    Returns:
        DataFrame with added signal columns
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Trend signals
    df = _add_trend_signals(df)
    
    # Momentum signals
    df = _add_momentum_signals(df)
    
    # Reversal signals
    df = _add_reversal_signals(df)
    
    # Breakout signals
    df = _add_breakout_signals(df)
    
    # Volume signals
    df = _add_volume_signals(df)
    
    # Volatility signals
    df = _add_volatility_signals(df)
    
    # Combined signals
    df = _add_combined_signals(df)
    
    return df

def _add_trend_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend-following signals"""
    
    # Golden Cross (SMA50 > SMA200)
    if 'sma50' in df.columns and 'sma200' in df.columns:
        df['golden_cross'] = df['sma50'] > df['sma200']
        df['golden_cross_signal'] = (df['sma50'] > df['sma200']) & (df['sma50'].shift(1) <= df['sma200'].shift(1))
    
    # Death Cross (SMA50 < SMA200)
    if 'sma50' in df.columns and 'sma200' in df.columns:
        df['death_cross'] = df['sma50'] < df['sma200']
        df['death_cross_signal'] = (df['sma50'] < df['sma200']) & (df['sma50'].shift(1) >= df['sma200'].shift(1))
    
    # EMA Crossover (EMA8 > EMA21)
    if 'ema8' in df.columns and 'ema21' in df.columns:
        df['ema_bullish'] = df['ema8'] > df['ema21']
        df['ema_cross_up'] = (df['ema8'] > df['ema21']) & (df['ema8'].shift(1) <= df['ema21'].shift(1))
        df['ema_cross_down'] = (df['ema8'] < df['ema21']) & (df['ema8'].shift(1) >= df['ema21'].shift(1))
    
    # Price above/below key moving averages
    if 'sma200' in df.columns:
        df['above_sma200'] = df['close'] > df['sma200']
    if 'sma50' in df.columns:
        df['above_sma50'] = df['close'] > df['sma50']
    if 'sma20' in df.columns:
        df['above_sma20'] = df['close'] > df['sma20']
    
    return df

def _add_momentum_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum-based signals"""
    
    # RSI signals
    if 'rsi14' in df.columns:
        df['rsi_oversold'] = df['rsi14'] < 30
        df['rsi_overbought'] = df['rsi14'] > 70
        df['rsi_bullish_divergence'] = _detect_bullish_divergence(df, 'rsi14')
        df['rsi_bearish_divergence'] = _detect_bearish_divergence(df, 'rsi14')
    
    # MACD signals
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_bullish'] = df['macd'] > df['macd_signal']
        df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        df['macd_above_zero'] = df['macd'] > 0
    
    # Stochastic signals
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        df['stoch_oversold'] = (df['stoch_k'] < 20) & (df['stoch_d'] < 20)
        df['stoch_overbought'] = (df['stoch_k'] > 80) & (df['stoch_d'] > 80)
        df['stoch_cross_up'] = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        df['stoch_cross_down'] = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
    
    return df

def _add_reversal_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add reversal signals"""
    
    # RSI reversal signals
    if 'rsi14' in df.columns:
        df['rsi_reversal_long'] = (df['rsi14'] < 30) & (df['rsi14'].shift(1) >= 30)
        df['rsi_reversal_short'] = (df['rsi14'] > 70) & (df['rsi14'].shift(1) <= 70)
    
    # Bollinger Band reversal signals
    if 'bb_position' in df.columns:
        df['bb_reversal_long'] = (df['bb_position'] < 0.1) & (df['close'] > df['close'].shift(1))
        df['bb_reversal_short'] = (df['bb_position'] > 0.9) & (df['close'] < df['close'].shift(1))
    
    # Hammer/Doji patterns (simplified)
    df['hammer'] = _detect_hammer(df)
    df['doji'] = _detect_doji(df)
    
    # Engulfing patterns
    df['bullish_engulfing'] = _detect_bullish_engulfing(df)
    df['bearish_engulfing'] = _detect_bearish_engulfing(df)
    
    return df

def _add_breakout_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add breakout signals"""
    
    # Price breakouts
    window = 20
    df['resistance_level'] = df['high'].rolling(window=window).max().shift(1)
    df['support_level'] = df['low'].rolling(window=window).min().shift(1)
    
    df['breakout_up'] = df['close'] > df['resistance_level']
    df['breakout_down'] = df['close'] < df['support_level']
    
    # Volume confirmation for breakouts
    if 'rel_vol_30d' in df.columns:
        df['breakout_up_volume'] = df['breakout_up'] & (df['rel_vol_30d'] > 1.2)
        df['breakout_down_volume'] = df['breakout_down'] & (df['rel_vol_30d'] > 1.2)
    
    # Bollinger Band breakouts
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_breakout_up'] = df['close'] > df['bb_upper']
        df['bb_breakout_down'] = df['close'] < df['bb_lower']
    
    # Opening Range Breakout (if available)
    if 'orb_breakout_up' in df.columns:
        df['orb_breakout_up'] = df['orb_breakout_up']
    if 'orb_breakout_down' in df.columns:
        df['orb_breakout_down'] = df['orb_breakout_down']
    
    return df

def _add_volume_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based signals"""
    
    # Volume spike
    if 'rel_vol_30d' in df.columns:
        df['volume_spike'] = df['rel_vol_30d'] > 1.5
        df['volume_surge'] = df['rel_vol_30d'] > 2.0
    
    # Volume trend
    if 'volume_sma_10' in df.columns and 'volume_sma_30' in df.columns:
        df['volume_trend_up'] = df['volume_sma_10'] > df['volume_sma_30']
        df['volume_trend_down'] = df['volume_sma_10'] < df['volume_sma_30']
    
    # Volume-price divergence
    df['volume_price_divergence'] = _detect_volume_price_divergence(df)
    
    return df

def _add_volatility_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility-based signals"""
    
    # ATR expansion/contraction
    if 'atr14' in df.columns:
        atr_sma = df['atr14'].rolling(window=20).mean()
        df['atr_expansion'] = df['atr14'] > atr_sma * 1.2
        df['atr_contraction'] = df['atr14'] < atr_sma * 0.8
    
    # Volatility breakout
    if 'volatility_20d' in df.columns:
        vol_sma = df['volatility_20d'].rolling(window=50).mean()
        df['vol_breakout'] = df['volatility_20d'] > vol_sma * 1.5
    
    # Gap signals
    if 'gap_pct' in df.columns:
        df['gap_up'] = df['gap_pct'] > 2.0
        df['gap_down'] = df['gap_pct'] < -2.0
    
    return df

def _add_combined_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined signal strategies"""
    
    # Bullish trend strategy
    bullish_conditions = []
    if 'above_sma200' in df.columns:
        bullish_conditions.append(df['above_sma200'])
    if 'ema_bullish' in df.columns:
        bullish_conditions.append(df['ema_bullish'])
    if 'macd_bullish' in df.columns:
        bullish_conditions.append(df['macd_bullish'])
    
    if bullish_conditions:
        df['bullish_trend'] = pd.concat(bullish_conditions, axis=1).all(axis=1)
    
    # Bearish trend strategy
    bearish_conditions = []
    if 'above_sma200' in df.columns:
        bearish_conditions.append(~df['above_sma200'])
    if 'ema_bullish' in df.columns:
        bearish_conditions.append(~df['ema_bullish'])
    if 'macd_bullish' in df.columns:
        bearish_conditions.append(~df['macd_bullish'])
    
    if bearish_conditions:
        df['bearish_trend'] = pd.concat(bearish_conditions, axis=1).all(axis=1)
    
    # Reversal strategy
    reversal_conditions = []
    if 'rsi_oversold' in df.columns:
        reversal_conditions.append(df['rsi_oversold'])
    if 'above_sma50' in df.columns:
        reversal_conditions.append(df['above_sma50'])
    if 'volume_spike' in df.columns:
        reversal_conditions.append(df['volume_spike'])
    
    if reversal_conditions:
        df['reversal_long'] = pd.concat(reversal_conditions, axis=1).all(axis=1)
    
    # Breakout strategy
    breakout_conditions = []
    if 'breakout_up' in df.columns:
        breakout_conditions.append(df['breakout_up'])
    if 'volume_spike' in df.columns:
        breakout_conditions.append(df['volume_spike'])
    if 'above_sma200' in df.columns:
        breakout_conditions.append(df['above_sma200'])
    
    if breakout_conditions:
        df['breakout_long'] = pd.concat(breakout_conditions, axis=1).all(axis=1)
    
    return df

# Helper functions for pattern detection

def _detect_bullish_divergence(df: pd.DataFrame, indicator_col: str, lookback: int = 5) -> pd.Series:
    """Detect bullish divergence between price and indicator"""
    if len(df) < lookback * 2:
        return pd.Series(False, index=df.index)
    
    price_lows = df['close'].rolling(window=lookback).min()
    indicator_lows = df[indicator_col].rolling(window=lookback).min()
    
    # Find recent low in price
    recent_price_low = price_lows.iloc[-lookback:].min()
    recent_indicator_low = indicator_lows.iloc[-lookback:].min()
    
    # Find previous low
    prev_price_low = price_lows.iloc[-lookback*2:-lookback].min()
    prev_indicator_low = indicator_lows.iloc[-lookback*2:-lookback].min()
    
    # Bullish divergence: price makes lower low, indicator makes higher low
    divergence = (recent_price_low < prev_price_low) & (recent_indicator_low > prev_indicator_low)
    
    return pd.Series(divergence, index=df.index)

def _detect_bearish_divergence(df: pd.DataFrame, indicator_col: str, lookback: int = 5) -> pd.Series:
    """Detect bearish divergence between price and indicator"""
    if len(df) < lookback * 2:
        return pd.Series(False, index=df.index)
    
    price_highs = df['close'].rolling(window=lookback).max()
    indicator_highs = df[indicator_col].rolling(window=lookback).max()
    
    # Find recent high in price
    recent_price_high = price_highs.iloc[-lookback:].max()
    recent_indicator_high = indicator_highs.iloc[-lookback:].max()
    
    # Find previous high
    prev_price_high = price_highs.iloc[-lookback*2:-lookback].max()
    prev_indicator_high = indicator_highs.iloc[-lookback*2:-lookback].max()
    
    # Bearish divergence: price makes higher high, indicator makes lower high
    divergence = (recent_price_high > prev_price_high) & (recent_indicator_high < prev_indicator_high)
    
    return pd.Series(divergence, index=df.index)

def _detect_hammer(df: pd.DataFrame) -> pd.Series:
    """Detect hammer candlestick pattern"""
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    
    # Hammer: small body, long lower shadow, short upper shadow
    hammer = (body < (df['high'] - df['low']) * 0.3) & \
             (lower_shadow > body * 2) & \
             (upper_shadow < body)
    
    return hammer

def _detect_doji(df: pd.DataFrame) -> pd.Series:
    """Detect doji candlestick pattern"""
    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']
    
    # Doji: very small body relative to total range
    doji = body < (total_range * 0.1)
    
    return doji

def _detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Detect bullish engulfing pattern"""
    if len(df) < 2:
        return pd.Series(False, index=df.index)
    
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    curr_open = df['open']
    curr_close = df['close']
    
    # Previous candle is bearish (red)
    prev_bearish = prev_close < prev_open
    
    # Current candle is bullish (green)
    curr_bullish = curr_close > curr_open
    
    # Current candle engulfs previous candle
    engulfing = (curr_open < prev_close) & (curr_close > prev_open)
    
    return prev_bearish & curr_bullish & engulfing

def _detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Detect bearish engulfing pattern"""
    if len(df) < 2:
        return pd.Series(False, index=df.index)
    
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    curr_open = df['open']
    curr_close = df['close']
    
    # Previous candle is bullish (green)
    prev_bullish = prev_close > prev_open
    
    # Current candle is bearish (red)
    curr_bearish = curr_close < curr_open
    
    # Current candle engulfs previous candle
    engulfing = (curr_open > prev_close) & (curr_close < prev_open)
    
    return prev_bullish & curr_bearish & engulfing

def _detect_volume_price_divergence(df: pd.DataFrame) -> pd.Series:
    """Detect volume-price divergence"""
    if 'rel_vol_30d' not in df.columns:
        return pd.Series(False, index=df.index)
    
    # Price making higher highs but volume decreasing
    price_higher = df['close'] > df['close'].shift(5)
    volume_lower = df['rel_vol_30d'] < df['rel_vol_30d'].shift(5)
    
    return price_higher & volume_lower

def get_signal_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get a summary of current signals"""
    if df.empty:
        return {}
    
    latest = df.iloc[-1]
    summary = {}
    
    # Trend signals
    trend_signals = ['golden_cross', 'death_cross', 'ema_bullish', 'above_sma200', 'above_sma50']
    for signal in trend_signals:
        if signal in df.columns:
            summary[signal] = latest[signal]
    
    # Momentum signals
    momentum_signals = ['rsi_oversold', 'rsi_overbought', 'macd_bullish', 'macd_above_zero']
    for signal in momentum_signals:
        if signal in df.columns:
            summary[signal] = latest[signal]
    
    # Reversal signals
    reversal_signals = ['rsi_reversal_long', 'rsi_reversal_short', 'hammer', 'doji']
    for signal in reversal_signals:
        if signal in df.columns:
            summary[signal] = latest[signal]
    
    # Breakout signals
    breakout_signals = ['breakout_up', 'breakout_down', 'bb_breakout_up', 'bb_breakout_down']
    for signal in breakout_signals:
        if signal in df.columns:
            summary[signal] = latest[signal]
    
    # Volume signals
    volume_signals = ['volume_spike', 'volume_surge', 'volume_trend_up']
    for signal in volume_signals:
        if signal in df.columns:
            summary[signal] = latest[signal]
    
    # Combined signals
    combined_signals = ['bullish_trend', 'bearish_trend', 'reversal_long', 'breakout_long']
    for signal in combined_signals:
        if signal in df.columns:
            summary[signal] = latest[signal]
    
    return summary
