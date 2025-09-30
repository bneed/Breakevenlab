"""
Ranking and scoring system for TradeScrubber
Computes Trade Readiness Score and ranks tickers by trade potential
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

def rank_universe(
    prices_dict: Dict[str, pd.DataFrame], 
    ml_predictions: Optional[Dict[str, Dict[str, float]]] = None,
    strategy_preset: str = "default"
) -> pd.DataFrame:
    """
    Rank universe of tickers by trade readiness score
    
    Args:
        prices_dict: Dictionary mapping ticker to DataFrame with OHLCV + indicators + signals
        ml_predictions: Optional ML predictions for each ticker
        strategy_preset: Strategy preset to use for scoring
        
    Returns:
        DataFrame sorted by score with ticker, direction, score, reasons, etc.
    """
    if not prices_dict:
        return pd.DataFrame()
    
    rows = []
    
    for ticker, df in prices_dict.items():
        if df.empty:
            continue
            
        try:
            # Get latest data point
            latest = df.iloc[-1]
            
            # Get ML predictions for this ticker
            ml_pred = ml_predictions.get(ticker, {}) if ml_predictions else {}
            
            # Score the ticker
            score_result = score_ticker(df, ml_pred, strategy_preset)
            
            # Add ticker info
            score_result['ticker'] = ticker
            score_result['price'] = latest.get('close', 0)
            score_result['volume'] = latest.get('volume', 0)
            score_result['market_cap'] = latest.get('market_cap', 0)  # If available
            
            rows.append(score_result)
            
        except Exception as e:
            logger.error(f"Error scoring ticker {ticker}: {e}")
            continue
    
    if not rows:
        return pd.DataFrame()
    
    # Create DataFrame and sort by score
    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    return result_df

def score_ticker(
    df: pd.DataFrame, 
    ml_pred: Optional[Dict[str, float]] = None,
    strategy_preset: str = "default"
) -> Dict[str, Any]:
    """
    Score a single ticker based on technical analysis and ML predictions
    
    Args:
        df: DataFrame with OHLCV + indicators + signals
        ml_pred: ML predictions for this ticker
        strategy_preset: Strategy preset to use
        
    Returns:
        Dictionary with score, direction, reasons, etc.
    """
    if df.empty:
        return {
            'score': 0,
            'direction': 'neutral',
            'confidence': 0,
            'reasons': ['No data available'],
            'entry_zone': None,
            'stop_loss': None,
            'target': None
        }
    
    latest = df.iloc[-1]
    ml_pred = ml_pred or {}
    
    # Initialize scoring components
    score = 0
    reasons = []
    confidence_factors = []
    
    # 1. Technical Alignment Score (40% weight)
    tech_score, tech_reasons = _calculate_technical_score(latest)
    score += tech_score * 0.4
    reasons.extend(tech_reasons)
    confidence_factors.append(tech_score / 100)
    
    # 2. Momentum/Volume Score (25% weight)
    momentum_score, momentum_reasons = _calculate_momentum_score(latest)
    score += momentum_score * 0.25
    reasons.extend(momentum_reasons)
    confidence_factors.append(momentum_score / 100)
    
    # 3. ML Prediction Score (25% weight)
    ml_score, ml_reasons = _calculate_ml_score(ml_pred)
    score += ml_score * 0.25
    reasons.extend(ml_reasons)
    confidence_factors.append(ml_score / 100)
    
    # 4. Trend Quality Score (10% weight)
    trend_score, trend_reasons = _calculate_trend_quality_score(latest)
    score += trend_score * 0.1
    reasons.extend(trend_reasons)
    confidence_factors.append(trend_score / 100)
    
    # Determine direction
    direction = _determine_direction(latest, ml_pred)
    
    # Calculate confidence
    confidence = np.mean(confidence_factors) if confidence_factors else 0
    
    # Calculate entry/stop/target levels
    entry_zone, stop_loss, target = _calculate_trade_levels(df, direction)
    
    return {
        'score': round(score, 1),
        'direction': direction,
        'confidence': round(confidence, 2),
        'reasons': reasons,
        'entry_zone': entry_zone,
        'stop_loss': stop_loss,
        'target': target,
        'technical_score': round(tech_score, 1),
        'momentum_score': round(momentum_score, 1),
        'ml_score': round(ml_score, 1),
        'trend_score': round(trend_score, 1)
    }

def _calculate_technical_score(latest: pd.Series) -> Tuple[float, List[str]]:
    """Calculate technical alignment score (0-100)"""
    score = 0
    reasons = []
    
    # Moving average alignment
    if latest.get('sma50', 0) > 0 and latest.get('sma200', 0) > 0:
        if latest['sma50'] > latest['sma200']:
            score += 15
            reasons.append('SMA50 > SMA200 (uptrend)')
        else:
            score -= 10
            reasons.append('SMA50 < SMA200 (downtrend)')
    
    # EMA momentum
    if latest.get('ema8', 0) > 0 and latest.get('ema21', 0) > 0:
        if latest['ema8'] > latest['ema21']:
            score += 10
            reasons.append('EMA8 > EMA21 (momentum)')
        else:
            score -= 5
            reasons.append('EMA8 < EMA21 (weak momentum)')
    
    # Price position relative to moving averages
    if latest.get('price_above_sma50', False):
        score += 5
        reasons.append('Price > SMA50')
    
    if latest.get('price_above_sma200', False):
        score += 10
        reasons.append('Price > SMA200')
    
    # RSI signals
    rsi = latest.get('rsi14', 50)
    if rsi < 30:
        score += 10
        reasons.append('RSI oversold (reversal potential)')
    elif rsi > 70:
        score -= 5
        reasons.append('RSI overbought (caution)')
    elif 40 <= rsi <= 60:
        score += 5
        reasons.append('RSI neutral (healthy)')
    
    # MACD signals
    if latest.get('macd_bullish', False):
        score += 8
        reasons.append('MACD bullish')
    elif latest.get('macd_cross_up', False):
        score += 12
        reasons.append('MACD cross up')
    
    # Bollinger Bands
    bb_pos = latest.get('bb_position', 0.5)
    if bb_pos < 0.2:
        score += 8
        reasons.append('Near BB lower band (oversold)')
    elif bb_pos > 0.8:
        score -= 3
        reasons.append('Near BB upper band (overbought)')
    
    return min(max(score, 0), 100), reasons

def _calculate_momentum_score(latest: pd.Series) -> Tuple[float, List[str]]:
    """Calculate momentum and volume score (0-100)"""
    score = 0
    reasons = []
    
    # Volume analysis
    rel_vol = latest.get('rel_vol_30d', 1.0)
    if rel_vol > 2.0:
        score += 20
        reasons.append(f'Volume surge ({rel_vol:.1f}x)')
    elif rel_vol > 1.5:
        score += 15
        reasons.append(f'Volume spike ({rel_vol:.1f}x)')
    elif rel_vol > 1.2:
        score += 10
        reasons.append(f'Above avg volume ({rel_vol:.1f}x)')
    elif rel_vol < 0.5:
        score -= 10
        reasons.append(f'Low volume ({rel_vol:.1f}x)')
    
    # ATR volatility
    atr_pct = latest.get('atr_pct', 0)
    if 1.0 <= atr_pct <= 3.0:
        score += 10
        reasons.append(f'Healthy volatility ({atr_pct:.1f}%)')
    elif atr_pct > 5.0:
        score -= 5
        reasons.append(f'High volatility ({atr_pct:.1f}%)')
    
    # Price momentum
    if latest.get('above_sma20', False) and latest.get('above_sma50', False):
        score += 15
        reasons.append('Strong price momentum')
    elif latest.get('above_sma50', False):
        score += 10
        reasons.append('Moderate price momentum')
    
    # Gap analysis
    gap_pct = latest.get('gap_pct', 0)
    if abs(gap_pct) > 2.0:
        if gap_pct > 0:
            score += 8
            reasons.append(f'Gap up ({gap_pct:.1f}%)')
        else:
            score -= 5
            reasons.append(f'Gap down ({gap_pct:.1f}%)')
    
    return min(max(score, 0), 100), reasons

def _calculate_ml_score(ml_pred: Dict[str, float]) -> Tuple[float, List[str]]:
    """Calculate ML prediction score (0-100)"""
    if not ml_pred:
        return 50, ['No ML predictions available']
    
    score = 50  # Neutral baseline
    reasons = []
    
    # Up probability
    up_prob = ml_pred.get('up_prob', 0.5)
    if up_prob > 0.7:
        score += 25
        reasons.append(f'High ML up probability ({up_prob:.2f})')
    elif up_prob > 0.6:
        score += 15
        reasons.append(f'Good ML up probability ({up_prob:.2f})')
    elif up_prob < 0.3:
        score -= 15
        reasons.append(f'Low ML up probability ({up_prob:.2f})')
    
    # Expected move
    exp_move = ml_pred.get('expected_move', 0)
    if abs(exp_move) > 0.05:  # 5% expected move
        if exp_move > 0:
            score += 15
            reasons.append(f'Positive expected move ({exp_move:.1%})')
        else:
            score -= 10
            reasons.append(f'Negative expected move ({exp_move:.1%})')
    
    # Model confidence
    confidence = ml_pred.get('confidence', 0.5)
    if confidence > 0.8:
        score += 10
        reasons.append(f'High ML confidence ({confidence:.2f})')
    elif confidence < 0.3:
        score -= 5
        reasons.append(f'Low ML confidence ({confidence:.2f})')
    
    return min(max(score, 0), 100), reasons

def _calculate_trend_quality_score(latest: pd.Series) -> Tuple[float, List[str]]:
    """Calculate trend quality score (0-100)"""
    score = 50  # Neutral baseline
    reasons = []
    
    # SMA200 slope (if we had historical data, we'd calculate slope)
    # For now, use price position relative to SMA200
    if latest.get('price_above_sma200', False):
        score += 20
        reasons.append('Above SMA200 (long-term uptrend)')
    else:
        score -= 15
        reasons.append('Below SMA200 (long-term downtrend)')
    
    # Volatility regime
    atr_pct = latest.get('atr_pct', 0)
    if 1.0 <= atr_pct <= 2.5:
        score += 15
        reasons.append('Optimal volatility regime')
    elif atr_pct > 4.0:
        score -= 10
        reasons.append('High volatility regime')
    
    # Trend consistency (simplified)
    if latest.get('golden_cross', False):
        score += 15
        reasons.append('Golden cross (strong trend)')
    elif latest.get('death_cross', False):
        score -= 15
        reasons.append('Death cross (weak trend)')
    
    return min(max(score, 0), 100), reasons

def _determine_direction(latest: pd.Series, ml_pred: Dict[str, float]) -> str:
    """Determine trade direction based on signals and ML predictions"""
    
    # Check for strong bullish signals
    bullish_signals = 0
    bearish_signals = 0
    
    # Technical signals
    if latest.get('ema_bullish', False):
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    if latest.get('price_above_sma50', False):
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    if latest.get('macd_bullish', False):
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # ML predictions
    up_prob = ml_pred.get('up_prob', 0.5)
    if up_prob > 0.6:
        bullish_signals += 1
    elif up_prob < 0.4:
        bearish_signals += 1
    
    # Determine direction
    if bullish_signals > bearish_signals + 1:
        return 'long'
    elif bearish_signals > bullish_signals + 1:
        return 'short'
    else:
        return 'neutral'

def _calculate_trade_levels(df: pd.DataFrame, direction: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculate entry zone, stop loss, and target levels"""
    if df.empty:
        return None, None, None
    
    latest = df.iloc[-1]
    current_price = latest.get('close', 0)
    
    if current_price <= 0:
        return None, None, None
    
    # Use ATR for level calculations
    atr = latest.get('atr14', current_price * 0.02)  # Default 2% if no ATR
    
    if direction == 'long':
        # Entry zone: current price to slightly above
        entry_zone = current_price * 1.01
        
        # Stop loss: below recent low or ATR-based
        recent_low = df['low'].tail(10).min()
        stop_loss = min(recent_low * 0.98, current_price - (atr * 2))
        
        # Target: 2:1 or 3:1 risk/reward
        risk = current_price - stop_loss
        target = current_price + (risk * 2.5)
        
    elif direction == 'short':
        # Entry zone: current price to slightly below
        entry_zone = current_price * 0.99
        
        # Stop loss: above recent high or ATR-based
        recent_high = df['high'].tail(10).max()
        stop_loss = max(recent_high * 1.02, current_price + (atr * 2))
        
        # Target: 2:1 or 3:1 risk/reward
        risk = stop_loss - current_price
        target = current_price - (risk * 2.5)
        
    else:  # neutral
        return None, None, None
    
    return round(entry_zone, 2), round(stop_loss, 2), round(target, 2)

def filter_by_strategy(
    ranked_df: pd.DataFrame, 
    strategy_preset: str = "default",
    min_score: float = 50.0,
    max_results: int = 20
) -> pd.DataFrame:
    """Filter ranked results by strategy preset and criteria"""
    
    if ranked_df.empty:
        return ranked_df
    
    # Apply minimum score filter
    filtered_df = ranked_df[ranked_df['score'] >= min_score].copy()
    
    # Apply strategy-specific filters
    if strategy_preset == "reversal":
        # Look for oversold conditions
        filtered_df = filtered_df[
            (filtered_df['rsi_oversold'] == True) |
            (filtered_df['direction'] == 'long')
        ]
    elif strategy_preset == "breakout":
        # Look for breakouts with volume
        filtered_df = filtered_df[
            (filtered_df['breakout_up'] == True) |
            (filtered_df['volume_spike'] == True)
        ]
    elif strategy_preset == "trend":
        # Look for strong trends
        filtered_df = filtered_df[
            (filtered_df['above_sma200'] == True) &
            (filtered_df['direction'] == 'long')
        ]
    
    # Limit results
    filtered_df = filtered_df.head(max_results)
    
    return filtered_df

def get_ranking_summary(ranked_df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics for ranked results"""
    if ranked_df.empty:
        return {}
    
    return {
        'total_tickers': len(ranked_df),
        'avg_score': round(ranked_df['score'].mean(), 1),
        'max_score': round(ranked_df['score'].max(), 1),
        'min_score': round(ranked_df['score'].min(), 1),
        'long_signals': len(ranked_df[ranked_df['direction'] == 'long']),
        'short_signals': len(ranked_df[ranked_df['direction'] == 'short']),
        'neutral_signals': len(ranked_df[ranked_df['direction'] == 'neutral']),
        'high_confidence': len(ranked_df[ranked_df['confidence'] > 0.7]),
        'top_ticker': ranked_df.iloc[0]['ticker'] if len(ranked_df) > 0 else None,
        'top_score': ranked_df.iloc[0]['score'] if len(ranked_df) > 0 else 0
    }
