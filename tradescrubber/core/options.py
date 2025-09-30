"""
Options analysis for TradeScrubber
Handles options chains, break-even analysis, expected value, and risk/reward calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class OptionsAnalyzer:
    """Options chain analyzer and calculator"""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% default risk-free rate
    
    def analyze_options_chain(
        self, 
        ticker: str, 
        current_price: float,
        options_data: Dict[str, pd.DataFrame],
        direction: str = "long",
        target_delta: float = 0.3,
        dte_range: Tuple[int, int] = (14, 45)
    ) -> Dict[str, Any]:
        """
        Analyze options chain and recommend trades
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            options_data: Dictionary with 'calls' and 'puts' DataFrames
            direction: Trade direction ('long' or 'short')
            target_delta: Target delta for options selection
            dte_range: Days to expiration range (min, max)
            
        Returns:
            Dictionary with recommended options trades
        """
        if not options_data or 'calls' not in options_data or 'puts' not in options_data:
            return {"error": "No options data available"}
        
        calls_df = options_data['calls']
        puts_df = options_data['puts']
        
        if calls_df.empty and puts_df.empty:
            return {"error": "Empty options chain"}
        
        # Filter by DTE range
        calls_filtered = self._filter_by_dte(calls_df, dte_range)
        puts_filtered = self._filter_by_dte(puts_df, dte_range)
        
        recommendations = {
            'ticker': ticker,
            'current_price': current_price,
            'direction': direction,
            'calls': [],
            'puts': [],
            'spreads': [],
            'straddles': []
        }
        
        # Analyze calls
        if not calls_filtered.empty:
            call_recs = self._analyze_calls(calls_filtered, current_price, direction, target_delta)
            recommendations['calls'] = call_recs
        
        # Analyze puts
        if not puts_filtered.empty:
            put_recs = self._analyze_puts(puts_filtered, current_price, direction, target_delta)
            recommendations['puts'] = put_recs
        
        # Analyze spreads
        if not calls_filtered.empty and not puts_filtered.empty:
            spread_recs = self._analyze_spreads(calls_filtered, puts_filtered, current_price, direction)
            recommendations['spreads'] = spread_recs
        
        # Analyze straddles
        if not calls_filtered.empty and not puts_filtered.empty:
            straddle_recs = self._analyze_straddles(calls_filtered, puts_filtered, current_price)
            recommendations['straddles'] = straddle_recs
        
        return recommendations
    
    def _filter_by_dte(self, df: pd.DataFrame, dte_range: Tuple[int, int]) -> pd.DataFrame:
        """Filter options by days to expiration"""
        if df.empty or 'expiration' not in df.columns:
            return df
        
        min_dte, max_dte = dte_range
        today = datetime.now().date()
        
        # Calculate DTE for each option
        df = df.copy()
        df['dte'] = df['expiration'].apply(
            lambda x: (datetime.strptime(x, '%Y-%m-%d').date() - today).days
        )
        
        return df[(df['dte'] >= min_dte) & (df['dte'] <= max_dte)]
    
    def _analyze_calls(
        self, 
        calls_df: pd.DataFrame, 
        current_price: float, 
        direction: str,
        target_delta: float
    ) -> List[Dict[str, Any]]:
        """Analyze call options"""
        recommendations = []
        
        # ATM call
        atm_call = self._find_atm_option(calls_df, current_price, 'call')
        if atm_call is not None:
            rec = self._analyze_single_option(atm_call, current_price, 'call', direction)
            rec['type'] = 'ATM Call'
            recommendations.append(rec)
        
        # Target delta call
        delta_call = self._find_delta_option(calls_df, target_delta, 'call')
        if delta_call is not None:
            rec = self._analyze_single_option(delta_call, current_price, 'call', direction)
            rec['type'] = f'Call (Δ={target_delta})'
            recommendations.append(rec)
        
        # OTM call (0.2 delta)
        otm_call = self._find_delta_option(calls_df, 0.2, 'call')
        if otm_call is not None:
            rec = self._analyze_single_option(otm_call, current_price, 'call', direction)
            rec['type'] = 'OTM Call (Δ=0.2)'
            recommendations.append(rec)
        
        return recommendations
    
    def _analyze_puts(
        self, 
        puts_df: pd.DataFrame, 
        current_price: float, 
        direction: str,
        target_delta: float
    ) -> List[Dict[str, Any]]:
        """Analyze put options"""
        recommendations = []
        
        # ATM put
        atm_put = self._find_atm_option(puts_df, current_price, 'put')
        if atm_put is not None:
            rec = self._analyze_single_option(atm_put, current_price, 'put', direction)
            rec['type'] = 'ATM Put'
            recommendations.append(rec)
        
        # Target delta put
        delta_put = self._find_delta_option(puts_df, target_delta, 'put')
        if delta_put is not None:
            rec = self._analyze_single_option(delta_put, current_price, 'put', direction)
            rec['type'] = f'Put (Δ={target_delta})'
            recommendations.append(rec)
        
        # OTM put (0.2 delta)
        otm_put = self._find_delta_option(puts_df, 0.2, 'put')
        if otm_put is not None:
            rec = self._analyze_single_option(otm_put, current_price, 'put', direction)
            rec['type'] = 'OTM Put (Δ=0.2)'
            recommendations.append(rec)
        
        return recommendations
    
    def _analyze_spreads(
        self, 
        calls_df: pd.DataFrame, 
        puts_df: pd.DataFrame, 
        current_price: float,
        direction: str
    ) -> List[Dict[str, Any]]:
        """Analyze spread strategies"""
        recommendations = []
        
        # Bull Call Spread
        bull_call_spread = self._create_bull_call_spread(calls_df, current_price)
        if bull_call_spread:
            recommendations.append(bull_call_spread)
        
        # Bear Put Spread
        bear_put_spread = self._create_bear_put_spread(puts_df, current_price)
        if bear_put_spread:
            recommendations.append(bear_put_spread)
        
        # Iron Condor
        iron_condor = self._create_iron_condor(calls_df, puts_df, current_price)
        if iron_condor:
            recommendations.append(iron_condor)
        
        return recommendations
    
    def _analyze_straddles(
        self, 
        calls_df: pd.DataFrame, 
        puts_df: pd.DataFrame, 
        current_price: float
    ) -> List[Dict[str, Any]]:
        """Analyze straddle strategies"""
        recommendations = []
        
        # Long Straddle
        long_straddle = self._create_long_straddle(calls_df, puts_df, current_price)
        if long_straddle:
            recommendations.append(long_straddle)
        
        # Short Straddle
        short_straddle = self._create_short_straddle(calls_df, puts_df, current_price)
        if short_straddle:
            recommendations.append(short_straddle)
        
        return recommendations
    
    def _find_atm_option(self, df: pd.DataFrame, current_price: float, option_type: str) -> Optional[pd.Series]:
        """Find at-the-money option"""
        if df.empty:
            return None
        
        # Find closest strike to current price
        if option_type == 'call':
            df_sorted = df[df['strike'] >= current_price].sort_values('strike')
        else:  # put
            df_sorted = df[df['strike'] <= current_price].sort_values('strike', ascending=False)
        
        if df_sorted.empty:
            return None
        
        return df_sorted.iloc[0]
    
    def _find_delta_option(self, df: pd.DataFrame, target_delta: float, option_type: str) -> Optional[pd.Series]:
        """Find option with target delta"""
        if df.empty or 'delta' not in df.columns:
            return None
        
        # For puts, delta is negative, so we look for -target_delta
        if option_type == 'put':
            target_delta = -target_delta
        
        # Find closest delta
        df['delta_diff'] = abs(df['delta'] - target_delta)
        closest = df.loc[df['delta_diff'].idxmin()]
        
        return closest
    
    def _analyze_single_option(
        self, 
        option: pd.Series, 
        current_price: float, 
        option_type: str,
        direction: str
    ) -> Dict[str, Any]:
        """Analyze a single option"""
        strike = option.get('strike', 0)
        bid = option.get('bid', 0)
        ask = option.get('ask', 0)
        mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else option.get('lastPrice', 0)
        
        # Calculate break-even
        if option_type == 'call':
            breakeven = strike + mid_price
        else:  # put
            breakeven = strike - mid_price
        
        # Calculate risk/reward
        if direction == 'long':
            if option_type == 'call':
                max_profit = float('inf')  # Unlimited upside
                max_loss = mid_price
                risk_reward = float('inf') if max_loss > 0 else 0
            else:  # put
                max_profit = strike - mid_price
                max_loss = mid_price
                risk_reward = max_profit / max_loss if max_loss > 0 else 0
        else:  # short
            if option_type == 'call':
                max_profit = mid_price
                max_loss = float('inf')  # Unlimited downside
                risk_reward = 0
            else:  # put
                max_profit = mid_price
                max_loss = strike - mid_price
                risk_reward = max_profit / max_loss if max_loss > 0 else 0
        
        # Calculate Greeks (if available)
        delta = option.get('delta', 0)
        gamma = option.get('gamma', 0)
        theta = option.get('theta', 0)
        vega = option.get('vega', 0)
        rho = option.get('rho', 0)
        
        # Calculate probability of profit (simplified)
        prob_profit = self._calculate_probability_of_profit(
            current_price, strike, mid_price, option_type, direction
        )
        
        return {
            'strike': strike,
            'premium': mid_price,
            'breakeven': breakeven,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'risk_reward': risk_reward,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'prob_profit': prob_profit,
            'dte': option.get('dte', 0),
            'volume': option.get('volume', 0),
            'open_interest': option.get('openInterest', 0)
        }
    
    def _create_bull_call_spread(self, calls_df: pd.DataFrame, current_price: float) -> Optional[Dict[str, Any]]:
        """Create bull call spread recommendation"""
        if calls_df.empty:
            return None
        
        # Find ATM and OTM calls
        atm_call = self._find_atm_option(calls_df, current_price, 'call')
        otm_call = self._find_delta_option(calls_df, 0.3, 'call')
        
        if atm_call is None or otm_call is None:
            return None
        
        # Calculate spread metrics
        long_strike = atm_call['strike']
        short_strike = otm_call['strike']
        long_premium = (atm_call.get('bid', 0) + atm_call.get('ask', 0)) / 2
        short_premium = (otm_call.get('bid', 0) + otm_call.get('ask', 0)) / 2
        
        net_debit = long_premium - short_premium
        max_profit = short_strike - long_strike - net_debit
        max_loss = net_debit
        breakeven = long_strike + net_debit
        
        return {
            'strategy': 'Bull Call Spread',
            'long_strike': long_strike,
            'short_strike': short_strike,
            'net_debit': net_debit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'risk_reward': max_profit / max_loss if max_loss > 0 else 0
        }
    
    def _create_bear_put_spread(self, puts_df: pd.DataFrame, current_price: float) -> Optional[Dict[str, Any]]:
        """Create bear put spread recommendation"""
        if puts_df.empty:
            return None
        
        # Find ATM and OTM puts
        atm_put = self._find_atm_option(puts_df, current_price, 'put')
        otm_put = self._find_delta_option(puts_df, 0.3, 'put')
        
        if atm_put is None or otm_put is None:
            return None
        
        # Calculate spread metrics
        long_strike = atm_put['strike']
        short_strike = otm_put['strike']
        long_premium = (atm_put.get('bid', 0) + atm_put.get('ask', 0)) / 2
        short_premium = (otm_put.get('bid', 0) + otm_put.get('ask', 0)) / 2
        
        net_debit = long_premium - short_premium
        max_profit = long_strike - short_strike - net_debit
        max_loss = net_debit
        breakeven = long_strike - net_debit
        
        return {
            'strategy': 'Bear Put Spread',
            'long_strike': long_strike,
            'short_strike': short_strike,
            'net_debit': net_debit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'risk_reward': max_profit / max_loss if max_loss > 0 else 0
        }
    
    def _create_iron_condor(
        self, 
        calls_df: pd.DataFrame, 
        puts_df: pd.DataFrame, 
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Create iron condor recommendation"""
        if calls_df.empty or puts_df.empty:
            return None
        
        # Find options for iron condor
        put_short = self._find_delta_option(puts_df, 0.2, 'put')
        put_long = self._find_delta_option(puts_df, 0.1, 'put')
        call_short = self._find_delta_option(calls_df, 0.2, 'call')
        call_long = self._find_delta_option(calls_df, 0.1, 'call')
        
        if any(x is None for x in [put_short, put_long, call_short, call_long]):
            return None
        
        # Calculate net credit
        put_short_premium = (put_short.get('bid', 0) + put_short.get('ask', 0)) / 2
        put_long_premium = (put_long.get('bid', 0) + put_long.get('ask', 0)) / 2
        call_short_premium = (call_short.get('bid', 0) + call_short.get('ask', 0)) / 2
        call_long_premium = (call_long.get('bid', 0) + call_long.get('ask', 0)) / 2
        
        net_credit = (put_short_premium + call_short_premium) - (put_long_premium + call_long_premium)
        max_profit = net_credit
        max_loss = (call_short['strike'] - call_long['strike']) - net_credit
        
        return {
            'strategy': 'Iron Condor',
            'put_short_strike': put_short['strike'],
            'put_long_strike': put_long['strike'],
            'call_short_strike': call_short['strike'],
            'call_long_strike': call_long['strike'],
            'net_credit': net_credit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'risk_reward': max_profit / max_loss if max_loss > 0 else 0
        }
    
    def _create_long_straddle(
        self, 
        calls_df: pd.DataFrame, 
        puts_df: pd.DataFrame, 
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Create long straddle recommendation"""
        if calls_df.empty or puts_df.empty:
            return None
        
        # Find ATM options
        atm_call = self._find_atm_option(calls_df, current_price, 'call')
        atm_put = self._find_atm_option(puts_df, current_price, 'put')
        
        if atm_call is None or atm_put is None:
            return None
        
        # Calculate straddle metrics
        call_premium = (atm_call.get('bid', 0) + atm_call.get('ask', 0)) / 2
        put_premium = (atm_put.get('bid', 0) + atm_put.get('ask', 0)) / 2
        
        total_cost = call_premium + put_premium
        breakeven_up = current_price + total_cost
        breakeven_down = current_price - total_cost
        max_loss = total_cost
        
        return {
            'strategy': 'Long Straddle',
            'strike': atm_call['strike'],
            'total_cost': total_cost,
            'breakeven_up': breakeven_up,
            'breakeven_down': breakeven_down,
            'max_loss': max_loss,
            'max_profit': float('inf')
        }
    
    def _create_short_straddle(
        self, 
        calls_df: pd.DataFrame, 
        puts_df: pd.DataFrame, 
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Create short straddle recommendation"""
        if calls_df.empty or puts_df.empty:
            return None
        
        # Find ATM options
        atm_call = self._find_atm_option(calls_df, current_price, 'call')
        atm_put = self._find_atm_option(puts_df, current_price, 'put')
        
        if atm_call is None or atm_put is None:
            return None
        
        # Calculate straddle metrics
        call_premium = (atm_call.get('bid', 0) + atm_call.get('ask', 0)) / 2
        put_premium = (atm_put.get('bid', 0) + atm_put.get('ask', 0)) / 2
        
        total_credit = call_premium + put_premium
        breakeven_up = current_price + total_credit
        breakeven_down = current_price - total_credit
        max_profit = total_credit
        max_loss = float('inf')
        
        return {
            'strategy': 'Short Straddle',
            'strike': atm_call['strike'],
            'total_credit': total_credit,
            'breakeven_up': breakeven_up,
            'breakeven_down': breakeven_down,
            'max_profit': max_profit,
            'max_loss': max_loss
        }
    
    def _calculate_probability_of_profit(
        self, 
        current_price: float, 
        strike: float, 
        premium: float, 
        option_type: str,
        direction: str
    ) -> float:
        """Calculate probability of profit (simplified)"""
        if direction == 'long':
            if option_type == 'call':
                breakeven = strike + premium
                # Simplified: assume 50% chance if breakeven is close to current price
                if breakeven <= current_price * 1.05:
                    return 0.6
                elif breakeven <= current_price * 1.1:
                    return 0.5
                else:
                    return 0.4
            else:  # put
                breakeven = strike - premium
                if breakeven >= current_price * 0.95:
                    return 0.6
                elif breakeven >= current_price * 0.9:
                    return 0.5
                else:
                    return 0.4
        else:  # short
            if option_type == 'call':
                breakeven = strike + premium
                if breakeven >= current_price * 1.1:
                    return 0.7
                elif breakeven >= current_price * 1.05:
                    return 0.6
                else:
                    return 0.5
            else:  # put
                breakeven = strike - premium
                if breakeven <= current_price * 0.9:
                    return 0.7
                elif breakeven <= current_price * 0.95:
                    return 0.6
                else:
                    return 0.5
        
        return 0.5  # Default 50%

def calculate_expected_value(
    current_price: float,
    strike: float,
    premium: float,
    option_type: str,
    direction: str,
    expected_move: float = 0.0,
    volatility: float = 0.2
) -> Dict[str, float]:
    """
    Calculate expected value of an option trade
    
    Args:
        current_price: Current stock price
        strike: Option strike price
        premium: Option premium
        option_type: 'call' or 'put'
        direction: 'long' or 'short'
        expected_move: Expected price move (as decimal, e.g., 0.05 for 5%)
        volatility: Implied volatility (as decimal)
        
    Returns:
        Dictionary with expected value metrics
    """
    
    # Calculate potential outcomes
    if direction == 'long':
        if option_type == 'call':
            breakeven = strike + premium
            max_profit = float('inf')
            max_loss = premium
        else:  # put
            breakeven = strike - premium
            max_profit = strike - premium
            max_loss = premium
    else:  # short
        if option_type == 'call':
            breakeven = strike + premium
            max_profit = premium
            max_loss = float('inf')
        else:  # put
            breakeven = strike - premium
            max_profit = premium
            max_loss = strike - premium
    
    # Simplified expected value calculation
    # This is a basic implementation - in practice, you'd use more sophisticated models
    
    # Probability of profit (simplified)
    if direction == 'long':
        prob_profit = 0.5 - (abs(expected_move) * 0.1)  # Adjust based on expected move
    else:
        prob_profit = 0.5 + (abs(expected_move) * 0.1)
    
    prob_profit = max(0.1, min(0.9, prob_profit))  # Clamp between 10% and 90%
    
    # Expected value
    if direction == 'long':
        if option_type == 'call':
            expected_profit = max_profit * prob_profit - max_loss * (1 - prob_profit)
        else:  # put
            expected_profit = max_profit * prob_profit - max_loss * (1 - prob_profit)
    else:  # short
        expected_profit = max_profit * prob_profit - max_loss * (1 - prob_profit)
    
    return {
        'expected_value': expected_profit,
        'prob_profit': prob_profit,
        'breakeven': breakeven,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'risk_reward': max_profit / max_loss if max_loss > 0 and max_profit != float('inf') else 0
    }

# Global options analyzer instance
options_analyzer = OptionsAnalyzer()

# Convenience functions
def analyze_options_chain(
    ticker: str, 
    current_price: float,
    options_data: Dict[str, pd.DataFrame],
    direction: str = "long",
    target_delta: float = 0.3
) -> Dict[str, Any]:
    """Analyze options chain and recommend trades"""
    return options_analyzer.analyze_options_chain(
        ticker, current_price, options_data, direction, target_delta
    )

def calculate_expected_value(
    current_price: float,
    strike: float,
    premium: float,
    option_type: str,
    direction: str,
    expected_move: float = 0.0
) -> Dict[str, float]:
    """Calculate expected value of an option trade"""
    return calculate_expected_value(
        current_price, strike, premium, option_type, direction, expected_move
    )
