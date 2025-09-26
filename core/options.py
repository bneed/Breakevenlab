"""
Options pricing, Greeks, and P/L calculations for Break-even Lab
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import streamlit as st

class OptionsPricer:
    """Black-Scholes options pricing and Greeks calculator"""
    
    @staticmethod
    def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """
        Black-Scholes option pricing formula
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        
        Returns:
            Option price
        """
        if T <= 0:
            # At expiration
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks for an option
        
        Returns:
            Dictionary with delta, gamma, theta, vega, rho
        """
        if T <= 0:
            # At expiration, most Greeks are 0
            if option_type == 'call':
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0
            
            return {
                'delta': delta,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        theta_part1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if option_type == 'call':
            theta_part2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta_part2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        theta = (theta_part1 + theta_part2) / 365  # Per day
        
        # Vega (same for calls and puts)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in volatility
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% change in rate
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    @staticmethod
    def calculate_implied_volatility(S: float, K: float, T: float, r: float, market_price: float, option_type: str = 'call') -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Returns:
            Implied volatility
        """
        if T <= 0:
            return 0.0
        
        # Initial guess
        sigma = 0.3
        
        for _ in range(100):  # Max iterations
            price = OptionsPricer.black_scholes(S, K, T, r, sigma, option_type)
            
            # Calculate vega for Newton-Raphson
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            
            if vega == 0:
                break
            
            # Newton-Raphson update
            diff = price - market_price
            if abs(diff) < 0.01:  # Convergence
                break
            
            sigma = sigma - diff / vega
            
            # Keep sigma in reasonable bounds
            sigma = max(0.01, min(5.0, sigma))
        
        return sigma

class OptionsStrategy:
    """Multi-leg options strategy analysis"""
    
    def __init__(self):
        self.legs = []
    
    def add_leg(self, option_type: str, quantity: int, strike: float, expiration: str, 
                entry_price: float, action: str = 'buy'):
        """
        Add a leg to the strategy
        
        Args:
            option_type: 'call' or 'put'
            quantity: Number of contracts (positive for long, negative for short)
            strike: Strike price
            expiration: Expiration date
            entry_price: Entry price per contract
            action: 'buy' or 'sell'
        """
        if action == 'sell':
            quantity = -quantity
        
        self.legs.append({
            'type': option_type,
            'quantity': quantity,
            'strike': strike,
            'expiration': expiration,
            'entry_price': entry_price
        })
    
    def calculate_pnl_at_expiration(self, stock_price: float) -> float:
        """Calculate P/L at expiration for given stock price"""
        total_pnl = 0.0
        
        for leg in self.legs:
            if leg['type'] == 'call':
                intrinsic_value = max(stock_price - leg['strike'], 0)
            else:
                intrinsic_value = max(leg['strike'] - stock_price, 0)
            
            # P/L = (intrinsic_value - entry_price) * quantity * 100
            leg_pnl = (intrinsic_value - leg['entry_price']) * leg['quantity'] * 100
            total_pnl += leg_pnl
        
        return total_pnl
    
    def calculate_current_pnl(self, stock_price: float, time_to_exp: float, 
                            risk_free_rate: float = 0.05, volatility: float = 0.3) -> float:
        """Calculate current P/L using Black-Scholes pricing"""
        total_pnl = 0.0
        
        for leg in self.legs:
            # Calculate theoretical value
            theoretical_value = OptionsPricer.black_scholes(
                stock_price, leg['strike'], time_to_exp, risk_free_rate, volatility, leg['type']
            )
            
            # P/L = (theoretical_value - entry_price) * quantity * 100
            leg_pnl = (theoretical_value - leg['entry_price']) * leg['quantity'] * 100
            total_pnl += leg_pnl
        
        return total_pnl
    
    def calculate_total_greeks(self, stock_price: float, time_to_exp: float, 
                             risk_free_rate: float = 0.05, volatility: float = 0.3) -> Dict[str, float]:
        """Calculate total Greeks for the strategy"""
        total_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        
        for leg in self.legs:
            greeks = OptionsPricer.calculate_greeks(
                stock_price, leg['strike'], time_to_exp, risk_free_rate, volatility, leg['type']
            )
            
            # Multiply by quantity and contract multiplier
            for greek in total_greeks:
                total_greeks[greek] += greeks[greek] * leg['quantity'] * 100
        
        return total_greeks
    
    def find_break_even_points(self, min_price: float = 0, max_price: float = 1000, 
                             step: float = 0.1) -> List[float]:
        """Find break-even points for the strategy"""
        break_even_points = []
        
        for price in np.arange(min_price, max_price, step):
            pnl = self.calculate_pnl_at_expiration(price)
            if abs(pnl) < 1.0:  # Within $1 of break-even
                break_even_points.append(price)
        
        return break_even_points
    
    def get_max_profit_loss(self) -> Tuple[float, float]:
        """Get maximum profit and loss for the strategy"""
        # This is a simplified version - in practice, you'd need to consider
        # the specific strategy type and calculate accordingly
        
        max_profit = float('inf')
        max_loss = float('inf')
        
        # For now, return placeholder values
        # In a full implementation, you'd analyze the strategy structure
        return max_profit, max_loss

def create_strategy_from_input(legs_data: List[Dict]) -> OptionsStrategy:
    """Create an options strategy from user input data"""
    strategy = OptionsStrategy()
    
    for leg in legs_data:
        strategy.add_leg(
            option_type=leg['type'],
            quantity=leg['quantity'],
            strike=leg['strike'],
            expiration=leg['expiration'],
            entry_price=leg['entry_price'],
            action=leg.get('action', 'buy')
        )
    
    return strategy
