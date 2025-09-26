"""
Backtesting engine for Break-even Lab
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import streamlit as st
from .data import data_manager
import ta

class BacktestEngine:
    """Backtesting engine for trading strategies"""
    
    def __init__(self):
        self.data_manager = data_manager
    
    def get_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data"""
        df = data.copy()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        
        # Simple Moving Averages
        df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        
        # Volume indicators
        df['volume_sma'] = ta.volume.VolumeSMAIndicator(df['Close'], df['Volume']).volume_sma()
        
        return df
    
    def rsi_strategy(self, data: pd.DataFrame, rsi_oversold: float = 30, 
                    rsi_overbought: float = 70, position_size: float = 1.0) -> pd.DataFrame:
        """
        RSI mean reversion strategy
        
        Args:
            data: Price data with RSI indicator
            rsi_oversold: RSI level to buy
            rsi_overbought: RSI level to sell
            position_size: Position size as fraction of portfolio
        
        Returns:
            DataFrame with trades and performance metrics
        """
        df = data.copy()
        df['position'] = 0
        df['trade_pnl'] = 0.0
        df['cumulative_pnl'] = 0.0
        
        position = 0
        entry_price = 0
        cumulative_pnl = 0
        
        for i in range(1, len(df)):
            current_rsi = df['rsi'].iloc[i]
            current_price = df['Close'].iloc[i]
            
            # Buy signal: RSI oversold and not in position
            if current_rsi < rsi_oversold and position == 0:
                position = position_size
                entry_price = current_price
                df.loc[df.index[i], 'position'] = position
            
            # Sell signal: RSI overbought and in position
            elif current_rsi > rsi_overbought and position > 0:
                trade_pnl = (current_price - entry_price) * position
                cumulative_pnl += trade_pnl
                
                df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                df.loc[df.index[i], 'cumulative_pnl'] = cumulative_pnl
                df.loc[df.index[i], 'position'] = 0
                
                position = 0
                entry_price = 0
            
            else:
                df.loc[df.index[i], 'position'] = position
                df.loc[df.index[i], 'cumulative_pnl'] = cumulative_pnl
        
        return df
    
    def macd_strategy(self, data: pd.DataFrame, position_size: float = 1.0) -> pd.DataFrame:
        """
        MACD crossover strategy
        
        Args:
            data: Price data with MACD indicators
            position_size: Position size as fraction of portfolio
        
        Returns:
            DataFrame with trades and performance metrics
        """
        df = data.copy()
        df['position'] = 0
        df['trade_pnl'] = 0.0
        df['cumulative_pnl'] = 0.0
        
        position = 0
        entry_price = 0
        cumulative_pnl = 0
        
        for i in range(1, len(df)):
            current_macd = df['macd'].iloc[i]
            current_signal = df['macd_signal'].iloc[i]
            prev_macd = df['macd'].iloc[i-1]
            prev_signal = df['macd_signal'].iloc[i-1]
            current_price = df['Close'].iloc[i]
            
            # Buy signal: MACD crosses above signal
            if current_macd > current_signal and prev_macd <= prev_signal and position == 0:
                position = position_size
                entry_price = current_price
                df.loc[df.index[i], 'position'] = position
            
            # Sell signal: MACD crosses below signal
            elif current_macd < current_signal and prev_macd >= prev_signal and position > 0:
                trade_pnl = (current_price - entry_price) * position
                cumulative_pnl += trade_pnl
                
                df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                df.loc[df.index[i], 'cumulative_pnl'] = cumulative_pnl
                df.loc[df.index[i], 'position'] = 0
                
                position = 0
                entry_price = 0
            
            else:
                df.loc[df.index[i], 'position'] = position
                df.loc[df.index[i], 'cumulative_pnl'] = cumulative_pnl
        
        return df
    
    def covered_call_strategy(self, data: pd.DataFrame, strike_delta: float = 0.3, 
                            roll_dte: int = 7, position_size: float = 1.0) -> pd.DataFrame:
        """
        Covered call strategy backtest
        
        Args:
            data: Price data
            strike_delta: Target delta for strike selection
            roll_dte: Days to expiration to roll
            position_size: Position size as fraction of portfolio
        
        Returns:
            DataFrame with trades and performance metrics
        """
        df = data.copy()
        df['position'] = 0
        df['option_position'] = 0
        df['trade_pnl'] = 0.0
        df['cumulative_pnl'] = 0.0
        
        position = 0
        option_position = 0
        entry_price = 0
        option_premium = 0
        cumulative_pnl = 0
        
        # Simplified covered call logic
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            
            # Entry: Buy stock and sell call
            if position == 0:
                position = position_size
                option_position = -position_size  # Short call
                entry_price = current_price
                option_premium = current_price * 0.02  # Assume 2% premium
                
                df.loc[df.index[i], 'position'] = position
                df.loc[df.index[i], 'option_position'] = option_position
            
            # Exit: Close positions
            elif position > 0:
                # Simplified exit logic - close after 30 days
                if i % 30 == 0:
                    trade_pnl = (current_price - entry_price) * position + option_premium
                    cumulative_pnl += trade_pnl
                    
                    df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                    df.loc[df.index[i], 'cumulative_pnl'] = cumulative_pnl
                    df.loc[df.index[i], 'position'] = 0
                    df.loc[df.index[i], 'option_position'] = 0
                    
                    position = 0
                    option_position = 0
                    entry_price = 0
                    option_premium = 0
            
            else:
                df.loc[df.index[i], 'position'] = position
                df.loc[df.index[i], 'option_position'] = option_position
                df.loc[df.index[i], 'cumulative_pnl'] = cumulative_pnl
        
        return df
    
    def calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for the backtest"""
        if data.empty or 'cumulative_pnl' not in data.columns:
            return {}
        
        # Basic metrics
        total_return = data['cumulative_pnl'].iloc[-1]
        initial_capital = 10000  # Assume $10,000 starting capital
        final_capital = initial_capital + total_return
        
        # Calculate returns
        data['returns'] = data['cumulative_pnl'].pct_change().fillna(0)
        
        # CAGR
        years = len(data) / 252  # Assuming 252 trading days per year
        if years > 0:
            cagr = (final_capital / initial_capital) ** (1 / years) - 1
        else:
            cagr = 0
        
        # Maximum drawdown
        peak = data['cumulative_pnl'].expanding().max()
        drawdown = (data['cumulative_pnl'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate
        trades = data[data['trade_pnl'] != 0]['trade_pnl']
        if len(trades) > 0:
            winning_trades = len(trades[trades > 0])
            win_rate = winning_trades / len(trades)
        else:
            win_rate = 0
        
        # Average trade
        avg_trade = trades.mean() if len(trades) > 0 else 0
        
        # Sharpe ratio (simplified)
        if data['returns'].std() > 0:
            sharpe_ratio = data['returns'].mean() / data['returns'].std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        return {
            'total_return': total_return,
            'cagr': cagr * 100,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate * 100,
            'avg_trade': avg_trade,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(trades),
            'winning_trades': len(trades[trades > 0]) if len(trades) > 0 else 0,
            'losing_trades': len(trades[trades < 0]) if len(trades) > 0 else 0
        }
    
    def run_backtest(self, symbol: str, strategy: str, start_date: str = None, 
                    end_date: str = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Run a backtest for a given symbol and strategy
        
        Args:
            symbol: Stock symbol
            strategy: Strategy name ('rsi', 'macd', 'covered_call')
            start_date: Start date for backtest
            end_date: End date for backtest
            **kwargs: Strategy-specific parameters
        
        Returns:
            Tuple of (results DataFrame, performance metrics)
        """
        # Get data
        data = self.data_manager.get_stock_data(symbol, "1y")
        if data.empty:
            return pd.DataFrame(), {}
        
        # Add technical indicators
        data = self.get_technical_indicators(data)
        
        # Run strategy
        if strategy == 'rsi':
            results = self.rsi_strategy(data, **kwargs)
        elif strategy == 'macd':
            results = self.macd_strategy(data, **kwargs)
        elif strategy == 'covered_call':
            results = self.covered_call_strategy(data, **kwargs)
        else:
            st.error(f"Unknown strategy: {strategy}")
            return pd.DataFrame(), {}
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(results)
        
        return results, metrics

# Global backtest engine instance
backtest_engine = BacktestEngine()

def run_backtest(symbol: str, strategy: str, start_date: str = None, 
                end_date: str = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run a backtest for a given symbol and strategy"""
    return backtest_engine.run_backtest(symbol, strategy, start_date, end_date, **kwargs)
