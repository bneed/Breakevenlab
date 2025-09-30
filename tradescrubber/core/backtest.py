"""
Backtesting engine for TradeScrubber
Implements walk-forward backtesting with various strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BacktestEngine:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.commission = 0.001  # 0.1% commission per trade
        self.slippage = 0.0005   # 0.05% slippage per trade
    
    def backtest_strategy(
        self,
        prices_dict: Dict[str, pd.DataFrame],
        strategy_name: str = "default",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        position_size: float = 0.1,  # 10% of capital per trade
        max_positions: int = 5
    ) -> Dict[str, Any]:
        """
        Backtest a trading strategy
        
        Args:
            prices_dict: Dictionary mapping ticker to DataFrame with OHLCV + indicators + signals
            strategy_name: Name of strategy to backtest
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            position_size: Fraction of capital to risk per trade
            max_positions: Maximum number of concurrent positions
            
        Returns:
            Dictionary with backtest results and metrics
        """
        if not prices_dict:
            return {"error": "No price data available"}
        
        # Filter data by date range
        filtered_data = self._filter_by_date_range(prices_dict, start_date, end_date)
        
        if not filtered_data:
            return {"error": "No data in specified date range"}
        
        # Get strategy rules
        strategy_rules = self._get_strategy_rules(strategy_name)
        
        # Run backtest
        trades, equity_curve = self._run_backtest(
            filtered_data, strategy_rules, position_size, max_positions
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, trades)
        
        return {
            'strategy': strategy_name,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_capital': equity_curve.iloc[-1]['capital'],
            'total_return': (equity_curve.iloc[-1]['capital'] - self.initial_capital) / self.initial_capital,
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics
        }
    
    def _filter_by_date_range(
        self, 
        prices_dict: Dict[str, pd.DataFrame], 
        start_date: Optional[str], 
        end_date: Optional[str]
    ) -> Dict[str, pd.DataFrame]:
        """Filter price data by date range"""
        filtered_data = {}
        
        for ticker, df in prices_dict.items():
            if df.empty:
                continue
            
            filtered_df = df.copy()
            
            if start_date:
                start_dt = pd.to_datetime(start_date)
                filtered_df = filtered_df[filtered_df.index >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                filtered_df = filtered_df[filtered_df.index <= end_dt]
            
            if not filtered_df.empty:
                filtered_data[ticker] = filtered_df
        
        return filtered_data
    
    def _get_strategy_rules(self, strategy_name: str) -> Dict[str, Any]:
        """Get strategy rules for backtesting"""
        strategies = {
            'default': {
                'entry_long': ['bullish_trend', 'above_sma50'],
                'entry_short': ['bearish_trend', 'below_sma50'],
                'exit_long': ['bearish_trend', 'below_sma20'],
                'exit_short': ['bullish_trend', 'above_sma20'],
                'stop_loss_pct': 0.05,  # 5% stop loss
                'take_profit_pct': 0.15  # 15% take profit
            },
            'reversal': {
                'entry_long': ['rsi_oversold', 'above_sma200'],
                'entry_short': ['rsi_overbought', 'below_sma200'],
                'exit_long': ['rsi_overbought'],
                'exit_short': ['rsi_oversold'],
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.12
            },
            'breakout': {
                'entry_long': ['breakout_up', 'volume_spike'],
                'entry_short': ['breakout_down', 'volume_spike'],
                'exit_long': ['below_sma20'],
                'exit_short': ['above_sma20'],
                'stop_loss_pct': 0.06,
                'take_profit_pct': 0.20
            },
            'trend': {
                'entry_long': ['golden_cross', 'above_sma200'],
                'entry_short': ['death_cross', 'below_sma200'],
                'exit_long': ['death_cross'],
                'exit_short': ['golden_cross'],
                'stop_loss_pct': 0.10,
                'take_profit_pct': 0.30
            }
        }
        
        return strategies.get(strategy_name, strategies['default'])
    
    def _run_backtest(
        self,
        prices_dict: Dict[str, pd.DataFrame],
        strategy_rules: Dict[str, Any],
        position_size: float,
        max_positions: int
    ) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """Run the actual backtest"""
        
        # Initialize tracking variables
        capital = self.initial_capital
        positions = {}  # {ticker: position_info}
        trades = []
        equity_curve = []
        
        # Get all unique dates across all tickers
        all_dates = set()
        for df in prices_dict.values():
            all_dates.update(df.index)
        
        all_dates = sorted(list(all_dates))
        
        # Process each date
        for date in all_dates:
            daily_capital = capital
            
            # Update existing positions
            positions_to_close = []
            for ticker, position in positions.items():
                if ticker not in prices_dict or date not in prices_dict[ticker].index:
                    continue
                
                current_price = prices_dict[ticker].loc[date, 'close']
                position_value = position['shares'] * current_price
                daily_capital += position_value
                
                # Check exit conditions
                should_exit = self._check_exit_conditions(
                    position, current_price, prices_dict[ticker].loc[date], strategy_rules
                )
                
                if should_exit:
                    # Close position
                    trade_pnl = position_value - position['cost']
                    trade_pnl -= position_value * self.commission  # Commission
                    
                    trades.append({
                        'ticker': ticker,
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'shares': position['shares'],
                        'pnl': trade_pnl,
                        'return_pct': trade_pnl / position['cost'],
                        'side': position['side']
                    })
                    
                    capital += trade_pnl
                    positions_to_close.append(ticker)
            
            # Remove closed positions
            for ticker in positions_to_close:
                del positions[ticker]
            
            # Look for new entry signals
            if len(positions) < max_positions:
                for ticker, df in prices_dict.items():
                    if ticker in positions or date not in df.index:
                        continue
                    
                    current_row = df.loc[date]
                    current_price = current_row['close']
                    
                    # Check entry conditions
                    entry_side = self._check_entry_conditions(current_row, strategy_rules)
                    
                    if entry_side:
                        # Calculate position size
                        position_value = capital * position_size
                        shares = int(position_value / current_price)
                        
                        if shares > 0:
                            cost = shares * current_price
                            cost += cost * self.commission  # Commission
                            
                            positions[ticker] = {
                                'shares': shares,
                                'entry_price': current_price,
                                'entry_date': date,
                                'cost': cost,
                                'side': entry_side,
                                'stop_loss': current_price * (1 - strategy_rules['stop_loss_pct']) if entry_side == 'long' else current_price * (1 + strategy_rules['stop_loss_pct']),
                                'take_profit': current_price * (1 + strategy_rules['take_profit_pct']) if entry_side == 'long' else current_price * (1 - strategy_rules['take_profit_pct'])
                            }
                            
                            capital -= cost
            
            # Calculate total portfolio value
            total_value = capital
            for ticker, position in positions.items():
                if ticker in prices_dict and date in prices_dict[ticker].index:
                    current_price = prices_dict[ticker].loc[date, 'close']
                    total_value += position['shares'] * current_price
            
            equity_curve.append({
                'date': date,
                'capital': total_value,
                'positions': len(positions)
            })
        
        # Close any remaining positions at the end
        for ticker, position in positions.items():
            if ticker in prices_dict:
                last_date = prices_dict[ticker].index[-1]
                last_price = prices_dict[ticker].loc[last_date, 'close']
                position_value = position['shares'] * last_price
                trade_pnl = position_value - position['cost']
                trade_pnl -= position_value * self.commission
                
                trades.append({
                    'ticker': ticker,
                    'entry_date': position['entry_date'],
                    'exit_date': last_date,
                    'entry_price': position['entry_price'],
                    'exit_price': last_price,
                    'shares': position['shares'],
                    'pnl': trade_pnl,
                    'return_pct': trade_pnl / position['cost'],
                    'side': position['side']
                })
        
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('date', inplace=True)
        
        return trades, equity_df
    
    def _check_entry_conditions(self, row: pd.Series, strategy_rules: Dict[str, Any]) -> Optional[str]:
        """Check if entry conditions are met"""
        
        # Check long entry conditions
        long_conditions = strategy_rules.get('entry_long', [])
        if long_conditions and all(row.get(condition, False) for condition in long_conditions):
            return 'long'
        
        # Check short entry conditions
        short_conditions = strategy_rules.get('entry_short', [])
        if short_conditions and all(row.get(condition, False) for condition in short_conditions):
            return 'short'
        
        return None
    
    def _check_exit_conditions(
        self, 
        position: Dict[str, Any], 
        current_price: float, 
        current_row: pd.Series,
        strategy_rules: Dict[str, Any]
    ) -> bool:
        """Check if exit conditions are met"""
        
        side = position['side']
        
        # Check stop loss
        if side == 'long' and current_price <= position['stop_loss']:
            return True
        elif side == 'short' and current_price >= position['stop_loss']:
            return True
        
        # Check take profit
        if side == 'long' and current_price >= position['take_profit']:
            return True
        elif side == 'short' and current_price <= position['take_profit']:
            return True
        
        # Check strategy exit conditions
        if side == 'long':
            exit_conditions = strategy_rules.get('exit_long', [])
        else:
            exit_conditions = strategy_rules.get('exit_short', [])
        
        if exit_conditions and any(current_row.get(condition, False) for condition in exit_conditions):
            return True
        
        return False
    
    def _calculate_metrics(self, equity_curve: pd.DataFrame, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate backtest metrics"""
        
        if equity_curve.empty:
            return {}
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1]['capital'] - self.initial_capital) / self.initial_capital
        
        # Calculate daily returns
        equity_curve['daily_return'] = equity_curve['capital'].pct_change()
        daily_returns = equity_curve['daily_return'].dropna()
        
        # CAGR
        days = len(equity_curve)
        years = days / 365.25
        cagr = (equity_curve.iloc[-1]['capital'] / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Maximum drawdown
        rolling_max = equity_curve['capital'].expanding().max()
        drawdown = (equity_curve['capital'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade metrics
        if trades:
            trade_returns = [trade['return_pct'] for trade in trades]
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]
            
            win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf')
            
            total_trades = len(trades)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            total_trades = 0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'avg_trade': np.mean([trade['return_pct'] for trade in trades]) if trades else 0
        }
    
    def compare_strategies(
        self,
        prices_dict: Dict[str, pd.DataFrame],
        strategies: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Compare multiple strategies"""
        
        results = []
        
        for strategy in strategies:
            result = self.backtest_strategy(
                prices_dict, strategy, start_date, end_date
            )
            
            if 'error' not in result:
                metrics = result['metrics']
                results.append({
                    'strategy': strategy,
                    'total_return': metrics.get('total_return', 0),
                    'cagr': metrics.get('cagr', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'total_trades': metrics.get('total_trades', 0)
                })
        
        return pd.DataFrame(results)

# Global backtest engine instance
backtest_engine = BacktestEngine()

# Convenience functions
def backtest_strategy(
    prices_dict: Dict[str, pd.DataFrame],
    strategy_name: str = "default",
    **kwargs
) -> Dict[str, Any]:
    """Backtest a trading strategy"""
    return backtest_engine.backtest_strategy(prices_dict, strategy_name, **kwargs)

def compare_strategies(
    prices_dict: Dict[str, pd.DataFrame],
    strategies: List[str],
    **kwargs
) -> pd.DataFrame:
    """Compare multiple strategies"""
    return backtest_engine.compare_strategies(prices_dict, strategies, **kwargs)
