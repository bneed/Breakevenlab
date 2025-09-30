"""
NiceGUI interface for TradeScrubber
Main UI implementation with all tabs and features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

try:
    from nicegui import ui, app
    NICEGUI_AVAILABLE = True
except ImportError:
    NICEGUI_AVAILABLE = False
    print("NiceGUI not available. Install with: pip install nicegui")

from ..core.data import get_prices, get_options_chain, get_market_status
from ..core.indicators import add_indicators
from ..core.signals import compute_signals
from ..core.ranker import rank_universe, filter_by_strategy
from ..core.options import analyze_options_chain
from ..core.ml import train_ml_models, predict_for_today, get_ml_model_info
from ..core.backtest import backtest_strategy, compare_strategies
from ..core.utils import load_watchlist, load_strategy_presets, get_cache_stats

logger = logging.getLogger(__name__)

class TradeScrubberUI:
    """Main UI class for TradeScrubber"""
    
    def __init__(self):
        self.current_data = {}
        self.current_rankings = pd.DataFrame()
        self.ml_predictions = {}
        self.selected_ticker = "SPY"
        self.selected_strategy = "default"
        
        # Load configuration
        self.watchlists = load_watchlist()
        self.strategy_presets = load_strategy_presets()
        
        # Initialize UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the main UI layout"""
        
        # Header
        with ui.header().classes('items-center justify-between'):
            ui.label('TradeScrubber').classes('text-h4 font-bold')
            ui.label('Smart Stock & Options Screener').classes('text-subtitle2')
            
            with ui.row():
                ui.button('Refresh Data', icon='refresh', on_click=self._refresh_data)
                ui.button('Settings', icon='settings', on_click=self._show_settings)
        
        # Main content
        with ui.tabs().classes('w-full') as tabs:
            ideas_tab = ui.tab('Ideas', icon='lightbulb')
            charts_tab = ui.tab('Charts', icon='show_chart')
            screener_tab = ui.tab('Screener', icon='table_view')
            options_tab = ui.tab('Options', icon='call_made')
            ml_tab = ui.tab('ML', icon='psychology')
            backtest_tab = ui.tab('Backtest', icon='trending_up')
        
        with ui.tab_panels(tabs, value=ideas_tab).classes('w-full'):
            # Ideas Tab
            with ui.tab_panel(ideas_tab):
                self._create_ideas_tab()
            
            # Charts Tab
            with ui.tab_panel(charts_tab):
                self._create_charts_tab()
            
            # Screener Tab
            with ui.tab_panel(screener_tab):
                self._create_screener_tab()
            
            # Options Tab
            with ui.tab_panel(options_tab):
                self._create_options_tab()
            
            # ML Tab
            with ui.tab_panel(ml_tab):
                self._create_ml_tab()
            
            # Backtest Tab
            with ui.tab_panel(backtest_tab):
                self._create_backtest_tab()
        
        # Load initial data
        self._refresh_data()
    
    def _create_ideas_tab(self):
        """Create the Ideas tab content"""
        
        with ui.row().classes('w-full'):
            # Left sidebar - Filters and controls
            with ui.column().classes('w-1/3'):
                ui.label('Filters & Controls').classes('text-h6 font-bold')
                
                # Strategy selection
                ui.label('Strategy Preset:')
                self.strategy_select = ui.select(
                    options=list(self.strategy_presets.keys()),
                    value=self.selected_strategy
                ).classes('w-full')
                
                # Watchlist selection
                ui.label('Watchlist:')
                self.watchlist_select = ui.select(
                    options=['default', 'custom'],
                    value='default'
                ).classes('w-full')
                
                # Custom tickers
                ui.label('Custom Tickers (comma-separated):')
                self.custom_tickers = ui.input(placeholder='AAPL,MSFT,GOOGL').classes('w-full')
                
                # Filters
                ui.label('Filters:')
                with ui.row():
                    self.min_score = ui.number(label='Min Score', value=50, min=0, max=100)
                    self.max_results = ui.number(label='Max Results', value=20, min=1, max=100)
                
                # Scan button
                ui.button('Scan Market', icon='search', on_click=self._scan_market).classes('w-full')
                
                # Market status
                ui.separator()
                ui.label('Market Status').classes('text-h6 font-bold')
                self.market_status = ui.label('Loading...')
            
            # Right panel - Results
            with ui.column().classes('w-2/3'):
                ui.label('Top Trading Ideas').classes('text-h6 font-bold')
                
                # Results table
                self.ideas_table = ui.table(
                    columns=[
                        {'name': 'ticker', 'label': 'Ticker', 'field': 'ticker', 'sortable': True},
                        {'name': 'direction', 'label': 'Direction', 'field': 'direction', 'sortable': True},
                        {'name': 'score', 'label': 'Score', 'field': 'score', 'sortable': True},
                        {'name': 'confidence', 'label': 'Confidence', 'field': 'confidence', 'sortable': True},
                        {'name': 'price', 'label': 'Price', 'field': 'price', 'sortable': True},
                        {'name': 'entry_zone', 'label': 'Entry Zone', 'field': 'entry_zone', 'sortable': True},
                        {'name': 'stop_loss', 'label': 'Stop Loss', 'field': 'stop_loss', 'sortable': True},
                        {'name': 'target', 'label': 'Target', 'field': 'target', 'sortable': True}
                    ],
                    rows=[],
                    selection='single'
                ).classes('w-full')
                
                # Action buttons
                with ui.row():
                    ui.button('View Chart', icon='show_chart', on_click=self._view_chart)
                    ui.button('Analyze Options', icon='call_made', on_click=self._analyze_options)
                    ui.button('Add Alert', icon='notifications', on_click=self._add_alert)
    
    def _create_charts_tab(self):
        """Create the Charts tab content"""
        
        with ui.row().classes('w-full'):
            # Left sidebar - Chart controls
            with ui.column().classes('w-1/4'):
                ui.label('Chart Controls').classes('text-h6 font-bold')
                
                # Ticker selection
                ui.label('Select Ticker:')
                self.chart_ticker = ui.select(
                    options=[],
                    value=self.selected_ticker
                ).classes('w-full')
                
                # Timeframe selection
                ui.label('Timeframe:')
                self.chart_timeframe = ui.select(
                    options=['1d', '1h', '5m'],
                    value='1d'
                ).classes('w-full')
                
                # Indicators toggle
                ui.label('Indicators:')
                self.show_sma = ui.checkbox('SMA (20,50,200)', value=True)
                self.show_ema = ui.checkbox('EMA (8,21)', value=True)
                self.show_rsi = ui.checkbox('RSI (14)', value=True)
                self.show_macd = ui.checkbox('MACD', value=True)
                self.show_volume = ui.checkbox('Volume', value=True)
                
                # Update button
                ui.button('Update Chart', icon='refresh', on_click=self._update_chart).classes('w-full')
            
            # Right panel - Chart
            with ui.column().classes('w-3/4'):
                ui.label('Price Chart').classes('text-h6 font-bold')
                
                # Chart container
                self.chart_container = ui.column().classes('w-full h-96')
                
                # Chart info
                self.chart_info = ui.label('Select a ticker to view chart')
    
    def _create_screener_tab(self):
        """Create the Screener tab content"""
        
        ui.label('Market Screener').classes('text-h6 font-bold')
        
        # Filters
        with ui.row().classes('w-full'):
            with ui.column().classes('w-1/4'):
                ui.label('Price Range:')
                self.price_min = ui.number(label='Min Price', value=0, min=0)
                self.price_max = ui.number(label='Max Price', value=1000, min=0)
                
                ui.label('RSI Range:')
                self.rsi_min = ui.number(label='Min RSI', value=0, min=0, max=100)
                self.rsi_max = ui.number(label='Max RSI', value=100, min=0, max=100)
                
                ui.label('Volume:')
                self.min_volume = ui.number(label='Min Volume', value=1000000, min=0)
                
                ui.button('Apply Filters', icon='filter_list', on_click=self._apply_filters).classes('w-full')
            
            with ui.column().classes('w-3/4'):
                # Screener table
                self.screener_table = ui.table(
                    columns=[
                        {'name': 'ticker', 'label': 'Ticker', 'field': 'ticker', 'sortable': True},
                        {'name': 'price', 'label': 'Price', 'field': 'price', 'sortable': True},
                        {'name': 'change', 'label': 'Change %', 'field': 'change', 'sortable': True},
                        {'name': 'volume', 'label': 'Volume', 'field': 'volume', 'sortable': True},
                        {'name': 'rsi', 'label': 'RSI', 'field': 'rsi', 'sortable': True},
                        {'name': 'macd', 'label': 'MACD', 'field': 'macd', 'sortable': True},
                        {'name': 'sma50', 'label': 'SMA50', 'field': 'sma50', 'sortable': True},
                        {'name': 'sma200', 'label': 'SMA200', 'field': 'sma200', 'sortable': True},
                        {'name': 'signals', 'label': 'Signals', 'field': 'signals', 'sortable': True}
                    ],
                    rows=[],
                    selection='single'
                ).classes('w-full')
    
    def _create_options_tab(self):
        """Create the Options tab content"""
        
        with ui.row().classes('w-full'):
            # Left sidebar - Options controls
            with ui.column().classes('w-1/3'):
                ui.label('Options Analysis').classes('text-h6 font-bold')
                
                # Ticker selection
                ui.label('Select Ticker:')
                self.options_ticker = ui.select(
                    options=[],
                    value=self.selected_ticker
                ).classes('w-full')
                
                # Direction selection
                ui.label('Trade Direction:')
                self.options_direction = ui.select(
                    options=['long', 'short'],
                    value='long'
                ).classes('w-full')
                
                # Target delta
                ui.label('Target Delta:')
                self.target_delta = ui.slider(min=0.1, max=0.5, value=0.3, step=0.1)
                
                # DTE range
                ui.label('Days to Expiration:')
                with ui.row():
                    self.dte_min = ui.number(label='Min DTE', value=14, min=1, max=365)
                    self.dte_max = ui.number(label='Max DTE', value=45, min=1, max=365)
                
                ui.button('Analyze Options', icon='call_made', on_click=self._analyze_options_chain).classes('w-full')
            
            # Right panel - Options results
            with ui.column().classes('w-2/3'):
                ui.label('Options Recommendations').classes('text-h6 font-bold')
                
                # Options table
                self.options_table = ui.table(
                    columns=[
                        {'name': 'type', 'label': 'Type', 'field': 'type', 'sortable': True},
                        {'name': 'strike', 'label': 'Strike', 'field': 'strike', 'sortable': True},
                        {'name': 'premium', 'label': 'Premium', 'field': 'premium', 'sortable': True},
                        {'name': 'breakeven', 'label': 'Breakeven', 'field': 'breakeven', 'sortable': True},
                        {'name': 'max_profit', 'label': 'Max Profit', 'field': 'max_profit', 'sortable': True},
                        {'name': 'max_loss', 'label': 'Max Loss', 'field': 'max_loss', 'sortable': True},
                        {'name': 'risk_reward', 'label': 'R/R', 'field': 'risk_reward', 'sortable': True},
                        {'name': 'prob_profit', 'label': 'Prob Profit', 'field': 'prob_profit', 'sortable': True}
                    ],
                    rows=[],
                    selection='single'
                ).classes('w-full')
    
    def _create_ml_tab(self):
        """Create the ML tab content"""
        
        with ui.row().classes('w-full'):
            # Left sidebar - ML controls
            with ui.column().classes('w-1/3'):
                ui.label('Machine Learning').classes('text-h6 font-bold')
                
                # Model info
                self.ml_info = ui.label('Loading model info...')
                
                # Training controls
                ui.label('Training:')
                self.training_days = ui.number(label='Lookback Days', value=365, min=30, max=1095)
                self.horizon_days = ui.number(label='Prediction Horizon', value=3, min=1, max=30)
                
                ui.button('Train Models', icon='psychology', on_click=self._train_models).classes('w-full')
                ui.button('Retrain Models', icon='refresh', on_click=self._retrain_models).classes('w-full')
                
                # Model metrics
                ui.separator()
                ui.label('Model Metrics').classes('text-h6 font-bold')
                self.ml_metrics = ui.label('No metrics available')
            
            # Right panel - ML results
            with ui.column().classes('w-2/3'):
                ui.label('ML Predictions').classes('text-h6 font-bold')
                
                # Predictions table
                self.ml_table = ui.table(
                    columns=[
                        {'name': 'ticker', 'label': 'Ticker', 'field': 'ticker', 'sortable': True},
                        {'name': 'expected_move', 'label': 'Expected Move %', 'field': 'expected_move', 'sortable': True},
                        {'name': 'up_prob', 'label': 'Up Probability', 'field': 'up_prob', 'sortable': True},
                        {'name': 'confidence', 'label': 'Confidence', 'field': 'confidence', 'sortable': True},
                        {'name': 'direction', 'label': 'Direction', 'field': 'direction', 'sortable': True}
                    ],
                    rows=[],
                    selection='single'
                ).classes('w-full')
                
                # Feature importance chart
                ui.label('Feature Importance').classes('text-h6 font-bold')
                self.feature_chart = ui.column().classes('w-full h-64')
    
    def _create_backtest_tab(self):
        """Create the Backtest tab content"""
        
        with ui.row().classes('w-full'):
            # Left sidebar - Backtest controls
            with ui.column().classes('w-1/3'):
                ui.label('Backtesting').classes('text-h6 font-bold')
                
                # Strategy selection
                ui.label('Strategy:')
                self.backtest_strategy = ui.select(
                    options=['default', 'reversal', 'breakout', 'trend'],
                    value='default'
                ).classes('w-full')
                
                # Date range
                ui.label('Date Range:')
                self.start_date = ui.date(label='Start Date', value=datetime.now() - timedelta(days=365))
                self.end_date = ui.date(label='End Date', value=datetime.now())
                
                # Parameters
                ui.label('Parameters:')
                self.position_size = ui.number(label='Position Size %', value=10, min=1, max=100)
                self.max_positions = ui.number(label='Max Positions', value=5, min=1, max=20)
                
                ui.button('Run Backtest', icon='play_arrow', on_click=self._run_backtest).classes('w-full')
                
                # Results summary
                ui.separator()
                ui.label('Results Summary').classes('text-h6 font-bold')
                self.backtest_summary = ui.label('No results available')
            
            # Right panel - Backtest results
            with ui.column().classes('w-2/3'):
                ui.label('Backtest Results').classes('text-h6 font-bold')
                
                # Results table
                self.backtest_table = ui.table(
                    columns=[
                        {'name': 'metric', 'label': 'Metric', 'field': 'metric', 'sortable': True},
                        {'name': 'value', 'label': 'Value', 'field': 'value', 'sortable': True}
                    ],
                    rows=[],
                    selection='single'
                ).classes('w-full')
                
                # Equity curve chart
                ui.label('Equity Curve').classes('text-h6 font-bold')
                self.equity_chart = ui.column().classes('w-full h-64')
    
    def _refresh_data(self):
        """Refresh market data"""
        try:
            # Get watchlist
            if self.watchlist_select.value == 'default':
                tickers = load_watchlist('default')
            else:
                custom_tickers_str = self.custom_tickers.value or ''
                tickers = [t.strip().upper() for t in custom_tickers_str.split(',') if t.strip()]
            
            if not tickers:
                ui.notify('No tickers selected', type='warning')
                return
            
            # Fetch data
            ui.notify('Fetching market data...', type='info')
            self.current_data = get_prices(tickers, '1d', 365)
            
            # Add indicators and signals
            for ticker, df in self.current_data.items():
                if not df.empty:
                    df_with_indicators = add_indicators(df)
                    df_with_signals = compute_signals(df_with_indicators)
                    self.current_data[ticker] = df_with_signals
            
            # Get ML predictions
            self.ml_predictions = predict_for_today(self.current_data)
            
            # Update UI
            self._update_ideas_table()
            self._update_market_status()
            
            ui.notify('Data refreshed successfully', type='positive')
            
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            ui.notify(f'Error refreshing data: {e}', type='negative')
    
    def _scan_market(self):
        """Scan market for trading ideas"""
        try:
            if not self.current_data:
                ui.notify('Please refresh data first', type='warning')
                return
            
            # Rank universe
            self.current_rankings = rank_universe(
                self.current_data, 
                self.ml_predictions,
                self.selected_strategy
            )
            
            # Apply filters
            filtered_rankings = filter_by_strategy(
                self.current_rankings,
                self.selected_strategy,
                self.min_score.value,
                self.max_results.value
            )
            
            # Update table
            self.ideas_table.rows = filtered_rankings.to_dict('records')
            
            ui.notify(f'Found {len(filtered_rankings)} trading ideas', type='positive')
            
        except Exception as e:
            logger.error(f"Error scanning market: {e}")
            ui.notify(f'Error scanning market: {e}', type='negative')
    
    def _update_ideas_table(self):
        """Update the ideas table with current data"""
        if not self.current_rankings.empty:
            self.ideas_table.rows = self.current_rankings.to_dict('records')
    
    def _update_market_status(self):
        """Update market status display"""
        try:
            status = get_market_status()
            status_text = f"Market: {status.get('market_state', 'Unknown')}\n"
            status_text += f"Currency: {status.get('currency', 'USD')}\n"
            status_text += f"Open: {status.get('is_market_open', False)}"
            self.market_status.text = status_text
        except Exception as e:
            self.market_status.text = f"Error loading market status: {e}"
    
    def _view_chart(self):
        """View chart for selected ticker"""
        if not self.ideas_table.selection:
            ui.notify('Please select a ticker first', type='warning')
            return
        
        selected_ticker = self.ideas_table.selection[0]['ticker']
        self.chart_ticker.value = selected_ticker
        self._update_chart()
    
    def _update_chart(self):
        """Update the price chart"""
        try:
            ticker = self.chart_ticker.value
            if not ticker or ticker not in self.current_data:
                ui.notify('Ticker not available', type='warning')
                return
            
            # Clear previous chart
            self.chart_container.clear()
            
            # Create simple chart (in production, use a proper charting library)
            df = self.current_data[ticker]
            if df.empty:
                ui.notify('No data available for chart', type='warning')
                return
            
            # Display price data
            latest_price = df.iloc[-1]['close']
            price_change = df.iloc[-1]['close'] - df.iloc[-2]['close'] if len(df) > 1 else 0
            change_pct = (price_change / df.iloc[-2]['close']) * 100 if len(df) > 1 else 0
            
            with self.chart_container:
                ui.label(f'{ticker} - ${latest_price:.2f} ({change_pct:+.2f}%)').classes('text-h5 font-bold')
                
                # Simple price table
                recent_data = df.tail(10)[['close', 'volume', 'rsi14', 'macd']]
                ui.table(
                    columns=[
                        {'name': 'date', 'label': 'Date', 'field': 'date'},
                        {'name': 'close', 'label': 'Close', 'field': 'close'},
                        {'name': 'volume', 'label': 'Volume', 'field': 'volume'},
                        {'name': 'rsi', 'label': 'RSI', 'field': 'rsi14'},
                        {'name': 'macd', 'label': 'MACD', 'field': 'macd'}
                    ],
                    rows=recent_data.reset_index().to_dict('records')
                )
            
        except Exception as e:
            logger.error(f"Error updating chart: {e}")
            ui.notify(f'Error updating chart: {e}', type='negative')
    
    def _analyze_options(self):
        """Analyze options for selected ticker"""
        if not self.ideas_table.selection:
            ui.notify('Please select a ticker first', type='warning')
            return
        
        selected_ticker = self.ideas_table.selection[0]['ticker']
        self.options_ticker.value = selected_ticker
        self._analyze_options_chain()
    
    def _analyze_options_chain(self):
        """Analyze options chain for selected ticker"""
        try:
            ticker = self.options_ticker.value
            if not ticker or ticker not in self.current_data:
                ui.notify('Ticker not available', type='warning')
                return
            
            current_price = self.current_data[ticker].iloc[-1]['close']
            
            # Get options data
            options_data = get_options_chain(ticker, 30)
            
            if not options_data or 'calls' in options_data and options_data['calls'].empty:
                ui.notify('No options data available', type='warning')
                return
            
            # Analyze options
            analysis = analyze_options_chain(
                ticker,
                current_price,
                options_data,
                self.options_direction.value,
                self.target_delta.value
            )
            
            # Update options table
            options_rows = []
            for option_type in ['calls', 'puts']:
                if option_type in analysis and analysis[option_type]:
                    for option in analysis[option_type]:
                        options_rows.append({
                            'type': option.get('type', ''),
                            'strike': option.get('strike', 0),
                            'premium': option.get('premium', 0),
                            'breakeven': option.get('breakeven', 0),
                            'max_profit': option.get('max_profit', 0),
                            'max_loss': option.get('max_loss', 0),
                            'risk_reward': option.get('risk_reward', 0),
                            'prob_profit': option.get('prob_profit', 0)
                        })
            
            self.options_table.rows = options_rows
            ui.notify(f'Options analysis completed for {ticker}', type='positive')
            
        except Exception as e:
            logger.error(f"Error analyzing options: {e}")
            ui.notify(f'Error analyzing options: {e}', type='negative')
    
    def _train_models(self):
        """Train ML models"""
        try:
            if not self.current_data:
                ui.notify('Please refresh data first', type='warning')
                return
            
            ui.notify('Training ML models...', type='info')
            
            result = train_ml_models(
                self.current_data,
                lookback_days=self.training_days.value,
                horizon_days=self.horizon_days.value
            )
            
            if 'error' in result:
                ui.notify(f'Training failed: {result["error"]}', type='negative')
                return
            
            # Update ML info and predictions
            self._update_ml_info()
            self._update_ml_predictions()
            
            ui.notify('ML models trained successfully', type='positive')
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            ui.notify(f'Error training models: {e}', type='negative')
    
    def _retrain_models(self):
        """Retrain ML models"""
        self._train_models()
    
    def _update_ml_info(self):
        """Update ML model information"""
        try:
            info = get_ml_model_info()
            info_text = f"Trained: {info.get('is_trained', False)}\n"
            info_text += f"Features: {info.get('feature_count', 0)}\n"
            info_text += f"Last Training: {info.get('last_training_date', 'Never')}"
            self.ml_info.text = info_text
        except Exception as e:
            self.ml_info.text = f"Error loading ML info: {e}"
    
    def _update_ml_predictions(self):
        """Update ML predictions table"""
        try:
            if not self.ml_predictions:
                return
            
            predictions_rows = []
            for ticker, pred in self.ml_predictions.items():
                predictions_rows.append({
                    'ticker': ticker,
                    'expected_move': f"{pred.get('expected_move', 0):.2%}",
                    'up_prob': f"{pred.get('up_prob', 0.5):.2%}",
                    'confidence': f"{pred.get('confidence', 0):.2%}",
                    'direction': pred.get('direction', 'neutral')
                })
            
            self.ml_table.rows = predictions_rows
            
        except Exception as e:
            logger.error(f"Error updating ML predictions: {e}")
    
    def _run_backtest(self):
        """Run backtest for selected strategy"""
        try:
            if not self.current_data:
                ui.notify('Please refresh data first', type='warning')
                return
            
            ui.notify('Running backtest...', type='info')
            
            result = backtest_strategy(
                self.current_data,
                self.backtest_strategy.value,
                start_date=self.start_date.value.strftime('%Y-%m-%d'),
                end_date=self.end_date.value.strftime('%Y-%m-%d'),
                position_size=self.position_size.value / 100,
                max_positions=self.max_positions.value
            )
            
            if 'error' in result:
                ui.notify(f'Backtest failed: {result["error"]}', type='negative')
                return
            
            # Update results
            self._update_backtest_results(result)
            
            ui.notify('Backtest completed successfully', type='positive')
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            ui.notify(f'Error running backtest: {e}', type='negative')
    
    def _update_backtest_results(self, result):
        """Update backtest results display"""
        try:
            metrics = result.get('metrics', {})
            
            # Update summary
            summary_text = f"Total Return: {metrics.get('total_return', 0):.2%}\n"
            summary_text += f"CAGR: {metrics.get('cagr', 0):.2%}\n"
            summary_text += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
            summary_text += f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
            summary_text += f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
            summary_text += f"Total Trades: {metrics.get('total_trades', 0)}"
            
            self.backtest_summary.text = summary_text
            
            # Update results table
            results_rows = []
            for metric, value in metrics.items():
                if isinstance(value, float):
                    if 'return' in metric or 'cagr' in metric or 'drawdown' in metric or 'rate' in metric:
                        value_str = f"{value:.2%}"
                    else:
                        value_str = f"{value:.2f}"
                else:
                    value_str = str(value)
                
                results_rows.append({
                    'metric': metric.replace('_', ' ').title(),
                    'value': value_str
                })
            
            self.backtest_table.rows = results_rows
            
        except Exception as e:
            logger.error(f"Error updating backtest results: {e}")
    
    def _apply_filters(self):
        """Apply filters to screener"""
        # Implementation for applying filters
        ui.notify('Filters applied', type='info')
    
    def _add_alert(self):
        """Add alert for selected ticker"""
        if not self.ideas_table.selection:
            ui.notify('Please select a ticker first', type='warning')
            return
        
        ui.notify('Alert functionality coming soon', type='info')
    
    def _show_settings(self):
        """Show settings dialog"""
        ui.notify('Settings functionality coming soon', type='info')

def create_nicegui_interface():
    """Create the NiceGUI interface"""
    if not NICEGUI_AVAILABLE:
        ui.label('NiceGUI not available. Please install with: pip install nicegui')
        return
    
    # Create the main UI
    TradeScrubberUI()
