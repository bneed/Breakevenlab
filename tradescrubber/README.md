# TradeScrubber

**Smart Stock & Options Screener with Timing**

TradeScrubber is a production-ready web application that scans the market, filters noise, ranks tickers by trade readiness, and tells you what to trade (stock/option) and when. It features fast data processing, clean UI, and is fully deployable.

## Features

### 🎯 Core Features
- **Market Scanning**: Scan hundreds of tickers in real-time
- **Technical Analysis**: 20+ technical indicators (SMA, EMA, RSI, MACD, VWAP, ATR, etc.)
- **Signal Detection**: Advanced signal detection with pattern recognition
- **Smart Ranking**: ML-powered trade readiness scoring (0-100)
- **Options Analysis**: Comprehensive options chain analysis with Greeks
- **Machine Learning**: RandomForest models for price prediction
- **Backtesting**: Walk-forward backtesting with multiple strategies
- **Real-time Data**: yfinance integration with optional API providers

### 📊 UI Features
- **Modern Interface**: Clean, responsive design with NiceGUI (Streamlit fallback)
- **Multiple Tabs**: Ideas, Charts, Screener, Options, ML, Backtest
- **Real-time Updates**: Live market data and signal updates
- **Interactive Charts**: Price charts with technical indicators
- **Strategy Presets**: Pre-configured trading strategies
- **Custom Watchlists**: Create and manage custom ticker lists

### 🤖 Machine Learning
- **Price Prediction**: Predict 1-5 day price movements
- **Direction Classification**: Up/down probability scoring
- **Feature Engineering**: 50+ technical and fundamental features
- **Model Training**: Automated model training and retraining
- **Performance Metrics**: R², accuracy, feature importance

### 📈 Backtesting
- **Multiple Strategies**: Default, Reversal, Breakout, Trend following
- **Risk Management**: Stop-loss, take-profit, position sizing
- **Performance Metrics**: CAGR, Sharpe ratio, max drawdown, win rate
- **Strategy Comparison**: Compare multiple strategies side-by-side

## Installation

### Prerequisites
- Python 3.10+
- pip package manager

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/tradescrubber.git
cd tradescrubber
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment (optional)**
```bash
cp env.example .env
# Edit .env with your API keys and preferences
```

4. **Run the application**
```bash
python app.py
```

The application will start on `http://localhost:8080` (NiceGUI) or `http://localhost:8501` (Streamlit).

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Data Sources
DATA_SOURCE=yfinance  # Options: yfinance, polygon, alpaca

# API Keys (optional)
POLYGON_API_KEY=your_polygon_api_key_here
ALPACA_API_KEY_ID=your_alpaca_key_id_here
ALPACA_API_SECRET_KEY=your_alpaca_secret_key_here
TRADIER_API_KEY=your_tradier_api_key_here

# Configuration
TZ=America/New_York
CACHE_TTL_MIN=15
```

### Data Sources

**yfinance (Default)**
- Free, no API key required
- EOD and delayed intraday data
- Basic options chains
- Rate limited

**Polygon.io (Optional)**
- Real-time and historical data
- Advanced options data
- Requires API key
- Higher rate limits

**Alpaca (Optional)**
- Real-time market data
- Paper and live trading
- Requires API key
- Professional features

## Usage

### 1. Ideas Tab
- Select a watchlist or enter custom tickers
- Choose a strategy preset
- Set minimum score and maximum results
- Click "Scan Market" to find trading ideas
- View ranked results with entry/stop/target levels

### 2. Charts Tab
- Select a ticker to view price chart
- Toggle technical indicators on/off
- View recent price data and signals
- Analyze price action and patterns

### 3. Screener Tab
- Filter stocks by price, RSI, volume, etc.
- Sort by any column
- View technical indicators for all tickers
- Export filtered results

### 4. Options Tab
- Select a ticker for options analysis
- Choose trade direction (long/short)
- Set target delta and DTE range
- View options recommendations with Greeks

### 5. ML Tab
- Train ML models on historical data
- View model performance metrics
- See ML predictions for all tickers
- Monitor feature importance

### 6. Backtest Tab
- Select strategy and date range
- Set position size and max positions
- Run backtest simulation
- View performance metrics and equity curve

## Strategy Presets

### Default Strategy
- Balanced approach with trend following
- Minimum score: 50
- All RSI levels
- Basic volume filter

### Reversal Strategy
- Look for oversold conditions
- RSI < 35
- Above SMA50
- Volume confirmation

### Breakout Strategy
- Breakouts above resistance
- RSI > 50
- Above SMA200
- High volume (1.5x average)

### Trend Following
- Strong trends with momentum
- Golden cross signals
- EMA alignment
- Above SMA200

## API Integration

### Polygon.io
```python
# Set in .env
POLYGON_API_KEY=your_key_here
DATA_SOURCE=polygon
```

### Alpaca
```python
# Set in .env
ALPACA_API_KEY_ID=your_key_id
ALPACA_API_SECRET_KEY=your_secret_key
DATA_SOURCE=alpaca
```

### Tradier (Options)
```python
# Set in .env
TRADIER_API_KEY=your_key_here
```

## Development

### Project Structure
```
tradescrubber/
├── app.py                 # Main application launcher
├── core/                  # Core modules
│   ├── data.py           # Data ingestion
│   ├── indicators.py     # Technical indicators
│   ├── signals.py        # Signal detection
│   ├── ranker.py         # Scoring and ranking
│   ├── options.py        # Options analysis
│   ├── ml.py            # Machine learning
│   ├── backtest.py      # Backtesting engine
│   └── utils.py         # Utilities
├── ui/                   # User interface
│   ├── nicegui_interface.py
│   └── streamlit_interface.py
├── presets/              # Configuration files
│   ├── watchlists.yaml
│   └── strategies.yaml
├── models/               # ML model storage
├── data_cache/          # Data cache
└── requirements.txt
```

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black tradescrubber/
flake8 tradescrubber/
```

## Deployment

### Local Development
```bash
python app.py
```

### Docker
```bash
docker build -t tradescrubber .
docker run -p 8080:8080 tradescrubber
```

### Cloud Deployment
- **Streamlit Cloud**: Deploy directly from GitHub
- **Heroku**: Use Procfile for deployment
- **AWS/GCP/Azure**: Container-based deployment

## Performance

### Data Processing
- **Caching**: 15-minute TTL for market data
- **Parallel Processing**: Multi-threaded data fetching
- **Memory Efficient**: Pandas DataFrames with optimized dtypes

### ML Models
- **Training Time**: ~2-5 minutes for 1 year of data
- **Prediction Time**: <1 second for 100+ tickers
- **Model Size**: ~10-50MB per model

### UI Performance
- **Load Time**: <3 seconds for initial data
- **Refresh Rate**: Real-time updates every 15 minutes
- **Responsiveness**: <500ms for user interactions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- **Documentation**: [Wiki](https://github.com/yourusername/tradescrubber/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/tradescrubber/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/tradescrubber/discussions)

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Always do your own research and consult with a financial advisor before making investment decisions. Trading involves risk and you may lose money.

## Changelog

### v1.0.0 (2024-01-01)
- Initial release
- Core features implemented
- NiceGUI and Streamlit interfaces
- ML models and backtesting
- Options analysis
- Multiple data sources

---

**TradeScrubber** - Making market analysis smarter, faster, and more accessible.
