# Break-even Lab 📊

Professional-grade trading and finance tools for options analysis, backtesting, and portfolio management.

## 🚀 Features

### Tier 0 (Free Tools)
- **Options Break-even Calculator** - Multi-leg P/L analysis with Greeks
- **Position Sizing & Risk Management** - Kelly fraction, fixed-fractional sizing
- **Greeks Viewer** - Delta, Gamma, Theta, Vega, Rho analysis
- **IV Rank Screener** - High implied volatility rank stocks

### Tier 1 (Pro Features)
- **Earnings Sentiment Analysis** - AI-powered transcript analysis
- **Backtest Lite** - Strategy backtesting with daily bars
- **Email Alerts** - IVR cross, earnings notifications

### Tier 2 (Premium Features)
- **Wheel Strategy Planner** - Cash-secured put → covered call simulator
- **Volatility Surface Explorer** - By expiry/strike analysis
- **Portfolio Risk Dashboard** - Exposure by delta/theta/vega

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python with FastAPI (optional)
- **Data Sources**: yfinance, Alpha Vantage, Finnhub
- **Charts**: Plotly
- **Database**: SQLite
- **Authentication**: Stripe + custom auth
- **Deployment**: Streamlit Community Cloud

## 📦 Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/breakeven-lab.git
   cd breakeven-lab
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   streamlit run app/app.py
   ```

### Environment Variables

Create a `.env` file with the following variables:

```env
# API Keys (optional - app works without them)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key

# Stripe (for Pro features)
STRIPE_PUBLISHABLE_KEY=your_stripe_publishable_key
STRIPE_SECRET_KEY=your_stripe_secret_key
STRIPE_WEBHOOK_SECRET=your_stripe_webhook_secret

# Email (for alerts)
RESEND_API_KEY=your_resend_api_key

# App Configuration
SECRET_KEY=your_secret_key
```

## 🚀 Deployment

### Streamlit Community Cloud

1. **Fork this repository** to your GitHub account

2. **Connect to Streamlit Community Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your forked repository
   - Set the main file path to `app/app.py`

3. **Configure environment variables**
   - In the Streamlit Community Cloud dashboard
   - Go to your app settings
   - Add the environment variables from your `.env` file

4. **Deploy**
   - Click "Deploy"
   - Your app will be available at `https://your-app-name.streamlit.app`

### Alternative Deployment Options

#### Heroku
```bash
# Install Heroku CLI
# Create Procfile
echo "web: streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
heroku config:set ALPHA_VANTAGE_API_KEY=your_key
git push heroku main
```

#### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📁 Project Structure

```
breakeven-lab/
├── app/                      # Streamlit application
│   ├── __init__.py
│   ├── app.py               # Main app router
│   ├── pages/               # Individual pages
│   │   ├── breakeven_calculator.py
│   │   ├── position_sizing.py
│   │   ├── greeks_viewer.py
│   │   ├── ivr_screener.py
│   │   ├── earnings_sentiment.py
│   │   ├── backtest_lite.py
│   │   ├── strategy_planner.py
│   │   └── portfolio_dashboard.py
│   └── components/          # Shared components
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── auth.py             # Authentication & subscriptions
│   ├── data.py             # Data fetching & management
│   ├── options.py          # Options pricing & Greeks
│   ├── screeners.py        # Stock screening
│   ├── backtest.py         # Backtesting engine
│   └── earnings.py         # Earnings sentiment analysis
├── assets/                 # Static assets
├── .streamlit/            # Streamlit configuration
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore file
└── README.md           # This file
```

## 🔧 Configuration

### Streamlit Configuration

The app uses `.streamlit/config.toml` for configuration:

```toml
[global]
developmentMode = false

[server]
headless = true
port = 8501
enableCORS = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### Database

The app uses SQLite for user management and alerts. The database is automatically created on first run.

## 📊 Usage

### Free Tools
1. **Break-even Calculator**: Enter multi-leg options strategies and analyze P/L
2. **Position Sizing**: Calculate optimal position sizes using various methods
3. **Greeks Viewer**: Analyze options Greeks across different scenarios
4. **IV Rank Screener**: Find stocks with high implied volatility rank

### Pro Features
1. **Earnings Sentiment**: Analyze earnings call transcripts for sentiment
2. **Backtest Lite**: Test trading strategies with historical data
3. **Strategy Planner**: Plan advanced options strategies
4. **Portfolio Dashboard**: Monitor portfolio risk and exposure

## 🔐 Authentication & Subscriptions

- **Free Tier**: Access to basic tools
- **Pro Tier**: $12-19/month for advanced features
- **Founders Plan**: $5/month for early adopters

Authentication is handled through Stripe Checkout and custom session management.

## 📈 Data Sources

- **yfinance**: Stock prices and basic data
- **Alpha Vantage**: Advanced market data (optional)
- **Finnhub**: Real-time data (optional)

The app works with free data sources, but premium data sources provide better accuracy.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for educational purposes only. Not investment advice. Options trading involves significant risk and may not be suitable for all investors. Always do your own research and consider consulting with a financial advisor.

## 🆘 Support

- **Documentation**: Check this README and inline help
- **Issues**: Open an issue on GitHub
- **Email**: support@breakevenlab.com (coming soon)

## 🗺️ Roadmap

### Phase 1 (Current)
- [x] Core tools and calculators
- [x] Basic screening functionality
- [x] Streamlit deployment

### Phase 2 (Next)
- [ ] Stripe integration for Pro features
- [ ] Email alert system
- [ ] Advanced backtesting
- [ ] Mobile optimization

### Phase 3 (Future)
- [ ] Real-time data feeds
- [ ] Advanced portfolio analytics
- [ ] Social features
- [ ] API access

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Charts powered by [Plotly](https://plotly.com)
- Data from [yfinance](https://github.com/ranaroussi/yfinance)
- Options pricing using Black-Scholes model

---

**Break-even Lab** - Professional trading tools for everyone 📊
