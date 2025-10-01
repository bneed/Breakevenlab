# 🤖 Low Cap Trading Bot

**AI-Powered Stock Screener & Trading Recommendations for Robinhood**

A comprehensive trading bot system designed specifically for low to mid market cap stocks ($1-10) with high volume and volatility. Perfect for Robinhood traders looking to capitalize on high-movement opportunities.

## 🎯 Key Features

### 🤖 Trading Bot
- **Smart Screening**: Automatically scans 200+ low cap stocks
- **AI Scoring**: ML-powered trade readiness scoring (0-100)
- **Buy/Sell Recommendations**: Specific entry and exit prices
- **Risk Management**: Built-in stop loss and position sizing
- **Real-time Analysis**: Live technical indicators and signals

### 📡 Market Scanner
- **Continuous Monitoring**: Scans market every 5-30 minutes
- **Alert System**: Notifications for high-scoring opportunities
- **Live Charts**: Real-time visualization of market data
- **Custom Filters**: Price, market cap, volume, and volatility filters

### 🛡️ Risk Manager
- **Position Sizing**: Kelly Criterion and risk-based sizing
- **Portfolio Heat**: Track total portfolio risk exposure
- **Correlation Analysis**: Diversification recommendations
- **Risk Scoring**: Overall portfolio risk assessment

## 🚀 Quick Start

### 1. Run the Application
```bash
cd app
streamlit run app.py
```

### 2. Navigate to Trading Bot
- Select "🤖 Trading Bot" from the navigation menu
- Configure your screening parameters in the sidebar
- Click "🔄 Scan Market" to find opportunities

### 3. Set Up Risk Management
- Go to "🛡️ Risk Manager"
- Enter your account value
- Set risk parameters (default: 2% per trade)
- Calculate position sizes for recommended stocks

### 4. Monitor with Market Scanner
- Navigate to "📡 Market Scanner"
- Start the scanner for continuous monitoring
- Set up alerts for high-scoring opportunities

## 📊 How It Works

### Stock Universe
The bot focuses on low to mid market cap stocks across multiple sectors:
- **Biotech/Pharma**: CRTX, OCGN, BNTX, MRNA, NVAX, etc.
- **Tech/Small Cap**: PLTR, SOFI, AFRM, UPST, LC, HOOD, etc.
- **Energy/Resources**: FCEL, PLUG, BLDP, BE, RUN, etc.
- **Financial Services**: SOFI, AFRM, UPST, LC, HOOD, etc.
- **Healthcare**: Various biotech and pharma stocks
- **Retail/Consumer**: GME, AMC, BB, NOK, BBBY, etc.
- **Crypto/Blockchain**: COIN, HOOD, SQ, PYPL, etc.
- **EV/Transportation**: TSLA, NIO, XPEV, LI, RIVN, etc.

### Screening Criteria
- **Price Range**: $1-10 (configurable)
- **Market Cap**: $100M - $10B (configurable)
- **Volume**: 1.5x+ average volume
- **Volatility**: 2%+ daily volatility
- **Technical Score**: 60+ (configurable)

### Scoring System
The AI scoring system evaluates stocks based on:

1. **Technical Alignment (40%)**
   - Moving average alignment
   - Price position relative to key levels
   - RSI signals
   - MACD signals
   - Bollinger Band position

2. **Momentum/Volume (25%)**
   - Volume spikes
   - Price momentum
   - Volume trend analysis

3. **ML Predictions (25%)**
   - Price direction prediction
   - Confidence levels
   - Expected move magnitude

4. **Trend Quality (10%)**
   - Overall trend strength
   - Signal consistency

### Trading Recommendations
Based on the score, the bot provides:

- **STRONG BUY (80-100)**: High conviction opportunities
- **BUY (70-79)**: Good opportunities
- **WEAK BUY (60-69)**: Watch closely
- **HOLD/AVOID (<60)**: Avoid or hold existing positions

Each recommendation includes:
- Specific buy price (2% below current for better entry)
- Target sell price (8-15% profit target)
- Stop loss price (5-8% risk)
- Risk/reward ratio
- Reasoning for the recommendation

## 🛡️ Risk Management

### Position Sizing
- **Kelly Criterion**: Optimal position sizing based on win rate and risk/reward
- **Risk Per Trade**: Default 2% of account value
- **Max Portfolio Risk**: 10% total exposure
- **Correlation Analysis**: Diversification recommendations

### Stop Losses
- **ATR-based**: Uses Average True Range for dynamic stops
- **Percentage-based**: Fixed percentage stops
- **Support/Resistance**: Key level stops

### Portfolio Heat
- **Total Risk**: Sum of all position risks
- **Risk Score**: 0-100 overall portfolio risk
- **Recommendations**: Automated risk management suggestions

## 📈 Technical Indicators

The bot uses 20+ technical indicators:

### Trend Indicators
- Simple Moving Averages (20, 50, 200)
- Exponential Moving Averages (8, 21)
- Golden Cross / Death Cross
- Price position relative to MAs

### Momentum Indicators
- RSI (14-period)
- MACD (12, 26, 9)
- Stochastic Oscillator
- Momentum signals

### Volume Indicators
- Volume ratio vs average
- Volume spikes
- Volume trend analysis
- Volume-price divergence

### Volatility Indicators
- Average True Range (ATR)
- Bollinger Bands
- Volatility expansion/contraction
- Gap analysis

### Pattern Recognition
- Hammer/Doji patterns
- Engulfing patterns
- Breakout patterns
- Divergence detection

## 🎯 Trading Strategies

### 1. Breakout Strategy
- Look for stocks breaking above resistance
- Confirm with volume spikes
- Target: 10-15% moves
- Stop: Below breakout level

### 2. Reversal Strategy
- Find oversold conditions (RSI < 30)
- Look for bullish divergence
- Target: 8-12% bounces
- Stop: Below recent low

### 3. Momentum Strategy
- Follow strong trends
- Enter on pullbacks to moving averages
- Target: 15-25% moves
- Stop: Below key moving average

### 4. Volatility Strategy
- Trade volatility expansion
- Use Bollinger Band breakouts
- Target: 10-20% moves
- Stop: Back inside bands

## 📱 Robinhood Integration

### Best Practices
- **Use Limit Orders**: Never use market orders
- **Set Alerts**: Use Robinhood's alert system
- **Trade During Market Hours**: Best fills during regular hours
- **Keep Cash Available**: Always have some cash for opportunities

### Common Mistakes to Avoid
- **FOMO Buying**: Don't chase after big moves
- **No Stop Losses**: Always use stop losses
- **Overtrading**: Don't take too many positions
- **Ignoring Risk**: Respect position sizing rules

## 🔧 Configuration

### Scanner Settings
- **Scan Interval**: 5-30 minutes
- **Min Score Threshold**: 50-100
- **Min Volume Ratio**: 1.0-5.0x
- **Min Volatility**: 1-10%

### Price Filters
- **Min Price**: $0.01+
- **Max Price**: $10.00
- **Min Market Cap**: $100M+
- **Max Market Cap**: $10B

### Risk Parameters
- **Risk Per Trade**: 0.5-5.0%
- **Max Portfolio Risk**: 5-20%
- **Max Kelly Fraction**: 5-50%

## 📊 Performance Metrics

### Key Metrics to Track
- **Win Rate**: Percentage of profitable trades
- **Average Win**: Average profit per winning trade
- **Average Loss**: Average loss per losing trade
- **Risk/Reward Ratio**: Average risk vs reward
- **Portfolio Heat**: Total risk exposure
- **Correlation Risk**: Position correlation

### Success Indicators
- **Consistent Profits**: Regular monthly gains
- **Low Drawdowns**: Minimal portfolio declines
- **Good Risk/Reward**: 1:2 or better ratios
- **Diversified Positions**: Low correlation between trades

## 🚨 Alerts & Notifications

### Alert Types
- **High Score Alerts**: Stocks scoring 80+
- **Volume Spikes**: Unusual volume activity
- **Breakout Alerts**: Price breakouts with volume
- **Reversal Alerts**: Oversold/overbought conditions

### Alert Management
- **Dismiss Alerts**: Remove alerts you've seen
- **Alert History**: Track all past alerts
- **Custom Thresholds**: Set your own alert levels

## 📚 Educational Resources

### Trading Concepts
- **Technical Analysis**: Chart patterns and indicators
- **Risk Management**: Position sizing and stop losses
- **Market Psychology**: Understanding market behavior
- **Portfolio Theory**: Diversification and correlation

### Strategy Guides
- **Breakout Trading**: How to trade breakouts
- **Reversal Trading**: Finding reversal opportunities
- **Momentum Trading**: Following strong trends
- **Volatility Trading**: Trading volatility expansion

## ⚠️ Disclaimer

**This tool is for educational purposes only and is not investment advice.**

- Past performance does not guarantee future results
- All trading involves risk of loss
- Only trade with money you can afford to lose
- Always do your own research before trading
- Consider consulting with a financial advisor

## 🆘 Support

### Common Issues
- **No Results**: Try adjusting filters or lowering thresholds
- **Slow Performance**: Reduce scan frequency or ticker list
- **Data Errors**: Check internet connection and API limits

### Getting Help
- Check the Strategy Guide tab in each tool
- Review the configuration settings
- Start with conservative parameters
- Gradually increase complexity as you learn

## 🔄 Updates & Maintenance

### Regular Updates
- **Ticker Universe**: Updated quarterly
- **Technical Indicators**: Enhanced regularly
- **ML Models**: Retrained weekly
- **Risk Models**: Updated based on market conditions

### Maintenance Tasks
- **Clear Old Alerts**: Remove dismissed alerts
- **Update Settings**: Adjust parameters as needed
- **Monitor Performance**: Track your trading results
- **Review Strategies**: Adapt to market changes

---

**Happy Trading! 🚀**

Remember: The key to successful trading is not just finding good opportunities, but managing risk properly. Use the tools provided, but always make your own decisions and never risk more than you can afford to lose.
