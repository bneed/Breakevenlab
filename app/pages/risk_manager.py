"""
Risk Management and Position Sizing for Trading Bot
Provides Kelly Criterion, risk-adjusted position sizing, and portfolio management
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class RiskManager:
    """Risk management and position sizing calculator"""
    
    def __init__(self):
        self.default_risk_per_trade = 0.02  # 2% risk per trade
        self.max_portfolio_risk = 0.10  # 10% max portfolio risk
        self.kelly_threshold = 0.25  # Max Kelly fraction
    
    def calculate_kelly_fraction(self, 
                               win_rate: float, 
                               avg_win: float, 
                               avg_loss: float) -> float:
        """
        Calculate Kelly Criterion fraction
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount
            
        Returns:
            Kelly fraction (0-1)
        """
        if avg_loss == 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly = (b * p - q) / b
        
        # Cap at threshold to prevent over-leveraging
        return min(max(kelly, 0), self.kelly_threshold)
    
    def calculate_position_size(self, 
                              account_value: float,
                              stock_price: float,
                              stop_loss_price: float,
                              risk_per_trade: float = None,
                              kelly_fraction: float = None) -> Dict[str, float]:
        """
        Calculate position size based on risk management rules
        
        Args:
            account_value: Total account value
            stock_price: Current stock price
            stop_loss_price: Stop loss price
            risk_per_trade: Risk per trade (default: 2%)
            kelly_fraction: Kelly fraction (optional)
            
        Returns:
            Dictionary with position sizing details
        """
        if risk_per_trade is None:
            risk_per_trade = self.default_risk_per_trade
        
        # Calculate risk per share
        risk_per_share = abs(stock_price - stop_loss_price)
        
        if risk_per_share == 0:
            return {
                'shares': 0,
                'dollar_amount': 0,
                'risk_amount': 0,
                'risk_percentage': 0,
                'kelly_fraction': 0,
                'recommendation': 'No position - no risk defined'
            }
        
        # Calculate position size based on risk
        risk_amount = account_value * risk_per_trade
        shares_by_risk = risk_amount / risk_per_share
        
        # Calculate position size based on Kelly (if provided)
        if kelly_fraction is not None:
            shares_by_kelly = (account_value * kelly_fraction) / stock_price
            shares = min(shares_by_kelly, shares_by_risk)
        else:
            shares = shares_by_risk
        
        # Round down to whole shares
        shares = math.floor(shares)
        
        # Calculate actual values
        dollar_amount = shares * stock_price
        actual_risk_amount = shares * risk_per_share
        actual_risk_percentage = actual_risk_amount / account_value
        
        # Determine recommendation
        if shares == 0:
            recommendation = "No position - risk too high"
        elif actual_risk_percentage > self.max_portfolio_risk:
            recommendation = f"Reduce position - risk {actual_risk_percentage:.1%} > {self.max_portfolio_risk:.1%}"
        elif actual_risk_percentage < risk_per_trade * 0.5:
            recommendation = "Consider larger position - low risk"
        else:
            recommendation = "Good position size"
        
        return {
            'shares': shares,
            'dollar_amount': dollar_amount,
            'risk_amount': actual_risk_amount,
            'risk_percentage': actual_risk_percentage,
            'kelly_fraction': kelly_fraction or 0,
            'recommendation': recommendation
        }
    
    def calculate_portfolio_heat(self, positions: List[Dict]) -> Dict[str, float]:
        """
        Calculate portfolio heat (total risk exposure)
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Dictionary with portfolio heat metrics
        """
        total_risk = sum(pos.get('risk_amount', 0) for pos in positions)
        total_value = sum(pos.get('dollar_amount', 0) for pos in positions)
        
        if total_value == 0:
            return {
                'total_risk': 0,
                'total_value': 0,
                'portfolio_heat': 0,
                'risk_per_trade_avg': 0,
                'recommendation': 'No positions'
            }
        
        portfolio_heat = total_risk / total_value
        avg_risk_per_trade = total_risk / len(positions) if positions else 0
        
        # Determine recommendation
        if portfolio_heat > self.max_portfolio_risk:
            recommendation = f"Reduce exposure - heat {portfolio_heat:.1%} > {self.max_portfolio_risk:.1%}"
        elif portfolio_heat > self.max_portfolio_risk * 0.8:
            recommendation = "Monitor closely - approaching max heat"
        else:
            recommendation = "Portfolio heat is healthy"
        
        return {
            'total_risk': total_risk,
            'total_value': total_value,
            'portfolio_heat': portfolio_heat,
            'risk_per_trade_avg': avg_risk_per_trade,
            'recommendation': recommendation
        }
    
    def calculate_correlation_risk(self, 
                                 positions: List[Dict], 
                                 correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate correlation-adjusted portfolio risk
        
        Args:
            positions: List of position dictionaries
            correlation_matrix: Correlation matrix between positions
            
        Returns:
            Dictionary with correlation risk metrics
        """
        if len(positions) < 2:
            return {
                'correlation_risk': 0,
                'max_correlation': 0,
                'recommendation': 'Need at least 2 positions for correlation analysis'
            }
        
        # Calculate weighted correlation
        symbols = [pos.get('symbol', '') for pos in positions]
        weights = [pos.get('dollar_amount', 0) for pos in positions]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return {
                'correlation_risk': 0,
                'max_correlation': 0,
                'recommendation': 'No position values'
            }
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        # Calculate portfolio correlation
        portfolio_correlation = 0
        max_correlation = 0
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i < j and symbol1 in correlation_matrix.columns and symbol2 in correlation_matrix.columns:
                    corr = correlation_matrix.loc[symbol1, symbol2]
                    portfolio_correlation += weights[i] * weights[j] * corr
                    max_correlation = max(max_correlation, abs(corr))
        
        # Determine recommendation
        if max_correlation > 0.8:
            recommendation = "High correlation risk - diversify positions"
        elif max_correlation > 0.6:
            recommendation = "Moderate correlation - monitor closely"
        else:
            recommendation = "Good diversification"
        
        return {
            'correlation_risk': portfolio_correlation,
            'max_correlation': max_correlation,
            'recommendation': recommendation
        }
    
    def generate_risk_report(self, 
                           account_value: float,
                           positions: List[Dict],
                           correlation_matrix: pd.DataFrame = None) -> Dict:
        """
        Generate comprehensive risk report
        
        Args:
            account_value: Total account value
            positions: List of current positions
            correlation_matrix: Optional correlation matrix
            
        Returns:
            Comprehensive risk report
        """
        # Calculate portfolio heat
        heat_metrics = self.calculate_portfolio_heat(positions)
        
        # Calculate correlation risk if matrix provided
        correlation_metrics = {}
        if correlation_matrix is not None:
            correlation_metrics = self.calculate_correlation_risk(positions, correlation_matrix)
        
        # Calculate individual position metrics
        position_metrics = []
        for pos in positions:
            pos_metrics = self.calculate_position_size(
                account_value=account_value,
                stock_price=pos.get('price', 0),
                stop_loss_price=pos.get('stop_loss', 0),
                risk_per_trade=pos.get('risk_per_trade', self.default_risk_per_trade)
            )
            pos_metrics['symbol'] = pos.get('symbol', 'Unknown')
            position_metrics.append(pos_metrics)
        
        # Overall risk assessment
        risk_score = self._calculate_risk_score(heat_metrics, correlation_metrics, position_metrics)
        
        return {
            'account_value': account_value,
            'portfolio_heat': heat_metrics,
            'correlation_risk': correlation_metrics,
            'position_metrics': position_metrics,
            'risk_score': risk_score,
            'recommendations': self._generate_recommendations(heat_metrics, correlation_metrics, position_metrics)
        }
    
    def _calculate_risk_score(self, heat_metrics: Dict, correlation_metrics: Dict, position_metrics: List[Dict]) -> float:
        """Calculate overall risk score (0-100, lower is better)"""
        score = 0
        
        # Portfolio heat component (40% weight)
        heat = heat_metrics.get('portfolio_heat', 0)
        if heat > self.max_portfolio_risk:
            score += 40
        elif heat > self.max_portfolio_risk * 0.8:
            score += 20
        else:
            score += heat / self.max_portfolio_risk * 20
        
        # Correlation risk component (30% weight)
        max_corr = correlation_metrics.get('max_correlation', 0)
        if max_corr > 0.8:
            score += 30
        elif max_corr > 0.6:
            score += 15
        else:
            score += max_corr * 15
        
        # Individual position risk component (30% weight)
        high_risk_positions = sum(1 for pos in position_metrics if pos.get('risk_percentage', 0) > self.default_risk_per_trade * 1.5)
        total_positions = len(position_metrics)
        if total_positions > 0:
            score += (high_risk_positions / total_positions) * 30
        
        return min(score, 100)
    
    def _generate_recommendations(self, heat_metrics: Dict, correlation_metrics: Dict, position_metrics: List[Dict]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Portfolio heat recommendations
        heat = heat_metrics.get('portfolio_heat', 0)
        if heat > self.max_portfolio_risk:
            recommendations.append(f"🚨 Reduce portfolio exposure - currently {heat:.1%}")
        elif heat > self.max_portfolio_risk * 0.8:
            recommendations.append(f"⚠️ Monitor portfolio heat - currently {heat:.1%}")
        
        # Correlation recommendations
        max_corr = correlation_metrics.get('max_correlation', 0)
        if max_corr > 0.8:
            recommendations.append("🚨 High correlation detected - diversify positions")
        elif max_corr > 0.6:
            recommendations.append("⚠️ Moderate correlation - consider diversification")
        
        # Individual position recommendations
        high_risk_positions = [pos for pos in position_metrics if pos.get('risk_percentage', 0) > self.default_risk_per_trade * 1.5]
        if high_risk_positions:
            recommendations.append(f"⚠️ {len(high_risk_positions)} position(s) exceed recommended risk")
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ Portfolio risk is within acceptable limits")
        
        return recommendations

def show_risk_manager():
    """Display the risk management interface"""
    
    st.title("🛡️ Risk Manager")
    st.markdown("**Position sizing and risk management for your trading bot**")
    
    # Initialize risk manager
    if 'risk_manager' not in st.session_state:
        st.session_state.risk_manager = RiskManager()
    
    risk_manager = st.session_state.risk_manager
    
    # Sidebar for risk parameters
    with st.sidebar:
        st.header("⚙️ Risk Parameters")
        
        # Account value
        account_value = st.number_input(
            "Account Value ($)",
            min_value=100.0,
            value=10000.0,
            step=100.0,
            help="Total account value for position sizing"
        )
        
        # Risk per trade
        risk_per_trade = st.slider(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Percentage of account to risk per trade"
        ) / 100
        
        # Max portfolio risk
        max_portfolio_risk = st.slider(
            "Max Portfolio Risk (%)",
            min_value=5.0,
            max_value=20.0,
            value=10.0,
            step=1.0,
            help="Maximum total portfolio risk"
        ) / 100
        
        # Kelly threshold
        kelly_threshold = st.slider(
            "Max Kelly Fraction (%)",
            min_value=5.0,
            max_value=50.0,
            value=25.0,
            step=5.0,
            help="Maximum Kelly fraction to prevent over-leveraging"
        ) / 100
        
        # Update risk manager settings
        risk_manager.default_risk_per_trade = risk_per_trade
        risk_manager.max_portfolio_risk = max_portfolio_risk
        risk_manager.kelly_threshold = kelly_threshold
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Position Sizing", "📈 Portfolio Heat", "🎯 Risk Report"])
    
    with tab1:
        st.subheader("📊 Position Sizing Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Stock Information**")
            stock_price = st.number_input(
                "Stock Price ($)",
                min_value=0.01,
                value=5.0,
                step=0.01,
                help="Current stock price"
            )
            
            stop_loss = st.number_input(
                "Stop Loss ($)",
                min_value=0.01,
                value=4.5,
                step=0.01,
                help="Stop loss price"
            )
            
            kelly_fraction = st.number_input(
                "Kelly Fraction (optional)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                help="Kelly fraction if known (0 to disable)"
            )
        
        with col2:
            st.markdown("**Position Size Results**")
            
            if st.button("Calculate Position Size"):
                position_size = risk_manager.calculate_position_size(
                    account_value=account_value,
                    stock_price=stock_price,
                    stop_loss_price=stop_loss,
                    risk_per_trade=risk_per_trade,
                    kelly_fraction=kelly_fraction if kelly_fraction > 0 else None
                )
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Shares", f"{position_size['shares']:,}")
                    st.metric("Dollar Amount", f"${position_size['dollar_amount']:,.2f}")
                
                with col2:
                    st.metric("Risk Amount", f"${position_size['risk_amount']:,.2f}")
                    st.metric("Risk %", f"{position_size['risk_percentage']:.1%}")
                
                st.info(f"**Recommendation:** {position_size['recommendation']}")
                
                if position_size['kelly_fraction'] > 0:
                    st.metric("Kelly Fraction", f"{position_size['kelly_fraction']:.1%}")
    
    with tab2:
        st.subheader("📈 Portfolio Heat Analysis")
        
        # Portfolio positions (mock data for demo)
        st.markdown("**Current Positions**")
        
        # Initialize positions in session state
        if 'portfolio_positions' not in st.session_state:
            st.session_state.portfolio_positions = []
        
        # Add position form
        with st.expander("➕ Add Position"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                symbol = st.text_input("Symbol", value="AAPL")
            with col2:
                price = st.number_input("Price", value=150.0, step=0.01)
            with col3:
                stop_loss = st.number_input("Stop Loss", value=140.0, step=0.01)
            with col4:
                shares = st.number_input("Shares", value=10, step=1)
            
            if st.button("Add Position"):
                position = {
                    'symbol': symbol,
                    'price': price,
                    'stop_loss': stop_loss,
                    'shares': shares,
                    'dollar_amount': shares * price,
                    'risk_amount': shares * abs(price - stop_loss)
                }
                st.session_state.portfolio_positions.append(position)
                st.success(f"Added {symbol} position")
                st.rerun()
        
        # Display current positions
        if st.session_state.portfolio_positions:
            positions_df = pd.DataFrame(st.session_state.portfolio_positions)
            st.dataframe(positions_df, use_container_width=True)
            
            # Calculate portfolio heat
            heat_metrics = risk_manager.calculate_portfolio_heat(st.session_state.portfolio_positions)
            
            # Display heat metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Value", f"${heat_metrics['total_value']:,.2f}")
            with col2:
                st.metric("Total Risk", f"${heat_metrics['total_risk']:,.2f}")
            with col3:
                st.metric("Portfolio Heat", f"{heat_metrics['portfolio_heat']:.1%}")
            with col4:
                st.metric("Avg Risk/Trade", f"${heat_metrics['risk_per_trade_avg']:,.2f}")
            
            st.info(f"**Recommendation:** {heat_metrics['recommendation']}")
            
            # Risk gauge
            heat_pct = heat_metrics['portfolio_heat'] * 100
            if heat_pct > 10:
                st.error(f"🚨 High Risk: {heat_pct:.1f}%")
            elif heat_pct > 7:
                st.warning(f"⚠️ Medium Risk: {heat_pct:.1f}%")
            else:
                st.success(f"✅ Low Risk: {heat_pct:.1f}%")
            
            # Clear positions button
            if st.button("🗑️ Clear All Positions"):
                st.session_state.portfolio_positions = []
                st.rerun()
        
        else:
            st.info("No positions added yet. Use the form above to add positions.")
    
    with tab3:
        st.subheader("🎯 Comprehensive Risk Report")
        
        if st.session_state.portfolio_positions:
            # Generate risk report
            risk_report = risk_manager.generate_risk_report(
                account_value=account_value,
                positions=st.session_state.portfolio_positions
            )
            
            # Risk score
            risk_score = risk_report['risk_score']
            st.metric("Overall Risk Score", f"{risk_score:.0f}/100")
            
            if risk_score > 80:
                st.error("🚨 High Risk Portfolio")
            elif risk_score > 60:
                st.warning("⚠️ Medium Risk Portfolio")
            else:
                st.success("✅ Low Risk Portfolio")
            
            # Recommendations
            st.markdown("**Risk Management Recommendations**")
            for rec in risk_report['recommendations']:
                st.write(rec)
            
            # Detailed metrics
            st.markdown("**Detailed Metrics**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Portfolio Heat**")
                heat = risk_report['portfolio_heat']
                st.write(f"- Total Value: ${heat['total_value']:,.2f}")
                st.write(f"- Total Risk: ${heat['total_risk']:,.2f}")
                st.write(f"- Portfolio Heat: {heat['portfolio_heat']:.1%}")
            
            with col2:
                st.markdown("**Position Details**")
                for pos in risk_report['position_metrics']:
                    st.write(f"- {pos['symbol']}: {pos['shares']} shares, {pos['risk_percentage']:.1%} risk")
        
        else:
            st.info("Add positions in the Portfolio Heat tab to generate a risk report.")

if __name__ == "__main__":
    show_risk_manager()
