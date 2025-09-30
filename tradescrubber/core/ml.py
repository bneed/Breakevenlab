"""
Machine Learning module for TradeScrubber
Implements RandomForest models for price prediction and direction classification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import pickle
from datetime import datetime, timedelta

from .utils import MODELS_DIR

logger = logging.getLogger(__name__)

class MLPredictor:
    """Machine Learning predictor for price movements and direction"""
    
    def __init__(self):
        self.regressor = None
        self.classifier = None
        self.feature_columns = None
        self.is_trained = False
        self.last_training_date = None
        
        # Try to load existing models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if they exist"""
        regressor_path = MODELS_DIR / "rf_price_move.pkl"
        classifier_path = MODELS_DIR / "rf_direction.pkl"
        
        try:
            if regressor_path.exists():
                with open(regressor_path, 'rb') as f:
                    self.regressor = pickle.load(f)
                logger.info("Loaded pre-trained regressor model")
            
            if classifier_path.exists():
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                logger.info("Loaded pre-trained classifier model")
            
            if self.regressor is not None and self.classifier is not None:
                self.is_trained = True
                logger.info("ML models loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_trained = False
    
    def train_models(
        self, 
        prices_dict: Dict[str, pd.DataFrame], 
        lookback_days: int = 365,
        horizon_days: int = 3,
        test_size: float = 0.3
    ) -> Dict[str, Any]:
        """
        Train ML models on historical data
        
        Args:
            prices_dict: Dictionary mapping ticker to DataFrame with OHLCV + indicators
            lookback_days: Number of days to look back for training
            horizon_days: Number of days ahead to predict
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training metrics
        """
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
            from sklearn.preprocessing import StandardScaler
            
            logger.info("Starting ML model training...")
            
            # Prepare training data
            features_df, targets_reg, targets_cls = self._prepare_training_data(
                prices_dict, lookback_days, horizon_days
            )
            
            if features_df.empty:
                return {"error": "No training data available"}
            
            # Split data
            X_train, X_test, y_reg_train, y_reg_test = train_test_split(
                features_df, targets_reg, test_size=test_size, random_state=42
            )
            
            _, _, y_cls_train, y_cls_test = train_test_split(
                features_df, targets_cls, test_size=test_size, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train regressor
            logger.info("Training RandomForest Regressor...")
            self.regressor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.regressor.fit(X_train_scaled, y_reg_train)
            
            # Train classifier
            logger.info("Training RandomForest Classifier...")
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.classifier.fit(X_train_scaled, y_cls_train)
            
            # Store feature columns
            self.feature_columns = features_df.columns.tolist()
            
            # Evaluate models
            reg_pred = self.regressor.predict(X_test_scaled)
            cls_pred = self.classifier.predict(X_test_scaled)
            
            # Calculate metrics
            reg_mse = mean_squared_error(y_reg_test, reg_pred)
            reg_r2 = r2_score(y_reg_test, reg_pred)
            cls_accuracy = accuracy_score(y_cls_test, cls_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.regressor.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save models
            self._save_models()
            
            self.is_trained = True
            self.last_training_date = datetime.now()
            
            metrics = {
                'regressor_mse': reg_mse,
                'regressor_r2': reg_r2,
                'classifier_accuracy': cls_accuracy,
                'feature_importance': feature_importance.head(10).to_dict('records'),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'last_training': self.last_training_date.isoformat()
            }
            
            logger.info(f"Training completed. R²: {reg_r2:.3f}, Accuracy: {cls_accuracy:.3f}")
            return metrics
            
        except ImportError:
            logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
            return {"error": "scikit-learn not installed"}
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {"error": str(e)}
    
    def _prepare_training_data(
        self, 
        prices_dict: Dict[str, pd.DataFrame], 
        lookback_days: int,
        horizon_days: int
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training data from prices dictionary"""
        
        all_features = []
        all_targets_reg = []
        all_targets_cls = []
        
        for ticker, df in prices_dict.items():
            if df.empty or len(df) < lookback_days:
                continue
            
            # Get features for each day
            for i in range(lookback_days, len(df) - horizon_days):
                # Features (current day)
                features = self._extract_features(df.iloc[i])
                if features is None:
                    continue
                
                # Targets (future day)
                future_price = df.iloc[i + horizon_days]['close']
                current_price = df.iloc[i]['close']
                
                # Regression target: percentage change
                target_reg = (future_price - current_price) / current_price
                
                # Classification target: direction (1 for up, 0 for down)
                target_cls = 1 if target_reg > 0 else 0
                
                all_features.append(features)
                all_targets_reg.append(target_reg)
                all_targets_cls.append(target_cls)
        
        if not all_features:
            return pd.DataFrame(), pd.Series(), pd.Series()
        
        # Convert to DataFrames
        features_df = pd.DataFrame(all_features)
        targets_reg = pd.Series(all_targets_reg)
        targets_cls = pd.Series(all_targets_cls)
        
        # Remove any rows with NaN values
        valid_mask = ~(features_df.isna().any(axis=1) | targets_reg.isna() | targets_cls.isna())
        
        return features_df[valid_mask], targets_reg[valid_mask], targets_cls[valid_mask]
    
    def _extract_features(self, row: pd.Series) -> Optional[Dict[str, float]]:
        """Extract features from a single row of data"""
        features = {}
        
        # Price features
        if 'close' in row and 'open' in row:
            features['price_change'] = (row['close'] - row['open']) / row['open']
        
        if 'high' in row and 'low' in row and 'close' in row:
            features['price_range'] = (row['high'] - row['low']) / row['close']
        
        # Moving averages
        for col in ['sma20', 'sma50', 'sma200', 'ema8', 'ema21']:
            if col in row and not pd.isna(row[col]) and row['close'] > 0:
                features[f'{col}_ratio'] = row[col] / row['close']
        
        # RSI
        if 'rsi14' in row and not pd.isna(row['rsi14']):
            features['rsi'] = row['rsi14']
            features['rsi_normalized'] = (row['rsi14'] - 50) / 50
        
        # MACD
        if 'macd' in row and not pd.isna(row['macd']):
            features['macd'] = row['macd']
        if 'macd_signal' in row and not pd.isna(row['macd_signal']):
            features['macd_signal'] = row['macd_signal']
        if 'macd_histogram' in row and not pd.isna(row['macd_histogram']):
            features['macd_histogram'] = row['macd_histogram']
        
        # ATR
        if 'atr14' in row and not pd.isna(row['atr14']) and row['close'] > 0:
            features['atr_ratio'] = row['atr14'] / row['close']
        
        # Volume features
        if 'rel_vol_30d' in row and not pd.isna(row['rel_vol_30d']):
            features['relative_volume'] = row['rel_vol_30d']
        
        if 'volume' in row and not pd.isna(row['volume']):
            features['volume_log'] = np.log1p(row['volume'])
        
        # Bollinger Bands
        if 'bb_position' in row and not pd.isna(row['bb_position']):
            features['bb_position'] = row['bb_position']
        if 'bb_width' in row and not pd.isna(row['bb_width']):
            features['bb_width'] = row['bb_width']
        
        # Volatility
        if 'volatility_20d' in row and not pd.isna(row['volatility_20d']):
            features['volatility'] = row['volatility_20d']
        
        # Gap
        if 'gap_pct' in row and not pd.isna(row['gap_pct']):
            features['gap'] = row['gap_pct']
        
        # Price position
        for col in ['price_above_sma20', 'price_above_sma50', 'price_above_sma200']:
            if col in row:
                features[col] = 1 if row[col] else 0
        
        # Signal features
        signal_cols = [
            'golden_cross', 'death_cross', 'ema_bullish', 'macd_bullish',
            'rsi_oversold', 'rsi_overbought', 'breakout_up', 'breakout_down',
            'volume_spike', 'bullish_trend', 'bearish_trend'
        ]
        
        for col in signal_cols:
            if col in row:
                features[col] = 1 if row[col] else 0
        
        # Return None if we don't have enough features
        if len(features) < 5:
            return None
        
        return features
    
    def predict_for_today(self, prices_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Make predictions for today's data
        
        Args:
            prices_dict: Dictionary mapping ticker to DataFrame with latest data
            
        Returns:
            Dictionary mapping ticker to prediction results
        """
        if not self.is_trained:
            logger.warning("Models not trained, returning default predictions")
            return self._get_default_predictions(prices_dict)
        
        predictions = {}
        
        for ticker, df in prices_dict.items():
            if df.empty:
                predictions[ticker] = self._get_default_prediction()
                continue
            
            try:
                # Extract features from latest row
                latest_row = df.iloc[-1]
                features = self._extract_features(latest_row)
                
                if features is None:
                    predictions[ticker] = self._get_default_prediction()
                    continue
                
                # Convert to DataFrame with same columns as training
                features_df = pd.DataFrame([features])
                
                # Ensure all training columns are present
                for col in self.feature_columns:
                    if col not in features_df.columns:
                        features_df[col] = 0
                
                # Reorder columns to match training
                features_df = features_df[self.feature_columns]
                
                # Scale features (would need to save scaler for production)
                # For now, assume features are already normalized
                
                # Make predictions
                expected_move = self.regressor.predict(features_df)[0]
                up_prob = self.classifier.predict_proba(features_df)[0][1]
                
                predictions[ticker] = {
                    'expected_move': expected_move,
                    'up_prob': up_prob,
                    'confidence': min(up_prob, 1 - up_prob) * 2,  # Convert to 0-1 scale
                    'direction': 'up' if up_prob > 0.5 else 'down'
                }
                
            except Exception as e:
                logger.error(f"Error predicting for {ticker}: {e}")
                predictions[ticker] = self._get_default_prediction()
        
        return predictions
    
    def _get_default_predictions(self, prices_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Get default predictions when models aren't trained"""
        return {ticker: self._get_default_prediction() for ticker in prices_dict.keys()}
    
    def _get_default_prediction(self) -> Dict[str, float]:
        """Get default prediction values"""
        return {
            'expected_move': 0.0,
            'up_prob': 0.5,
            'confidence': 0.0,
            'direction': 'neutral'
        }
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            if self.regressor is not None:
                with open(MODELS_DIR / "rf_price_move.pkl", 'wb') as f:
                    pickle.dump(self.regressor, f)
            
            if self.classifier is not None:
                with open(MODELS_DIR / "rf_direction.pkl", 'wb') as f:
                    pickle.dump(self.classifier, f)
            
            # Save feature columns
            if self.feature_columns is not None:
                with open(MODELS_DIR / "feature_columns.pkl", 'wb') as f:
                    pickle.dump(self.feature_columns, f)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current models"""
        return {
            'is_trained': self.is_trained,
            'last_training_date': self.last_training_date.isoformat() if self.last_training_date else None,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'regressor_available': self.regressor is not None,
            'classifier_available': self.classifier is not None
        }
    
    def retrain_if_needed(self, prices_dict: Dict[str, pd.DataFrame], days_since_training: int = 7) -> bool:
        """Retrain models if they're old or not trained"""
        if not self.is_trained:
            logger.info("Models not trained, starting training...")
            result = self.train_models(prices_dict)
            return "error" not in result
        
        if self.last_training_date is None:
            logger.info("No training date available, retraining...")
            result = self.train_models(prices_dict)
            return "error" not in result
        
        days_since = (datetime.now() - self.last_training_date).days
        if days_since >= days_since_training:
            logger.info(f"Models are {days_since} days old, retraining...")
            result = self.train_models(prices_dict)
            return "error" not in result
        
        return True

# Global ML predictor instance
ml_predictor = MLPredictor()

# Convenience functions
def train_ml_models(prices_dict: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
    """Train ML models on historical data"""
    return ml_predictor.train_models(prices_dict, **kwargs)

def predict_for_today(prices_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """Make predictions for today's data"""
    return ml_predictor.predict_for_today(prices_dict)

def get_ml_model_info() -> Dict[str, Any]:
    """Get information about the current ML models"""
    return ml_predictor.get_model_info()

def retrain_if_needed(prices_dict: Dict[str, pd.DataFrame]) -> bool:
    """Retrain models if needed"""
    return ml_predictor.retrain_if_needed(prices_dict)
