"""
Core utilities for TradeScrubber
Handles environment loading, caching, logging, and configuration
"""

import os
import json
import logging
import functools
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable
import pandas as pd

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data_cache"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "presets"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

def load_env():
    """Load environment variables from .env file"""
    try:
        from dotenv import load_dotenv
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment from {env_path}")
        else:
            print("No .env file found, using system environment variables")
    except ImportError:
        print("python-dotenv not installed, using system environment variables")

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("tradescrubber")
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

def disk_cache(key: str, ttl_minutes: int = 15) -> Callable:
    """
    Decorator for disk-based caching with TTL
    
    Args:
        key: Cache key prefix
        ttl_minutes: Time to live in minutes
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{key}_{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            cache_file = CACHE_DIR / f"{cache_key}.json"
            
            # Check if cache exists and is valid
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    # Check TTL
                    cache_time = datetime.fromisoformat(cache_data['timestamp'])
                    if datetime.now() - cache_time < timedelta(minutes=ttl_minutes):
                        return cache_data['data']
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Invalid cache file, remove it
                    cache_file.unlink(missing_ok=True)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Save to cache
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': result
            }
            
            try:
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, default=str)
            except (TypeError, ValueError) as e:
                # If result is not JSON serializable, skip caching
                print(f"Warning: Could not cache result for {func.__name__}: {e}")
            
            return result
        
        return wrapper
    return decorator

def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value from environment"""
    return os.getenv(key, default)

def get_data_source() -> str:
    """Get configured data source"""
    return get_config("DATA_SOURCE", "yfinance")

def get_cache_ttl() -> int:
    """Get cache TTL in minutes"""
    return int(get_config("CACHE_TTL_MIN", "15"))

def get_timezone() -> str:
    """Get configured timezone"""
    return get_config("TZ", "America/New_York")

def load_watchlist(preset: str = "default") -> list:
    """Load watchlist from presets"""
    watchlist_file = CONFIG_DIR / "watchlists.yaml"
    
    if not watchlist_file.exists():
        # Create default watchlist
        default_watchlist = {
            "default": [
                "SPY", "QQQ", "IWM", "DIA", "VTI", "VEA", "VWO", "BND", "TLT", "GLD",
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "INTC",
                "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE",
                "CRM", "NKE", "ABT", "TMO", "ACN", "WMT", "VZ", "KO", "PEP", "MRK",
                "T", "CSCO", "ABBV", "CVX", "LLY", "DHR", "NEE", "TXN", "QCOM", "HON"
            ]
        }
        
        try:
            import yaml
            with open(watchlist_file, 'w') as f:
                yaml.dump(default_watchlist, f)
        except ImportError:
            # Fallback to JSON if YAML not available
            with open(watchlist_file.with_suffix('.json'), 'w') as f:
                json.dump(default_watchlist, f, indent=2)
    
    try:
        import yaml
        with open(watchlist_file, 'r') as f:
            watchlists = yaml.safe_load(f)
        return watchlists.get(preset, watchlists.get("default", []))
    except (ImportError, FileNotFoundError):
        # Fallback to JSON
        json_file = watchlist_file.with_suffix('.json')
        if json_file.exists():
            with open(json_file, 'r') as f:
                watchlists = json.load(f)
            return watchlists.get(preset, watchlists.get("default", []))
    
    # Ultimate fallback
    return ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

def load_strategy_presets() -> dict:
    """Load strategy presets from YAML"""
    strategy_file = CONFIG_DIR / "strategies.yaml"
    
    if not strategy_file.exists():
        # Create default strategies
        default_strategies = {
            "reversal": {
                "name": "Reversal Strategy",
                "description": "Look for oversold conditions with bullish divergence",
                "filters": {
                    "rsi_min": 0,
                    "rsi_max": 35,
                    "price_above_sma200": False,
                    "min_volume_ratio": 1.0
                }
            },
            "breakout": {
                "name": "Breakout Strategy", 
                "description": "Look for breakouts above resistance with volume",
                "filters": {
                    "rsi_min": 50,
                    "rsi_max": 100,
                    "price_above_sma200": True,
                    "min_volume_ratio": 1.5
                }
            },
            "trend": {
                "name": "Trend Following",
                "description": "Follow strong trends with momentum",
                "filters": {
                    "rsi_min": 40,
                    "rsi_max": 80,
                    "price_above_sma200": True,
                    "min_volume_ratio": 1.2
                }
            }
        }
        
        try:
            import yaml
            with open(strategy_file, 'w') as f:
                yaml.dump(default_strategies, f)
        except ImportError:
            # Fallback to JSON
            with open(strategy_file.with_suffix('.json'), 'w') as f:
                json.dump(default_strategies, f, indent=2)
    
    try:
        import yaml
        with open(strategy_file, 'r') as f:
            return yaml.safe_load(f)
    except (ImportError, FileNotFoundError):
        # Fallback to JSON
        json_file = strategy_file.with_suffix('.json')
        if json_file.exists():
            with open(json_file, 'r') as f:
                return json.load(f)
    
    return {}

def clear_cache():
    """Clear all cached data"""
    for cache_file in CACHE_DIR.glob("*.json"):
        cache_file.unlink()
    print(f"Cleared {len(list(CACHE_DIR.glob('*.json')))} cache files")

def get_cache_stats() -> dict:
    """Get cache statistics"""
    cache_files = list(CACHE_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in cache_files)
    
    return {
        "total_files": len(cache_files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "cache_dir": str(CACHE_DIR)
    }
