from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import ta
import ta.momentum
import ta.trend
import ta.volatility
import logging
import warnings

# Setup logging
logger = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=FutureWarning)


class TradingStrategy:
    """Base trading strategy class"""
    
    def __init__(self, name="BaseStrategy"):
        self.name = name
        self.risk_manager = None
    
    @property
    def strategy_name(self):
        """Property to access strategy name (for compatibility)"""
        return self.name
        
    def set_risk_manager(self, risk_manager):
        """Set the risk manager for this strategy"""
        self.risk_manager = risk_manager
        
    def get_signal(self, klines):
        """Get trading signal from klines data. Override in subclasses."""
        return None
        
        
    def add_indicators(self, df):
        """Add technical indicators to dataframe. Override in subclasses."""
        return df


class SmartTrendCatcher(TradingStrategy):
    """
    EMA Alignment Strategy with ADX Trend Strength Filter:
    
    Core Strategy:
    - Generates signals for every candle based on EMA alignment
    - Uses 10 and 30 EMA configuration
    - ADX trend strength filter: ADX <= 20 = HOLD (weak trend)
    - Continuous signal generation for each candle
    
    Signal Generation (Every Candle):
    - BUY: 10 EMA > 30 EMA AND ADX > 20 (bullish alignment with strong trend)
    - SELL: 10 EMA < 30 EMA AND ADX > 20 (bearish alignment with strong trend)  
    - HOLD: ADX <= 20 (weak trend) OR no clear EMA alignment

    Benefits of ADX Integration:
    - Filters out signals during weak trending conditions
    - ADX <= 20 indicates sideways/choppy market conditions
    - Reduces false signals in ranging markets
    - Only trades when trend strength is sufficient
    """
    
    def __init__(self, 
                 ema_slow=30,               # Slow EMA (30 period)
                 ema_fast=10,               # Fast EMA (10 period)
                 adx_period=14,             # ADX period (14 is standard)
                 adx_threshold=20.0):       # ADX threshold for trend strength
        
        super().__init__("SmartTrendCatcher")
        
        # Parameter validation
        if ema_slow <= 0 or ema_fast <= 0:
            raise ValueError("EMA periods must be positive")
        if ema_fast >= ema_slow:
            raise ValueError("Fast EMA must be less than slow EMA")
        if adx_period <= 0:
            raise ValueError("ADX period must be positive")
        if adx_threshold < 0:
            raise ValueError("ADX threshold must be non-negative")
        
        # Store parameters
        self.ema_slow = ema_slow
        self.ema_fast = ema_fast
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self._warning_count = 0
        
        logger.info(f"{self.name} initialized with:")
        logger.info(f"  EMA Alignment: {ema_fast}/{ema_slow}")
        logger.info(f"  ADX Period: {adx_period}")
        logger.info(f"  ADX Threshold: <= {adx_threshold} (HOLD), > {adx_threshold} (Allow signals)")
        logger.info(f"  Signal Generation: Every candle with ADX trend strength filter")
    
    def add_indicators(self, df):
        """Add EMA and ADX indicators for alignment strategy with trend strength filter"""
        try:
            # Ensure sufficient data
            min_required = max(self.ema_slow, self.ema_fast, self.adx_period) + 5
            if len(df) < min_required:
                logger.warning(f"Insufficient data: need {min_required}, got {len(df)}")
                return df
            
            # Data cleaning
            if df['close'].isna().any():
                logger.warning("Found NaN values in close prices, cleaning data")
                df['close'] = df['close'].interpolate(method='linear').bfill().ffill()
                
            # Ensure positive prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (df[col] <= 0).any():
                    logger.warning(f"Found zero or negative values in {col}, using interpolation")
                    df[col] = df[col].replace(0, np.nan)
                    df[col] = df[col].interpolate(method='linear').bfill().ffill()
                    
            # Calculate EMA indicators
            df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=self.ema_slow)
            df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=self.ema_fast)
            
            # Calculate ADX indicator
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=self.adx_period)
            
            # Handle NaN values
            df['ema_slow'] = df['ema_slow'].interpolate(method='linear').bfill()
            df['ema_fast'] = df['ema_fast'].interpolate(method='linear').bfill()
            df['adx'] = df['adx'].interpolate(method='linear').bfill()
            
            # EMA alignment conditions
            df['fast_above_slow'] = df['ema_fast'] > df['ema_slow']
            df['fast_below_slow'] = df['ema_fast'] < df['ema_slow']
            
            # ADX trend strength condition (ADX > threshold for strong trend)
            df['strong_trend'] = df['adx'] > self.adx_threshold
            
            # Generate signals with ADX filter
            df['buy_signal'] = df['fast_above_slow'] & df['strong_trend']
            df['sell_signal'] = df['fast_below_slow'] & df['strong_trend']
            df['hold_signal'] = ~df['strong_trend'] | ~(df['buy_signal'] | df['sell_signal'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding EMA and ADX indicators: {e}")
            return df
    
    def get_signal(self, klines):
        """Generate EMA alignment signals with ADX trend strength filter"""
        try:
            min_required = max(self.ema_slow, self.ema_fast, self.adx_period) + 5
            if not klines or len(klines) < min_required:
                # Show warning every 10th time to reduce log spam
                if self._warning_count % 10 == 0:
                    logger.warning(f"Insufficient data for EMA+ADX signal (need {min_required}, have {len(klines) if klines else 0})")
                self._warning_count += 1
                return None
            
            # Convert and validate data
            df = pd.DataFrame(klines)
            if len(df.columns) != 12:
                logger.error(f"Invalid klines format: expected 12 columns, got {len(df.columns)}")
                return None
                
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                         'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
            
            # Data cleaning
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    logger.warning(f"Cleaning NaN values in {col}")
                    df[col] = df[col].interpolate(method='linear').bfill().ffill()
            
            # Final validation after cleaning
            if df[numeric_columns].isna().any().any():
                logger.error("Failed to clean price data after interpolation")
                return None
            
            # Add EMA and ADX indicators
            df = self.add_indicators(df)
            
            if len(df) < 2:
                return None
            
            latest = df.iloc[-1]
            
            # Validate required columns
            required_columns = ['buy_signal', 'sell_signal', 'hold_signal', 'ema_fast', 'ema_slow', 'adx', 'strong_trend']
            
            for col in required_columns:
                if col not in df.columns or pd.isna(latest[col]):
                    logger.warning(f"Missing or invalid indicator: {col}")
                    return None
            
            # Generate signal based on EMA alignment AND ADX trend strength
            signal = None
            
            # Check ADX first - if ADX <= threshold, always HOLD
            if latest['adx'] <= self.adx_threshold:
                signal = 'HOLD'
                logger.info(f"⚪ HOLD Signal - Weak Trend (ADX Filter)")
                logger.info(f"   ADX: {latest['adx']:.2f} <= {self.adx_threshold} (weak trend)")
                logger.info(f"   Fast EMA (10): {latest['ema_fast']:.6f}, Slow EMA (30): {latest['ema_slow']:.6f}")
                logger.info(f"   Current Price: {latest['close']:.6f}")
            
            # BUY Signal: Fast EMA above Slow EMA AND ADX > threshold
            elif latest['buy_signal']:
                signal = 'BUY'
                logger.info(f"🟢 BUY Signal - EMA Bullish Alignment + Strong Trend")
                logger.info(f"   ADX: {latest['adx']:.2f} > {self.adx_threshold} (strong trend)")
                logger.info(f"   Fast EMA (10): {latest['ema_fast']:.6f} > Slow EMA (30): {latest['ema_slow']:.6f}")
                logger.info(f"   Current Price: {latest['close']:.6f}")
            
            # SELL Signal: Fast EMA below Slow EMA AND ADX > threshold
            elif latest['sell_signal']:
                signal = 'SELL'
                logger.info(f"🔴 SELL Signal - EMA Bearish Alignment + Strong Trend")
                logger.info(f"   ADX: {latest['adx']:.2f} > {self.adx_threshold} (strong trend)")
                logger.info(f"   Fast EMA (10): {latest['ema_fast']:.6f} < Slow EMA (30): {latest['ema_slow']:.6f}")
                logger.info(f"   Current Price: {latest['close']:.6f}")
            
            # HOLD Signal: Strong trend but no clear EMA alignment
            else:
                signal = 'HOLD'
                logger.info(f"⚪ HOLD Signal - No Clear EMA Alignment")
                logger.info(f"   ADX: {latest['adx']:.2f} > {self.adx_threshold} (strong trend)")
                logger.info(f"   Fast EMA (10): {latest['ema_fast']:.6f}, Slow EMA (30): {latest['ema_slow']:.6f}")
                logger.info(f"   Current Price: {latest['close']:.6f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in EMA+ADX signal generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
# Factory function to get a strategy by name
def get_strategy(strategy_name):
    """Factory function to get a strategy by name"""
    # Import EMA and ADX config values
    try:
        from modules.config import FAST_EMA, SLOW_EMA, ADX_PERIOD, ADX_THRESHOLD
    except ImportError:
        # Fallback values
        FAST_EMA = 10
        SLOW_EMA = 30
        ADX_PERIOD = 14
        ADX_THRESHOLD = 20.0
    
    strategies = {
        'SmartTrendCatcher': SmartTrendCatcher(
            ema_slow=SLOW_EMA,
            ema_fast=FAST_EMA,
            adx_period=ADX_PERIOD,
            adx_threshold=ADX_THRESHOLD
        ),
    }
    
    if strategy_name in strategies:
        return strategies[strategy_name]
    
    logger.warning(f"Strategy {strategy_name} not found. Defaulting to SmartTrendCatcher.")
    return strategies['SmartTrendCatcher']


def get_strategy_for_symbol(symbol, strategy_name=None):
    """Get the appropriate strategy based on the trading symbol"""
    # If a specific strategy is requested, use it
    if strategy_name:
        return get_strategy(strategy_name)
    
    # Default to SmartTrendCatcher for any symbol
    return get_strategy('SmartTrendCatcher')