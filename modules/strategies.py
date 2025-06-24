from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import pandas_ta as ta
import logging
import warnings
import traceback

# Setup logging
logger = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import EMA and ADX config values at module level
try:
    from modules.config import FAST_EMA, SLOW_EMA, ADX_PERIOD, ADX_THRESHOLD
except ImportError:
    # Fallback values
    FAST_EMA = 10
    SLOW_EMA = 30
    ADX_PERIOD = 14
    ADX_THRESHOLD = 20


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
    EMA Crossover + ADX Trend Strategy:
    
    Core Strategy:
    - Combines EMA crossover signals with ADX trend strength filtering
    - Uses 10 and 30 EMA configuration with 14-period ADX
    - ADX <= 20 forces HOLD signal (weak trend condition)
    - Only generates BUY/SELL when EMA crossover occurs AND ADX confirms strength
    
    Technical Indicators (pandas_ta):
    - EMA (Exponential Moving Average): Fast (10) and Slow (30) periods
    - ADX (Average Directional Index): 14-period for trend strength
    - Standard implementations using pandas_ta library
    
    Signal Generation Logic:
    - BUY: Fast EMA crosses ABOVE Slow EMA AND ADX > 20 (bullish crossover + strong trend)
    - SELL: Fast EMA crosses BELOW Slow EMA AND ADX > 20 (bearish crossover + strong trend)  
    - HOLD: No crossover OR ADX <= 20 (no signal or weak trend)

    Benefits of EMA Crossover + ADX Strategy:
    - Filters out weak trends and sideways markets
    - Reduces false signals during consolidation
    - Only trades on actual trend changes (crossovers)
    - Combines trend direction change (EMA crossover) with trend strength (ADX)
    """
    
    def __init__(self, 
                 ema_slow=30,               # Slow EMA (30 period)
                 ema_fast=10,               # Fast EMA (10 period)
                 adx_period=14,             # ADX period (14 standard)
                 adx_threshold=20):  # ADX threshold from config (<=20 = HOLD)
        
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
        logger.info(f"  EMA Crossover: {ema_fast}/{ema_slow} (pandas_ta implementation)")
        logger.info(f"  ADX Filter: {adx_period}-period, threshold <= {adx_threshold} = HOLD")
        logger.info(f"  Signal Generation: EMA crossover + ADX strength confirmation")
    
    def add_indicators(self, df):
        """Add EMA and ADX indicators for crossover strategy"""
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
                    
            # Calculate EMA indicators using pandas_ta
            df['ema_slow'] = ta.ema(df['close'], length=self.ema_slow)
            df['ema_fast'] = ta.ema(df['close'], length=self.ema_fast)
            
            # Calculate ADX indicator using pandas_ta
            adx_result = ta.adx(high=df['high'], low=df['low'], close=df['close'], length=self.adx_period)
            df['adx'] = adx_result[f'ADX_{self.adx_period}']
            
            # Handle NaN values
            df['ema_slow'] = df['ema_slow'].interpolate(method='linear').bfill()
            df['ema_fast'] = df['ema_fast'].interpolate(method='linear').bfill()
            df['adx'] = df['adx'].interpolate(method='linear').bfill()
            
            # EMA crossover detection (need at least 2 periods to detect crossover)
            if len(df) >= 2:
                # Current period alignment
                df['fast_above_slow'] = df['ema_fast'] > df['ema_slow']
                df['fast_below_slow'] = df['ema_fast'] < df['ema_slow']
                
                # Previous period alignment
                df['fast_above_slow_prev'] = df['fast_above_slow'].shift(1)
                df['fast_below_slow_prev'] = df['fast_below_slow'].shift(1)
                
                # Crossover detection
                df['bullish_crossover'] = (df['fast_above_slow'] & df['fast_below_slow_prev'])  # Fast crosses above slow
                df['bearish_crossover'] = (df['fast_below_slow'] & df['fast_above_slow_prev'])  # Fast crosses below slow
            else:
                # Not enough data for crossover detection
                df['fast_above_slow'] = df['ema_fast'] > df['ema_slow']
                df['fast_below_slow'] = df['ema_fast'] < df['ema_slow']
                df['fast_above_slow_prev'] = False
                df['fast_below_slow_prev'] = False
                df['bullish_crossover'] = False
                df['bearish_crossover'] = False
            
            # ADX strength condition
            df['adx_strong'] = df['adx'] > self.adx_threshold
            df['adx_weak'] = df['adx'] <= self.adx_threshold
            
            # Generate signals based on EMA crossover AND ADX strength
            df['buy_signal'] = df['bullish_crossover'] & df['adx_strong']
            df['sell_signal'] = df['bearish_crossover'] & df['adx_strong']
            df['hold_signal'] = df['adx_weak'] | (~df['bullish_crossover'] & ~df['bearish_crossover'])  # ADX weak OR no crossover
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding EMA crossover+ADX indicators: {e}")
            return df
    
    def get_signal(self, klines):
        """Generate EMA crossover+ADX filtered signals"""
        try:
            min_required = max(self.ema_slow, self.ema_fast, self.adx_period) + 5
            if not klines or len(klines) < min_required:
                # Show warning every 10th time to reduce log spam
                if self._warning_count % 10 == 0:
                    logger.warning(f"Insufficient data for EMA crossover+ADX signal (need {min_required}, have {len(klines) if klines else 0})")
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
            
            # Add EMA crossover and ADX indicators
            df = self.add_indicators(df)
            
            if len(df) < 2:
                return None
            
            latest = df.iloc[-1]
            
            # Validate required columns for crossover detection
            required_columns = ['buy_signal', 'sell_signal', 'hold_signal', 'ema_fast', 'ema_slow', 'adx', 'bullish_crossover', 'bearish_crossover']
            
            for col in required_columns:
                if col not in df.columns or pd.isna(latest[col]):
                    logger.warning(f"Missing or invalid indicator: {col}")
                    return None
            
            # Generate signal based on EMA crossover+ADX combination
            signal = None
            
            # BUY Signal: Bullish crossover AND ADX > threshold
            if latest['buy_signal']:
                signal = 'BUY'
                logger.info(f"🟢 BUY Signal - EMA Bullish Crossover + Strong Trend")
                logger.info(f"   Fast EMA ({self.ema_fast}) crossed ABOVE Slow EMA ({self.ema_slow})")
                logger.info(f"   Fast EMA: {latest['ema_fast']:.6f} > Slow EMA: {latest['ema_slow']:.6f}")
                logger.info(f"   ADX ({self.adx_period}): {latest['adx']:.2f} > {self.adx_threshold} (strong trend)")
                logger.info(f"   Current Price: {latest['close']:.6f}")
            
            # SELL Signal: Bearish crossover AND ADX > threshold
            elif latest['sell_signal']:
                signal = 'SELL'
                logger.info(f"🔴 SELL Signal - EMA Bearish Crossover + Strong Trend")
                logger.info(f"   Fast EMA ({self.ema_fast}) crossed BELOW Slow EMA ({self.ema_slow})")
                logger.info(f"   Fast EMA: {latest['ema_fast']:.6f} < Slow EMA: {latest['ema_slow']:.6f}")
                logger.info(f"   ADX ({self.adx_period}): {latest['adx']:.2f} > {self.adx_threshold} (strong trend)")
                logger.info(f"   Current Price: {latest['close']:.6f}")
            
            # HOLD Signal: No crossover OR ADX <= threshold (weak trend)
            else:
                signal = 'HOLD'
                if latest['hold_signal'] and latest['adx'] <= self.adx_threshold:
                    logger.info(f"⚪ HOLD Signal - Weak Trend (ADX Filter)")
                    logger.info(f"   ADX ({self.adx_period}): {latest['adx']:.2f} <= {self.adx_threshold} (weak trend)")
                else:
                    logger.info(f"⚪ HOLD Signal - No EMA Crossover")
                    logger.info(f"   No crossover detected (waiting for Fast EMA to cross Slow EMA)")
                logger.info(f"   Fast EMA ({self.ema_fast}): {latest['ema_fast']:.6f}, Slow EMA ({self.ema_slow}): {latest['ema_slow']:.6f}")
                logger.info(f"   ADX ({self.adx_period}): {latest['adx']:.2f}")
                logger.info(f"   Current Price: {latest['close']:.6f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in EMA crossover+ADX signal generation: {e}")
            logger.error(traceback.format_exc())
            return None


# Factory function to get a strategy by name
def get_strategy(strategy_name):
    """Factory function to get a strategy by name"""
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