from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import ta
import ta.momentum
import ta.trend
import ta.volatility
import logging
import warnings
import traceback

# Setup logging
logger = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import EMA and ADX config values at module level
try:
    from modules.config import FAST_EMA, SLOW_EMA, ADX_SMOOTHING, ADX_DI_LENGTH, ADX_THRESHOLD
except ImportError:
    # Fallback values
    FAST_EMA = 10
    SLOW_EMA = 30
    ADX_SMOOTHING = 14
    ADX_DI_LENGTH = 20
    ADX_THRESHOLD = 20.0

# Import EMA and ADX config values at module level
try:
    from modules.config import FAST_EMA, SLOW_EMA, ADX_SMOOTHING, ADX_DI_LENGTH, ADX_THRESHOLD
except ImportError:
    # Fallback values
    FAST_EMA = 10
    SLOW_EMA = 30
    ADX_SMOOTHING = 14
    ADX_DI_LENGTH = 20
    ADX_THRESHOLD = 20.0


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
    EMA Alignment Strategy with Pure ADX Trend Strength Filter:
    
    Core Strategy:
    - Generates signals for every candle based on EMA alignment
    - Uses 10 and 30 EMA configuration
    - Pure ADX by Minhaz trend strength filter: ADX <= 20 = HOLD (weak trend)
    - Continuous signal generation for each candle
    
    Pure ADX Implementation:
    - Custom ADX calculation based on Pine Script version 6
    - ADX Smoothing: 14 periods (adxlen in Pine Script)
    - DI Length: 20 periods (dilen in Pine Script)
    - More precise directional movement calculation
    - Uses RMA (Relative Moving Average) for smoothing
    
    Signal Generation (Every Candle):
    - BUY: 10 EMA > 30 EMA AND Pure ADX > 20 (bullish alignment with strong trend)
    - SELL: 10 EMA < 30 EMA AND Pure ADX > 20 (bearish alignment with strong trend)  
    - HOLD: Pure ADX <= 20 (weak trend) OR no clear EMA alignment

    Benefits of Pure ADX Integration:
    - Filters out signals during weak trending conditions
    - Pure ADX <= 20 indicates sideways/choppy market conditions
    - Reduces false signals in ranging markets
    - Only trades when trend strength is sufficient
    - More accurate directional movement calculation than standard ADX
    """
    
    def __init__(self, 
                 ema_slow=30,               # Slow EMA (30 period)
                 ema_fast=10,               # Fast EMA (10 period)
                 adx_smoothing=14,          # ADX smoothing period (equivalent to adxlen in Pine Script)
                 adx_di_length=20,          # DI length period (equivalent to dilen in Pine Script)
                 adx_threshold=20.0):       # ADX threshold for trend strength
        
        super().__init__("SmartTrendCatcher")
        
        # Parameter validation
        if ema_slow <= 0 or ema_fast <= 0:
            raise ValueError("EMA periods must be positive")
        if ema_fast >= ema_slow:
            raise ValueError("Fast EMA must be less than slow EMA")
        if adx_smoothing <= 0 or adx_di_length <= 0:
            raise ValueError("ADX parameters must be positive")
        if adx_threshold < 0:
            raise ValueError("ADX threshold must be non-negative")
        
        # Store parameters
        self.ema_slow = ema_slow
        self.ema_fast = ema_fast
        self.adx_smoothing = adx_smoothing
        self.adx_di_length = adx_di_length
        self.adx_threshold = adx_threshold
        self._warning_count = 0
        
        logger.info(f"{self.name} initialized with:")
        logger.info(f"  EMA Alignment: {ema_fast}/{ema_slow}")
        logger.info(f"  ADX Smoothing: {adx_smoothing}")
        logger.info(f"  ADX DI Length: {adx_di_length}")
        logger.info(f"  ADX Threshold: <= {adx_threshold} (HOLD), > {adx_threshold} (Allow signals)")
        logger.info(f"  Signal Generation: Every candle with ADX trend strength filter")
    
    def add_indicators(self, df):
        """Add EMA and ADX indicators for alignment strategy with trend strength filter"""
        try:
            # Ensure sufficient data
            min_required = max(self.ema_slow, self.ema_fast, self.adx_smoothing, self.adx_di_length) + 5
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
            
            # Calculate custom ADX indicator (Pure ADX by Minhaz implementation)
            df['adx'] = self._calculate_pure_adx(df, self.adx_di_length, self.adx_smoothing)
            
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
            df['hold_signal'] = ~df['strong_trend']  # Simplified: HOLD when trend is weak
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding EMA and ADX indicators: {e}")
            return df
    
    def _calculate_pure_adx(self, df, di_length=20, adx_smoothing=14):
        """
        Calculate Pure ADX by Minhaz - Custom implementation based on Pine Script
        //@version=6
        indicator("Pure ADX By Minhaz", format=format.price, precision=2)
        
        Args:
            df: DataFrame with OHLC data
            di_length: DI Length (dilen in Pine Script, default 20)
            adx_smoothing: ADX Smoothing (adxlen in Pine Script, default 14)
        
        Returns:
            ADX series
        """
        try:
            # Calculate price changes
            high_change = df['high'].diff()
            low_change = -df['low'].diff()
            
            # Calculate directional movements (plusDM and minusDM)
            plus_dm = np.where(
                (high_change > low_change) & (high_change > 0),
                high_change,
                0
            )
            minus_dm = np.where(
                (low_change > high_change) & (low_change > 0),
                low_change,
                0
            )
            
            # Calculate True Range (TR)
            tr1 = df['high'] - df['low']
            tr2 = np.abs(df['high'] - df['close'].shift(1))
            tr3 = np.abs(df['low'] - df['close'].shift(1))
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate smoothed values using RMA (Relative Moving Average)
            # RMA in Pine Script is equivalent to EWM with alpha = 1/length
            alpha_di = 1.0 / di_length
            alpha_adx = 1.0 / adx_smoothing
            
            # Smoothed True Range
            tr_rma = pd.Series(true_range).ewm(alpha=alpha_di, adjust=False).mean()
            
            # Smoothed Plus DM and Minus DM
            plus_dm_rma = pd.Series(plus_dm).ewm(alpha=alpha_di, adjust=False).mean()
            minus_dm_rma = pd.Series(minus_dm).ewm(alpha=alpha_di, adjust=False).mean()
            
            # Calculate Plus DI and Minus DI
            # Avoid division by zero in TR calculations
            tr_rma_safe = tr_rma.replace(0, np.finfo(float).eps)  # Replace zeros with smallest float
            plus_di = 100 * plus_dm_rma / tr_rma_safe
            minus_di = 100 * minus_dm_rma / tr_rma_safe
            
            # Handle division by zero and fill NaN values
            plus_di = plus_di.fillna(0)
            minus_di = minus_di.fillna(0)
            
            # Calculate DX (Directional Index)
            di_sum = plus_di + minus_di
            di_diff = np.abs(plus_di - minus_di)
            
            # Avoid division by zero
            dx = np.where(di_sum == 0, 0, 100 * di_diff / di_sum)
            
            # Calculate ADX using RMA smoothing
            adx = pd.Series(dx).ewm(alpha=alpha_adx, adjust=False).mean()
            
            # Fill initial NaN values with 0
            adx = adx.fillna(0)
            
            logger.debug(f"Pure ADX calculated with DI Length: {di_length}, ADX Smoothing: {adx_smoothing}")
            
            return adx
            
        except Exception as e:
            logger.error(f"Error calculating Pure ADX: {e}")
            # Fallback to standard ADX
            return ta.trend.adx(df['high'], df['low'], df['close'], window=adx_smoothing)
    
    def get_signal(self, klines):
        """Generate EMA alignment signals with ADX trend strength filter"""
        try:
            min_required = max(self.ema_slow, self.ema_fast, self.adx_smoothing, self.adx_di_length) + 5
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
                logger.info(f"⚪ HOLD Signal - Weak Trend (Pure ADX Filter)")
                logger.info(f"   Pure ADX: {latest['adx']:.2f} <= {self.adx_threshold} (weak trend)")
                logger.info(f"   Fast EMA ({self.ema_fast}): {latest['ema_fast']:.6f}, Slow EMA ({self.ema_slow}): {latest['ema_slow']:.6f}")
                logger.info(f"   Current Price: {latest['close']:.6f}")
            
            # BUY Signal: Fast EMA above Slow EMA AND ADX > threshold
            elif latest['buy_signal']:
                signal = 'BUY'
                logger.info(f"🟢 BUY Signal - EMA Bullish Alignment + Strong Trend")
                logger.info(f"   Pure ADX: {latest['adx']:.2f} > {self.adx_threshold} (strong trend)")
                logger.info(f"   Fast EMA ({self.ema_fast}): {latest['ema_fast']:.6f} > Slow EMA ({self.ema_slow}): {latest['ema_slow']:.6f}")
                logger.info(f"   Current Price: {latest['close']:.6f}")
            
            # SELL Signal: Fast EMA below Slow EMA AND ADX > threshold
            elif latest['sell_signal']:
                signal = 'SELL'
                logger.info(f"🔴 SELL Signal - EMA Bearish Alignment + Strong Trend")
                logger.info(f"   Pure ADX: {latest['adx']:.2f} > {self.adx_threshold} (strong trend)")
                logger.info(f"   Fast EMA ({self.ema_fast}): {latest['ema_fast']:.6f} < Slow EMA ({self.ema_slow}): {latest['ema_slow']:.6f}")
                logger.info(f"   Current Price: {latest['close']:.6f}")
            
            # HOLD Signal: Strong trend but no clear EMA alignment
            else:
                signal = 'HOLD'
                logger.info(f"⚪ HOLD Signal - No Clear EMA Alignment")
                logger.info(f"   Pure ADX: {latest['adx']:.2f} > {self.adx_threshold} (strong trend)")
                logger.info(f"   Fast EMA ({self.ema_fast}): {latest['ema_fast']:.6f}, Slow EMA ({self.ema_slow}): {latest['ema_slow']:.6f}")
                logger.info(f"   Current Price: {latest['close']:.6f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in EMA+ADX signal generation: {e}")
            logger.error(traceback.format_exc())
            return None


# Factory function to get a strategy by name
def get_strategy(strategy_name):
    """Factory function to get a strategy by name"""
    strategies = {
        'SmartTrendCatcher': SmartTrendCatcher(
            ema_slow=SLOW_EMA,
            ema_fast=FAST_EMA,
            adx_smoothing=ADX_SMOOTHING,
            adx_di_length=ADX_DI_LENGTH,
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