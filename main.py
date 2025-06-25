#!/usr/bin/env python3
"""
Trading Bot with Enhanced Deduplication

Recent improvements to prevent duplicate processing:
1. Order update deduplication - prevents duplicate order logs and processing
2. Account update throttling - prevents spam from duplicate balance updates  
3. Price update throttling - limits kline update logs to every 5 seconds
4. Chart generation throttling - prevents excessive chart creation
5. Telegram notification deduplication - prevents duplicate messages

These changes eliminate the duplicate logging and processing issues while
maintaining full functionality and real-time responsiveness.
"""

import os
import logging
import time
import signal
import schedule
import argparse
import json
import traceback 
import requests
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from modules.binance_client import BinanceClient
from modules.risk_manager import RiskManager
from modules.strategies import get_strategy
from modules.backtest import Backtester
from modules.websocket_handler import BinanceWebSocketManager
from modules.config import (
    TRADING_SYMBOL, TIMEFRAME, STRATEGY, LOG_LEVEL,
    USE_TELEGRAM, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    SEND_DAILY_REPORT, DAILY_REPORT_TIME, AUTO_COMPOUND,
    MULTI_INSTANCE_MODE, MAX_POSITIONS_PER_SYMBOL,
    USE_TAKE_PROFIT, TAKE_PROFIT_PCT,
    UPDATE_TRAILING_ON_HOLD,
    # Add these to your config.py:
    # BACKTEST_BEFORE_LIVE = True
    # BACKTEST_MIN_PROFIT_PCT = 5.0
    # BACKTEST_MIN_WIN_RATE = 50.0
    # BACKTEST_PERIOD = "30 days"
)

# Additional imports for WebSocket-only buffer
from collections import deque
import threading

# Try to import the new config variables, use defaults if not available
try:
    from modules.config import (
        BACKTEST_BEFORE_LIVE, 
        BACKTEST_MIN_PROFIT_PCT, 
        BACKTEST_MIN_WIN_RATE,
        BACKTEST_PERIOD
    )
except ImportError:
    # Default values if not in config
    BACKTEST_BEFORE_LIVE = False  # Changed to False to skip backtest by default
    BACKTEST_MIN_PROFIT_PCT = 5.0
    BACKTEST_MIN_WIN_RATE = 50.0
    BACKTEST_PERIOD = "15 days"

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RollingCandleBuffer:
    """
    WebSocket-Only Rolling Buffer for Real-Time Signal Generation
    
    Maintains a rolling buffer of candles updated exclusively by WebSocket data.
    Uses REST API only once at initialization to seed the buffer.
    After initialization, all signal generation uses WebSocket data only.
    """
    
    def __init__(self, symbol, timeframe, buffer_size=60):
        self.symbol = symbol
        self.timeframe = timeframe
        self.buffer_size = buffer_size
        
        # Rolling buffer - uses deque for efficient append/pop operations
        self.candle_buffer = deque(maxlen=buffer_size)
        
        # State tracking
        self.is_initialized = False
        self.last_candle_time = None
        self.buffer_lock = threading.Lock()
        
        # Statistics for monitoring
        self.total_updates = 0
        self.signals_generated = 0
        
        logger.info(f"🔄 Created rolling buffer for {symbol} ({timeframe}) - size: {buffer_size}")
    
    def initialize_from_api(self, binance_client):
        """
        ONE-TIME INITIALIZATION: Seed buffer with REST API data
        After this, buffer is updated exclusively by WebSocket
        """
        logger.info(f"🚀 Initializing WebSocket-only buffer for {self.symbol}...")
        
        try:
            # Calculate appropriate time period based on timeframe to get enough candles
            # Need at least 60 candles for proper EMA calculations
            timeframe_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
                '12h': 720, '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
            }
            
            minutes_per_candle = timeframe_minutes.get(self.timeframe, 15)  # Default to 15m
            required_minutes = minutes_per_candle * self.buffer_size  # Get full buffer worth of data
            required_hours = max(1, required_minutes // 60)  # At least 1 hour
            
            start_str = f"{required_hours} hours ago"
            logger.info(f"📊 Getting {self.buffer_size} candles for {self.timeframe} timeframe ({start_str})")
            
            # ONE-TIME API call to seed the buffer
            initial_klines = binance_client.get_historical_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                start_str=start_str,
                limit=self.buffer_size
            )
            
            if not initial_klines or len(initial_klines) < 50:
                logger.error(f"❌ Failed to initialize buffer: {len(initial_klines) if initial_klines else 0} candles")
                return False
            
            # Convert API data to standardized format and populate buffer
            with self.buffer_lock:
                self.candle_buffer.clear()
                for kline in initial_klines:
                    candle = self._format_api_candle(kline)
                    self.candle_buffer.append(candle)
                    
            self.last_candle_time = self.candle_buffer[-1][6]  # close_time
            self.is_initialized = True
            
            logger.info(f"✅ Buffer initialized with {len(self.candle_buffer)} candles")
            logger.info(f"📈 Last candle time: {datetime.fromtimestamp(self.last_candle_time/1000)}")
            logger.info("🔥 BUFFER READY - Switching to WebSocket-only mode!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize buffer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _format_api_candle(self, kline):
        """Convert API kline to standard format matching WebSocket structure"""
        return [
            int(kline[0]),      # open_time
            float(kline[1]),    # open
            float(kline[2]),    # high  
            float(kline[3]),    # low
            float(kline[4]),    # close
            float(kline[5]),    # volume
            int(kline[6]),      # close_time
            float(kline[7]),    # quote_asset_volume
            int(kline[8]),      # number_of_trades
            float(kline[9]),    # taker_buy_base_asset_volume
            float(kline[10]),   # taker_buy_quote_asset_volume
            0                   # ignore (unused)
        ]
    
    def update_from_websocket(self, kline_data):
        """
        Update buffer with new WebSocket candle data
        This is the ONLY way buffer gets updated after initialization
        """
        if not self.is_initialized:
            logger.warning("⚠️ Buffer not initialized - cannot update from WebSocket")
            return False
        
        try:
            # Log the incoming kline data for debugging
            logger.debug(f"🔍 WebSocket kline data: close_time={kline_data.get('close_time')}, "
                        f"is_closed={kline_data.get('is_closed', 'N/A')}, "
                        f"last_buffer_time={self.last_candle_time}")
            
            # Convert WebSocket data to standard format
            candle = [
                kline_data['open_time'],
                float(kline_data['open']),
                float(kline_data['high']),
                float(kline_data['low']),
                float(kline_data['close']),
                float(kline_data['volume']),
                kline_data['close_time'],
                0,  # placeholder
                0,  # placeholder  
                0,  # placeholder
                0,  # placeholder
                0   # placeholder
            ]
            
            with self.buffer_lock:
                current_close_time = kline_data['close_time']
                
                # Enhanced logic for determining if this is a new candle
                # Allow for small time differences and handle initialization edge cases
                is_new_candle = False
                
                if not self.last_candle_time:
                    # First WebSocket update
                    logger.info(f"🆕 First WebSocket candle update")
                    is_new_candle = True
                elif current_close_time > self.last_candle_time:
                    # Clearly newer candle
                    logger.info(f"🆕 New candle detected: {current_close_time} > {self.last_candle_time}")
                    is_new_candle = True
                elif current_close_time == self.last_candle_time:
                    # Same time - could be duplicate or update to same candle
                    logger.debug(f"🔄 Same candle time - checking if it's an update")
                    
                    # If we have existing candles, check if this is updating the last one
                    if len(self.candle_buffer) > 0:
                        last_candle = self.candle_buffer[-1]
                        last_buffer_close_time = last_candle[6]  # close_time is at index 6
                        
                        if current_close_time == last_buffer_close_time:
                            # Update the existing candle instead of adding a new one
                            logger.info(f"🔄 Updating existing candle with same close_time: {current_close_time}")
                            self.candle_buffer[-1] = candle
                            self.total_updates += 1
                            logger.info(f"📊 Buffer updated (same candle): {len(self.candle_buffer)} candles, "
                                      f"latest: O={kline_data['open']:.6f} C={kline_data['close']:.6f}")
                            return True
                        else:
                            logger.warning(f"⚠️ Time mismatch: WebSocket={current_close_time}, Buffer={last_buffer_close_time}")
                            is_new_candle = True
                    else:
                        # No candles in buffer yet
                        is_new_candle = True
                else:
                    # Current time is older than last time - this should not happen normally
                    logger.warning(f"⚠️ Received older candle: {current_close_time} < {self.last_candle_time}")
                    
                    # But allow it if the time difference is small (within 1 minute)
                    time_diff = self.last_candle_time - current_close_time
                    if time_diff < 60000:  # Less than 1 minute in milliseconds
                        logger.info(f"🔄 Allowing slightly older candle (diff: {time_diff}ms)")
                        is_new_candle = True
                    else:
                        logger.warning(f"� Rejecting significantly older candle (diff: {time_diff}ms)")
                        return False
                
                if is_new_candle:
                    self.candle_buffer.append(candle)
                    self.last_candle_time = current_close_time
                    self.total_updates += 1
                    
                    logger.info(f"📊 Buffer updated: {len(self.candle_buffer)} candles, "
                              f"latest: O={kline_data['open']:.6f} C={kline_data['close']:.6f} "
                              f"at {datetime.fromtimestamp(kline_data['close_time']/1000)}")
                    return True
                else:
                    logger.debug(f"🔄 No buffer update needed")
                    return False
                    
        except KeyError as e:
            logger.error(f"❌ Missing key in WebSocket data: {e}. Data: {kline_data}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to update buffer from WebSocket: {e}")
            logger.error(f"📋 WebSocket data: {kline_data}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_buffer_data(self):
        """
        Get current buffer data for strategy analysis
        Returns data in format expected by strategy (list of candles)
        """
        if not self.is_initialized:
            logger.error("❌ Buffer not initialized - cannot provide data")
            return None
        
        with self.buffer_lock:
            if len(self.candle_buffer) < 50:
                logger.error(f"❌ Insufficient buffer data: {len(self.candle_buffer)} candles (need 50+)")
                return None
            
            # Return copy of buffer data as list
            buffer_data = list(self.candle_buffer)
            
        logger.info(f"✅ Providing buffer data: {len(buffer_data)} candles for signal generation")
        return buffer_data
    
    def get_buffer_stats(self):
        """Get buffer statistics for monitoring"""
        with self.buffer_lock:
            return {
                'size': len(self.candle_buffer),
                'initialized': self.is_initialized,
                'total_updates': self.total_updates,
                'signals_generated': self.signals_generated,
                'last_candle_time': self.last_candle_time
            }
    
    def is_ready_for_signals(self):
        """Check if buffer has sufficient data for signal generation"""
        return (self.is_initialized and 
                len(self.candle_buffer) >= 50)


def recover_rolling_buffer():
    """
    Recover rolling buffer after WebSocket disconnection
    Re-initializes buffer with fresh API data to ensure continuity
    """
    global rolling_buffer, binance_client
    
    if not rolling_buffer or not binance_client:
        logger.error("❌ Cannot recover buffer - components not initialized")
        return False
    
    logger.warning("🔄 Recovering rolling buffer after WebSocket issue...")
    
    try:
        # Re-initialize buffer with fresh data
        if rolling_buffer.initialize_from_api(binance_client):
            logger.info("✅ Rolling buffer recovered successfully")
            return True
        else:
            logger.error("❌ Failed to recover rolling buffer")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during buffer recovery: {e}")
        return False


def monitor_buffer_health():
    """
    Monitor rolling buffer health and performance
    Called periodically to ensure buffer is working correctly
    """
    global rolling_buffer
    
    if not rolling_buffer:
        logger.warning("⚠️ Rolling buffer not initialized")
        return False
    
    try:
        stats = rolling_buffer.get_buffer_stats()
        
        logger.info(f"📊 Buffer Health Check:")
        logger.info(f"   Size: {stats['size']}/{rolling_buffer.buffer_size}")
        logger.info(f"   Initialized: {stats['initialized']}")
        logger.info(f"   Total Updates: {stats['total_updates']}")
        logger.info(f"   Signals Generated: {stats['signals_generated']}")
        logger.info(f"   Ready for Signals: {rolling_buffer.is_ready_for_signals()}")
        
        if stats['last_candle_time']:
            last_update = datetime.fromtimestamp(stats['last_candle_time']/1000)
            age = datetime.now() - last_update
            logger.info(f"   Last Update: {last_update} ({age.total_seconds():.0f}s ago)")
            
            # Check if buffer data is getting stale (more than 10 minutes for higher timeframes)
            # Adjust staleness threshold based on timeframe
            timeframe_minutes = {
                '1m': 2, '3m': 5, '5m': 8, '15m': 20, '30m': 35,
                '1h': 65, '2h': 125, '4h': 245, '6h': 365, '8h': 485,
                '12h': 725, '1d': 1445, '3d': 4325, '1w': 10085, '1M': 43205
            }
            
            max_age = timeframe_minutes.get(TIMEFRAME, 20) * 60  # Convert to seconds
            
            if age.total_seconds() > max_age:
                logger.warning(f"⚠️ Buffer data is stale: {age.total_seconds():.0f}s old "
                              f"(max expected: {max_age}s for {TIMEFRAME})")
                return False
        
        return stats['initialized'] and stats['size'] >= 50
        
    except Exception as e:
        logger.error(f"❌ Error checking buffer health: {e}")
        return False


def test_strategies():
    """Test loading the strategies"""
    from modules.strategies import get_strategy
    
    # Test RAYSOL strategy
    raysol_strategy = get_strategy('SmartTrendCatcher')
    logger.info(f"Loaded strategy: {raysol_strategy.strategy_name}")
    
    return raysol_strategy

# Global variables
running = True
binance_client = None
risk_manager = None
strategy = None
websocket_manager = None
rolling_buffer = None  # WebSocket-only rolling buffer for signals
klines_data = {}
new_candle_received = {}
stats = {
    'total_trades': 0,
    'winning_trades': 0,
    'losing_trades': 0,
    'total_profit': 0,
    'start_balance': 0,
    'current_balance': 0,
    'daily_profit': 0,
    'last_trade_time': None,
    'last_report_time': None
}

class TelegramNotifier:
    def __init__(self):
        self.enabled = USE_TELEGRAM and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
        
    def send_message(self, message):
        """Send message to Telegram"""
        if not self.enabled:
            return
            
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            
            # Escape Markdown special characters to avoid parsing errors
            # Replace characters that could cause Markdown parsing errors
            for char in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
                message = message.replace(char, '\\' + char)
                
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "MarkdownV2"
            }
            response = requests.post(url, json=payload)
            if response.status_code != 200:
                logger.error(f"Failed to send Telegram message: {response.text}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False
            
    def send_photo(self, photo_path, caption=None):
        """Send photo to Telegram with optional caption"""
        if not self.enabled:
            return
            
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            
            files = {'photo': open(photo_path, 'rb')}
            data = {"chat_id": TELEGRAM_CHAT_ID}
            
            if caption:
                # Escape Markdown special characters
                for char in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
                    caption = caption.replace(char, '\\' + char)
                data["caption"] = caption
                data["parse_mode"] = "MarkdownV2"
                
            response = requests.post(url, files=files, data=data)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram photo: {e}")
            return False
            
    def send_plain_message(self, message):
        """Send message to Telegram without markdown parsing"""
        if not self.enabled:
            return
            
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message
            }
            response = requests.post(url, json=payload)
            if response.status_code != 200:
                logger.error(f"Failed to send Telegram message: {response.text}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False


def setup():
    """Initialize the trading bot"""
    global binance_client, risk_manager, strategy, stats, websocket_manager, rolling_buffer
    
    logger.info("Setting up trading bot...")
    
    # Create necessary directories
    state_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state')
    os.makedirs(state_dir, exist_ok=True)
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Initialize Binance client
    try:
        binance_client = BinanceClient()
        logger.info("Binance client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Binance client: {e}")
        exit(1)
    
    # Initialize risk manager with current account balance
    risk_manager = RiskManager(binance_client)
    
    # Update initial balance for compounding
    if AUTO_COMPOUND:
        risk_manager.update_balance_for_compounding()
    
    # Get the selected trading strategy
    strategy = get_strategy(STRATEGY)
    logger.info(f"Using trading strategy: {strategy.strategy_name}")
    
    # Connect the risk manager to the strategy for adaptive risk management
    strategy.set_risk_manager(risk_manager)
    logger.info(f"Connected risk manager to strategy for adaptive risk management")
    
    # Initialize futures settings for the trading symbol
    try:
        binance_client.initialize_futures(TRADING_SYMBOL)
    except Exception as e:
        logger.error(f"Failed to initialize futures: {e}")
        exit(1)
    
    # Initialize WebSocket manager
    websocket_manager = BinanceWebSocketManager()
    websocket_manager.add_symbol(TRADING_SYMBOL)
    
    # Register WebSocket callbacks
    websocket_manager.register_callback('kline', on_kline_closed)
    websocket_manager.register_callback('kline_update', on_kline_update)
    websocket_manager.register_callback('book_ticker', on_book_ticker)
    websocket_manager.register_callback('account_update', on_account_update)
    websocket_manager.register_callback('order_update', on_order_update)
    websocket_manager.register_callback('trade', on_trade)
    
    # Start WebSocket connections
    websocket_manager.start()
    logger.info("WebSocket connections started")
    
    # 🚀 Initialize WebSocket-only rolling buffer for signal generation
    logger.info("🔄 Initializing WebSocket-only rolling buffer...")
    rolling_buffer = RollingCandleBuffer(TRADING_SYMBOL, TIMEFRAME, buffer_size=60)
    
    # Seed buffer with ONE-TIME REST API call
    if not rolling_buffer.initialize_from_api(binance_client):
        logger.error("❌ Failed to initialize rolling buffer - cannot continue")
        exit(1)
    
    logger.info("✅ WebSocket-only buffer initialized successfully")
    logger.info("🔥 Bot ready for real-time signal generation using WebSocket data only!")
    
    # Initialize klines_data with initial data from REST API (legacy support)
    initialize_klines_data()
    
    # Record starting balance - try multiple times if needed
    for attempt in range(3):
        try:
            account_balance = binance_client.get_account_balance()
            if account_balance > 0:
                stats['start_balance'] = account_balance
                stats['current_balance'] = account_balance
                logger.info(f"Starting balance: {stats['start_balance']} USDT")
                break
            else:
                logger.warning(f"Got zero balance on attempt {attempt+1}, retrying...")
                time.sleep(2)
        except Exception as e:
            logger.error(f"Error getting balance on attempt {attempt+1}: {e}")
            time.sleep(2)
            
    # If we still have zero balance, try loading from config
    if stats['start_balance'] <= 0:
        # Use the initial balance from config as fallback
        try:
            from modules.config import INITIAL_BALANCE
            stats['start_balance'] = INITIAL_BALANCE
            stats['current_balance'] = INITIAL_BALANCE
            logger.warning(f"Failed to get balance from API, using configured balance: {INITIAL_BALANCE} USDT")
        except ImportError:
            # Default if INITIAL_BALANCE not in config
            stats['start_balance'] = 50.0
            stats['current_balance'] = 50.0
            logger.warning("Failed to get balance from API and no INITIAL_BALANCE in config. Using default: 50.0 USDT")
    
    # Make sure total_profit is reset properly
    stats['total_profit'] = 0.0
    stats['daily_profit'] = 0.0
    stats['last_report_time'] = datetime.now()
    
    # Set new_candle_received flag to False initially
    new_candle_received[TRADING_SYMBOL] = False
    
    # Send startup notification
    notifier = TelegramNotifier()
    notifier.send_message(f"🤖 *Trading Bot Started*\n"
                         f"Symbol: {TRADING_SYMBOL}\n"
                         f"Strategy: {strategy.strategy_name}\n"
                         f"Timeframe: {TIMEFRAME}\n"
                         f"Starting Balance: {stats['start_balance']} USDT\n"
                         f"Auto-Compound: {'Enabled' if AUTO_COMPOUND else 'Disabled'}\n"
                         f"Adaptive Risk Management: Enabled\n"
                         f"WebSocket-Only Signal Generation: Enabled ✅")


def initialize_state_file(force=False):
    """Initialize or repair the state file with proper balance values"""
    global stats
    
    state_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state')
    os.makedirs(state_dir, exist_ok=True)
    
    state_file = os.path.join(state_dir, 'trading_state.json')
    
    # Only initialize if the file doesn't exist or force=True
    if not os.path.exists(state_file) or force:
        logger.info("Initializing state file with current settings")
        
        # Try to get current balance from API
        current_balance = 0.0
        if binance_client:
            try:
                current_balance = binance_client.get_account_balance()
                logger.info(f"Got balance from API: {current_balance} USDT")
            except Exception as e:
                logger.error(f"Could not get balance from API: {e}")
        
        # If API balance failed, use the INITIAL_BALANCE from config
        if current_balance <= 0:
            try:
                from modules.config import INITIAL_BALANCE
                current_balance = INITIAL_BALANCE
                logger.info(f"Using configured initial balance: {INITIAL_BALANCE} USDT")
            except ImportError:
                current_balance = 50.0  # Default fallback
                logger.info(f"Using default initial balance: 50.0 USDT")
        
        # Update stats
        stats['start_balance'] = current_balance
        stats['current_balance'] = current_balance
        stats['total_profit'] = 0.0
        stats['daily_profit'] = 0.0
        stats['last_report_time'] = datetime.now()
        
        # Save to file - convert all datetime objects to ISO format
        json_stats = stats.copy()
        
        # Convert all datetime objects to ISO format string
        for key, value in json_stats.items():
            if isinstance(value, datetime):
                json_stats[key] = value.isoformat()
        
        try:
            with open(state_file, 'w') as f:
                json.dump(json_stats, f)
                
            logger.info(f"State file initialized with balance: {current_balance} USDT")
            return True
        except Exception as e:
            logger.error(f"Error saving state file: {e}")
            return False
    
    return False


def initialize_klines_data(timeframe=None):
    """Initialize minimal data for WebSocket startup - NOT for trading decisions"""
    global klines_data
    
    # Use provided timeframe or fall back to config default
    tf = timeframe or TIMEFRAME
    
    try:
        # 🚨 LIVE TRADING: Only get minimal data for WebSocket initialization
        # Trading decisions will ALWAYS use fresh API calls
        logger.info("📡 Getting minimal initialization data (NOT for trading decisions)")
        
        # Calculate required time based on timeframe to get at least 50 candles
        timeframe_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
            '12h': 720, '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        
        minutes_per_candle = timeframe_minutes.get(tf, 15)  # Default to 15m
        required_minutes = minutes_per_candle * 50  # Get 50 candles worth of data
        required_hours = max(1, required_minutes // 60)  # At least 1 hour
        
        start_str = f"{required_hours} hours ago"
        logger.info(f"📡 Getting minimal {tf} initialization data from {start_str}")
        
        klines = binance_client.get_historical_klines(
            symbol=TRADING_SYMBOL,
            interval=tf,
            start_str=start_str,
            limit=50  # Minimal amount
        )
        
        if not klines or len(klines) < 10:
            logger.warning(f"Minimal initialization data obtained (got {len(klines) if klines else 0} candles)")
            klines_data[TRADING_SYMBOL] = []  # Start with empty if needed
            return
            
        klines_data[TRADING_SYMBOL] = klines
        logger.info(f"🔧 Minimal initialization complete: {len(klines)} candles (will be replaced with fresh data)")
        logger.info("⚠️ NOTE: This data is ONLY for initialization - trading uses fresh API calls")
    except Exception as e:
        logger.error(f"Error initializing klines data: {e}")


def on_kline_closed(symbol, kline_data):
    """Callback for when a kline (candlestick) closes - WebSocket-only buffer update"""
    global klines_data, new_candle_received, rolling_buffer
    
    logger.info(f"🕒 Kline closed for {symbol}: {datetime.fromtimestamp(kline_data['close_time']/1000)}")
    
    try:
        # Debug: Log the complete kline data structure
        logger.info(f"🔍 Complete kline data: {kline_data}")
        logger.info(f"🔍 Kline data keys: {list(kline_data.keys())}")
        logger.info(f"🔍 Is closed flag: {kline_data.get('is_closed', 'Missing')}")
        
        # 🚀 WEBSOCKET-ONLY TRADING: Update rolling buffer with new candle
        # No more REST API calls for signal generation!
        
        if not rolling_buffer:
            logger.error("❌ Rolling buffer is None - cannot process candle")
            return
            
        if not rolling_buffer.is_initialized:
            logger.error("❌ Rolling buffer not initialized - cannot process candle")
            return
        
        # Update the rolling buffer with new WebSocket candle
        logger.info(f"🔄 Attempting to update rolling buffer with closed candle...")
        buffer_updated = rolling_buffer.update_from_websocket(kline_data)
        
        if buffer_updated:
            logger.info(f"✅ Rolling buffer updated with closed candle")
            logger.info(f"📊 Candle: O={kline_data['open']:.6f} H={kline_data['high']:.6f} "
                      f"L={kline_data['low']:.6f} C={kline_data['close']:.6f}")
            
            # Trigger signal generation using buffer data
            new_candle_received[symbol] = True
            check_for_signals(symbol)
        else:
            logger.warning("⚠️ Failed to update rolling buffer")
            
            # Additional debugging - check buffer state
            if rolling_buffer:
                stats = rolling_buffer.get_buffer_stats()
                logger.info(f"🔍 Buffer stats: {stats}")
                
                # Check what the last candle time comparison looks like
                logger.info(f"🔍 Buffer last_candle_time: {rolling_buffer.last_candle_time}")
                logger.info(f"🔍 Incoming close_time: {kline_data.get('close_time', 'Missing')}")
                
                # Try to recover if buffer seems corrupted
                if stats['size'] == 0:
                    logger.warning("🔧 Buffer is empty - attempting recovery...")
                    recover_rolling_buffer()
    
    except Exception as e:
        logger.error(f"❌ Error processing kline close: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Try to recover buffer on any error
        logger.warning("🔧 Attempting buffer recovery due to error...")
        recover_rolling_buffer()


def on_kline_update(symbol, kline_data):
    """Callback for real-time kline updates (includes unclosed candles)"""
    # Add throttling to reduce log spam
    if not hasattr(on_kline_update, 'last_log_time'):
        on_kline_update.last_log_time = {}
    
    current_time = time.time()
    
    # Only log price updates every 5 seconds to reduce spam
    if symbol not in on_kline_update.last_log_time or \
       current_time - on_kline_update.last_log_time[symbol] >= 5:
        
        # Format for easy reading - real-time price updates with colored output
        price = kline_data['close']
        formatted_time = datetime.fromtimestamp(kline_data['close_time']/1000).strftime('%H:%M:%S')
        
        # Calculate price change percentage from open
        price_change = ((kline_data['close'] - kline_data['open']) / kline_data['open']) * 100
        direction = "▲" if price_change >= 0 else "▼"
        
        # Log real-time price updates (throttled)
        logger.info(f"📊 {symbol} | Price: {price:.2f} | {direction} {abs(price_change):.2f}% | O: {kline_data['open']:.2f} H: {kline_data['high']:.2f} L: {kline_data['low']:.2f} | Vol: {kline_data['volume']:.2f} | {formatted_time}")
        
        on_kline_update.last_log_time[symbol] = current_time
    
    # Store data for later use
    global klines_data
    if symbol in klines_data and klines_data[symbol]:
        # Update the latest candle with real-time data until it's closed
        if not kline_data['is_closed'] and len(klines_data[symbol]) > 0:
            # CRITICAL FIX: Keep numeric values consistent with REST API data format
            candle = [
                kline_data['open_time'],
                kline_data['open'],      # Keep as numeric, not string
                kline_data['high'],      # Keep as numeric, not string
                kline_data['low'],       # Keep as numeric, not string
                kline_data['close'],     # Keep as numeric, not string
                kline_data['volume'],    # Keep as numeric, not string
                kline_data['close_time'],
                0,    # Placeholder - keep as numeric
                0,    # Placeholder - keep as numeric
                0,    # Placeholder - keep as numeric
                0,    # Placeholder - keep as numeric
                0     # Placeholder - keep as numeric
            ]
            klines_data[symbol][-1] = candle


def on_book_ticker(symbol, ticker_data):
    """Callback for book ticker updates (best bid/ask)"""
    # Only log significant changes to avoid flooding
    # Store last values to track changes
    if not hasattr(on_book_ticker, 'last_values'):
        on_book_ticker.last_values = {}
    
    # Store last values per symbol
    last = on_book_ticker.last_values.get(symbol, {})
    bid_price = ticker_data['bid_price']
    ask_price = ticker_data['ask_price']
    
    # Only log if price changed significantly (0.05% or more)
    significant_change = False
    if symbol not in on_book_ticker.last_values:
        significant_change = True
    else:
        last_bid = last.get('bid_price', 0)
        last_ask = last.get('ask_price', 0)
        
        if last_bid > 0 and last_ask > 0:
            bid_change_pct = abs((bid_price - last_bid) / last_bid * 100)
            ask_change_pct = abs((ask_price - last_ask) / last_ask * 100)
            if bid_change_pct >= 0.05 or ask_change_pct >= 0.05:
                significant_change = True
    
    # Update last values
    on_book_ticker.last_values[symbol] = ticker_data
    
    if significant_change:
        spread = ask_price - bid_price
        spread_pct = (spread / ask_price) * 100
        
        logger.info(f"💹 {symbol} | Bid: {bid_price:.2f} ({ticker_data['bid_qty']:.4f}) | Ask: {ask_price:.2f} ({ticker_data['ask_qty']:.4f}) | Spread: {spread:.2f} ({spread_pct:.2f}%)")


def on_trade(symbol, trade_data):
    """Callback for real-time trades"""
    # Register this callback in main setup function
    price = trade_data['price']
    qty = trade_data['quantity']
    value = price * qty
    side = "BUY" if trade_data['buyer_maker'] else "SELL"
    
    # Only log trades above a certain value to avoid flooding
    if value > 10000:  # Only log significant trades (>$10,000)
        trade_time = datetime.fromtimestamp(trade_data['time']/1000).strftime('%H:%M:%S')
        logger.info(f"💰 Large {side} Trade | {symbol} | Price: {price:.2f} | Qty: {qty:.4f} | Value: ${value:.2f} | {trade_time}")


def on_account_update(balance_updates, position_updates):
    """Callback for account updates"""
    global stats
    
    # Add deduplication to prevent duplicate processing
    if not hasattr(on_account_update, 'last_update'):
        on_account_update.last_update = {'balance': 0, 'time': 0}
    
    current_time = time.time()
    
    logger.info(f"Account update received: {len(balance_updates)} balances, {len(position_updates)} positions")
    
    try:
        if 'USDT' in balance_updates:
            new_balance = balance_updates['USDT']
            
            # Only process if balance actually changed or it's been more than 5 seconds
            balance_changed = abs(new_balance - on_account_update.last_update['balance']) > 0.000001
            time_elapsed = current_time - on_account_update.last_update['time'] > 5
            
            if balance_changed or time_elapsed:
                if 'current_balance' in stats and stats['current_balance'] > 0:
                    balance_change = new_balance - stats['current_balance']
                    # Only add to daily profit if we're processing a realized profit/loss from a trade
                    # This helps avoid counting deposits/withdrawals as profit/loss
                    if len(position_updates) > 0 and balance_changed:
                        stats['daily_profit'] += balance_change
                        logger.info(f"Balance change from trade: {balance_change:.6f} USDT, Daily P/L: {stats['daily_profit']:.6f} USDT")
                    elif balance_changed:
                        logger.info(f"Balance change (non-trade): {balance_change:.6f} USDT")
                
                stats['current_balance'] = new_balance
                logger.info(f"Balance updated: {new_balance:.6f} USDT")
                
                # Update risk manager with new balance
                if risk_manager:
                    risk_manager.update_balance_for_compounding()
                
                # Update tracking
                on_account_update.last_update['balance'] = new_balance
                on_account_update.last_update['time'] = current_time
            else:
                logger.debug(f"Skipping duplicate balance update: {new_balance:.6f} USDT")
            
        # Log position updates with more precision
        for symbol, position in position_updates.items():
            position_amount = position['position_amount']
            entry_price = position['entry_price']
            unrealized_pnl = position['unrealized_pnl']
            logger.info(f"Position update for {symbol}: {position_amount} @ {entry_price} (Unrealized P/L: {unrealized_pnl:.6f} USDT)")
            
    except Exception as e:
        logger.error(f"Error processing account update: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def on_order_update(order_data):
    """Callback for order updates"""
    global stats
    
    # Add order deduplication to prevent duplicate processing
    if not hasattr(on_order_update, 'processed_orders'):
        on_order_update.processed_orders = set()
    
    try:
        symbol = order_data['symbol']
        status = order_data['order_status']
        side = order_data['side']
        order_type = order_data['type']
        filled_qty = order_data['filled_quantity']
        price = order_data['last_filled_price']
        order_id = order_data.get('order_id', 'unknown')
        
        # Create unique order key to prevent duplicate processing
        order_key = f"{order_id}_{status}_{filled_qty}_{price}"
        
        # Skip if we've already processed this exact order update
        if order_key in on_order_update.processed_orders:
            logger.debug(f"Skipping duplicate order update: {order_key}")
            return
        
        # Add to processed orders set
        on_order_update.processed_orders.add(order_key)
        
        # Clean up old processed orders to prevent memory buildup (keep last 1000)
        if len(on_order_update.processed_orders) > 1000:
            # Remove oldest half
            old_orders = list(on_order_update.processed_orders)[:500]
            for old_order in old_orders:
                on_order_update.processed_orders.discard(old_order)
        
        # Create a more visually informative log message
        if status == 'FILLED':
            # Highlight completed orders with visual indicators
            if order_type == 'MARKET':
                if side == 'BUY':
                    logger.info(f"🟢🟢🟢 EXECUTED {side} ORDER: {filled_qty} {symbol} @ {price} 🟢🟢🟢")
                elif side == 'SELL':
                    logger.info(f"🔴🔴🔴 EXECUTED {side} ORDER: {filled_qty} {symbol} @ {price} 🔴🔴🔴")
                else:
                    logger.info(f"⚫⚫⚫ EXECUTED {side} ORDER: {filled_qty} {symbol} @ {price} ⚫⚫⚫")
            elif order_type in ['STOP_MARKET']:
                logger.info(f"⚠️⚠️⚠️ EXECUTED {order_type} {side} ORDER: {filled_qty} {symbol} @ {price} ⚠️⚠️⚠️")
                
                # Check if we need to cancel any remaining orders after a stop loss is hit
                try:
                    position = binance_client.get_position_info(symbol)
                    # If position was closed or is very small (dust)
                    if not position or abs(position['position_amount']) < 0.000001:
                        logger.info(f"Position closed via {order_type}, canceling any remaining orders")
                        cancelled = binance_client.cancel_all_open_orders(symbol)
                        logger.info(f"Cancelled {cancelled} remaining orders for {symbol}")
                except Exception as e:
                    logger.error(f"Error checking position after {order_type}: {e}")
            else:
                logger.info(f"✅✅✅ EXECUTED {order_type} {side} ORDER: {filled_qty} {symbol} @ {price} ✅✅✅")
                
            # For filled orders, update trade statistics
            if order_type == 'MARKET':
                stats['total_trades'] += 1
                stats['last_trade_time'] = datetime.now()
                
                realized_profit = order_data['realized_profit']
                if realized_profit > 0:
                    stats['winning_trades'] += 1
                    logger.info(f"💲💲💲 PROFIT: +{realized_profit:.2f} USDT 💲💲💲")
                elif realized_profit < 0:
                    stats['losing_trades'] += 1
                    logger.info(f"💸💸💸 LOSS: {realized_profit:.2f} USDT 💸💸💸")
                
                # Get additional trade context
                signal_reason = "Unknown"
                    
                # Try to get strategy name
                strategy_name = STRATEGY
                if strategy and hasattr(strategy, 'strategy_name'):
                    strategy_name = strategy.strategy_name
                    
                # Store extended trade data
                trade_data = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': filled_qty, 
                    'price': price,
                    'realized_profit': realized_profit,
                    'commission': order_data['commission'],
                    'commission_asset': order_data['commission_asset'],
                    'timestamp': datetime.now().isoformat(),
                    'balance': stats['current_balance'],
                    'strategy': strategy_name,
                    'execution_type': order_type
                }
                
                save_trade(trade_data)
                
                # Show current account balance after trade
                current_balance = stats['current_balance']
                try:
                    current_balance = binance_client.get_account_balance()
                    if current_balance > 0:
                        # Calculate trade P&L by comparing balances (more accurate than just using realized_profit)
                        if stats['current_balance'] > 0:
                            trade_pnl = current_balance - stats['current_balance']
                            # Update total profit stats based on actual balance change
                            stats['total_profit'] = stats.get('total_profit', 0) + trade_pnl
                            logger.info(f"Trade P&L based on balance change: {trade_pnl:.6f} USDT")
                        
                        stats['current_balance'] = current_balance
                        logger.info(f"💰 ACCOUNT BALANCE: {current_balance:.6f} USDT 💰")
                except Exception as e:
                    logger.error(f"Failed to get account balance after trade: {e}")
                
                # Send detailed notification with trade info and account status (with deduplication)
                # Create notification key to prevent duplicate messages
                notification_key = f"{order_id}_{status}_{side}_{price}_{realized_profit}"
                
                if not hasattr(on_order_update, 'sent_notifications'):
                    on_order_update.sent_notifications = set()
                
                if notification_key not in on_order_update.sent_notifications:
                    on_order_update.sent_notifications.add(notification_key)
                    
                    # Clean up old notifications to prevent memory buildup
                    if len(on_order_update.sent_notifications) > 500:
                        old_notifications = list(on_order_update.sent_notifications)[:250]
                        for old_notif in old_notifications:
                            on_order_update.sent_notifications.discard(old_notif)
                    
                    notifier = TelegramNotifier()
                
                    if side == 'BUY':
                        message = f"🟢 *Position Opened*\n"
                    else:
                        message = f"🔴 *Position Opened*\n"
                        
                    # Basic trade info
                    message += f"Symbol: {symbol}\n" \
                              f"Side: {side}\n" \
                              f"Quantity: {filled_qty}\n" \
                              f"Price: {price}\n"
                    
                    # Add strategy info
                    message += f"Strategy: {strategy_name}\n"
                              
                    # Add profit/loss info if applicable
                    if realized_profit > 0:
                        message += f"Realized Profit: +{realized_profit:.2f} USDT 🎯\n"
                    elif realized_profit < 0:
                        message += f"Realized Loss: {realized_profit:.2f} USDT 📉\n"
                    
                    # Add current account info
                    message += f"\n*Account Status:*\n" \
                              f"Current Balance: {current_balance:.2f} USDT\n" \
                              f"Total Trades: {stats['total_trades']}\n" \
                              f"Win Rate: {(stats['winning_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0:.1f}%\n"
                    
                    # Add API & WebSocket status
                    ws_status = "Connected" if websocket_manager and websocket_manager.is_connected() else "Disconnected"
                    message += f"\n*Connection Status:*\n" \
                              f"WebSocket: {ws_status}"
                    
                    notifier.send_message(message)
                    
                    # For significant trades (profit/loss over certain threshold), send a chart
                    try:
                        significant_threshold = current_balance * 0.02  # 2% of balance
                        if abs(realized_profit) >= significant_threshold:
                            # Generate and send a trade chart for significant trades
                            chart_path = generate_trade_chart(symbol, side, price, realized_profit)
                            if chart_path:
                                caption = f"Trade Chart: {symbol} {side} @ {price}"
                                notifier.send_photo(chart_path, caption)
                    except Exception as chart_error:
                        logger.error(f"Error generating trade chart: {chart_error}")
                else:
                    logger.debug(f"Skipping duplicate notification for order: {notification_key}")
                    
            elif order_type in ['STOP_MARKET']:
                # For stop loss orders only (take profit functionality removed)
                event_type = "Stop Loss"
                
                # Calculate profit/loss if available
                profit_loss_msg = ""
                if 'realized_profit' in order_data and order_data['realized_profit'] != 0:
                    profit = order_data['realized_profit']
                    profit_loss_msg = f"P/L: {'+' if profit > 0 else ''}{profit:.2f} USDT"
                
                notifier = TelegramNotifier()
                message = f"{'⚠️' if order_type == 'STOP_MARKET' else '🎯'} *{event_type} Triggered*\n\n" \
                         f"Symbol: {symbol}\n" \
                         f"Price: {price}\n" \
                         f"Size: {filled_qty}\n" \
                         f"{profit_loss_msg}"
                
                notifier.send_message(message)
        else:
            # For other order statuses, just log basic info
            logger.info(f"Order Update: {symbol} {side} {order_type} {status} - Qty: {filled_qty}, Price: {price}")
                
    except Exception as e:
        logger.error(f"Error processing order update: {e}")
        import traceback
        logger.error(traceback.format_exc())


def check_for_signals(symbol=None):
    """Check for trading signals using WebSocket-only rolling buffer"""
    global klines_data, new_candle_received, rolling_buffer
    
    if not symbol:
        symbol = TRADING_SYMBOL
    
    # In multi-instance mode, ensure we only process signals for our designated symbol
    if MULTI_INSTANCE_MODE and symbol != TRADING_SYMBOL:
        logger.warning(f"Ignoring signal check for {symbol} - this instance is dedicated to {TRADING_SYMBOL}")
        return
    
    if not new_candle_received.get(symbol, False):
        return
    
    new_candle_received[symbol] = False
    
    logger.info(f"🎯 Checking for trading signals for {symbol} using WebSocket buffer")
    
    try:
        # � WEBSOCKET-ONLY SIGNAL GENERATION
        # Use rolling buffer data - NO REST API calls!
        
        if not rolling_buffer or not rolling_buffer.is_ready_for_signals():
            logger.error("❌ Rolling buffer not ready for signal generation")
            return
        
        # Get candle data from rolling buffer (WebSocket-only)
        klines = rolling_buffer.get_buffer_data()
        
        if not klines or len(klines) < 50:
            logger.error(f"❌ Insufficient WebSocket buffer data: {len(klines) if klines else 0} candles")
            return
        
        logger.info(f"✅ Using WebSocket buffer data: {len(klines)} candles (NO REST API)")
        
        # Get current price from WebSocket
        current_price = websocket_manager.get_last_kline(symbol).get('close', None)
        if not current_price:
            logger.error("❌ Failed to get current price from WebSocket")
            return
        
        logger.info(f"💰 Current price for {symbol}: {current_price}")
            
        # Get position info - REST API only for position management (not signals)
        position = binance_client.get_position_info(symbol)
        position_amount = position['position_amount'] if position else 0
        
        logger.info(f"📊 Current position amount: {position_amount}")
        
        # WebSocket buffer data verification 
        logger.info("🔍 === WEBSOCKET BUFFER DATA VERIFICATION ===")
        
        if klines and len(klines) > 0:
            latest_candle = klines[-1]
            if len(latest_candle) >= 5:
                buffer_close = float(latest_candle[4])
                logger.info(f"📊 Buffer latest close: {buffer_close:.6f}")
                logger.info(f"🌐 WebSocket current price: {current_price}")
                
                # Show recent buffer trend (last 3 candles)
                logger.info("📈 WebSocket buffer trend (last 3 candles):")
                for i in range(max(-3, -len(klines)), 0):
                    candle = klines[i]
                    if len(candle) >= 5:
                        try:
                            o_price = float(candle[1])
                            h_price = float(candle[2])
                            l_price = float(candle[3])
                            c_price = float(candle[4])
                            candle_time = datetime.fromtimestamp(candle[6]/1000).strftime('%H:%M:%S')
                            logger.info(f"   Buffer candle {i}: O={o_price:.6f} H={h_price:.6f} "
                                      f"L={l_price:.6f} C={c_price:.6f} @{candle_time}")
                        except (ValueError, TypeError) as e:
                            logger.error(f"   Buffer candle {i}: Data error - {e}")
        
        # Generate signal using WebSocket buffer data ONLY
        logger.info("🎯 Generating signal from WebSocket buffer (zero REST API calls)...")
        signal = strategy.get_signal(klines)
        
        # Update buffer stats
        rolling_buffer.signals_generated += 1
        
        logger.info(f"🎯 === WEBSOCKET-ONLY SIGNAL: {signal} ===")
        
        # Verify signal source and log appropriately
        if signal in ['BUY', 'SELL']:
            logger.info(f"✅ {signal} signal generated from WebSocket buffer (real-time)")
            logger.info(f"📊 Buffer stats: {rolling_buffer.get_buffer_stats()}")
        elif signal == 'HOLD':
            logger.info("⚪ HOLD signal - waiting for clear market direction")
        else:
            logger.warning(f"⚠️ Unexpected signal type: {signal}")
        
        # Execute trade based on signal (REST API only for order execution)
        if signal == 'BUY' and position_amount <= 0:
            logger.info("🟢 BUY signal detected with no current position")
            # Trade execution logic is handled below in the existing sophisticated system
        elif signal == 'SELL' and position_amount > 0:
            logger.info("🔴 SELL signal detected with existing position")
            # Trade execution logic is handled below in the existing sophisticated system
        elif signal in ['BUY', 'SELL']:
            logger.info(f"⚪ {signal} signal detected but position filter applied")
            logger.info(f"   Current position: {position_amount}")
        
    except Exception as e:
        logger.error(f"❌ Error checking signals: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return  # Early return to avoid executing trades if signal generation failed
        
    # Continue with existing sophisticated trade execution logic
    try:
        
        # Verify bot status before proceeding with trades
        if not binance_client:
            logger.error("Binance client not initialized. Cannot place trades.")
            return
        
        # Handle None signal - do nothing
        if signal is None:
            logger.info("No signal generated (None). Skipping trading action.")
            return
            
        # Handle HOLD signal - update trailing stop loss if enabled and position exists
        if signal == "HOLD":
            logger.info("HOLD signal received. No trading action taken.")
            if position and abs(position['position_amount']) > 0:
                position_side = "LONG" if position['position_amount'] > 0 else "SHORT"
                logger.info(f"Current position: {position_side} {abs(position['position_amount'])} {symbol} at {position.get('entry_price', 'unknown')} (unrealized PnL: {position.get('unrealized_pnl', 0):.4f})")
                
                # Update trailing stop loss on HOLD signal if enabled
                if UPDATE_TRAILING_ON_HOLD:
                    logger.info("Checking for trailing stop loss update on HOLD signal...")
                    
                    # Determine the correct side for stop loss calculation
                    stop_side = "BUY" if position['position_amount'] > 0 else "SELL"
                    
                    # Calculate new trailing stop loss
                    new_stop = risk_manager.adjust_stop_loss_for_trailing(
                        symbol, stop_side, current_price, position
                    )
                    
                    # Update stop loss if needed
                    if new_stop:
                        logger.info(f"Updating trailing stop loss on HOLD signal...")
                        # Cancel only stop loss orders (preserve take profit)
                        binance_client.cancel_stop_loss_orders_only(symbol)
                        time.sleep(0.5)  # Small delay to ensure orders are cancelled
                        
                        # Place new trailing stop loss
                        opposite_side = "SELL" if stop_side == "BUY" else "BUY"
                        sl_order = binance_client.place_stop_loss_order(
                            symbol, opposite_side, abs(position['position_amount']), new_stop
                        )
                        if sl_order:
                            logger.info(f"✅ Updated trailing stop loss to {new_stop} for {position_side} position on HOLD signal")
                        else:
                            logger.error(f"❌ Failed to update stop loss order at {new_stop}")
                    else:
                        logger.info("No trailing stop loss update needed at current price level")
                        
            else:
                logger.info("No open position. Waiting for clear BUY or SELL signal.")
            return
            
        # Process signals with normal logic (BUY signal creates LONG position, SELL signal creates SHORT position)
        if signal == "BUY":  # Process BUY signal as BUY order
            # If already in a LONG position, update trailing stop loss instead of ignoring
            if position_amount > 0:
                logger.info(f"Already in a LONG position ({position_amount}). Ignoring BUY signal. now update trailing stop loss")
                
                # Calculate trailing stop-loss for existing LONG position
                new_stop = risk_manager.adjust_stop_loss_for_trailing(
                    symbol, "BUY", current_price, position
                )
                
                # Check if stop loss needs updating
                if new_stop:
                    # Cancel only stop loss orders (preserve take profit)
                    binance_client.cancel_stop_loss_orders_only(symbol)
                    time.sleep(0.5)  # Small delay to ensure orders are cancelled
                    
                    # Place new trailing stop loss (only moves in favorable direction)
                    sl_order = binance_client.place_stop_loss_order(
                        symbol, "SELL", abs(position['position_amount']), new_stop
                    )
                    if sl_order:
                        logger.info(f"✅ Updated trailing stop loss to {new_stop} for LONG position")
                    else:
                        logger.error(f"❌ Failed to update stop loss order at {new_stop}")
                else:
                    logger.info("No trailing stop loss update needed for current price level")
                
                return
                
            # Handle SHORT → LONG transition
            elif position_amount < 0:
                logger.info(f"POSITION TRANSITION: Closing existing SHORT position ({position_amount}) before going LONG")
                
                # Cancel existing orders first
                cancelled = binance_client.cancel_position_orders(symbol)
                logger.info(f"Cancelled {cancelled} existing position orders")
                time.sleep(0.5)  # Small delay to ensure orders are cancelled
                
                # Close short position with market order
                close_amount = abs(position_amount)
                logger.info(f"Placing BUY order to close SHORT position: {close_amount} {symbol}")
                close_order = binance_client.place_market_order(symbol, "BUY", close_amount)
                
                if close_order:
                    order_id = close_order.get('orderId', 'unknown')
                    logger.info(f"✅ Successfully closed SHORT position with order ID: {order_id}")
                    
                    # Clear locked trailing stop since position is closed
                    risk_manager.clear_locked_trailing_stop(symbol)
                    
                    # Check if position was actually closed (sometimes there can be a delay)
                    time.sleep(1)  # Wait a moment for the order to process
                    position = binance_client.get_position_info(symbol)
                    if position and position['position_amount'] > 0.0001:  # Using a small threshold
                        logger.warning(f"⚠️ Position not fully closed. Remaining: {position['position_amount']}. Trying again...")
                        remaining = position['position_amount']
                        binance_client.place_market_order(symbol, "BUY", remaining)
                        time.sleep(1)  # Wait for the second attempt to process
                else:
                    logger.error("❌ Failed to close SHORT position! Cannot proceed with opening LONG position.")
                    return
            
            # Check if we should open a new position - at this point we know we don't have an existing LONG position
            logger.info(f"Opening new LONG position based on BUY signal (current position amount: {position_amount})")
            if risk_manager.should_open_position(symbol):
                stop_loss_price = risk_manager.calculate_stop_loss(symbol, "BUY", current_price)
                
                # Get current risk level based on market conditions
                current_risk = risk_manager.get_current_risk_level(symbol)
                logger.info(f"Opening BUY position with risk level: {current_risk:.4f} (position size multiplier: {risk_manager.position_size_multiplier:.2f})")
                
                quantity = risk_manager.calculate_position_size(
                    symbol, "BUY", current_price, stop_loss_price
                )
                logger.info(f"Calculated position size for BUY: {quantity} at price {current_price}")
                
                if quantity <= 0:
                    logger.warning(f"⚠️ Calculated quantity is too small or zero: {quantity}. Not placing BUY order.")
                    return
                
                # Place the market order to open long position
                logger.info(f"Placing LONG position: {quantity} {symbol} at ~{current_price}")
                order = binance_client.place_market_order(symbol, "BUY", quantity)
                
                if order:
                    order_id = order.get('orderId', 'unknown')
                    logger.info(f"✅ Successfully opened LONG position with order ID: {order_id}")
                    
                    # Try to verify the position was opened with retries
                    position_verified = False
                    new_position = None
                    entry_price = current_price  # Fallback to current price
                    position_amount = quantity  # Fallback to original quantity
                    
                    for attempt in range(3):
                        time.sleep(1 + attempt)  # Increasing wait time: 1s, 2s, 3s
                        try:
                            new_position = binance_client.get_position_info(symbol)
                            if new_position and new_position['position_amount'] > 0:
                                logger.info(f"Position verification successful. Amount: {new_position['position_amount']}")
                                entry_price = float(new_position.get('entry_price', current_price))
                                position_amount = new_position['position_amount']
                                position_verified = True
                                break
                            else:
                                logger.warning(f"Position verification attempt {attempt + 1}/3 failed. Retrying...")
                        except Exception as e:
                            logger.warning(f"Error during position verification attempt {attempt + 1}/3: {e}")
                    
                    if not position_verified:
                        logger.warning(f"⚠️ Position verification failed after 3 attempts. Using fallback values.")
                        logger.warning(f"Will attempt to place protective orders with fallback entry price: {entry_price}")
                    
                    # Always attempt to place protective orders, even if verification failed
                    try:
                        # Get recent klines for volatility calculation
                        recent_klines = klines_data.get(symbol, [])[-30:] if klines_data.get(symbol) else None
                        
                        # Place protective stop loss
                        stop_loss_price = risk_manager.calculate_volatility_based_stop_loss(
                            symbol, "BUY", entry_price, recent_klines
                        )
                        
                        if stop_loss_price:
                            sl_order = binance_client.place_stop_loss_order(
                                symbol, "SELL", position_amount, stop_loss_price
                            )
                            if sl_order:
                                logger.info(f"✅ Volatility-based stop loss placed at {stop_loss_price}")
                            else:
                                logger.error(f"❌ Failed to place stop loss at {stop_loss_price}")
                        
                        # Place take profit order if enabled
                        if USE_TAKE_PROFIT:
                            take_profit_price = risk_manager.calculate_take_profit(
                                symbol, "BUY", entry_price
                            )
                            
                            if take_profit_price:
                                tp_order = binance_client.place_take_profit_order(
                                    symbol, "SELL", position_amount, take_profit_price
                                )
                                if tp_order:
                                    logger.info(f"✅ Fixed take profit placed at {take_profit_price} ({TAKE_PROFIT_PCT*100:.1f}%)")
                                else:
                                    logger.error(f"❌ Failed to place take profit at {take_profit_price}")
                                    
                    except Exception as e:
                        logger.error(f"❌ Error placing protective orders: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                else:
                    logger.error(f"❌ Failed to place BUY order!")
                    
        elif signal == "SELL":  # Process SELL signal as SELL order
            # If already in a SHORT position, update trailing stop loss instead of ignoring
            if position_amount < 0:
                logger.info(f"Already in a SHORT position ({position_amount}). Ignoring SELL signal. now update trailing stop loss")
                
                # Calculate trailing stop-loss for existing SHORT position
                new_stop = risk_manager.adjust_stop_loss_for_trailing(
                    symbol, "SELL", current_price, position
                )
                
                # Check if stop loss needs updating
                if new_stop:
                    # Cancel only stop loss orders (preserve take profit)
                    binance_client.cancel_stop_loss_orders_only(symbol)
                    time.sleep(0.5)  # Small delay to ensure orders are cancelled
                    
                    # Place new trailing stop loss (only moves in favorable direction)
                    sl_order = binance_client.place_stop_loss_order(
                        symbol, "BUY", abs(position['position_amount']), new_stop
                    )
                    if sl_order:
                        logger.info(f"✅ Updated trailing stop loss to {new_stop} for SHORT position")
                    else:
                        logger.error(f"❌ Failed to update stop loss order at {new_stop}")
                else:
                    logger.info("No trailing stop loss update needed for current price level")
                
                return
                
            # Handle LONG → SHORT transition
            elif position_amount > 0:
                logger.info(f"POSITION TRANSITION: Closing existing LONG position ({position_amount}) before going SHORT")
                
                # Cancel existing orders first
                cancelled = binance_client.cancel_position_orders(symbol)
                logger.info(f"Cancelled {cancelled} existing position orders")
                time.sleep(0.5)  # Small delay to ensure orders are cancelled
                
                # Close long position with market order
                close_amount = position_amount
                logger.info(f"Placing SELL order to close LONG position: {close_amount} {symbol}")
                close_order = binance_client.place_market_order(symbol, "SELL", close_amount)
                
                if close_order:
                    order_id = close_order.get('orderId', 'unknown')
                    logger.info(f"✅ Successfully closed LONG position with order ID: {order_id}")
                    
                    # Clear locked trailing stop since position is closed
                    risk_manager.clear_locked_trailing_stop(symbol)
                    
                    # Check if position was actually closed (sometimes there can be a delay)
                    time.sleep(1)  # Wait a moment for the order to process
                    position = binance_client.get_position_info(symbol)
                    if position and position['position_amount'] > 0.0001:  # Using a small threshold
                        logger.warning(f"⚠️ Position not fully closed. Remaining: {position['position_amount']}. Trying again...")
                        remaining = position['position_amount']
                        binance_client.place_market_order(symbol, "SELL", remaining)
                        time.sleep(1)  # Wait for the second attempt to process
                else:
                    logger.error("❌ Failed to close LONG position! Cannot proceed with opening SHORT position.")
                    return
                
            # Check if we should open a new position - at this point we know we don't have an existing SHORT position
            logger.info(f"Opening new SHORT position based on SELL signal (current position amount: {position_amount})")  
            if risk_manager.should_open_position(symbol):
                stop_loss_price = risk_manager.calculate_stop_loss(symbol, "SELL", current_price)
                
                # Get current risk level based on market conditions
                current_risk = risk_manager.get_current_risk_level(symbol)
                logger.info(f"Opening SELL position with risk level: {current_risk:.4f} (position size multiplier: {risk_manager.position_size_multiplier:.2f})")
                
                quantity = risk_manager.calculate_position_size(
                    symbol, "SELL", current_price, stop_loss_price
                )
                
                logger.info(f"Calculated position size for SELL: {quantity} at price {current_price}")
                
                if quantity <= 0:
                    logger.warning(f"⚠️ Calculated quantity is too small or zero: {quantity}. Not placing SELL order.")
                    return
                
                # Place the market order to open short position
                logger.info(f"Placing SHORT position: {quantity} {symbol} at ~{current_price}")
                order = binance_client.place_market_order(symbol, "SELL", quantity)
                
                if order:
                    order_id = order.get('orderId', 'unknown')
                    logger.info(f"✅ Successfully opened SHORT position with order ID: {order_id}")
                    
                    # Try to verify the position was opened with retries
                    position_verified = False
                    new_position = None
                    entry_price = current_price  # Fallback to current price
                    position_amount = quantity  # Fallback to original quantity
                    
                    for attempt in range(3):
                        time.sleep(1 + attempt)  # Increasing wait time: 1s, 2s, 3s
                        try:
                            new_position = binance_client.get_position_info(symbol)
                            if new_position and new_position['position_amount'] < 0:
                                logger.info(f"Position verification successful. Amount: {new_position['position_amount']}")
                                entry_price = float(new_position.get('entry_price', current_price))
                                position_amount = abs(new_position['position_amount'])  # Use absolute value for order quantity
                                position_verified = True
                                break
                            else:
                                logger.warning(f"Position verification attempt {attempt + 1}/3 failed. Retrying...")
                        except Exception as e:
                            logger.warning(f"Error during position verification attempt {attempt + 1}/3: {e}")
                    
                    if not position_verified:
                        logger.warning(f"⚠️ Position verification failed after 3 attempts. Using fallback values.")
                        logger.warning(f"Will attempt to place protective orders with fallback entry price: {entry_price}")
                    
                    # Always attempt to place protective orders, even if verification failed
                    try:
                        # Get recent klines for volatility calculation
                        recent_klines = klines_data.get(symbol, [])[-30:] if klines_data.get(symbol) else None
                        
                        # Place protective stop loss
                        stop_loss_price = risk_manager.calculate_volatility_based_stop_loss(
                            symbol, "SELL", entry_price, recent_klines
                        )
                        
                        if stop_loss_price:
                            sl_order = binance_client.place_stop_loss_order(
                                symbol, "BUY", position_amount, stop_loss_price
                            )
                            if sl_order:
                                logger.info(f"✅ Volatility-based stop loss placed at {stop_loss_price}")
                            else:
                                logger.error(f"❌ Failed to place stop loss at {stop_loss_price}")
                        
                        # Place take profit order if enabled
                        if USE_TAKE_PROFIT:
                            take_profit_price = risk_manager.calculate_take_profit(
                                symbol, "SELL", entry_price
                            )
                            
                            if take_profit_price:
                                tp_order = binance_client.place_take_profit_order(
                                    symbol, "BUY", position_amount, take_profit_price
                                )
                                if tp_order:
                                    logger.info(f"✅ Fixed take profit placed at {take_profit_price} ({TAKE_PROFIT_PCT*100:.1f}%)")
                                else:
                                    logger.error(f"❌ Failed to place take profit at {take_profit_price}")
                                    
                    except Exception as e:
                        logger.error(f"❌ Error placing protective orders: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                else:
                    logger.error(f"❌ Failed to place SELL order!")
        
        # Handle trailing stops and take profits for existing positions - for current symbol only in multi-instance mode
        if position and abs(position['position_amount']) > 0 and position['symbol'] == symbol:
            position_side = "LONG" if position['position_amount'] > 0 else "SHORT"
            logger.info(f"Managing existing {position_side} position for {symbol} with size {abs(position['position_amount'])}, received {signal} signal")
            
            side = "BUY" if position['position_amount'] > 0 else "SELL"
            opposite_side = "SELL" if side == "BUY" else "BUY"
            
            # Calculate trailing stop-loss specifically for this symbol
            new_stop = risk_manager.adjust_stop_loss_for_trailing(
                symbol, side, current_price, position
            )
            
            # Check if stop loss needs updating (take profit functionality removed)
            if new_stop:
                # Only cancel stop loss orders for this specific symbol (preserve take profit)
                binance_client.cancel_stop_loss_orders_only(symbol)
                time.sleep(0.5)  # Small delay to ensure orders are cancelled
                
                # Place new stop loss (only moves in favorable direction)
                stop_loss_price = new_stop
                sl_order = binance_client.place_stop_loss_order(
                    symbol, opposite_side, abs(position['position_amount']), stop_loss_price
                )
                if sl_order:
                    logger.info(f"Updated trailing stop loss to {stop_loss_price} (moved in favorable direction)")
                else:
                    logger.error(f"Failed to update stop loss order at {stop_loss_price}")
    
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


def generate_performance_report():
    global stats
    
    logger.info("Generating performance report...")
    
    try:
        # Force refresh the current balance from the API
        if binance_client:
            current_balance = binance_client.get_account_balance()
            logger.info(f"Raw balance from API: {current_balance} USDT")
            
            # Only update if we got a valid balance (including very small amounts)
            if current_balance > 0:
                # Display with more precision for very small balances
                stats['current_balance'] = current_balance
                
                # If we didn't have a start balance before, set it now
                if stats['start_balance'] <= 0:
                    stats['start_balance'] = current_balance
                    logger.info(f"Setting initial balance to {current_balance} USDT")
        else:
            logger.error("Binance client not initialized, cannot get balance")
            current_balance = stats['current_balance']  # Use last known balance
            
    except Exception as e:
        logger.error(f"Error getting current balance: {e}")
        current_balance = stats['current_balance']  # Use last known balance
    
    # Calculate profit metrics with proper handling of small values
    profit_loss = current_balance - stats['start_balance']
    profit_pct = (profit_loss / stats['start_balance']) * 100 if stats['start_balance'] > 0 else 0
    
    # More accurate daily profit percent calculation
    daily_profit_pct = (stats['daily_profit'] / (current_balance - stats['daily_profit'])) * 100 if (current_balance - stats['daily_profit']) > 0 else 0
    
    # If we have total profit directly from trade stats, use that for more accuracy
    if 'total_profit' in stats and stats['total_profit'] != 0:
        logger.info(f"Using accumulated trade profit: {stats['total_profit']:.6f} USDT (vs balance diff: {profit_loss:.6f} USDT)")
        # Don't replace profit_loss with total_profit as they measure different things
        # profit_loss is the change in account value, while total_profit is from trades only
    
    # Load state from file to ensure we have the most recent data
    state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state', 'trading_state.json')
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                saved_stats = json.load(f)
                # Update our stats with any values from the file that might be more accurate
                if saved_stats.get('total_trades', 0) > stats['total_trades']:
                    stats['total_trades'] = saved_stats['total_trades']
                if saved_stats.get('winning_trades', 0) > stats['winning_trades']:
                    stats['winning_trades'] = saved_stats['winning_trades']
                if saved_stats.get('losing_trades', 0) > stats['losing_trades']:
                    stats['losing_trades'] = saved_stats['losing_trades']
        except Exception as e:
            logger.error(f"Error loading state file for report: {e}")
    
    # Use more decimal places for very small balances (< 1.0 USDT)
    balance_format = '.6f' if current_balance < 1.0 else '.2f'
    start_balance_format = '.6f' if stats['start_balance'] < 1.0 else '.2f'
    
    report = f"""
    ===== PERFORMANCE REPORT =====
    Time: {datetime.now()}
    
    Starting Balance: {stats['start_balance']:{start_balance_format}} USDT
    Current Balance: {current_balance:{balance_format}} USDT
    
    Total Profit/Loss: {profit_loss:{balance_format}} USDT ({profit_pct:.2f}%)
    Daily Profit/Loss: {stats['daily_profit']:{balance_format}} USDT ({daily_profit_pct:.2f}%)
    
    Total Trades: {stats['total_trades']}
    Winning Trades: {stats['winning_trades']}
    Losing Trades: {stats['losing_trades']}
    
    Win Rate: {(stats['winning_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0:.2f}%
    """
    
    logger.info(report)
    
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    report_file = os.path.join(report_dir, f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(report_file, 'w') as f:
        f.write(report)
        
    try:
        chart_path = None
        if stats['total_trades'] > 0:
            chart_path = generate_equity_chart(report_dir)
    except Exception as e:
        logger.error(f"Failed to generate chart: {e}")
        chart_path = None
        
    notifier = TelegramNotifier()
    
    # Use different precision for Telegram message too
    telegram_report = f"📊 *Daily Performance Report*\n\n" \
                     f"*Starting Balance:* {stats['start_balance']:{start_balance_format}} USDT\n" \
                     f"*Current Balance:* {current_balance:{balance_format}} USDT\n\n" \
                     f"*Total Profit/Loss:* {profit_loss:{balance_format}} USDT ({profit_pct:.2f}%)\n" \
                     f"*Daily Profit/Loss:* {stats['daily_profit']:{balance_format}} USDT ({daily_profit_pct:.2f}%)\n\n" \
                     f"*Total Trades:* {stats['total_trades']}\n" \
                     f"*Win Rate:* {(stats['winning_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0:.2f}%"
    
    notifier.send_message(telegram_report)
    
    if chart_path and os.path.exists(chart_path):
        notifier.send_photo(chart_path, "Equity Curve")
        
    stats['daily_profit'] = 0.0
    stats['last_report_time'] = datetime.now()
    
    return report_file


def generate_equity_chart(output_dir):
    try:
        trades_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state')
        trades_file = os.path.join(trades_dir, 'trades.json')
        
        if not os.path.exists(trades_file):
            logger.warning("No trade history found for chart generation")
            return None
            
        with open(trades_file, 'r') as f:
            trades = json.load(f)
            
        df = pd.DataFrame(trades)
        
        df['date'] = pd.to_datetime(df['timestamp'])
        df.set_index('date', inplace=True)
        df = df.sort_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['balance'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Balance (USDT)')
        plt.grid(True)
        
        chart_path = os.path.join(output_dir, f'equity_{datetime.now().strftime("%Y%m%d")}.png')
        plt.savefig(chart_path)
        plt.close()
        
        return chart_path
    except Exception as e:
        logger.error(f"Failed to generate equity chart: {e}")
        return None


def send_daily_report():
    """Send daily performance report"""
    if not SEND_DAILY_REPORT:
        return
        
    logger.info("Sending daily performance report...")
    generate_performance_report()


def send_status_report():
    """Send a status report with API connection, WebSocket status, and account balance"""
    
    try:
        logger.info("Generating connection status report")
        
        # Check if websocket is connected
        ws_status = "Connected" if websocket_manager and websocket_manager.is_connected() else "Disconnected"
        
        # Check API connection by attempting to get server time
        api_status = "Connected"
        try:
            binance_client.client.get_server_time()
        except Exception as e:
            api_status = f"Error: {str(e)[:50]}..."
        
        # Get current balance
        balance = binance_client.get_account_balance()
        
        # Get current positions
        positions = []
        try:
            position_info = binance_client.get_position_info(TRADING_SYMBOL)
            if position_info and abs(position_info.get('position_amount', 0)) > 0:
                side = "LONG" if position_info['position_amount'] > 0 else "SHORT"
                positions.append(f"{TRADING_SYMBOL}: {position_info['position_amount']} ({side})")
        except Exception as e:
            positions.append(f"Error getting positions: {str(e)[:30]}...")
        
        # Get current price
        price = "Unknown"
        try:
            price = binance_client.get_current_price(TRADING_SYMBOL)
        except:
            pass
        
        # Create status report
        status_message = (
            f"🔄 *Status Report*\n\n"
            f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"*Connection Status:*\n"
            f"API: {api_status}\n"
            f"WebSocket: {ws_status}\n\n"
            f"*Account:*\n"
            f"Balance: {balance:.4f} USDT\n"
            f"Current {TRADING_SYMBOL} Price: {price}\n\n"
        )
        
        if positions:
            status_message += f"*Active Positions:*\n"
            for pos in positions:
                status_message += f"{pos}\n"
        else:
            status_message += "*No active positions*\n"
        
        status_message += f"\n*Trading Stats:*\n"
        status_message += f"Total Trades: {stats['total_trades']}\n"
        status_message += f"Win Rate: {(stats['winning_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0:.1f}%\n"
        
        # Send the message using plaintext to avoid formatting issues
        notifier = TelegramNotifier()
        notifier.send_plain_message(status_message)
        
        logger.info("Status report sent successfully")
        
    except Exception as e:
        logger.error(f"Error generating status report: {e}")


def handle_exit(signal, frame):
    """Handle exit gracefully"""
    global running, websocket_manager

    logger.info("Shutdown signal received. Cleaning up...")
    running = False
    
    # Send notification that bot is stopping
    try:
        notifier = TelegramNotifier()
        notifier.send_message("🛑 *Trading Bot Stopping*\n\n"
                            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Trading Symbol: {TRADING_SYMBOL}\n"
                            f"Current Balance: {stats['current_balance']:.2f} USDT\n\n"
                            "Bot has received shutdown signal and is stopping gracefully.")
    except Exception as e:
        logger.error(f"Failed to send stop notification: {e}")
    
    if websocket_manager:
        websocket_manager.stop()
        logger.info("WebSocket connections closed")


def save_state():
    """Save the current state to a file for possible restart"""
    state_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state')
    os.makedirs(state_dir, exist_ok=True)
    
    json_stats = stats.copy()
    if json_stats['last_trade_time']:
        json_stats['last_trade_time'] = json_stats['last_trade_time'].isoformat()
    if json_stats['last_report_time']:
        json_stats['last_report_time'] = json_stats['last_report_time'].isoformat()
    
    state_file = os.path.join(state_dir, 'trading_state.json')
    with open(state_file, 'w') as f:
        json.dump(json_stats, f)
        
    logger.info("Trading state saved")


def save_trade(trade_data):
    """Save trade to trade history"""
    trade_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state')
    os.makedirs(trade_dir, exist_ok=True)
    
    if 'timestamp' not in trade_data:
        trade_data['timestamp'] = datetime.now().isoformat()
    
    # Make sure we have the latest balance
    if 'balance' not in trade_data or trade_data['balance'] <= 0:
        try:
            current_balance = binance_client.get_account_balance()
            if current_balance > 0:
                trade_data['balance'] = current_balance
            else:
                trade_data['balance'] = stats['current_balance']  # Fallback to stats
        except Exception:
            trade_data['balance'] = stats['current_balance']  # Fallback to stats if API call fails
    
    trades_file = os.path.join(trade_dir, 'trades.json')
    
    trades = []
    if os.path.exists(trades_file):
        with open(trades_file, 'r') as f:
            try:
                trades = json.load(f)
            except json.JSONDecodeError:
                trades = []
    
    trades.append(trade_data)
    
    with open(trades_file, 'w') as f:
        json.dump(trades, f)


def load_state():
    """Load saved state if exists"""
    state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state', 'trading_state.json')
    if os.path.exists(state_file) and os.path.getsize(state_file) > 0:
        try:
            with open(state_file, 'r') as f:
                loaded_stats = json.load(f)
                
            if 'last_trade_time' in loaded_stats and loaded_stats['last_trade_time']:
                loaded_stats['last_trade_time'] = datetime.fromisoformat(loaded_stats['last_trade_time'])
            if 'last_report_time' in loaded_stats and loaded_stats['last_report_time']:
                loaded_stats['last_report_time'] = datetime.fromisoformat(loaded_stats['last_report_time'])
                
            logger.info("Loaded previous trading state")
            return loaded_stats
        except json.JSONDecodeError as e:
            logger.error(f"Error loading state file (corrupted JSON): {e}")
            logger.info("Creating a new state file")
            # Create a backup of the corrupted file
            if os.path.exists(state_file):
                backup_file = f"{state_file}.bak.{int(time.time())}"
                try:
                    os.rename(state_file, backup_file)
                    logger.info(f"Backed up corrupted state file to {backup_file}")
                except Exception as rename_error:
                    logger.error(f"Failed to backup corrupted state file: {rename_error}")
            return None
    
    logger.info("No previous state found or file is empty")
    return None


def run_backtest(symbol, timeframe, strategy_name, start_date, end_date=None, save_results=True):
    """Run backtest using historical data"""
    logger.info(f"Starting backtest for {symbol} with {strategy_name} strategy")
    
    try:
        binance = BinanceClient()
        
        # Handle relative date strings like "30 days" or "1 year ago"
        if isinstance(start_date, str) and any(word in start_date for word in ['day', 'week', 'month', 'year']):
            # For Binance API, we can use relative dates directly
            api_start_date = start_date
            
            # For the Backtester, convert to YYYY-MM-DD format
            # Extract the numeric value and time unit
            parts = start_date.split()
            
            # Default values
            num = 30  # Default to 30 days if parsing fails
            unit = 'days'
            
            if len(parts) >= 2 and parts[0].isdigit():
                num = int(parts[0])
                unit = parts[1].lower()
                
                # Calculate actual date
                if unit.startswith('day'):
                    backtest_start_date = (datetime.now() - timedelta(days=num)).strftime('%Y-%m-%d')
                elif unit.startswith('week'):
                    backtest_start_date = (datetime.now() - timedelta(weeks=num)).strftime('%Y-%m-%d')
                elif unit.startswith('month'):
                    backtest_start_date = (datetime.now() - timedelta(days=num*30)).strftime('%Y-%m-%d')
                elif unit.startswith('year'):
                    backtest_start_date = (datetime.now() - timedelta(days=num*365)).strftime('%Y-%m-%d')
                else:
                    # Default fallback
                    backtest_start_date = (datetime.now() - timedelta(days=num)).strftime('%Y-%m-%d')
            else:
                # Default fallback if we can't parse the input
                backtest_start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                logger.warning(f"Couldn't parse date format '{start_date}', using past 30 days as default")
        else:
            # If it's already in YYYY-MM-DD format, use it directly for both
            api_start_date = start_date
            backtest_start_date = start_date
            
        logger.info(f"Fetching historical data from: {api_start_date} (as {backtest_start_date})")
        logger.info(f"Using {symbol} on {timeframe} timeframe")
        
        try:
            logger.info("Starting historical data request - this may take some time...")
            # Calculate required candles for the period
            if isinstance(start_date, str) and any(word in start_date for word in ['day', 'week', 'month', 'year']):
                # Calculate candles needed based on timeframe and period
                if '15m' in timeframe:
                    candles_per_day = 96  # 24h * 4 (15min intervals)
                elif '5m' in timeframe:
                    candles_per_day = 288  # 24h * 12 (5min intervals)
                elif '1h' in timeframe:
                    candles_per_day = 24  # 24 hours
                elif '4h' in timeframe:
                    candles_per_day = 6   # 6 * 4h intervals
                elif '1d' in timeframe:
                    candles_per_day = 1   # 1 daily candle
                else:
                    candles_per_day = 96  # Default to 15m
                
                # Calculate total candles needed (add buffer for weekends/holidays)
                total_days = num if 'num' in locals() else 90
                required_candles = int(total_days * candles_per_day * 1.2)  # 20% buffer
                
                # Binance API has maximum limits, cap at 1500 for most intervals
                required_candles = min(required_candles, 1500)
                
                logger.info(f"Requesting {required_candles} candles for {total_days} days on {timeframe} timeframe")
            else:
                # Default for date range backtests
                required_candles = 1500
            
            # Use calculated limit with Binance API limits in mind
            klines = binance.get_historical_klines(
                symbol=symbol,
                interval=timeframe,
                start_str=api_start_date,
                end_str=end_date,
                limit=required_candles  # Use calculated limit with API constraints
            )
            logger.info(f"Successfully retrieved {len(klines)} historical candles")
        except Exception as api_error:
            logger.error(f"API Error fetching klines: {api_error}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
            return None
        
        if not klines or len(klines) < 100:
            logger.error(f"Not enough historical data for backtest. Got {len(klines) if klines else 0} candles.")
            return None
            
        backtester = Backtester(strategy_name, symbol, timeframe, backtest_start_date, end_date)
        
        df = backtester.load_historical_data(klines)
        
        results = backtester.run(df)
        
        if save_results and results:
            output_dir = backtester.save_results(results)
            
            summary = backtester.generate_summary_report(results)
            
            summary_path = os.path.join(output_dir, 'summary.md')
            with open(summary_path, 'w') as f:
                f.write(summary)
                
            print("\n" + summary + "\n")
            
            if USE_TELEGRAM:
                notifier = TelegramNotifier()
                notifier.send_message(f"🔍 *Backtest Completed*\n\n{summary}")
                
                equity_chart = os.path.join(output_dir, 'plots', 'equity_curve.png')
                if os.path.exists(equity_chart):
                    notifier.send_photo(equity_chart, f"Equity Curve - {symbol} {strategy_name}")
            
        return results
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        return None


def validate_backtest_results(results):
    """
    Validate backtest results against minimum performance criteria
    Returns (is_valid, message)
    """
    if not results:
        return False, "Backtest failed to produce results"
    
    # Extract key performance metrics
    total_return_pct = results.get('total_return', 0)
    win_rate = results.get('win_rate', 0)
    total_trades = results.get('total_trades', 0)
    
    # Check if strategy meets minimum criteria
    reasons = []
    
    if total_trades < 5:
        reasons.append(f"Too few trades ({total_trades} < 5)")
    
    if total_return_pct < BACKTEST_MIN_PROFIT_PCT:
        reasons.append(f"Profit too low ({total_return_pct:.2f}% < {BACKTEST_MIN_PROFIT_PCT}%)")
    
    if win_rate < BACKTEST_MIN_WIN_RATE:
        reasons.append(f"Win rate too low ({win_rate:.2f}% < {BACKTEST_MIN_WIN_RATE}%)")
    
    if reasons:
        return False, "Strategy validation failed: " + ", ".join(reasons)
    
    return True, "Strategy passed validation"

def run_safety_backtest(symbol, timeframe, strategy_name):
    """
    Run a backtest to check strategy performance before live trading
    """
    logger.info("Running safety backtest before starting live trading...")
    
    # Use past period defined in config for backtest
    start_date = BACKTEST_PERIOD
    
    # Run the backtest
    results = run_backtest(
        symbol=symbol,
        timeframe=timeframe,
        strategy_name=strategy_name,
        start_date=start_date,
        save_results=True
    )
    
    if not results:
        return False, "Backtest failed to complete"
    
    # Validate the results
    is_valid, message = validate_backtest_results(results)
    
    if is_valid:
        logger.info(f"Strategy validation passed: {symbol} with {strategy_name}")
    else:
        logger.warning(f"Strategy validation failed: {message}")
        
        # Send alert if Telegram is enabled
        notifier = TelegramNotifier()
        notifier.send_message(f"⚠️ *Strategy Validation Failed*\n\n"
                              f"Symbol: {symbol}\n"
                              f"Strategy: {strategy_name}\n"
                              f"Reason: {message}\n\n"
                              f"Bot will not start live trading unless you use --skip-validation")
    
    return is_valid, message

def perform_test_trade(symbol=TRADING_SYMBOL):
    """
    Perform a small test trade to verify trading functionality
    Returns True if successful, False otherwise
    """
    logger.info(f"Performing test trade on {symbol} to verify trading functionality")
    
    try:
        # Step 1: Get current price with retries
        retry_count = 3
        current_price = None
        
        for attempt in range(retry_count):
            try:
                current_price = binance_client.get_current_price(symbol)
                if current_price:
                    break
                logger.warning(f"Got empty price on attempt {attempt+1}/{retry_count}, retrying...")
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Error getting current price (attempt {attempt+1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2)
        
        if not current_price:
            logger.error("Could not get current price for test trade after multiple attempts")
            return False
            
        # Step 2: Get current account balance with retries
        retry_count = 3
        initial_balance = None
        
        for attempt in range(retry_count):
            try:
                initial_balance = binance_client.get_account_balance()
                if initial_balance > 0:
                    break
                logger.warning(f"Got zero balance on attempt {attempt+1}/{retry_count}, retrying...")
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Error getting account balance (attempt {attempt+1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2)
        
        if not initial_balance or initial_balance <= 0:
            logger.error("Could not get valid account balance for test trade")
            return False
            
        logger.info(f"Balance before test trade: {initial_balance} USDT")
        
        # Step 3: Get symbol info with retries and fallback
        symbol_info = None
        qty_precision = 3  # Default if we can't get from API
        min_qty = 0.001    # Default if we can't get from API
        min_notional = 100.0  # Binance futures default is 100 USDT
        
        for attempt in range(3):
            try:
                symbol_info = binance_client.get_symbol_info(symbol)
                if symbol_info:
                    qty_precision = symbol_info.get('quantity_precision', 3)
                    min_qty = float(symbol_info.get('min_qty', 0.001))
                    min_notional = float(symbol_info.get('min_notional', 100.0))
                    logger.info(f"Symbol info: precision={qty_precision}, min_qty={min_qty}, min_notional={min_notional}")
                    break
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Error getting symbol info (attempt {attempt+1}/3): {e}")
                time.sleep(2)
                
        if not symbol_info:
            logger.warning("Could not get symbol info, using fallback values: precision=3, min_qty=0.001, min_notional=100.0")
            
            # For common coins, use known precision values
            if symbol == "BTCUSDT":
                qty_precision = 3
                min_qty = 0.001
                min_notional = 100.0
            elif symbol == "ETHUSDT":
                qty_precision = 3  
                min_qty = 0.001
                min_notional = 100.0
            elif symbol == "ETHUSDT":
                qty_precision = 1
                min_qty = 0.1
                min_notional = 100.0
            elif symbol == "BNBUSDT":
                qty_precision = 2
                min_qty = 0.01
                min_notional = 100.0
            # For others, use conservative defaults
            else:
                qty_precision = 3
                min_qty = 0.001
                min_notional = 100.0
        
        # Step 4: Calculate test trade size to meet minimum notional requirement
        # CRITICAL FIX: Ensure order size meets minimum notional requirement
        # Calculate the minimum quantity needed to meet min_notional
        min_notional_qty = min_notional / current_price
        
        # Apply precision to this quantity
        import math
        multiplier = 10 ** qty_precision
        min_notional_qty = math.ceil(min_notional_qty * multiplier) / multiplier
        
        # Verify the minimum quantity is at least the exchange's min_qty
        test_qty = max(min_notional_qty, min_qty)
        
        # Calculate test value using 1.5% of account (but don't go below min notional)
        account_percent_qty = (initial_balance * 0.015) / current_price
        account_percent_qty = math.floor(account_percent_qty * multiplier) / multiplier
        
        # Only use account percentage if it's large enough to meet min notional
        if account_percent_qty * current_price >= min_notional and account_percent_qty >= min_qty:
            test_qty = account_percent_qty
        
        # Don't use more than 5% of account balance
        max_qty = (initial_balance * 0.05) / current_price
        max_qty = math.floor(max_qty * multiplier) / multiplier
        test_qty = min(test_qty, max_qty)
        
        # Verify test_qty meets both min_qty and min_notional requirements
        order_value = test_qty * current_price
        
        logger.info(f"Final test order: {test_qty} {symbol} at ~{current_price} = {order_value:.2f} USDT")
        
        # Final safety check - if order value is too small, adjust to minimum required
        if test_qty <= 0:
            logger.warning(f"Calculated quantity is too small or zero: {test_qty}")
            logger.info("Automatically using minimum quantity required")
            test_qty = min_qty
            order_value = test_qty * current_price
            
        if order_value < min_notional:
            logger.warning(f"Calculated order value is too small: {order_value:.2f} USDT (min required: {min_notional} USDT)")
            logger.info("Automatically adjusting order quantity to meet minimum notional requirement")
            
            # Calculate the minimum quantity needed to meet min_notional requirement
            test_qty = min_notional / current_price
            # Apply precision
            multiplier = 10 ** qty_precision
            test_qty = math.ceil(test_qty * multiplier) / multiplier
            # Make sure it meets min_qty requirement too
            test_qty = max(test_qty, min_qty)
            # Recalculate order value with adjusted quantity
            order_value = test_qty * current_price
            
            logger.info(f"Adjusted test order: {test_qty} {symbol} at ~{current_price} = {order_value:.2f} USDT")
        
        # Check if we have enough balance for the test trade
        if order_value > initial_balance * 0.9:
            logger.error(f"Account balance too low for test trade. Required: {order_value:.2f} USDT, Available: {initial_balance} USDT")
            return False
        
        # Step 5: Place a market BUY order with retry
        logger.info(f"Executing test BUY order: {test_qty} {symbol} at ~{current_price}")
        
        buy_order = None
        for attempt in range(3):
            try:
                buy_order = binance_client.place_market_order(symbol, "BUY", test_qty)
                if buy_order:
                    break
                logger.warning(f"Test BUY order attempt {attempt+1}/3 returned empty result, retrying...")
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Error placing test BUY order (attempt {attempt+1}/3): {e}")
                if attempt < 2:  # Less than max retries
                    time.sleep(2)
        
        if not buy_order:
            logger.error("Test BUY order failed after multiple attempts")
            return False
            
        logger.info(f"Test BUY order successful: {buy_order.get('orderId', 'unknown')}")
        
        # Wait a moment for the order to fully process
        time.sleep(3)  # Increased from 2 to 3 seconds
        
        # Step 6: Check position - more robust approach to verify position is open
        position_verified = False
        position_qty = test_qty
        
        # Method 1: Try to get direct position info
        try:
            position = binance_client.get_position_info(symbol)
            if position and position.get('position_amount', 0) > 0:
                position_verified = True
                position_qty = float(position.get('position_amount', test_qty))
                logger.info(f"Test position opened: {position_qty} {symbol} @ {position.get('entry_price')}")
        except Exception as e:
            logger.warning(f"Error getting position info: {e}")
        
        # Method 2: If get_position_info fails, verify by checking account balance change
        if not position_verified:
            try:
                # Check if balance has changed, which indicates position was opened
                current_balance = binance_client.get_account_balance()
                if current_balance < initial_balance:  # Balance decreased = position likely opened
                    logger.info(f"Position verified via balance change: {initial_balance} → {current_balance}")
                    position_verified = True
            except Exception as e:
                logger.warning(f"Error checking balance for position verification: {e}")
        
        # Method 3: Use directly the saved order details
        if not position_verified and buy_order and 'fills' in buy_order:
            filled_qty = 0
            for fill in buy_order.get('fills', []):
                try:
                    filled_qty += float(fill.get('qty', 0))
                except (ValueError, TypeError):
                    pass
                    
            if filled_qty > 0:
                logger.info(f"Position verified via order fills: {filled_qty} {symbol}")
                position_verified = True
                # Use the filled quantity for selling later
                position_qty = filled_qty
        
        # Final fallback - assume position opened (if we can't verify but buy order was accepted)
        if not position_verified and buy_order:
            logger.warning("Could not definitively verify position was opened. Assuming successful buy and continuing with test sell.")
            position_verified = True
        
        # If we couldn't verify position and no buy order succeeded, abort
        if not position_verified and not buy_order:
            logger.error("Could not verify position was opened and no successful buy order. Aborting test.")
            return False
        
        # Step 7: Place a SELL order to close the position
        logger.info(f"Executing test SELL order to close position: {position_qty} {symbol}")
        
        sell_order = None
        for attempt in range(3):
            try:
                sell_order = binance_client.place_market_order(symbol, "SELL", position_qty)
                if sell_order:
                    break
                logger.warning(f"Test SELL order attempt {attempt+1}/3 returned empty result, retrying...")
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Error placing test SELL order (attempt {attempt+1}/3): {e}")
                if attempt < 2:  # Less than max retries
                    time.sleep(2)
        
        if not sell_order:
            logger.error("Test SELL order failed. You may need to manually close the position!")
            return False
            
        logger.info(f"Test SELL order successful: {sell_order.get('orderId', 'unknown')}")
        
        # Wait a moment for the order to fully process
        time.sleep(3)  # Increased from 2 to 3 seconds
        
        # Step 8: Check final balance and report results
        final_balance = None
        for attempt in range(3):
            try:
                final_balance = binance_client.get_account_balance()
                if final_balance > 0:
                    break
                time.sleep(1)
            except:
                if attempt < 2:  # Less than max retries
                    time.sleep(1)
        
        if final_balance and final_balance > 0:
            balance_diff = final_balance - initial_balance
            logger.info(f"Balance after test trade: {final_balance} USDT (Change: {balance_diff} USDT)")
        else:
            logger.warning("Could not get final balance after test trade")
        
        # Consider the test successful if we got this far with a sell order
        logger.info(f"Test trade completed successfully")
        
        # Notify on Telegram if enabled
        try:
            notifier = TelegramNotifier()
            notifier.send_message(f"✅ *Test Trade Completed*\n\n"
                                f"Symbol: {symbol}\n"
                                f"Test Size: {position_qty} (Value: {order_value:.2f} USDT)\n"
                                f"Trading functionality verified successfully.")
        except:
            logger.warning("Failed to send telegram notification")
        
        return True
    except Exception as e:
        logger.error(f"Error during test trade: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Notify on Telegram if enabled
        try:
            notifier = TelegramNotifier()
            notifier.send_message(f"❌ *Test Trade Failed*\n\n"
                                f"Symbol: {symbol}\n"
                                f"Error: {str(e)}\n\n"
                                f"Please check logs and resolve issues before starting live trading.")
        except:
            pass
        
        return False


def generate_trade_chart(symbol, side, price, profit_loss=None):
    """
    Generates a price chart showing the trade entry/exit and recent price action
    
    Args:
        symbol: Trading symbol
        side: Trade side (BUY/SELL)
        price: Trade execution price
        profit_loss: Optional profit/loss value
    
    Returns:
        Path to saved chart image or None if failed
    """
    # Add throttling to prevent excessive chart generation
    if not hasattr(generate_trade_chart, 'last_chart_time'):
        generate_trade_chart.last_chart_time = {}
    
    current_time = time.time()
    chart_key = f"{symbol}_{side}_{price}"
    
    # Only generate chart if it's been more than 10 seconds since last chart for this trade
    if chart_key in generate_trade_chart.last_chart_time:
        time_since_last = current_time - generate_trade_chart.last_chart_time[chart_key]
        if time_since_last < 10:
            logger.debug(f"Skipping chart generation - too soon since last chart ({time_since_last:.1f}s)")
            return None
    
    generate_trade_chart.last_chart_time[chart_key] = current_time
    
    try:
        # Create charts directory if needed
        charts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # Get recent price data for chart
        recent_klines = klines_data.get(symbol, [])
        if not recent_klines or len(recent_klines) < 20:
            logger.warning("Not enough historical data to generate trade chart")
            return None
            
        # Convert to dataframe
        df = pd.DataFrame(recent_klines[-100:], columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        df['timestamp'] = pd.to_datetime(df['open_time'].astype(float), unit='ms')
        
        # Create the chart
        plt.figure(figsize=(10, 6))
        
        # Plot price action
        plt.plot(df['timestamp'], df['close'], label='Price', color='blue', alpha=0.7)
        
        # Mark the trade
        trade_color = 'green' if side == 'BUY' else 'red'
        trade_marker = '^' if side == 'BUY' else 'v'
        trade_label = f"{side} @ {price}"
        
        # Find closest timestamp to add marker
        closest_time = df['timestamp'].iloc[-1]
        
        plt.scatter([closest_time], [price], color=trade_color, s=100, marker=trade_marker, label=trade_label)
        
        # Add horizontal line at trade price
        plt.axhline(y=price, color=trade_color, linestyle='--', alpha=0.5)
        
        # Add profit/loss info if available
        if profit_loss is not None:
            profit_label = f"P/L: {'+' if profit_loss > 0 else ''}{profit_loss:.2f} USDT"
            plt.annotate(profit_label, 
                         xy=(df['timestamp'].iloc[-5], price), 
                         xytext=(0, 10 if profit_loss > 0 else -20),
                         textcoords='offset points',
                         color='green' if profit_loss > 0 else 'red',
                         fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='green' if profit_loss > 0 else 'red'))
        
        # Set title and labels
        plt.title(f"{symbol} Trade - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(charts_dir, f"trade_{symbol}_{side}_{int(time.time())}.png")
        plt.savefig(chart_path)
        plt.close()
        
        logger.info(f"Trade chart saved to {chart_path}")
        return chart_path
        
    except Exception as e:
        logger.error(f"Error generating trade chart: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def round_quantity(quantity, symbol):
    """Round quantity based on symbol precision"""
    try:
        symbol_info = binance_client.get_symbol_info(symbol)
        if symbol_info and 'quantity_precision' in symbol_info:
            precision = symbol_info['quantity_precision']
            return round(quantity, precision)
        return quantity  # Return as is if we couldn't get precision
    except Exception as e:
        logger.warning(f"Error rounding quantity: {e}")
        return quantity  # Return original quantity on error

def debug_buffer_state():
    """Debug function to check current buffer and WebSocket state"""
    global rolling_buffer, websocket_manager
    
    logger.info("🔍 === DEBUG BUFFER STATE ===")
    
    # Check rolling buffer
    if rolling_buffer:
        try:
            stats = rolling_buffer.get_buffer_stats()
            logger.info(f"📊 Buffer Stats: {stats}")
            
            # Check if buffer has data
            if hasattr(rolling_buffer, 'candle_buffer') and rolling_buffer.candle_buffer:
                latest_candle = list(rolling_buffer.candle_buffer)[-1] if rolling_buffer.candle_buffer else None
                if latest_candle and len(latest_candle) >= 7:
                    latest_time = datetime.fromtimestamp(latest_candle[6]/1000)
                    logger.info(f"📈 Latest Buffer Candle: time={latest_time}, close={latest_candle[4]}")
                else:
                    logger.warning("⚠️ Buffer has no valid candles")
            
        except Exception as e:
            logger.error(f"❌ Error checking buffer: {e}")
    else:
        logger.error("❌ Rolling buffer is None")
    
    # Check WebSocket state
    if websocket_manager:
        try:
            is_connected = websocket_manager.is_connected()
            logger.info(f"🌐 WebSocket Connected: {is_connected}")
            
            # Check last kline data
            if hasattr(websocket_manager, 'last_kline_data'):
                last_kline = websocket_manager.last_kline_data.get(TRADING_SYMBOL, {})
                if last_kline:
                    last_time = datetime.fromtimestamp(last_kline.get('close_time', 0)/1000)
                    logger.info(f"📊 Last WebSocket Kline: time={last_time}, close={last_kline.get('close', 'N/A')}")
                else:
                    logger.warning("⚠️ No WebSocket kline data for symbol")
        except Exception as e:
            logger.error(f"❌ Error checking WebSocket: {e}")
    else:
        logger.error("❌ WebSocket manager is None")
    
    logger.info("🔍 === END DEBUG ===")

def main():
    """Main function to start the trading bot"""
    global running, stats, websocket_manager
    
    parser = argparse.ArgumentParser(description='Binance Futures Trading Bot')
    parser.add_argument('--backtest', action='store_true', help='Run in backtest mode only')
    parser.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, default=TRADING_SYMBOL, help='Trading symbol for backtest')
    parser.add_argument('--timeframe', type=str, default=TIMEFRAME, help='Timeframe for trading (e.g. 1m, 5m, 15m, 1h)')
    parser.add_argument('--strategy', type=str, default=STRATEGY, help='Strategy for backtest')
    parser.add_argument('--report', action='store_true', help='Generate performance report only')
    parser.add_argument('--interval', type=int, default=5, help='Trading check interval in minutes')
    parser.add_argument('--skip-validation', action='store_true', help='Skip strategy validation before live trading')
    parser.add_argument('--skip-test-trade', action='store_true', help='Skip test trade before live trading')
    parser.add_argument('--small-account', action='store_true', help='Run with small account (under $45) - skips test trade and uses adjusted risk')
    parser.add_argument('--force-balance', action='store_true', help='Force initialization of balance from config file')
    parser.add_argument('--test-trade', action='store_true', help='Run test trade only and exit')
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # For the report command, make sure we can access the config
    if args.report:
        setup()
        generate_performance_report()
        return
    
    # Run only in backtest mode
    if args.backtest:
        symbol = args.symbol or TRADING_SYMBOL
        timeframe = args.timeframe or TIMEFRAME
        strategy = args.strategy or STRATEGY
        start_date = args.start_date or "30 days ago"
        
        run_backtest(
            symbol=symbol,
            timeframe=timeframe,
            strategy_name=strategy,
            start_date=start_date,
            end_date=args.end_date
        )
        return
    
    # Run only a test trade
    if args.test_trade:
        setup()
        symbol = args.symbol or TRADING_SYMBOL
        perform_test_trade(symbol)
        return
        
  
    
    # Initialize basic setup
    setup()
    
    # Auto-detect small account
    try:
        account_balance = binance_client.get_account_balance()
        if account_balance < 50:
            logger.info(f"Small account detected (${account_balance:.2f}), automatically enabling small account mode")
            args.small_account = True
    except Exception as e:
        logger.error(f"Error detecting account size: {e}")
    
    # Load previous state if available
    previous_state = load_state()
    if previous_state:
        stats.update(previous_state)
    
    # Force initialize state file with proper balance if requested
    if args.force_balance:
        logger.info("Forcing state file initialization with current balance")
        initialize_state_file(force=True)
        
    # Get symbol info for this trading pair
    symbol = args.symbol or TRADING_SYMBOL
    timeframe = args.timeframe or TIMEFRAME
    strategy_name = args.strategy or STRATEGY
    
    # When running without explicit arguments, always send notification, test a trade, then run bot
    if not (args.backtest or args.report or args.test_trade):
        # Step 1: Send startup notification
        logger.info("===== STEP 1: SENDING STARTUP NOTIFICATION =====")
        notifier = TelegramNotifier()
        notifier.send_message(f"🚀 *Trading Bot Initializing*\n\n"
                            f"Symbol: {symbol}\n"
                            f"Strategy: {strategy_name}\n"
                            f"Timeframe: {timeframe}\n"
                            f"Starting Balance: {stats['current_balance']:.2f} USDT\n\n"
                            f"Starting sequence: Notification → Test Trade → Live Trading")
        
        # Step 2: Always run a test trade to verify trading functionality
        logger.info("===== STEP 2: PERFORMING TEST TRADE =====")
        test_trade_success = perform_test_trade(symbol)
        
        if not test_trade_success:
            logger.warning("Test trade had issues, but continuing with live trading...")
            
        # Step 3: Start the actual trading bot (skip backtest validation)
        logger.info("===== STEP 3: STARTING LIVE TRADING BOT =====")
        notifier.send_message(f"✅ *Trading Bot Started*\n\n"
                            f"Symbol: {symbol}\n"
                            f"Strategy: {strategy_name}\n"
                            f"Timeframe: {timeframe}\n"
                            f"Starting Balance: {stats['current_balance']:.2f} USDT")
    else:
        # Original sequential flow for when arguments are provided
        # Step 1: Run a backtest first to validate the strategy
        if not args.skip_validation and BACKTEST_BEFORE_LIVE:
            logger.info("===== STEP 1: RUNNING STRATEGY VALIDATION BACKTEST =====")
            is_valid, validation_message = run_safety_backtest(symbol, timeframe, strategy_name)
            
            if not is_valid:
                logger.warning(f"Strategy validation failed: {validation_message}")
                logger.warning("You can override this with --skip-validation flag")
                
                # Notify user and wait for confirmation to continue
                notifier = TelegramNotifier()
                notifier.send_message(f"⚠️ *Strategy Validation Failed*\n\n"
                                    f"Symbol: {symbol}\n"
                                    f"Strategy: {strategy_name}\n"
                                    f"Reason: {validation_message}\n\n"
                                    f"Bot will continue anyway as validation is not strict. Add --skip-validation to bypass this check.")
                
                logger.warning("Continuing despite strategy validation failure after 5 seconds...")
                time.sleep(5)  # Brief pause to allow user to see the message
        else:
            logger.info("Strategy validation backtest skipped")
        
        # Step 2: Perform a test trade to verify API connectivity and permissions
        if not (args.skip_test_trade or args.small_account):
            logger.info("===== STEP 2: PERFORMING TEST TRADE =====")
            test_trade_success = perform_test_trade(symbol)
            
            if not test_trade_success:
                logger.error("Test trade failed! Please check API settings and permissions.")
                logger.error("You can bypass this check with --skip-test-trade flag")
                
                # Notify user and wait for confirmation to continue
                notifier = TelegramNotifier()
                notifier.send_message(f"⚠️ *Test Trade Failed*\n\n"
                                    f"Symbol: {symbol}\n"
                                    f"Check API keys, permissions, and account balance.\n\n"
                                    f"Bot will continue anyway with live trading. Use --skip-test-trade to bypass this check in the future.")
                
                logger.warning("Continuing despite test trade failure after 10 seconds...")
                time.sleep(10)  # Longer pause for test trade issues as they're more serious
        else:
            logger.info("Test trade skipped")
        
        # Step 3: Start the actual trading bot
        logger.info("===== STEP 3: STARTING LIVE TRADING BOT =====")
        notifier = TelegramNotifier()
        notifier.send_message(f"🚀 *Trading Bot Starting*\n\n"
                            f"Symbol: {symbol}\n"
                            f"Strategy: {strategy_name}\n"
                            f"Timeframe: {timeframe}\n"
                            f"Starting Balance: {stats['current_balance']:.2f} USDT")
    
    # Main trading loop
    check_interval = args.interval * 60  # Convert to seconds
    next_check = time.time()
    next_report = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
    last_status_report = time.time() - 7200  # Send status report after first 2 hours
    last_buffer_check = time.time()  # Buffer health monitoring
    
    # Begin regular trading cycle
    while running:
        try:
            current_time = time.time()
            
            # Check for trading signals
            if current_time >= next_check:
                logger.debug("Running regular trading check")
                check_for_signals()
                next_check = current_time + check_interval
            
            # Monitor buffer health every 5 minutes
            if current_time - last_buffer_check >= 300:  # 5 minutes
                buffer_healthy = monitor_buffer_health()
                if not buffer_healthy:
                    logger.warning("⚠️ Buffer health check failed - attempting recovery")
                    recover_rolling_buffer()
                last_buffer_check = current_time
            
            # Check if it's time for the daily report
            now = datetime.now()
            if now >= next_report:
                logger.info("Time for the daily report")
                send_daily_report()
                next_report = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
            
            # Send status report every 2 hours
            if current_time - last_status_report >= 2 * 3600:
                send_status_report()
                last_status_report = current_time
            
            # Save current state
            if int(current_time) % 600 == 0:  # Every 10 minutes
                save_state()
            
            # Sleep to avoid high CPU usage
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            logger.error(traceback.format_exc())
            time.sleep(60)  # Wait a minute before retrying on error
    
    # Clean up before exit
    if websocket_manager:
        websocket_manager.stop()
    
    # Final save of state
    save_state()
    
    logger.info("Trading bot stopped")
    
if __name__ == "__main__":
    main()