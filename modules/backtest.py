# filepath: /home/minhajulislam/tradingbot/modules/backtest.py
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any
import math

# Import strategy modules
from modules.strategies import get_strategy
from modules.config import (
    BACKTEST_INITIAL_BALANCE, BACKTEST_COMMISSION, LEVERAGE,
    USE_STOP_LOSS, STOP_LOSS_PCT, TRAILING_STOP, TRAILING_STOP_PCT,
    USE_TAKE_PROFIT, TAKE_PROFIT_PCT,
    FIXED_TRADE_PERCENTAGE, MARGIN_SAFETY_FACTOR,
    BACKTEST_MIN_PROFIT_PCT, BACKTEST_MIN_WIN_RATE, BACKTEST_MAX_DRAWDOWN, BACKTEST_MIN_PROFIT_FACTOR
)

# Set up logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')


class Position:
    """Class to represent a trading position"""
    
    def __init__(self, symbol: str, side: str, size: float, entry_price: float, 
                 timestamp: str, stop_loss: float = None, take_profit: float = None):
        self.symbol = symbol
        self.side = side  # 'BUY' or 'SELL'
        self.size = abs(size)  # Always positive
        self.entry_price = entry_price
        self.timestamp = timestamp
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.unrealized_pnl = 0.0
        self.max_profit = 0.0
        self.max_loss = 0.0
        
    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized P&L based on current price"""
        if self.side == 'BUY':
            # Long position
            pnl_per_unit = current_price - self.entry_price
        else:
            # Short position
            pnl_per_unit = self.entry_price - current_price
        
        self.unrealized_pnl = pnl_per_unit * self.size
        
        # Track max profit and loss
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
        if self.unrealized_pnl < self.max_loss:
            self.max_loss = self.unrealized_pnl
            
    def should_stop_loss(self, current_price: float) -> bool:
        """Check if position should be closed due to stop loss"""
        if not self.stop_loss:
            return False
            
        if self.side == 'BUY':
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss
            
    def should_take_profit(self, current_price: float) -> bool:
        """Check if position should be closed due to take profit"""
        if not self.take_profit:
            return False
            
        if self.side == 'BUY':
            return current_price >= self.take_profit
        else:
            return current_price <= self.take_profit
            
    def update_trailing_stop(self, current_price: float, trailing_pct: float):
        """Update trailing stop loss"""
        if not TRAILING_STOP:
            return
            
        if self.side == 'BUY':
            # For long positions, trail stop up
            new_stop = current_price * (1 - trailing_pct)
            if not self.stop_loss or new_stop > self.stop_loss:
                self.stop_loss = new_stop
        else:
            # For short positions, trail stop down
            new_stop = current_price * (1 + trailing_pct)
            if not self.stop_loss or new_stop < self.stop_loss:
                self.stop_loss = new_stop


class BacktestResults:
    """Class to store and analyze backtest results"""
    
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.positions = []
        self.daily_returns = []
        self.start_date = None
        self.end_date = None
        self.initial_balance = 0
        self.final_balance = 0
        
    def add_trade(self, trade: Dict):
        """Add a completed trade to results"""
        self.trades.append(trade)
        
    def add_equity_point(self, timestamp: str, balance: float, position_value: float = 0):
        """Add a point to the equity curve"""
        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': balance,
            'position_value': position_value,
            'total_equity': balance + position_value
        })
        
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return self._empty_metrics()
            
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        total_return = ((self.final_balance - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Drawdown analysis
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) > 0:
            equity_df['cummax'] = equity_df['total_equity'].cummax()
            equity_df['drawdown'] = (equity_df['total_equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
            max_drawdown = equity_df['drawdown'].min()
        else:
            max_drawdown = 0
            
        # Sharpe ratio (simplified using daily returns)
        if len(self.daily_returns) > 1:
            returns_std = np.std(self.daily_returns)
            avg_return = np.mean(self.daily_returns)
            sharpe_ratio = (avg_return / returns_std * np.sqrt(252)) if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Trade duration analysis
        if 'duration_hours' in trades_df.columns:
            avg_trade_duration = trades_df['duration_hours'].mean()
        else:
            avg_trade_duration = 0
            
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_duration': avg_trade_duration,
            'initial_balance': self.initial_balance,
            'final_balance': self.final_balance,
            'start_date': self.start_date,
            'end_date': self.end_date
        }
        
    def _empty_metrics(self) -> Dict:
        """Return empty metrics when no trades"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'avg_trade_duration': 0,
            'initial_balance': self.initial_balance,
            'final_balance': self.final_balance,
            'start_date': self.start_date,
            'end_date': self.end_date
        }


class Backtester:
    """Comprehensive backtesting engine for trading strategies"""
    
    def __init__(self, strategy_name: str, symbol: str, timeframe: str, 
                 start_date: str, end_date: str = None):
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # Initialize strategy
        self.strategy = get_strategy(strategy_name)
        
        # Backtesting parameters
        self.initial_balance = BACKTEST_INITIAL_BALANCE
        self.commission_rate = BACKTEST_COMMISSION
        self.leverage = LEVERAGE
        self.trade_percentage = FIXED_TRADE_PERCENTAGE
        
        # Current state
        self.current_balance = self.initial_balance
        self.current_position = None
        self.last_signal = None  # Track last processed signal
        self.ignored_signals = {'BUY': 0, 'SELL': 0}  # Count ignored duplicate signals
        self.results = BacktestResults()
        
        # Auto compounding tracking
        self.balance_history = []
        self.equity_curve = []
        self.peak_balance = self.initial_balance
        self.trades_count = 0
        self.total_compounded_profit = 0.0
        
        logger.info(f"Initialized Backtester for {symbol} using {strategy_name} strategy")
        logger.info(f"Period: {start_date} to {self.end_date}")
        logger.info(f"Initial balance: {self.initial_balance} USDT")
        logger.info("🔄 Auto compounding enabled - profits will increase position sizes")
        
    def load_historical_data(self, klines: List) -> pd.DataFrame:
        """Load and prepare historical data for backtesting"""
        try:
            if not klines or len(klines) < 50:
                raise ValueError(f"Insufficient historical data: {len(klines) if klines else 0} candles")
                
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Handle timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove any NaN values
            df = df.dropna(subset=numeric_columns)
            
            # Add technical indicators using strategy
            df = self.strategy.add_indicators(df)
            
            logger.info(f"Loaded {len(df)} candles for backtesting")
            logger.info(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise
            
    def calculate_position_size(self, price: float, stop_loss: float = None) -> float:
        """Calculate enhanced position size with dynamic sizing and improved risk management"""
        try:
            # Get dynamic position size multiplier from strategy if available
            base_percentage = self.trade_percentage
            if hasattr(self.strategy, 'get_position_size_multiplier'):
                multiplier = self.strategy.get_position_size_multiplier()
                adjusted_percentage = base_percentage * multiplier
            else:
                adjusted_percentage = base_percentage
            
            # Cap the adjusted percentage at 50% for safety
            adjusted_percentage = min(adjusted_percentage, 0.5)
            
            # Risk-based position sizing using stop loss
            if stop_loss and USE_STOP_LOSS:
                # Calculate risk per unit
                risk_per_unit = abs(price - stop_loss)
                risk_percentage = risk_per_unit / price
                
                # Maximum risk per trade (2% of balance)
                max_risk_amount = self.current_balance * 0.02
                
                # Calculate position size based on risk
                risk_based_quantity = max_risk_amount / risk_per_unit
                
                # Calculate percentage-based quantity
                trade_amount = self.current_balance * adjusted_percentage
                position_value = trade_amount * self.leverage
                percentage_based_quantity = position_value / price
                
                # Use the smaller of the two for risk management
                quantity = min(risk_based_quantity, percentage_based_quantity)
                
                logger.debug(f"Risk-based sizing: risk={risk_percentage:.3%}, "
                           f"risk_qty={risk_based_quantity:.6f}, "
                           f"pct_qty={percentage_based_quantity:.6f}, "
                           f"chosen={quantity:.6f}")
            else:
                # Standard percentage-based sizing
                trade_amount = self.current_balance * adjusted_percentage
                position_value = trade_amount * self.leverage
                quantity = position_value / price
            
            # Apply margin safety limits
            max_margin = self.current_balance * MARGIN_SAFETY_FACTOR
            max_position_value = max_margin * self.leverage
            max_quantity_by_margin = max_position_value / price
            
            if quantity > max_quantity_by_margin:
                quantity = max_quantity_by_margin
                logger.debug(f"Position size limited by margin safety: {quantity:.6f}")
            
            # Minimum position size check
            min_position_value = 10.0  # Minimum $10 position
            min_quantity = min_position_value / price
            
            if quantity < min_quantity:
                logger.warning(f"Position size too small: {quantity:.6f} < {min_quantity:.6f}")
                return 0
                
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating enhanced position size: {e}")
            return 0
            
    def calculate_stop_loss_price(self, entry_price: float, side: str, atr: float = None) -> float:
        """Calculate enhanced stop loss price with ATR-based dynamic levels"""
        if not USE_STOP_LOSS:
            return None
            
        try:
            # Base stop loss percentage
            base_stop_pct = STOP_LOSS_PCT
            
            # If ATR is available, use it for dynamic stop loss
            if atr and atr > 0:
                # ATR-based stop loss (1.5x ATR from entry)
                atr_stop_distance = atr * 1.5
                atr_stop_pct = atr_stop_distance / entry_price
                
                # Use the larger of base stop or ATR stop for better risk management
                stop_pct = max(base_stop_pct, atr_stop_pct)
                
                # Cap at maximum 5% stop loss
                stop_pct = min(stop_pct, 0.05)
                
                logger.debug(f"Dynamic stop loss: base={base_stop_pct:.3%}, "
                           f"atr_based={atr_stop_pct:.3%}, chosen={stop_pct:.3%}")
            else:
                stop_pct = base_stop_pct
            
            if side == 'BUY':
                return entry_price * (1 - stop_pct)
            else:
                return entry_price * (1 + stop_pct)
                
        except Exception as e:
            logger.error(f"Error calculating dynamic stop loss: {e}")
            # Fallback to basic calculation
            if side == 'BUY':
                return entry_price * (1 - STOP_LOSS_PCT)
            else:
                return entry_price * (1 + STOP_LOSS_PCT)
            
    def open_position(self, row: pd.Series, signal: str) -> bool:
        """Open a new position with enhanced risk management"""
        try:
            if self.current_position:
                logger.warning("Attempted to open position while one already exists")
                return False
                
            price = row['close']
            timestamp = row['timestamp']
            
            # Get ATR for dynamic stop loss if available
            atr = row.get('atr', None)
            
            # Calculate stop loss first (needed for position sizing)
            stop_loss = self.calculate_stop_loss_price(price, signal, atr)
            
            # Calculate position size with stop loss consideration
            quantity = self.calculate_position_size(price, stop_loss)
            
            if quantity <= 0:
                logger.warning(f"Position size too small: {quantity}")
                return False
                
            # Calculate commission
            position_value = quantity * price
            commission = position_value * self.commission_rate
            
            # Enhanced balance check
            required_margin = position_value / self.leverage
            total_required = required_margin + commission
            
            if total_required > self.current_balance * 0.95:  # Keep 5% buffer
                logger.warning(f"Insufficient balance: need {total_required:.2f}, have {self.current_balance:.2f}")
                return False
            
            # Calculate take profit if enabled
            take_profit = None
            if USE_TAKE_PROFIT:
                if signal == 'BUY':
                    take_profit = price * (1 + TAKE_PROFIT_PCT)
                else:  # signal == 'SELL'
                    take_profit = price * (1 - TAKE_PROFIT_PCT)
            
            # Create enhanced position
            self.current_position = Position(
                symbol=self.symbol,
                side=signal,
                size=quantity,
                entry_price=price,
                timestamp=timestamp,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Deduct commission and margin
            self.current_balance -= commission
            
            # Log enhanced position details with compounding info
            risk_pct = (abs(price - stop_loss) / price * 100) if stop_loss else 0
            growth_factor = self.current_balance / self.initial_balance
            
            logger.info(f"Opened {signal} position: {quantity:.6f} @ {price:.6f}")
            logger.info(f"  Stop Loss: {stop_loss:.6f} ({risk_pct:.2f}% risk)")
            if take_profit:
                tp_pct = (abs(take_profit - price) / price * 100)
                logger.info(f"  Take Profit: {take_profit:.6f} ({tp_pct:.1f}% target)")
            else:
                logger.info(f"  Take Profit: Disabled")
            logger.info(f"  Position Value: ${position_value:.2f}, Margin: ${required_margin:.2f}")
            logger.info(f"  🔄 Compounding Factor: {growth_factor:.2f}x (Balance: ${self.current_balance:.2f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening enhanced position: {e}")
            return False
            
    def close_position(self, row: pd.Series, reason: str = "Signal") -> bool:
        """Close the current position"""
        try:
            if not self.current_position:
                return False
                
            price = row['close']
            timestamp = row['timestamp']
            
            # Calculate P&L
            if self.current_position.side == 'BUY':
                pnl_per_unit = price - self.current_position.entry_price
            else:
                pnl_per_unit = self.current_position.entry_price - price
                
            gross_pnl = pnl_per_unit * self.current_position.size
            
            # Calculate commission
            position_value = self.current_position.size * price
            exit_commission = position_value * self.commission_rate
            
            # Net P&L after commission
            net_pnl = gross_pnl - exit_commission
            
            # Store previous balance for compounding calculation
            previous_balance = self.current_balance
            
            # Update balance (AUTO COMPOUNDING)
            self.current_balance += net_pnl
            
            # Track compounding metrics
            self.trades_count += 1
            if net_pnl > 0:
                self.total_compounded_profit += net_pnl
                # Update peak balance
                if self.current_balance > self.peak_balance:
                    self.peak_balance = self.current_balance
            
            # Add balance tracking for compounding visualization
            self.balance_history.append({
                'trade_number': self.trades_count,
                'previous_balance': previous_balance,
                'pnl': net_pnl,
                'new_balance': self.current_balance,
                'growth_factor': self.current_balance / self.initial_balance,
                'timestamp': timestamp
            })
            
            # Calculate trade duration
            duration = (timestamp - self.current_position.timestamp).total_seconds() / 3600  # hours
            
            # Record trade
            trade = {
                'symbol': self.symbol,
                'side': self.current_position.side,
                'size': self.current_position.size,
                'entry_price': self.current_position.entry_price,
                'exit_price': price,
                'entry_time': self.current_position.timestamp,
                'exit_time': timestamp,
                'duration_hours': duration,
                'gross_pnl': gross_pnl,
                'commission': exit_commission,
                'pnl': net_pnl,
                'return_pct': (net_pnl / (self.current_position.size * self.current_position.entry_price)) * 100,
                'close_reason': reason,
                'max_profit': self.current_position.max_profit,
                'max_loss': self.current_position.max_loss
            }
            
            self.results.add_trade(trade)
            
            # Enhanced logging with compounding information
            growth_pct = ((self.current_balance - previous_balance) / previous_balance) * 100
            total_growth_pct = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
            
            logger.debug(f"Closed {self.current_position.side} position: {net_pnl:.4f} USDT ({reason})")
            logger.info(f"💰 Balance: ${previous_balance:.2f} → ${self.current_balance:.2f} (+{growth_pct:+.2f}%)")
            logger.info(f"📈 Total Growth: {total_growth_pct:+.2f}% | Trades: {self.trades_count}")
            
            # Show compounding effect for profitable trades
            if net_pnl > 0:
                next_position_value = self.current_balance * self.trade_percentage * self.leverage
                logger.info(f"🔄 Auto Compounding: Next position value ~${next_position_value:.2f} (was ${previous_balance * self.trade_percentage * self.leverage:.2f})")
            
            # Clear position and signal tracking
            self.current_position = None
            self.last_signal = None  # Reset signal tracking when position is closed
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
            
    def update_position(self, row: pd.Series):
        """Update position with current market data"""
        if not self.current_position:
            return
            
        price = row['close']
        
        # Update unrealized P&L
        self.current_position.update_unrealized_pnl(price)
        
        # Update trailing stop
        if TRAILING_STOP:
            self.current_position.update_trailing_stop(price, TRAILING_STOP_PCT)
            
    def run(self, df: pd.DataFrame) -> Dict:
        """Run the backtest on historical data"""
        try:
            logger.info("Starting backtest execution...")
            
            # Initialize results
            self.results.start_date = df['timestamp'].iloc[0].strftime('%Y-%m-%d')
            self.results.end_date = df['timestamp'].iloc[-1].strftime('%Y-%m-%d')
            self.results.initial_balance = self.initial_balance
            
            # Track daily returns
            daily_balances = {}
            
            # Process each candle
            for i, row in df.iterrows():
                try:
                    timestamp = row['timestamp']
                    price = row['close']
                    
                    # Update current position if exists
                    if self.current_position:
                        self.update_position(row)
                        
                        # Check for take profit first (higher priority)
                        if self.current_position.should_take_profit(price):
                            self.close_position(row, "Take Profit")
                        # Check for stop loss
                        elif self.current_position.should_stop_loss(price):
                            self.close_position(row, "Stop Loss")
                    
                    # Get trading signal (only check if we have enough historical data)
                    if i >= 50:  # Need enough data for indicators
                        klines_subset = []
                        start_idx = max(0, i - 100)  # Use last 100 candles for signal (reduced from 200)
                        
                        for j in range(start_idx, i + 1):
                            row_data = df.iloc[j]
                            klines_subset.append([
                                int(row_data['timestamp'].timestamp() * 1000),
                                str(row_data['open']),
                                str(row_data['high']),
                                str(row_data['low']),
                                str(row_data['close']),
                                str(row_data['volume']),
                                int(row_data['timestamp'].timestamp() * 1000),
                                "0", "0", "0", "0", "0"
                            ])
                        
                        signal = self.strategy.get_signal(klines_subset)
                        
                        # Process signal - ENHANCED LOGIC TO PREVENT DUPLICATE POSITIONS
                        if signal:
                            logger.debug(f"Signal {signal} received at {timestamp}")
                            
                            if not self.current_position:
                                # No current position - open new position
                                success = self.open_position(row, signal)
                                if success:
                                    self.last_signal = signal
                                    logger.info(f"✅ Opened NEW {signal} position at {row['close']:.6f}")
                            else:
                                # Already have a position - check if signal is different
                                current_side = self.current_position.side
                                
                                if signal == current_side:
                                    # Same signal as current position - IGNORE to prevent duplicate
                                    self.ignored_signals[signal] += 1
                                    logger.debug(f"🔄 Duplicate {signal} signal ignored - already in {current_side} position (ignored {self.ignored_signals[signal]} times)")
                                    continue  # Skip processing this signal
                                
                                elif ((signal == 'BUY' and current_side == 'SELL') or
                                      (signal == 'SELL' and current_side == 'BUY')):
                                    # Opposite signal - close current and open new
                                    logger.info(f"🔄 Signal change: {current_side} → {signal}")
                                    self.close_position(row, "Signal Change")
                                    
                                    # Open new position with opposite signal
                                    success = self.open_position(row, signal)
                                    if success:
                                        self.last_signal = signal
                                        logger.info(f"✅ Switched to {signal} position at {row['close']:.6f}")
                                    else:
                                        logger.warning(f"❌ Failed to open {signal} position after signal change")
                    
                    # Calculate current equity
                    position_value = 0
                    if self.current_position:
                        position_value = self.current_position.unrealized_pnl
                        
                    total_equity = self.current_balance + position_value
                    
                    # Add to equity curve
                    self.results.add_equity_point(
                        timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        self.current_balance,
                        position_value
                    )
                    
                    # Track daily returns
                    date_str = timestamp.strftime('%Y-%m-%d')
                    daily_balances[date_str] = total_equity
                    
                except Exception as e:
                    logger.error(f"Error processing candle at index {i}: {e}")
                    continue
                    
            # Close any remaining position
            if self.current_position:
                self.close_position(df.iloc[-1], "End of Backtest")
                
            # Calculate daily returns
            dates = sorted(daily_balances.keys())
            for i in range(1, len(dates)):
                prev_balance = daily_balances[dates[i-1]]
                curr_balance = daily_balances[dates[i]]
                if prev_balance > 0:
                    daily_return = (curr_balance - prev_balance) / prev_balance
                    self.results.daily_returns.append(daily_return)
                    
            # Finalize results
            self.results.final_balance = self.current_balance
            
            # Calculate metrics
            metrics = self.results.calculate_metrics()
            
            # Add auto compounding statistics
            metrics.update({
                'compounding_factor': self.current_balance / self.initial_balance,
                'total_compounded_profit': self.total_compounded_profit,
                'peak_balance': self.peak_balance,
                'balance_growth_factor': self.current_balance / self.initial_balance,
                'avg_balance_per_trade': sum([h['new_balance'] for h in self.balance_history]) / len(self.balance_history) if self.balance_history else self.initial_balance
            })
            
            # Validate performance
            validation_results = self.validate_backtest_performance(metrics)
            
            logger.info("Backtest completed successfully")
            logger.info("="*60)
            logger.info("📊 AUTO COMPOUNDING RESULTS:")
            logger.info(f"💰 Initial Balance: ${self.initial_balance:.2f}")
            logger.info(f"💰 Final Balance: ${self.current_balance:.2f}")
            logger.info(f"📈 Compounding Factor: {metrics['compounding_factor']:.2f}x")
            logger.info(f"💵 Total Compounded Profit: ${self.total_compounded_profit:.2f}")
            logger.info(f"⬆️ Peak Balance: ${self.peak_balance:.2f}")
            logger.info("="*60)
            logger.info(f"Total trades: {metrics['total_trades']}")
            logger.info(f"Win rate: {metrics['win_rate']:.2f}%")
            logger.info(f"Total return: {metrics['total_return']:.2f}%")
            logger.info(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
            logger.info(f"Ignored duplicate signals - BUY: {self.ignored_signals['BUY']}, SELL: {self.ignored_signals['SELL']}")
            logger.info(f"Validation score: {validation_results['score']}/100")
            
            # Add validation results to metrics
            metrics['validation'] = validation_results
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during backtest execution: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    def save_results(self, results: Dict) -> str:
        """Save backtest results to files"""
        try:
            # Create output directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'backtest_results',
                f"{self.symbol}_{self.strategy_name}_{timestamp}"
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Save trades
            if self.results.trades:
                trades_df = pd.DataFrame(self.results.trades)
                trades_file = os.path.join(output_dir, 'trades.csv')
                trades_df.to_csv(trades_file, index=False)
                
            # Save equity curve
            if self.results.equity_curve:
                equity_df = pd.DataFrame(self.results.equity_curve)
                equity_file = os.path.join(output_dir, 'equity_curve.csv')
                equity_df.to_csv(equity_file, index=False)
                
            # Save auto compounding history
            if self.balance_history:
                compounding_df = pd.DataFrame(self.balance_history)
                compounding_file = os.path.join(output_dir, 'compounding_history.csv')
                compounding_df.to_csv(compounding_file, index=False)
                logger.info(f"💰 Compounding history saved to: compounding_history.csv")
                
            # Save results summary
            results_file = os.path.join(output_dir, 'results.json')
            with open(results_file, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_results = {}
                for key, value in results.items():
                    if isinstance(value, (np.integer, np.floating)):
                        json_results[key] = value.item()
                    elif isinstance(value, np.ndarray):
                        json_results[key] = value.tolist()
                    else:
                        json_results[key] = value
                json.dump(json_results, f, indent=2, default=str)
                
            # Generate plots
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            self._generate_equity_curve_plot(plots_dir)
            self._generate_drawdown_plot(plots_dir)
            self._generate_trade_analysis_plots(plots_dir)
            self._generate_compounding_plot(plots_dir)
            
            logger.info(f"Results saved to: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None
            
    def _generate_equity_curve_plot(self, plots_dir: str):
        """Generate equity curve plot"""
        try:
            if not self.results.equity_curve:
                return
                
            equity_df = pd.DataFrame(self.results.equity_curve)
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            
            plt.figure(figsize=(12, 6))
            plt.plot(equity_df['timestamp'], equity_df['total_equity'], label='Total Equity', linewidth=2)
            plt.plot(equity_df['timestamp'], equity_df['balance'], label='Cash Balance', alpha=0.7)
            
            plt.title(f'Equity Curve - {self.symbol} ({self.strategy_name})')
            plt.xlabel('Date')
            plt.ylabel('Balance (USDT)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_path = os.path.join(plots_dir, 'equity_curve.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating equity curve plot: {e}")
            
    def _generate_drawdown_plot(self, plots_dir: str):
        """Generate drawdown plot"""
        try:
            if not self.results.equity_curve:
                return
                
            equity_df = pd.DataFrame(self.results.equity_curve)
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            
            # Calculate drawdown
            equity_df['cummax'] = equity_df['total_equity'].cummax()
            equity_df['drawdown'] = (equity_df['total_equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
            
            plt.figure(figsize=(12, 6))
            plt.fill_between(equity_df['timestamp'], equity_df['drawdown'], 0, 
                           alpha=0.3, color='red', label='Drawdown')
            plt.plot(equity_df['timestamp'], equity_df['drawdown'], color='red', linewidth=1)
            
            plt.title(f'Drawdown Analysis - {self.symbol} ({self.strategy_name})')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_path = os.path.join(plots_dir, 'drawdown.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating drawdown plot: {e}")
            
    def _generate_trade_analysis_plots(self, plots_dir: str):
        """Generate trade analysis plots"""
        try:
            if not self.results.trades:
                return
                
            trades_df = pd.DataFrame(self.results.trades)
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # P&L distribution
            ax1.hist(trades_df['pnl'], bins=20, alpha=0.7, edgecolor='black')
            ax1.set_title('P&L Distribution')
            ax1.set_xlabel('P&L (USDT)')
            ax1.set_ylabel('Frequency')
            ax1.axvline(0, color='red', linestyle='--', alpha=0.7)
            
            # Win/Loss ratio
            win_loss_data = ['Wins', 'Losses']
            win_loss_counts = [len(trades_df[trades_df['pnl'] > 0]), len(trades_df[trades_df['pnl'] <= 0])]
            colors = ['green', 'red']
            ax2.pie(win_loss_counts, labels=win_loss_data, colors=colors, autopct='%1.1f%%')
            ax2.set_title('Win/Loss Ratio')
            
            # Trade duration
            ax3.hist(trades_df['duration_hours'], bins=20, alpha=0.7, edgecolor='black')
            ax3.set_title('Trade Duration Distribution')
            ax3.set_xlabel('Duration (Hours)')
            ax3.set_ylabel('Frequency')
            
            # Cumulative returns
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            ax4.plot(range(len(trades_df)), trades_df['cumulative_pnl'], linewidth=2)
            ax4.set_title('Cumulative P&L')
            ax4.set_xlabel('Trade Number')
            ax4.set_ylabel('Cumulative P&L (USDT)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = os.path.join(plots_dir, 'trade_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating trade analysis plots: {e}")
            
    def _generate_compounding_plot(self, plots_dir: str):
        """Generate auto compounding visualization plot"""
        try:
            if not self.balance_history:
                return
                
            compounding_df = pd.DataFrame(self.balance_history)
            
            # Create dual subplot for balance growth and growth factor
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Balance Growth Over Trades
            ax1.plot(compounding_df['trade_number'], compounding_df['new_balance'], 
                    marker='o', linewidth=2, markersize=4, color='green', label='Balance After Trade')
            ax1.axhline(y=self.initial_balance, color='blue', linestyle='--', 
                       alpha=0.7, label=f'Initial Balance (${self.initial_balance:.0f})')
            ax1.set_title(f'Auto Compounding: Balance Growth - {self.symbol} ({self.strategy_name})')
            ax1.set_xlabel('Trade Number')
            ax1.set_ylabel('Balance (USDT)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Highlight profitable vs losing trades
            for _, row in compounding_df.iterrows():
                color = 'green' if row['pnl'] > 0 else 'red'
                alpha = 0.6 if row['pnl'] > 0 else 0.3
                ax1.scatter(row['trade_number'], row['new_balance'], 
                           color=color, alpha=alpha, s=50)
            
            # Plot 2: Growth Factor Over Time
            ax2.plot(compounding_df['trade_number'], compounding_df['growth_factor'], 
                    marker='s', linewidth=2, markersize=4, color='purple', label='Growth Factor')
            ax2.axhline(y=1.0, color='blue', linestyle='--', alpha=0.7, label='Break-even (1.0x)')
            ax2.set_title('Compounding Growth Factor')
            ax2.set_xlabel('Trade Number')
            ax2.set_ylabel('Growth Factor (x)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add annotations for key milestones
            if compounding_df['growth_factor'].max() >= 2.0:
                milestone_2x = compounding_df[compounding_df['growth_factor'] >= 2.0].iloc[0]
                ax2.annotate(f'2x Growth\nTrade #{milestone_2x["trade_number"]}', 
                           xy=(milestone_2x['trade_number'], milestone_2x['growth_factor']),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            plt.tight_layout()
            
            plot_path = os.path.join(plots_dir, 'auto_compounding.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"🔄 Auto compounding plot saved to: auto_compounding.png")
            
        except Exception as e:
            logger.error(f"Error generating compounding plot: {e}")
            
    def generate_summary_report(self, results: Dict, validation: Dict = None) -> str:
        """Generate a comprehensive summary report with validation results"""
        try:
            # Basic performance section
            report = f"""
# Enhanced Backtest Report

## Strategy Configuration
- **Strategy**: {self.strategy_name}
- **Symbol**: {self.symbol}
- **Timeframe**: {self.timeframe}
- **Period**: {results['start_date']} to {results['end_date']}
- **Initial Balance**: {results['initial_balance']:.2f} USDT
- **Leverage**: {self.leverage}x

## Performance Summary
- **Final Balance**: {results['final_balance']:.2f} USDT
- **Total Return**: {results['total_return']:.2f}%
- **Total P&L**: {results['total_pnl']:.2f} USDT

## Auto Compounding Results 🔄
- **Compounding Factor**: {results.get('compounding_factor', 1.0):.2f}x
- **Total Compounded Profit**: {results.get('total_compounded_profit', 0):.2f} USDT
- **Peak Balance**: {results.get('peak_balance', results['final_balance']):.2f} USDT
- **Balance Growth Factor**: {results.get('balance_growth_factor', 1.0):.2f}x
- **Average Balance per Trade**: {results.get('avg_balance_per_trade', results['initial_balance']):.2f} USDT

## Trading Statistics
- **Total Trades**: {results['total_trades']}
- **Winning Trades**: {results['winning_trades']}
- **Losing Trades**: {results['losing_trades']}
- **Win Rate**: {results['win_rate']:.2f}%
- **Profit Factor**: {results['profit_factor']:.2f}
- **Ignored Duplicate Signals**: BUY: {self.ignored_signals['BUY']}, SELL: {self.ignored_signals['SELL']}

## Performance Metrics
- **Average Win**: {results['avg_win']:.2f} USDT
- **Average Loss**: {results['avg_loss']:.2f} USDT
- **Maximum Drawdown**: {results['max_drawdown']:.2f}%
- **Sharpe Ratio**: {results['sharpe_ratio']:.2f}
- **Average Trade Duration**: {results['avg_trade_duration']:.2f} hours

## Risk Management
- **Commission Rate**: {self.commission_rate:.4f} ({self.commission_rate*100:.2f}%)
- **Stop Loss**: {'Enabled' if USE_STOP_LOSS else 'Disabled'}
- **Trailing Stop**: {'Enabled' if TRAILING_STOP else 'Disabled'}
- **Base Position Size**: {self.trade_percentage:.1%} of balance
"""

            # Add validation results if available
            if validation:
                status_emoji = "✅" if validation['passed'] else "❌"
                report += f"""
## Strategy Validation {status_emoji}
- **Validation Status**: {"PASSED" if validation['passed'] else "FAILED"}
- **Performance Score**: {validation['score']}/100
- **Risk Assessment**: {validation['risk_assessment']}
"""
                
                if validation['issues']:
                    report += "\n### Issues Identified:\n"
                    for issue in validation['issues']:
                        report += f"- ⚠️ {issue}\n"
                
                if validation['recommendations']:
                    report += "\n### Recommendations:\n"
                    for rec in validation['recommendations']:
                        report += f"- 💡 {rec}\n"
            
            # Enhanced analysis section
            if results['total_trades'] > 0:
                avg_trade_return = results['total_return'] / results['total_trades']
                risk_reward_ratio = abs(results['avg_win'] / results['avg_loss']) if results['avg_loss'] != 0 else 0
                
                report += f"""
## Enhanced Analysis
- **Average Return per Trade**: {avg_trade_return:.2f}%
- **Risk-Reward Ratio**: {risk_reward_ratio:.2f}:1
- **Expectancy**: {(results['win_rate']/100 * results['avg_win']) + ((1-results['win_rate']/100) * results['avg_loss']):.2f} USDT
- **Kelly Criterion**: {((results['win_rate']/100 * risk_reward_ratio) - (1-results['win_rate']/100)) / risk_reward_ratio * 100 if risk_reward_ratio > 0 else 0:.1f}%
"""
            
            # Trading frequency analysis
            if 'start_date' in results and 'end_date' in results:
                try:
                    start_date = pd.to_datetime(results['start_date'])
                    end_date = pd.to_datetime(results['end_date'])
                    days = (end_date - start_date).days
                    if days > 0:
                        trades_per_day = results['total_trades'] / days
                        report += f"- **Trading Frequency**: {trades_per_day:.2f} trades/day\n"
                except:
                    pass
            
            report += f"""
---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Enhanced Backtesting Engine v2.0*
"""
            return report
            
        except Exception as e:
            logger.error(f"Error generating enhanced summary report: {e}")
            return f"Error generating report: {e}"
        
    def validate_backtest_performance(self, results: Dict) -> Dict:
        """Enhanced backtest validation with comprehensive performance checks"""
        try:
            validation_results = {
                'passed': False,
                'score': 0,
                'issues': [],
                'recommendations': [],
                'risk_assessment': 'HIGH'
            }
            
            # Import validation thresholds
            min_profit = BACKTEST_MIN_PROFIT_PCT
            min_win_rate = BACKTEST_MIN_WIN_RATE
            max_drawdown = BACKTEST_MAX_DRAWDOWN
            min_profit_factor = BACKTEST_MIN_PROFIT_FACTOR
            
            # 1. Profitability Check (30 points)
            profit_score = 0
            if results['total_return'] >= min_profit:
                profit_score = 30
            elif results['total_return'] >= min_profit * 0.7:
                profit_score = 20
            elif results['total_return'] >= 0:
                profit_score = 10
            else:
                validation_results['issues'].append(f"Negative returns: {results['total_return']:.2f}%")
            
            # 2. Win Rate Check (25 points)
            win_rate_score = 0
            if results['win_rate'] >= min_win_rate:
                win_rate_score = 25
            elif results['win_rate'] >= min_win_rate * 0.8:
                win_rate_score = 15
            elif results['win_rate'] >= 30:
                win_rate_score = 10
            else:
                validation_results['issues'].append(f"Low win rate: {results['win_rate']:.2f}%")
            
            # 3. Drawdown Check (25 points)
            drawdown_score = 0
            abs_drawdown = abs(results['max_drawdown'])
            if abs_drawdown <= max_drawdown:
                drawdown_score = 25
            elif abs_drawdown <= max_drawdown * 1.5:
                drawdown_score = 15
            elif abs_drawdown <= max_drawdown * 2:
                drawdown_score = 8
            else:
                validation_results['issues'].append(f"Excessive drawdown: {abs_drawdown:.2f}%")
            
            # 4. Profit Factor Check (20 points)
            pf_score = 0
            if results['profit_factor'] >= min_profit_factor:
                pf_score = 20
            elif results['profit_factor'] >= 1.0:
                pf_score = 10
            elif results['profit_factor'] >= 0.8:
                pf_score = 5
            else:
                validation_results['issues'].append(f"Poor profit factor: {results['profit_factor']:.2f}")
            
            # Calculate total score
            total_score = profit_score + win_rate_score + drawdown_score + pf_score
            validation_results['score'] = total_score
            
            # Additional checks
            trade_count = results['total_trades']
            avg_trade_duration = results.get('avg_trade_duration', 0)
            
            # Check for overtrading
            if trade_count > 100:  # More than 100 trades in backtest period
                validation_results['issues'].append(f"Potential overtrading: {trade_count} trades")
            
            # Check for extremely short trades (likely noise)
            if avg_trade_duration < 0.5:  # Less than 30 minutes
                validation_results['issues'].append(f"Very short trades: {avg_trade_duration:.2f} hours avg")
            
            # Check for insufficient data
            if trade_count < 10:
                validation_results['issues'].append(f"Insufficient trade data: {trade_count} trades")
            
            # Risk assessment
            if total_score >= 80:
                validation_results['risk_assessment'] = 'LOW'
                validation_results['passed'] = True
            elif total_score >= 60:
                validation_results['risk_assessment'] = 'MEDIUM'
                validation_results['passed'] = True
            elif total_score >= 40:
                validation_results['risk_assessment'] = 'HIGH'
            else:
                validation_results['risk_assessment'] = 'VERY HIGH'
            
            # Generate recommendations
            if profit_score < 20:
                validation_results['recommendations'].append("Improve signal quality and entry timing")
            if win_rate_score < 15:
                validation_results['recommendations'].append("Enhance signal filtering to reduce false positives")
            if drawdown_score < 15:
                validation_results['recommendations'].append("Implement better position sizing and stop losses")
            if pf_score < 10:
                validation_results['recommendations'].append("Optimize risk-reward ratio and exit strategy")
            if trade_count > 100:
                validation_results['recommendations'].append("Reduce trade frequency with stricter entry criteria")
            
            # Success message
            if validation_results['passed']:
                logger.info(f"✅ Backtest validation PASSED (Score: {total_score}/100, Risk: {validation_results['risk_assessment']})")
            else:
                logger.warning(f"❌ Backtest validation FAILED (Score: {total_score}/100, Risk: {validation_results['risk_assessment']})")
                logger.warning(f"Issues: {', '.join(validation_results['issues'])}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in backtest validation: {e}")
            return {
                'passed': False,
                'score': 0,
                'issues': [f"Validation error: {str(e)}"],
                'recommendations': ["Fix validation errors before proceeding"],
                'risk_assessment': 'VERY HIGH'
            }