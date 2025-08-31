import ccxt
import pandas as pd
import numpy as np
import time
import datetime
import json
import logging
from typing import Dict, List, Optional

# --- Configuration ---
class Config:
    EXCHANGE = 'binance'  # Use Binance for reliable data
    SYMBOL = 'ETH/USDT'
    TIMEFRAME = '5m'
    INITIAL_BALANCE = 200.0  # USDT starting balance
    
    # Strategy Parameters
    FAST_MA = 50
    SLOW_MA = 200
    RISK_PER_TRADE = 0.02  # 2% risk per trade
    STOP_LOSS_ATR_MULTIPLIER = 2.0
    TRAILING_STOP_PERCENT = 0.10
    
    # Trading Parameters
    COMMISSION_RATE = 0.001  # 0.1% commission
    SLIPPAGE = 0.001  # 0.1% slippage

# --- Realistic Simulation Trading Bot ---
class SimulationTradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        self.setup_exchange()
        self.initialize_state()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_simulation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_exchange(self):
        """Initialize exchange connection for market data only"""
        exchange_config = {
            'enableRateLimit': True,
            'timeout': 30000,
        }
        
        try:
            self.exchange = getattr(ccxt, self.config.EXCHANGE)(exchange_config)
            self.logger.info(f"Connected to {self.config.EXCHANGE} for market data")
        except Exception as e:
            self.logger.error(f"Failed to connect to exchange: {e}")
            raise
    
    def initialize_state(self):
        """Initialize bot state with demo trading"""
        self.balance = {
            'USDT': self.config.INITIAL_BALANCE,
            'ETH': 0.0
        }
        self.current_position = None
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.trade_count = 0
        self.trade_history = []
        self.portfolio_history = []
        self.start_time = datetime.datetime.now()
        
        self.logger.info(f"Initialized simulation with ${self.balance['USDT']:.2f} USDT")
    
    def fetch_ohlcv_data(self, limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV data from exchange"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.config.SYMBOL, 
                self.config.TIMEFRAME, 
                limit=limit
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        if len(df) < self.config.SLOW_MA:
            return {}
        
        # Moving Averages
        fast_ma = df['close'].rolling(window=self.config.FAST_MA).mean().iloc[-1]
        slow_ma = df['close'].rolling(window=self.config.SLOW_MA).mean().iloc[-1]
        
        # ATR for volatility
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean().iloc[-1]
        
        # Crossover signal
        crossover_signal = 1 if fast_ma > slow_ma else -1 if fast_ma < slow_ma else 0
        
        return {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'atr': atr,
            'crossover_signal': crossover_signal,
            'current_price': df['close'].iloc[-1],
            'signal_time': df.index[-1]
        }
    
    def calculate_position_size(self, current_price: float, atr: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = self.balance['USDT'] * self.config.RISK_PER_TRADE
        position_size = risk_amount / (atr * self.config.STOP_LOSS_ATR_MULTIPLIER)
        
        # Convert to amount of cryptocurrency
        crypto_amount = position_size / current_price
        
        # Apply slippage
        crypto_amount *= (1 - self.config.SLIPPAGE)
        
        # Ensure we don't exceed available balance
        max_affordable = self.balance['USDT'] / current_price
        crypto_amount = min(crypto_amount, max_affordable * 0.95)  # 5% buffer
        
        return crypto_amount
    
    def execute_buy(self, amount: float, current_price: float):
        """Execute simulated buy order"""
        try:
            # Calculate costs with commission and slippage
            effective_price = current_price * (1 + self.config.SLIPPAGE)
            cost = amount * effective_price
            commission = cost * self.config.COMMISSION_RATE
            total_cost = cost + commission
            
            if total_cost > self.balance['USDT']:
                self.logger.warning(f"Insufficient balance: ${self.balance['USDT']:.2f} < ${total_cost:.2f}")
                return False
            
            # Update balances
            self.balance['USDT'] -= total_cost
            self.balance['ETH'] += amount
            
            # Record trade
            trade = {
                'type': 'BUY',
                'amount': amount,
                'price': effective_price,
                'cost': cost,
                'commission': commission,
                'timestamp': datetime.datetime.now(),
                'balance_usdt': self.balance['USDT'],
                'balance_eth': self.balance['ETH']
            }
            self.trade_history.append(trade)
            
            self.logger.info(f"SIMULATED BUY: {amount:.6f} ETH @ ${effective_price:.2f}")
            self.logger.info(f"Cost: ${cost:.2f} + Commission: ${commission:.4f}")
            self.logger.info(f"New Balance: ${self.balance['USDT']:.2f} USDT + {self.balance['ETH']:.6f} ETH")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in buy execution: {e}")
            return False
    
    def execute_sell(self, amount: float, current_price: float):
        """Execute simulated sell order"""
        try:
            if amount > self.balance['ETH']:
                self.logger.warning(f"Insufficient ETH: {self.balance['ETH']:.6f} < {amount:.6f}")
                return False
            
            # Calculate proceeds with commission and slippage
            effective_price = current_price * (1 - self.config.SLIPPAGE)
            revenue = amount * effective_price
            commission = revenue * self.config.COMMISSION_RATE
            net_proceeds = revenue - commission
            
            # Update balances
            self.balance['USDT'] += net_proceeds
            self.balance['ETH'] -= amount
            
            # Calculate P&L
            entry_value = 0
            for trade in reversed(self.trade_history):
                if trade['type'] == 'BUY':
                    entry_value = trade['cost'] * (amount / trade['amount'])
                    break
            
            profit = revenue - entry_value - commission
            profit_percent = (profit / entry_value) * 100 if entry_value > 0 else 0
            
            # Record trade
            trade = {
                'type': 'SELL',
                'amount': amount,
                'price': effective_price,
                'revenue': revenue,
                'commission': commission,
                'profit': profit,
                'profit_percent': profit_percent,
                'timestamp': datetime.datetime.now(),
                'balance_usdt': self.balance['USDT'],
                'balance_eth': self.balance['ETH']
            }
            self.trade_history.append(trade)
            
            self.logger.info(f"SIMULATED SELL: {amount:.6f} ETH @ ${effective_price:.2f}")
            self.logger.info(f"Revenue: ${revenue:.2f} - Commission: ${commission:.4f}")
            self.logger.info(f"Profit: ${profit:.2f} ({profit_percent:+.2f}%)")
            self.logger.info(f"New Balance: ${self.balance['USDT']:.2f} USDT + {self.balance['ETH']:.6f} ETH")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in sell execution: {e}")
            return False
    
    def run_strategy(self):
        """Main strategy execution loop"""
        self.logger.info("Starting Realistic Trading Simulation...")
        self.logger.info(f"Symbol: {self.config.SYMBOL}")
        self.logger.info(f"Timeframe: {self.config.TIMEFRAME}")
        self.logger.info(f"Initial Balance: ${self.balance['USDT']:.2f}")
        self.logger.info(f"Commission: {self.config.COMMISSION_RATE*100:.1f}% + Slippage: {self.config.SLIPPAGE*100:.1f}%")
        
        while True:
            try:
                # Fetch market data
                df = self.fetch_ohlcv_data(limit=300)
                if df.empty:
                    self.logger.error("Failed to fetch data")
                    time.sleep(60)
                    continue
                
                # Calculate indicators
                indicators = self.calculate_indicators(df)
                if not indicators:
                    self.logger.warning("Not enough data for indicators")
                    time.sleep(60)
                    continue
                
                current_price = indicators['current_price']
                signal = indicators['crossover_signal']
                
                # Log market data
                self.logger.info(f"Price: ${current_price:.2f} | "
                               f"Fast MA: ${indicators['fast_ma']:.2f} | "
                               f"Slow MA: ${indicators['slow_ma']:.2f} | "
                               f"ATR: ${indicators['atr']:.2f} | "
                               f"Signal: {'BUY' if signal == 1 else 'SELL' if signal == -1 else 'NEUTRAL'}")
                
                # Record portfolio value
                portfolio_value = self.balance['USDT'] + (self.balance['ETH'] * current_price)
                self.portfolio_history.append({
                    'timestamp': datetime.datetime.now(),
                    'value': portfolio_value,
                    'price': current_price
                })
                
                # Trading logic
                if signal == 1 and self.balance['ETH'] == 0:  # Buy signal, no position
                    position_size = self.calculate_position_size(current_price, indicators['atr'])
                    
                    if position_size > 0:
                        self.logger.info(f"Calculated position size: {position_size:.6f} ETH")
                        success = self.execute_buy(position_size, current_price)
                        
                        if success:
                            self.current_position = 'long'
                            self.entry_price = current_price
                            self.stop_loss = current_price * (1 - self.config.STOP_LOSS_ATR_MULTIPLIER * indicators['atr'] / current_price)
                            self.trade_count += 1
                            self.logger.info(f"Stop loss set at: ${self.stop_loss:.2f}")
                
                elif signal == -1 and self.balance['ETH'] > 0:  # Sell signal, has position
                    self.logger.info(f"Executing SELL of {self.balance['ETH']:.6f} ETH")
                    success = self.execute_sell(self.balance['ETH'], current_price)
                    
                    if success:
                        self.current_position = None
                        self.entry_price = 0
                        self.stop_loss = 0
                
                # Check stop loss
                elif self.balance['ETH'] > 0 and current_price <= self.stop_loss:
                    self.logger.info(f"Stop loss triggered at ${current_price:.2f}")
                    success = self.execute_sell(self.balance['ETH'], current_price)
                    if success:
                        self.current_position = None
                        self.entry_price = 0
                        self.stop_loss = 0
                
                # Wait for next candle
                self.logger.info("Waiting for next 5-minute candle...")
                time.sleep(300)  # 5 minutes
                
            except KeyboardInterrupt:
                self.logger.info("Simulation stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(60)
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        if not self.trade_history:
            self.logger.info("No trades executed during simulation")
            return
        
        total_trades = len([t for t in self.trade_history if t['type'] == 'BUY'])
        winning_trades = len([t for t in self.trade_history if t.get('profit', 0) > 0 and t['type'] == 'SELL'])
        losing_trades = total_trades - winning_trades
        
        total_profit = sum(t.get('profit', 0) for t in self.trade_history if t['type'] == 'SELL')
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_profit / losing_trades if losing_trades > 0 else 0
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        final_balance = self.balance['USDT']
        total_return = ((final_balance / self.config.INITIAL_BALANCE) - 1) * 100
        
        print(f"\n{'='*80}")
        print("TRADING SIMULATION PERFORMANCE REPORT")
        print(f"{'='*80}")
        print(f"Simulation Period: {self.start_time} to {datetime.datetime.now()}")
        print(f"Total Runtime: {(datetime.datetime.now() - self.start_time)}")
        print(f"\nBALANCE:")
        print(f"Initial Balance: ${self.config.INITIAL_BALANCE:.2f}")
        print(f"Final Balance: ${final_balance:.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        
        print(f"\nTRADE STATISTICS:")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Profit/Loss: ${total_profit:+.2f}")
        print(f"Average Profit: ${avg_profit:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        
        if self.trade_history:
            print(f"\nLAST 5 TRADES:")
            for trade in self.trade_history[-5:]:
                if trade['type'] == 'BUY':
                    print(f"  BUY: {trade['amount']:.6f} ETH @ ${trade['price']:.2f}")
                else:
                    print(f"  SELL: {trade['amount']:.6f} ETH @ ${trade['price']:.2f} - PnL: ${trade['profit']:+.2f} ({trade['profit_percent']:+.2f}%)")

# --- Main Execution ---
if __name__ == "__main__":
    config = Config()
    bot = SimulationTradingBot(config)
    
    try:
        bot.run_strategy()
    except KeyboardInterrupt:
        bot.logger.info("Simulation stopped by user")
    finally:
        bot.generate_report()