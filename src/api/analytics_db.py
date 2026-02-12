"""
DuckDB Analytics Module

Manages connection to DuckDB and queries for backtest analytics.
Supports querying Parquet files directly.

Task Group: Sprint 6 (Analytics)
"""

import duckdb
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.getcwd(), "data", "backtests")
DB_PATH = os.path.join(os.getcwd(), "data", "analytics.duckdb")

def init_db():
    """Initialize DuckDB and mock data if needed."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check if we have data, if not generate some
    if not os.listdir(DATA_DIR):
        logger.info("No backtest data found. Generating mock parquet files...")
        generate_mock_data()
        
    logger.info(f"Analytics DB initialized. Watching {DATA_DIR}")

def get_connection():
    """Get DuckDB connection."""
    return duckdb.connect(DB_PATH)

def generate_mock_data():
    """Generate mock backtest results in Parquet format for testing."""
    strategies = ["MacdCross", "BollingerBreakout", "RsiMeanReversion"]
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD"]
    
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(10):  # Generate 10 mock backtest runs
        strat = random.choice(strategies)
        sym = random.choice(symbols)
        run_date = start_date + timedelta(days=i*3)
        
        # Generate trades
        trades = []
        balance = 10000.0
        
        for j in range(random.randint(20, 100)):
            entry_time = run_date + timedelta(hours=j*4)
            duration = random.randint(1, 48) # hours
            exit_time = entry_time + timedelta(hours=duration)
            
            direction = random.choice(["buy", "sell"])
            volume = 0.1
            
            # Random pnl based on 'strategy' logic (random walk)
            pnl = random.gauss(10, 50) if random.random() > 0.4 else random.gauss(-15, 40)
            balance += pnl
            
            trades.append({
                "run_id": f"run_{i}",
                "strategy": strat,
                "symbol": sym,
                "timestamp": run_date,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "direction": direction,
                "volume": volume,
                "profit": pnl,
                "balance": balance
            })
            
        df = pd.DataFrame(trades)
        filename = f"{DATA_DIR}/backtest_{min(1000 + i, 9999)}.parquet"
        df.to_parquet(filename)
        logger.info(f"Generated {filename}")

def query_backtests(limit: int = 50):
    """Query recent backtest runs."""
    con = get_connection()
    try:
        # Use read_parquet with pattern
        # Group by run to get summary
        query = f"""
            SELECT 
                run_id, 
                any_value(strategy) as strategy, 
                any_value(symbol) as symbol, 
                min(timestamp) as run_date,
                count(*) as total_trades,
                sum(profit) as total_pnl,
                max(balance) as max_equity,
                min(balance) as min_equity
            FROM '{DATA_DIR}/*.parquet'
            GROUP BY run_id
            ORDER BY run_date DESC
            LIMIT {limit}
        """
        return con.execute(query).df().to_dict(orient='records')
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []
    finally:
        con.close()

def query_trades(run_id: str):
    """Get trades for a specific run."""
    con = get_connection()
    try:
        query = f"""
            SELECT *
            FROM '{DATA_DIR}/*.parquet'
            WHERE run_id = '{run_id}'
            ORDER BY entry_time ASC
        """
        df = con.execute(query).df()
        # Convert timestamps for JSON serialization
        df['entry_time'] = df['entry_time'].astype(str)
        df['exit_time'] = df['exit_time'].astype(str)
        df['timestamp'] = df['timestamp'].astype(str)
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []
    finally:
        con.close()

def run_custom_query(sql: str):
    """Run a custom SQL query (safe mode)."""
    con = get_connection()
    try:
        # Basic protection: ensure reading from data dir or simple selects
        # In prod, this needs strict sandbox. For Copilot, we allow flexibility.
        if "DROP" in sql.upper() or "DELETE" in sql.upper():
            return {"error": "Destructive queries not allowed"}
            
        # Replace simplified table names with parquet paths if needed
        sql = sql.replace("backtests", f"'{DATA_DIR}/*.parquet'")
        
        df = con.execute(sql).df()
        return df.to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}
    finally:
        con.close()
