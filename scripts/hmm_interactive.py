#!/usr/bin/env python3
"""
QuantMindX — Regime Detection Training Console
================================================

Interactive CLI for training all regime detection models:
  - HMM (Hidden Markov Model)
  - MS-GARCH (Markov-Switching GARCH)
  - BOCPD (Bayesian Online Changepoint Detection)
  - Ensemble (combined voter)

Usage:
    python3 scripts/hmm_interactive.py

Requirements:
    pip install hmmlearn arch histdata yfinance pandas numpy pyarrow
"""

import sys
import os
import json
import pickle
import time
import zipfile
import logging
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*converging.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("regime-console")

# Silence noisy libraries during grid search
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)
logging.getLogger("hmmlearn.base").setLevel(logging.CRITICAL)
logging.getLogger("hmmlearn.hmm").setLevel(logging.CRITICAL)
logging.getLogger("arch").setLevel(logging.CRITICAL)
logging.getLogger("statsmodels").setLevel(logging.CRITICAL)


class _QuietTraining:
    """Context manager to suppress logger noise but allow stderr progress."""
    def __init__(self, label: str = "Training"):
        self._label = label
        self._prev_levels = {}
        self._noisy = ["hmmlearn", "hmmlearn.base", "hmmlearn.hmm",
                        "arch", "arch.univariate", "statsmodels",
                        "hmm-console", "regime-console",
                        "hmm_inference_server",
                        "src.risk.physics.bocpd",
                        "src.risk.physics.bocpd.detector",
                        "src.risk.physics.bocpd.hazard",
                        "src.risk.physics.bocpd.observation",
                        "src.risk.physics.msgarch",
                        "src.risk.physics.msgarch.trainer",
                        "src.risk.physics.msgarch.models"]

    def __enter__(self):
        for name in self._noisy:
            lg = logging.getLogger(name)
            self._prev_levels[name] = lg.level
            lg.setLevel(logging.CRITICAL)
        trainer_lg = logging.getLogger("src.risk.physics.hmm.trainer")
        self._prev_levels["trainer"] = trainer_lg.level
        trainer_lg.setLevel(logging.CRITICAL)
        print(f"  {self._label}")
        self._t0 = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self._t0
        # Clear any leftover progress line
        sys.stderr.write("\r" + " " * 60 + "\r")
        sys.stderr.flush()
        print(f"  {C.GREEN}done{C.RESET} ({elapsed:.0f}s)")
        for name, level in self._prev_levels.items():
            logging.getLogger(name).setLevel(level)

# Paths
CONFIG_PATH = PROJECT_ROOT / "config" / "hmm_config.json"
HMM_MODEL_DIR = PROJECT_ROOT / "models" / "hmm"
MSGARCH_MODEL_DIR = PROJECT_ROOT / "models" / "msgarch"
BOCPD_MODEL_DIR = PROJECT_ROOT / "models" / "bocpd"
DATA_DIR = PROJECT_ROOT / "data"
M5_DIR = DATA_DIR / "real_m5_multi_year"
M1_DIR = DATA_DIR / "histdata_m1"

# histdata.com supported pairs
HISTDATA_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "EURGBP", "EURJPY",
    "GBPJPY", "USDCHF", "USDCAD", "NZDUSD", "CADJPY", "AUDJPY"
]


class C:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def banner():
    print(f"""
{C.CYAN}╔══════════════════════════════════════════════════════════╗
║      {C.BOLD}QuantMindX Regime Detection Training Console{C.RESET}{C.CYAN}     ║
║        HMM  ·  MS-GARCH  ·  BOCPD  ·  Ensemble          ║
╚══════════════════════════════════════════════════════════╝{C.RESET}
""")


def menu():
    print(f"""
{C.BOLD}DATA:{C.RESET}
  {C.GREEN}1{C.RESET}   Download data            (histdata.com M1 → M5, yfinance gold)
  {C.GREEN}2{C.RESET}   Show data status          (cached data + models)

{C.BOLD}HMM:{C.RESET}
  {C.GREEN}10{C.RESET}  Train HMM (all symbols)   (adaptive grid search)
  {C.GREEN}11{C.RESET}  Train HMM (single)        (pick one pair)
  {C.GREEN}12{C.RESET}  Train HMM premium         (LONDON_OPEN, LDN_NY, TKY_LDN)
  {C.GREEN}13{C.RESET}  Train HMM full suite      (universal + all premium sessions)
  {C.GREEN}14{C.RESET}  Train HMM (new only)      (skip pairs that already have models)

{C.BOLD}MS-GARCH:{C.RESET}
  {C.MAGENTA}20{C.RESET}  Train MS-GARCH (all)      (regime-switching volatility)
  {C.MAGENTA}21{C.RESET}  Train MS-GARCH (single)   (pick one pair)
  {C.MAGENTA}22{C.RESET}  Train MS-GARCH premium    (premium sessions only)

{C.BOLD}BOCPD:{C.RESET}
  {C.BLUE}30{C.RESET}  Calibrate BOCPD (all)     (changepoint detection — fast)
  {C.BLUE}31{C.RESET}  Calibrate BOCPD (single)  (pick one pair)

{C.BOLD}ENSEMBLE:{C.RESET}
  {C.CYAN}40{C.RESET}  {C.BOLD}Train ALL models{C.RESET}          (HMM + MS-GARCH + BOCPD for all pairs)
  {C.CYAN}41{C.RESET}  Test ensemble              (load all models, run predictions)

{C.BOLD}MANAGE:{C.RESET}
  {C.GREEN}50{C.RESET}  View training report
  {C.GREEN}51{C.RESET}  Evaluate model             (load .pkl, show regimes + routing)
  {C.GREEN}52{C.RESET}  Config                     (view hmm_config.json)
  {C.GREEN}53{C.RESET}  Export for deploy           (package models for server)
  {C.YELLOW}54{C.RESET}  Cleanup old models         (keep latest per symbol, remove rest)

  {C.GREEN}q{C.RESET}   Quit
""")


# ═══════════════════════════════════════════════════════════════════════
# DATA COMMANDS
# ═══════════════════════════════════════════════════════════════════════
def cmd_download():
    print(f"\n{C.BOLD}── Download Market Data ──{C.RESET}")
    print(f"Source: histdata.com (M1 → M5 resample)")
    print(f"Available pairs: {', '.join(HISTDATA_PAIRS)}")
    print(f"Gold (XAUUSD) uses yfinance H1\n")

    default_pairs = "EURUSD,GBPUSD,USDJPY,AUDUSD,EURJPY,GBPJPY,EURGBP,USDCHF,XAUUSD,XAGUSD"
    pairs_input = input(f"Pairs [{C.DIM}{default_pairs}{C.RESET}]: ").strip()
    if not pairs_input:
        pairs_input = default_pairs
    pairs = [p.strip().upper() for p in pairs_input.split(",")]

    years_input = input(f"Years [{C.DIM}2020,2021,2022,2023,2024{C.RESET}]: ").strip()
    if not years_input:
        years_input = "2020,2021,2022,2023,2024"
    years = [y.strip() for y in years_input.split(",")]

    print(f"\n{C.CYAN}Downloading {len(pairs)} pairs × {len(years)} years...{C.RESET}\n")
    M1_DIR.mkdir(parents=True, exist_ok=True)
    M5_DIR.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        if pair == "XAUUSD":
            _download_gold()
        elif pair == "XAGUSD":
            _download_silver()
        elif pair in HISTDATA_PAIRS:
            _download_histdata_pair(pair, years)
        else:
            print(f"  {C.YELLOW}⚠ {pair} not supported, skipping{C.RESET}")

    print(f"\n{C.GREEN}✓ Download complete{C.RESET}")
    _show_data_status()


def _download_gold():
    try:
        import yfinance as yf
        print(f"  XAUUSD: downloading from yfinance (H1, ~2yr)...")
        data = yf.download("GC=F", period="730d", interval="1h", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        df = pd.DataFrame({
            "time": data.index, "open": data["Open"].values,
            "high": data["High"].values, "low": data["Low"].values,
            "close": data["Close"].values,
            "tick_volume": data.get("Volume", pd.Series(np.zeros(len(data)))).values
        }).dropna(subset=["close"])
        df.to_parquet(M5_DIR / "XAUUSD_H1.parquet", index=False)
        print(f"  {C.GREEN}✓ XAUUSD: {len(df):,} H1 bars{C.RESET}")
    except Exception as e:
        print(f"  {C.RED}✗ XAUUSD failed: {e}{C.RESET}")


def _download_silver():
    try:
        import yfinance as yf
        print(f"  XAGUSD: downloading from yfinance (H1, ~2yr)...")
        data = yf.download("SI=F", period="730d", interval="1h", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        df = pd.DataFrame({
            "time": data.index, "open": data["Open"].values,
            "high": data["High"].values, "low": data["Low"].values,
            "close": data["Close"].values,
            "tick_volume": data.get("Volume", pd.Series(np.zeros(len(data)))).values
        }).dropna(subset=["close"])
        df.to_parquet(M5_DIR / "XAGUSD_H1.parquet", index=False)
        print(f"  {C.GREEN}✓ XAGUSD: {len(df):,} H1 bars{C.RESET}")
    except Exception as e:
        print(f"  {C.RED}✗ XAGUSD failed: {e}{C.RESET}")


def _download_histdata_pair(pair: str, years: List[str]):
    from histdata import download_hist_data
    from histdata.api import Platform as P, TimeFrame as TF

    all_dfs = []
    for year in years:
        zip_name = f"DAT_ASCII_{pair}_M1_{year}.zip"
        zip_path = M1_DIR / zip_name

        if not zip_path.exists():
            try:
                print(f"  {pair} {year}: downloading...", end=" ", flush=True)
                download_hist_data(
                    year=year, pair=pair.lower(),
                    platform=P.GENERIC_ASCII, time_frame=TF.ONE_MINUTE,
                    output_directory=str(M1_DIR)
                )
                print(f"{C.GREEN}ok{C.RESET}")
                time.sleep(0.5)
            except Exception as e:
                print(f"{C.RED}failed: {e}{C.RESET}")
                continue
        else:
            print(f"  {pair} {year}: {C.DIM}cached{C.RESET}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, sep=';', header=None,
                                     names=['time', 'open', 'high', 'low', 'close', 'volume'])
                    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d %H%M%S')
                    all_dfs.append(df)
        except Exception as e:
            print(f"  {C.RED}  parse error: {e}{C.RESET}")

    if not all_dfs:
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values('time').drop_duplicates(subset='time').reset_index(drop=True)
    combined = combined.set_index('time')
    m5 = combined.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()
    m5.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']

    m5.to_parquet(M5_DIR / f"{pair}_M5.parquet", index=False)
    date_range = f"{m5['time'].iloc[0].date()} → {m5['time'].iloc[-1].date()}"
    print(f"  {C.GREEN}✓ {pair}: {len(m5):,} M5 bars ({date_range}){C.RESET}")


# ═══════════════════════════════════════════════════════════════════════
# HMM COMMANDS (10-13)
# ═══════════════════════════════════════════════════════════════════════
def cmd_hmm_train_all():
    print(f"\n{C.BOLD}── Train HMM (All Symbols) ──{C.RESET}")
    from src.risk.physics.hmm.trainer import HMMTrainer

    available = _find_available_data()
    if not available:
        print(f"{C.RED}No data found. Run download first.{C.RESET}")
        return

    print(f"Found data for: {', '.join(available.keys())}")
    proceed = input(f"Train all? [Y/n]: ").strip().lower()
    if proceed == 'n':
        return

    trainer = HMMTrainer(model_dir=HMM_MODEL_DIR)
    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = []

    for symbol, info in available.items():
        df = pd.read_parquet(info["path"])
        with _QuietTraining(f"HMM {symbol} ({len(df):,} bars, 81 configs)"):
            result = trainer.train(symbol, info["tf"], df, version=version)
        passed = result.get("anti_overfit_passed", False)
        ns = result.get("n_states", "?")
        gap = result.get("per_sample_gap_pct", "?")
        status = f"{C.GREEN}PASS{C.RESET}" if passed else f"{C.YELLOW}WARN{C.RESET}"
        print(f"    {status} {ns} states, gap={gap}%")
        results.append(result)

    _save_training_report("hmm", version, results)
    _print_results_table(results)


def cmd_hmm_train_single():
    print(f"\n{C.BOLD}── Train HMM (Single Symbol) ──{C.RESET}")
    available = _find_available_data()
    if not available:
        print(f"{C.RED}No data. Run download first.{C.RESET}")
        return

    symbol, info = _pick_symbol(available)
    if not symbol:
        return

    states_input = input(f"States to try [{C.DIM}3,4,5{C.RESET}]: ").strip()
    n_states_list = [int(x) for x in states_input.split(",")] if states_input else None

    from src.risk.physics.hmm.trainer import HMMTrainer
    trainer = HMMTrainer(model_dir=HMM_MODEL_DIR)
    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    df = pd.read_parquet(info["path"])
    result = trainer.train(symbol, info["tf"], df, n_states_list=n_states_list, version=version)
    _print_results_table([result])


def cmd_hmm_train_premium():
    print(f"\n{C.BOLD}── Train HMM Premium Sessions ──{C.RESET}")
    print(f"  Premium: TOKYO_LONDON_OVERLAP, LONDON_OPEN, LONDON_NY_OVERLAP")
    print(f"  + Combined PREMIUM model\n")

    from src.risk.physics.hmm.trainer import HMMTrainer

    available = _find_available_data()
    if not available:
        print(f"{C.RED}No data. Run download first.{C.RESET}")
        return

    pairs = _pick_pairs(available)
    if not pairs:
        return

    trainer = HMMTrainer(model_dir=HMM_MODEL_DIR)
    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_results = []

    for symbol in pairs:
        info = available[symbol]
        df = pd.read_parquet(info["path"])
        print(f"\n{C.BOLD}{symbol}{C.RESET} ({len(df):,} bars)")
        result = trainer.train_premium_sessions(symbol, info["tf"], df, version=version)
        all_results.append(result)
        for session in ["TOKYO_LONDON_OVERLAP", "LONDON_OPEN", "LONDON_NY_OVERLAP"]:
            result = trainer.train_session(symbol, info["tf"], df, session, version=version)
            all_results.append(result)

    _print_session_results(all_results)


def cmd_hmm_train_full_suite():
    print(f"\n{C.BOLD}── Train HMM Full Suite ──{C.RESET}")
    print(f"  Per symbol: Universal + Premium + 3 session-specific\n")

    from src.risk.physics.hmm.trainer import HMMTrainer

    available = _find_available_data()
    if not available:
        print(f"{C.RED}No data. Run download first.{C.RESET}")
        return

    pairs = _pick_pairs(available)
    if not pairs:
        return

    trainer = HMMTrainer(model_dir=HMM_MODEL_DIR)
    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_results = []

    for symbol in pairs:
        info = available[symbol]
        df = pd.read_parquet(info["path"])
        print(f"\n{'='*60}")
        print(f"  {C.BOLD}{symbol}{C.RESET} ({len(df):,} bars)")
        results = trainer.train_all_sessions(
            symbol, info["tf"], df,
            sessions=["UNIVERSAL", "PREMIUM_COMBINED",
                       "TOKYO_LONDON_OVERLAP", "LONDON_OPEN", "LONDON_NY_OVERLAP"],
            version=version
        )
        all_results.extend(results)

    _save_training_report("hmm", version, all_results)
    _print_session_results(all_results)


def cmd_hmm_train_new_only():
    """Train HMM only for pairs that don't have a model yet."""
    print(f"\n{C.BOLD}── Train HMM (New Pairs Only) ──{C.RESET}")
    print(f"  Skips symbols that already have trained HMM models\n")

    from src.risk.physics.hmm.trainer import HMMTrainer

    available = _find_available_data()
    if not available:
        print(f"{C.RED}No data found. Run download first.{C.RESET}")
        return

    # Find symbols that already have HMM models
    existing = set()
    if HMM_MODEL_DIR.exists():
        for pkl in HMM_MODEL_DIR.glob("*_M5_*.pkl"):
            sym = pkl.stem.split("_")[0]
            existing.add(sym)
        for pkl in HMM_MODEL_DIR.glob("*_H1_*.pkl"):
            sym = pkl.stem.split("_")[0]
            existing.add(sym)

    new_pairs = {s: info for s, info in available.items() if s not in existing}

    if not new_pairs:
        print(f"  {C.GREEN}All pairs already have HMM models!{C.RESET}")
        print(f"  Existing: {', '.join(sorted(existing))}")
        print(f"  Use command 10 to retrain all, or 11 to retrain a specific pair.")
        return

    print(f"  {C.DIM}Already trained:{C.RESET} {', '.join(sorted(existing))}")
    print(f"  {C.BOLD}Need training:{C.RESET}  {', '.join(sorted(new_pairs.keys()))}")
    proceed = input(f"\n  Train {len(new_pairs)} new pairs? [Y/n]: ").strip().lower()
    if proceed == 'n':
        return

    trainer = HMMTrainer(model_dir=HMM_MODEL_DIR)
    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = []

    for symbol, info in new_pairs.items():
        df = pd.read_parquet(info["path"])
        with _QuietTraining(f"HMM {symbol} ({len(df):,} bars, 81 configs)"):
            result = trainer.train(symbol, info["tf"], df, version=version)
        passed = result.get("anti_overfit_passed", False)
        ns = result.get("n_states", "?")
        gap = result.get("per_sample_gap_pct", "?")
        status = f"{C.GREEN}PASS{C.RESET}" if passed else f"{C.YELLOW}WARN{C.RESET}"
        print(f"    {status} {ns} states, gap={gap}%")
        results.append(result)

    _save_training_report("hmm", version, results)
    _print_hmm_results(results)


# ═══════════════════════════════════════════════════════════════════════
# MS-GARCH COMMANDS (20-22)
# ═══════════════════════════════════════════════════════════════════════
def cmd_msgarch_train_all():
    print(f"\n{C.BOLD}── Train MS-GARCH (All Symbols) ──{C.RESET}")
    print(f"  Regime-switching GARCH(1,1) volatility model\n")

    from src.risk.physics.msgarch.trainer import MSGARCHTrainer

    available = _find_available_data()
    if not available:
        print(f"{C.RED}No data. Run download first.{C.RESET}")
        return

    print(f"Found data for: {', '.join(available.keys())}")
    proceed = input(f"Train all? [Y/n]: ").strip().lower()
    if proceed == 'n':
        return

    trainer = MSGARCHTrainer(model_dir=MSGARCH_MODEL_DIR)
    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = []

    for symbol, info in available.items():
        df = pd.read_parquet(info["path"])
        with _QuietTraining(f"{C.MAGENTA}MS-GARCH {symbol}{C.RESET} ({len(df):,} bars)"):
            result = trainer.train(symbol, info["tf"], df, version=version)
        results.append(result)

    _save_training_report("msgarch", version, results)
    _print_msgarch_results(results)


def cmd_msgarch_train_single():
    print(f"\n{C.BOLD}── Train MS-GARCH (Single) ──{C.RESET}")
    available = _find_available_data()
    if not available:
        print(f"{C.RED}No data. Run download first.{C.RESET}")
        return

    symbol, info = _pick_symbol(available)
    if not symbol:
        return

    from src.risk.physics.msgarch.trainer import MSGARCHTrainer
    trainer = MSGARCHTrainer(model_dir=MSGARCH_MODEL_DIR)
    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    df = pd.read_parquet(info["path"])
    with _QuietTraining(f"{C.MAGENTA}MS-GARCH {symbol}{C.RESET} ({len(df):,} bars)"):
        result = trainer.train(symbol, info["tf"], df, version=version)
    _print_msgarch_results([result])


def cmd_msgarch_train_premium():
    print(f"\n{C.BOLD}── Train MS-GARCH Premium Sessions ──{C.RESET}")
    from src.risk.physics.msgarch.trainer import MSGARCHTrainer

    available = _find_available_data()
    if not available:
        print(f"{C.RED}No data. Run download first.{C.RESET}")
        return

    pairs = _pick_pairs(available)
    if not pairs:
        return

    trainer = MSGARCHTrainer(model_dir=MSGARCH_MODEL_DIR)
    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_results = []

    for symbol in pairs:
        info = available[symbol]
        df = pd.read_parquet(info["path"])
        print(f"\n{C.MAGENTA}{symbol}{C.RESET} ({len(df):,} bars)")
        result = trainer.train_premium_sessions(symbol, info["tf"], df, version=version)
        all_results.append(result)
        for session in ["TOKYO_LONDON_OVERLAP", "LONDON_OPEN", "LONDON_NY_OVERLAP"]:
            result = trainer.train_session(symbol, info["tf"], df, session, version=version)
            all_results.append(result)

    _print_session_results(all_results)


# ═══════════════════════════════════════════════════════════════════════
# BOCPD COMMANDS (30-31)
# ═══════════════════════════════════════════════════════════════════════
def cmd_bocpd_calibrate_all():
    print(f"\n{C.BOLD}── Calibrate BOCPD (All Symbols) ──{C.RESET}")
    print(f"  Bayesian Online Changepoint Detection")
    print(f"  No heavy training — just calibrating hazard rate\n")

    from src.risk.physics.bocpd.detector import calibrate_for_symbol

    available = _find_available_data()
    if not available:
        print(f"{C.RED}No data. Run download first.{C.RESET}")
        return

    BOCPD_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for symbol, info in available.items():
        df = pd.read_parquet(info["path"])
        print(f"\n{C.BLUE}{symbol}{C.RESET} ({len(df):,} bars) calibrating...", end=" ", flush=True)
        try:
            t0 = time.time()
            # Suppress BOCPD logger noise during calibration
            _bocpd_loggers = [
                "src.risk.physics.bocpd", "src.risk.physics.bocpd.detector",
                "src.risk.physics.bocpd.hazard", "src.risk.physics.bocpd.observation",
            ]
            _saved = {}
            for _ln in _bocpd_loggers:
                _lg = logging.getLogger(_ln)
                _saved[_ln] = _lg.level
                _lg.setLevel(logging.CRITICAL)

            detector = calibrate_for_symbol(symbol, info["tf"], df,
                                             model_dir=BOCPD_MODEL_DIR)

            for _ln, _lv in _saved.items():
                logging.getLogger(_ln).setLevel(_lv)

            elapsed = time.time() - t0
            model_info = detector.get_model_info()
            hazard_lam = model_info.get("hazard", {}).get("lambda", "?")
            print(f"{C.GREEN}done{C.RESET} ({elapsed:.1f}s, λ={hazard_lam})")
            results.append({
                "symbol": symbol, "timeframe": info["tf"],
                "hazard_lambda": hazard_lam,
                "elapsed": round(elapsed, 1),
                "passed": True,
            })
        except Exception as e:
            print(f"{C.RED}failed: {e}{C.RESET}")
            results.append({"symbol": symbol, "passed": False, "error": str(e)})

    _print_bocpd_results(results)


def cmd_bocpd_calibrate_single():
    print(f"\n{C.BOLD}── Calibrate BOCPD (Single) ──{C.RESET}")
    from src.risk.physics.bocpd.detector import calibrate_for_symbol

    available = _find_available_data()
    if not available:
        print(f"{C.RED}No data. Run download first.{C.RESET}")
        return

    symbol, info = _pick_symbol(available)
    if not symbol:
        return

    BOCPD_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(info["path"])
    print(f"\n{C.BLUE}Calibrating {symbol}...{C.RESET}")
    detector = calibrate_for_symbol(symbol, info["tf"], df, model_dir=BOCPD_MODEL_DIR)
    model_info = detector.get_model_info()
    print(f"\n  {C.GREEN}✓ Calibrated{C.RESET}")
    print(f"  Hazard λ:     {model_info.get('hazard', {}).get('lambda', '?')}")
    print(f"  Observation:  {model_info.get('observation', {}).get('type', '?')}")
    print(f"  Threshold:    {model_info.get('threshold', '?')}")


# ═══════════════════════════════════════════════════════════════════════
# ENSEMBLE COMMANDS (40-41)
# ═══════════════════════════════════════════════════════════════════════
def cmd_train_all_models():
    print(f"\n{C.CYAN}{C.BOLD}══════════════════════════════════════════{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}  FULL ENSEMBLE TRAINING — ALL MODELS{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}══════════════════════════════════════════{C.RESET}")
    print(f"\n  This will train for each symbol:")
    print(f"    {C.GREEN}1. HMM{C.RESET}      — Hidden Markov Model (adaptive grid)")
    print(f"    {C.MAGENTA}2. MS-GARCH{C.RESET} — Regime-switching volatility")
    print(f"    {C.BLUE}3. BOCPD{C.RESET}    — Changepoint detection (calibrate)")
    print()

    available = _find_available_data()
    if not available:
        print(f"{C.RED}No data. Run download first.{C.RESET}")
        return

    pairs = _pick_pairs(available)
    if not pairs:
        return

    est_time = len(pairs) * 5
    print(f"\n  {C.CYAN}Estimated: ~{est_time} minutes for {len(pairs)} pairs{C.RESET}")
    proceed = input(f"  Proceed? [Y/n]: ").strip().lower()
    if proceed == 'n':
        return

    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    hmm_results = []
    msgarch_results = []
    bocpd_results = []

    from src.risk.physics.hmm.trainer import HMMTrainer
    from src.risk.physics.msgarch.trainer import MSGARCHTrainer
    from src.risk.physics.bocpd.detector import calibrate_for_symbol

    hmm_trainer = HMMTrainer(model_dir=HMM_MODEL_DIR)
    msgarch_trainer = MSGARCHTrainer(model_dir=MSGARCH_MODEL_DIR)
    BOCPD_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for symbol in pairs:
        info = available[symbol]
        df = pd.read_parquet(info["path"])
        tf = info["tf"]

        print(f"\n{'='*60}")
        print(f"  {C.BOLD}{symbol}{C.RESET} ({len(df):,} bars, {tf})")
        print(f"{'='*60}")

        # HMM
        try:
            with _QuietTraining(f"{C.GREEN}[HMM]{C.RESET}      81 configs"):
                result = hmm_trainer.train(symbol, tf, df, version=version)
            hmm_results.append(result)
            passed = result.get("anti_overfit_passed", False)
            status = f"{C.GREEN}PASS{C.RESET}" if passed else f"{C.YELLOW}WARN{C.RESET}"
            print(f"             {status} {result.get('n_states', '?')} states, "
                  f"gap={result.get('per_sample_gap_pct', '?')}%")
        except Exception as e:
            print(f"  {C.GREEN}[HMM]{C.RESET}      {C.RED}FAIL: {e}{C.RESET}")
            hmm_results.append({"symbol": symbol, "passed": False})

        # MS-GARCH
        try:
            with _QuietTraining(f"{C.MAGENTA}[MS-GARCH]{C.RESET} fitting"):
                result = msgarch_trainer.train(symbol, tf, df, version=version)
            msgarch_results.append(result)
            passed = result.get("anti_overfit_passed", False)
            status = f"{C.GREEN}PASS{C.RESET}" if passed else f"{C.YELLOW}WARN{C.RESET}"
            print(f"             {status} gap={result.get('gap_pct', '?')}%")
        except Exception as e:
            print(f"  {C.MAGENTA}[MS-GARCH]{C.RESET} {C.RED}FAIL: {e}{C.RESET}")
            msgarch_results.append({"symbol": symbol, "passed": False})

        # BOCPD
        try:
            with _QuietTraining(f"{C.BLUE}[BOCPD]{C.RESET}    calibrating"):
                detector = calibrate_for_symbol(symbol, tf, df, model_dir=BOCPD_MODEL_DIR)
            lam = detector.get_model_info().get("hazard", {}).get("lambda", "?")
            print(f"             {C.GREEN}OK{C.RESET} λ={lam}")
            bocpd_results.append({"symbol": symbol, "hazard_lambda": lam, "passed": True})
        except Exception as e:
            print(f"  {C.BLUE}[BOCPD]{C.RESET}    {C.RED}FAIL: {e}{C.RESET}")
            bocpd_results.append({"symbol": symbol, "passed": False})

    # Summary
    print(f"\n{'='*60}")
    print(f"  {C.BOLD}TRAINING SUMMARY{C.RESET}")
    print(f"{'='*60}")

    _save_training_report("hmm", version, hmm_results)
    _save_training_report("msgarch", version, msgarch_results)

    hmm_pass = sum(1 for r in hmm_results if r.get("anti_overfit_passed", r.get("passed", False)))
    msg_pass = sum(1 for r in msgarch_results if r.get("anti_overfit_passed", r.get("passed", False)))
    boc_pass = sum(1 for r in bocpd_results if r.get("passed", False))

    print(f"  {C.GREEN}HMM{C.RESET}:      {hmm_pass}/{len(hmm_results)} passed")
    print(f"  {C.MAGENTA}MS-GARCH{C.RESET}: {msg_pass}/{len(msgarch_results)} passed")
    print(f"  {C.BLUE}BOCPD{C.RESET}:    {boc_pass}/{len(bocpd_results)} calibrated")
    print(f"\n  Models saved to:")
    print(f"    HMM:      {HMM_MODEL_DIR}")
    print(f"    MS-GARCH: {MSGARCH_MODEL_DIR}")
    print(f"    BOCPD:    {BOCPD_MODEL_DIR}")


def cmd_test_ensemble():
    print(f"\n{C.BOLD}── Test Ensemble Predictions ──{C.RESET}")
    print(f"  Loading all available models and running predictions\n")

    from src.risk.physics.hmm.trainer import extract_features_vectorized
    from src.risk.physics.ensemble import EnsembleVoter
    import glob

    available = _find_available_data()
    if not available:
        print(f"{C.RED}No data. Run download first.{C.RESET}")
        return

    symbol, info = _pick_symbol(available)
    if not symbol:
        return

    df = pd.read_parquet(info["path"])
    features = extract_features_vectorized(df)
    print(f"  {symbol}: {len(features):,} feature vectors")

    # Try loading models
    hmm_sensor = None
    msgarch_sensor = None
    bocpd_detector = None

    # HMM
    hmm_pkls = sorted(HMM_MODEL_DIR.glob(f"{symbol}_*.pkl"))
    if hmm_pkls:
        with open(hmm_pkls[-1], "rb") as f:
            bundle = pickle.load(f)
        # Create a simple wrapper for HMM predict
        hmm_sensor = _HMMWrapper(bundle)
        print(f"  {C.GREEN}[HMM]{C.RESET}      Loaded {hmm_pkls[-1].name}")

    # MS-GARCH
    msgarch_pkls = sorted(MSGARCH_MODEL_DIR.glob(f"{symbol}_*.pkl"))
    if msgarch_pkls:
        from src.risk.physics.msgarch import MSGARCHSensor
        msgarch_sensor = MSGARCHSensor(model_path=msgarch_pkls[-1])
        print(f"  {C.MAGENTA}[MS-GARCH]{C.RESET} Loaded {msgarch_pkls[-1].name}")

    # BOCPD
    bocpd_jsons = sorted(BOCPD_MODEL_DIR.glob(f"bocpd_{symbol}_*.json"))
    if bocpd_jsons:
        from src.risk.physics.bocpd import BOCPDDetector
        bocpd_detector = BOCPDDetector.load(bocpd_jsons[-1])
        # Warm up with 100 bars
        for i in range(min(100, len(features))):
            bocpd_detector.update(float(features[i, 0]))
        print(f"  {C.BLUE}[BOCPD]{C.RESET}    Loaded {bocpd_jsons[-1].name}")

    if not any([hmm_sensor, msgarch_sensor, bocpd_detector]):
        print(f"\n  {C.RED}No models found for {symbol}. Train first.{C.RESET}")
        return

    voter = EnsembleVoter(
        hmm_sensor=hmm_sensor,
        msgarch_sensor=msgarch_sensor,
        bocpd_detector=bocpd_detector,
        session_name="LONDON_OPEN"
    )

    # Run predictions on last 20 bars
    print(f"\n  {C.BOLD}Last 20 bars (LONDON_OPEN session):{C.RESET}\n")
    print(f"  {'Bar':>5}  {'Regime':<17}  {'Conf':>6}  {'Agree':>6}  {'Transition'}")
    print(f"  {'─'*5}  {'─'*17}  {'─'*6}  {'─'*6}  {'─'*10}")

    for i in range(-20, 0):
        pred = voter.predict_regime(features[i])
        regime = pred.get("regime_type", "?")
        conf = pred.get("confidence", 0)
        agree = pred.get("ensemble_agreement", 0)
        trans = pred.get("is_transition", False)

        regime_color = C.GREEN if "TREND" in regime else (C.RED if "CHAOS" in regime else C.YELLOW)
        trans_str = f"{C.RED}YES{C.RESET}" if trans else f"{C.DIM}no{C.RESET}"
        print(f"  {len(features)+i:>5}  {regime_color}{regime:<17}{C.RESET}  "
              f"{conf:>5.1%}  {agree:>5.1%}  {trans_str}")

    print(f"\n  {C.BOLD}Ensemble status:{C.RESET}")
    status = voter.get_model_info()
    ensemble_info = status.get("ensemble", {})
    print(f"    Session:     {ensemble_info.get('session', '?')}")
    print(f"    Premium:     {ensemble_info.get('is_premium_session', '?')}")
    print(f"    Predictions: {ensemble_info.get('prediction_count', 0)}")
    print(f"    Transitions: {ensemble_info.get('transition_count', 0)}")


class _HMMWrapper:
    """Minimal wrapper to make a loaded HMM bundle work with EnsembleVoter."""
    def __init__(self, bundle):
        self._bundle = bundle
        self._model = bundle["model"]
        self._mean = bundle["scaler_mean"]
        self._std = bundle["scaler_std"]
        self._labels = bundle["state_labels"]

    def predict_regime(self, features, cache_key=None):
        from src.risk.physics.hmm.trainer import _scale
        sc, _, _ = _scale(features.reshape(1, -1), self._mean, self._std)
        state = int(self._model.predict(sc)[0])
        probs = self._model.predict_proba(sc)[0]
        label = self._labels.get(state, self._labels.get(str(state), f"S{state}"))
        return {
            "regime_type": label,
            "confidence": float(probs[state]),
            "state": state,
            "probabilities": {str(s): float(probs[s]) for s in range(len(probs))},
        }

    def get_model_info(self):
        return {
            "loaded": True,
            "symbol": self._bundle.get("symbol"),
            "n_states": self._bundle.get("n_states"),
            "version": self._bundle.get("version"),
        }

    def is_model_loaded(self):
        return True


# ═══════════════════════════════════════════════════════════════════════
# MANAGEMENT COMMANDS (50-53)
# ═══════════════════════════════════════════════════════════════════════
def cmd_view_report():
    print(f"\n{C.BOLD}── Training Reports ──{C.RESET}")

    for name, model_dir in [("HMM", HMM_MODEL_DIR), ("MS-GARCH", MSGARCH_MODEL_DIR)]:
        report_path = model_dir / "training_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            passed = report.get('passed', 0)
            total = report.get('total', 0)
            color = C.GREEN if passed == total else C.YELLOW
            print(f"\n  {C.BOLD}{name}{C.RESET}: v{report.get('version', '?')} "
                  f"({report.get('trained_at', '?')[:19]})")
            print(f"    {color}{passed}/{total} passed{C.RESET}")
        else:
            jsons = sorted(model_dir.glob("*_v*.json")) if model_dir.exists() else []
            if jsons:
                results = []
                for jp in jsons:
                    if jp.name == "training_report.json":
                        continue
                    with open(jp) as f:
                        results.append(json.load(f))
                print(f"\n  {C.BOLD}{name}{C.RESET}: {len(results)} models from individual metadata")
                _print_results_table(results)

    # BOCPD
    bocpd_jsons = sorted(BOCPD_MODEL_DIR.glob("*.json")) if BOCPD_MODEL_DIR.exists() else []
    if bocpd_jsons:
        print(f"\n  {C.BOLD}BOCPD{C.RESET}: {len(bocpd_jsons)} calibrations")
        for j in bocpd_jsons:
            with open(j) as f:
                cal = json.load(f)
            lam = cal.get("hazard", {}).get("lambda", "?")
            print(f"    {j.name}: λ={lam}")


def cmd_evaluate():
    print(f"\n{C.BOLD}── Evaluate Model ──{C.RESET}")

    all_pkls = []
    for name, model_dir in [("HMM", HMM_MODEL_DIR), ("MSG", MSGARCH_MODEL_DIR)]:
        if model_dir.exists():
            for p in sorted(model_dir.glob("*.pkl")):
                all_pkls.append((name, p))

    if not all_pkls:
        print(f"{C.YELLOW}No .pkl models found{C.RESET}")
        return

    print("Available models:")
    for i, (typ, p) in enumerate(all_pkls):
        sz = p.stat().st_size
        print(f"  {C.GREEN}{i+1:2d}{C.RESET}  [{typ:3s}] {p.name} ({sz:,} bytes)")

    choice = input(f"\nPick model [1-{len(all_pkls)}]: ").strip()
    try:
        idx = int(choice) - 1
        model_type, pkl_path = all_pkls[idx]
    except (ValueError, IndexError):
        print(f"{C.RED}Invalid choice{C.RESET}")
        return

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    print(f"\n{C.BOLD}Model: {pkl_path.name}{C.RESET}")
    for key in ["symbol", "timeframe", "n_states", "covariance_type", "version",
                 "trained_at", "state_labels", "n_vol_regimes", "vol_regime_labels"]:
        if key in bundle:
            val = bundle[key]
            print(f"  {key}: {val}")

    if model_type == "HMM" and "model" in bundle:
        model = bundle["model"]
        ns = bundle.get("n_states", model.n_components)
        labels = bundle.get("state_labels", {})

        available = _find_available_data()
        sym = bundle.get("symbol", "")
        if sym in available:
            from src.risk.physics.hmm.trainer import extract_features_vectorized, _scale
            df = pd.read_parquet(available[sym]["path"])
            features = extract_features_vectorized(df)
            sc, _, _ = _scale(features, bundle["scaler_mean"], bundle["scaler_std"])
            states = model.predict(sc)

            print(f"\n  Inference on {len(sc):,} samples:")
            for s in range(ns):
                pct = np.mean(states == s) * 100
                label = labels.get(s, labels.get(str(s), f"S{s}"))
                bar = "█" * int(pct / 2)
                print(f"    {label:15s} {pct:5.1f}% {C.CYAN}{bar}{C.RESET}")


def cmd_config():
    print(f"\n{C.BOLD}── Configuration ──{C.RESET}")
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        print(json.dumps(cfg, indent=2))
    else:
        print(f"{C.YELLOW}Config not found at {CONFIG_PATH}{C.RESET}")


def cmd_export():
    print(f"\n{C.BOLD}── Export Models for Deployment ──{C.RESET}")
    import shutil

    export_dir = PROJECT_ROOT / "deploy" / "models"
    export_dir.mkdir(parents=True, exist_ok=True)

    n_exported = 0
    for name, model_dir, pattern in [("hmm", HMM_MODEL_DIR, "*.pkl"),
                                      ("msgarch", MSGARCH_MODEL_DIR, "*.pkl"),
                                      ("bocpd", BOCPD_MODEL_DIR, "*.json")]:
        if not model_dir.exists():
            continue
        dest = export_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        files = sorted(model_dir.glob(pattern))
        if files:
            print(f"\n  {C.BOLD}{name.upper()}{C.RESET}:")
            for f in files:
                shutil.copy2(f, dest / f.name)
                print(f"    {C.GREEN}✓{C.RESET} {f.name}")
                n_exported += 1
            # Also copy metadata jsons for HMM/MSGARCH
            if name in ("hmm", "msgarch"):
                for j in model_dir.glob("*.json"):
                    shutil.copy2(j, dest / j.name)

    if n_exported:
        print(f"\n  {C.GREEN}Exported {n_exported} files to {export_dir}{C.RESET}")
    else:
        print(f"  {C.YELLOW}No models to export{C.RESET}")


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
def _find_available_data() -> Dict:
    available = {}
    if not M5_DIR.exists():
        return available
    for f in sorted(M5_DIR.glob("*.parquet")):
        name = f.stem
        parts = name.split("_")
        symbol = parts[0]
        tf = parts[1] if len(parts) > 1 else "M5"
        df = pd.read_parquet(f)
        available[symbol] = {"path": str(f), "rows": len(df), "tf": tf}
    return available


def _pick_symbol(available):
    print("Available symbols:")
    syms = list(available.keys())
    for i, s in enumerate(syms):
        info = available[s]
        print(f"  {C.GREEN}{i+1}{C.RESET}  {s} ({info['rows']:,} bars, {info['tf']})")
    choice = input(f"\nPick symbol [1-{len(syms)}]: ").strip()
    try:
        idx = int(choice) - 1
        return syms[idx], available[syms[idx]]
    except (ValueError, IndexError):
        print(f"{C.RED}Invalid choice{C.RESET}")
        return None, None


def _pick_pairs(available):
    pairs_input = input(f"Pairs [{C.DIM}{','.join(available.keys())}{C.RESET}]: ").strip()
    if not pairs_input:
        return list(available.keys())
    pairs = [p.strip().upper() for p in pairs_input.split(",")]
    valid = [p for p in pairs if p in available]
    if not valid:
        print(f"{C.RED}No valid pairs selected{C.RESET}")
        return []
    return valid


def _show_data_status():
    print(f"\n  {C.BOLD}M5 Data{C.RESET} ({M5_DIR}):")
    if M5_DIR.exists():
        for f in sorted(M5_DIR.glob("*.parquet")):
            df = pd.read_parquet(f)
            sz = f.stat().st_size / 1024 / 1024
            time_col = 'time' if 'time' in df.columns else df.columns[0]
            date_range = f"{df[time_col].iloc[0]} → {df[time_col].iloc[-1]}" if len(df) > 0 else "empty"
            print(f"    {C.GREEN}{f.name:25s}{C.RESET}  {len(df):>10,} bars  "
                  f"{sz:.1f}MB  {C.DIM}{date_range}{C.RESET}")
    else:
        print(f"    {C.DIM}no data{C.RESET}")

    for name, model_dir in [("HMM Models", HMM_MODEL_DIR),
                             ("MS-GARCH Models", MSGARCH_MODEL_DIR),
                             ("BOCPD Calibrations", BOCPD_MODEL_DIR)]:
        print(f"\n  {C.BOLD}{name}{C.RESET} ({model_dir}):")
        if model_dir.exists():
            files = sorted(list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.json")))
            real_files = [f for f in files if f.name != "training_report.json"]
            if real_files:
                for p in real_files[:10]:
                    sz = p.stat().st_size
                    print(f"    {C.GREEN}{p.name:45s}{C.RESET}  {sz:,} bytes")
                if len(real_files) > 10:
                    print(f"    {C.DIM}... and {len(real_files)-10} more{C.RESET}")
            else:
                print(f"    {C.DIM}no models{C.RESET}")
        else:
            print(f"    {C.DIM}not yet created{C.RESET}")


def _save_training_report(model_type: str, version: str, results: List[Dict]):
    model_dir = HMM_MODEL_DIR if model_type == "hmm" else MSGARCH_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for r in results if r.get("anti_overfit_passed", r.get("passed", False)))
    report = {
        "model_type": model_type, "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed, "total": len(results),
        "results": results,
    }
    with open(model_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)


def _print_results_table(results: List[Dict]):
    print(f"\n  {'Symbol':<8} {'States':<7} {'Gap%':<8} {'Persist':<8} {'Status':<8}")
    print(f"  {'─'*8} {'─'*7} {'─'*8} {'─'*8} {'─'*8}")
    for r in results:
        sym = r.get("symbol", "?")
        ns = r.get("n_states", r.get("n_vol_regimes", "?"))
        gap = r.get("per_sample_gap_pct", r.get("gap_pct", "?"))
        persist = r.get("avg_persistence", "?")
        passed = r.get("anti_overfit_passed", r.get("passed", False))
        status = f"{C.GREEN}PASS{C.RESET}" if passed else f"{C.YELLOW}WARN{C.RESET}"
        gap_str = f"{gap:.1f}" if isinstance(gap, (int, float)) else str(gap)
        per_str = f"{persist:.3f}" if isinstance(persist, (int, float)) else str(persist)
        print(f"  {sym:<8} {str(ns):<7} {gap_str:<8} {per_str:<8} {status}")


def _print_session_results(results: List[Dict]):
    print(f"\n  {'Symbol':<8} {'Session':<25} {'States':<7} {'Gap%':<8} {'Status':<8}")
    print(f"  {'─'*8} {'─'*25} {'─'*7} {'─'*8} {'─'*8}")
    for r in results:
        sym = r.get("symbol", "?")
        session = r.get("session", "UNIVERSAL")[:24]
        ns = r.get("n_states", "?")
        gap = r.get("per_sample_gap_pct", r.get("gap_pct", "?"))
        passed = r.get("anti_overfit_passed", r.get("passed", False))
        warns = r.get("warnings", [])
        if "INSUFFICIENT_DATA" in str(warns):
            status = f"{C.RED}SKIP{C.RESET}"
        elif passed:
            status = f"{C.GREEN}PASS{C.RESET}"
        else:
            status = f"{C.YELLOW}WARN{C.RESET}"
        gap_str = f"{gap:.1f}" if isinstance(gap, (int, float)) else str(gap)
        print(f"  {sym:<8} {session:<25} {str(ns):<7} {gap_str:<8} {status}")


def _print_msgarch_results(results: List[Dict]):
    print(f"\n  {'Symbol':<8} {'Regimes':<8} {'Gap%':<8} {'BIC':<12} {'Size':<8} {'Status':<8}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*12} {'─'*8} {'─'*8}")
    for r in results:
        sym = r.get("symbol", "?")
        nr = r.get("n_regimes", r.get("n_vol_regimes", "?"))
        gap = r.get("gap_pct", "?")
        bic = r.get("garch_bic", r.get("bic", "?"))
        fsize = r.get("file_size_bytes", 0)
        passed = r.get("anti_overfit_passed", r.get("passed", False))
        status = f"{C.GREEN}PASS{C.RESET}" if passed else f"{C.YELLOW}WARN{C.RESET}"
        gap_str = f"{gap:.1f}" if isinstance(gap, (int, float)) else str(gap)
        bic_str = f"{bic:.1f}" if isinstance(bic, (int, float)) else str(bic)
        size_str = f"{fsize:,}B" if fsize else "?"
        print(f"  {sym:<8} {str(nr):<8} {gap_str:<8} {bic_str:<12} {size_str:<8} {status}")


def _print_bocpd_results(results: List[Dict]):
    print(f"\n  {'Symbol':<8} {'λ (hazard)':<12} {'Time':<8} {'Status'}")
    print(f"  {'─'*8} {'─'*12} {'─'*8} {'─'*8}")
    for r in results:
        sym = r.get("symbol", "?")
        lam = r.get("hazard_lambda", "?")
        elapsed = r.get("elapsed", "?")
        passed = r.get("passed", False)
        status = f"{C.GREEN}OK{C.RESET}" if passed else f"{C.RED}FAIL{C.RESET}"
        lam_str = f"{lam}" if isinstance(lam, str) else f"{lam:.0f}"
        elapsed_str = f"{elapsed}s" if isinstance(elapsed, (int, float)) else str(elapsed)
        print(f"  {sym:<8} {lam_str:<12} {elapsed_str:<8} {status}")


def cmd_cleanup():
    """Remove old model versions, keeping only the latest per symbol."""
    print(f"\n{C.BOLD}── Model Cleanup ──{C.RESET}")
    print(f"  Keeps only the LATEST model per symbol. Old versions are removed.\n")

    removed = 0
    freed = 0

    for name, model_dir in [("HMM", HMM_MODEL_DIR), ("MS-GARCH", MSGARCH_MODEL_DIR)]:
        if not model_dir.exists():
            continue

        pkls = sorted(model_dir.glob("*.pkl"))
        if not pkls:
            continue

        # Group by symbol
        by_symbol = {}
        for p in pkls:
            parts = p.stem.split("_")
            sym = parts[0]
            by_symbol.setdefault(sym, []).append(p)

        print(f"  {C.BOLD}{name}{C.RESET}:")
        for sym, files in by_symbol.items():
            if len(files) <= 1:
                print(f"    {sym}: 1 model (keeping)")
                continue

            # Sort by mtime, keep latest
            files.sort(key=lambda f: f.stat().st_mtime)
            keep = files[-1]
            old = files[:-1]
            print(f"    {sym}: {len(files)} models → keeping {keep.name}")

            for f in old:
                sz = f.stat().st_size
                freed += sz
                f.unlink()
                removed += 1
                # Remove matching .json too
                j = f.with_suffix(".json")
                if j.exists():
                    j.unlink()
                    removed += 1
                print(f"      {C.RED}✗{C.RESET} removed {f.name} ({sz:,} bytes)")

    # Also clean sessions dir
    sessions_dir = HMM_MODEL_DIR / "sessions"
    if sessions_dir.exists():
        sess_pkls = sorted(sessions_dir.glob("*.pkl"))
        by_key = {}
        for p in sess_pkls:
            # Group by everything except version
            parts = p.stem.rsplit("_v", 1)
            key = parts[0]
            by_key.setdefault(key, []).append(p)

        for key, files in by_key.items():
            if len(files) <= 1:
                continue
            files.sort(key=lambda f: f.stat().st_mtime)
            old = files[:-1]
            for f in old:
                freed += f.stat().st_size
                f.unlink()
                removed += 1
                j = f.with_suffix(".json")
                if j.exists():
                    j.unlink()
                    removed += 1

    if removed:
        print(f"\n  {C.GREEN}Removed {removed} old files, freed {freed:,} bytes ({freed/1024:.1f} KB){C.RESET}")
    else:
        print(f"\n  {C.DIM}Nothing to clean — only latest versions found{C.RESET}")


def cmd_data_status():
    print(f"\n{C.BOLD}── Data & Model Status ──{C.RESET}")
    _show_data_status()


# ═══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════
def main():
    banner()

    commands = {
        # Data
        "1": cmd_download, "2": cmd_data_status,
        # HMM
        "10": cmd_hmm_train_all, "11": cmd_hmm_train_single,
        "12": cmd_hmm_train_premium, "13": cmd_hmm_train_full_suite,
        "14": cmd_hmm_train_new_only,
        # MS-GARCH
        "20": cmd_msgarch_train_all, "21": cmd_msgarch_train_single,
        "22": cmd_msgarch_train_premium,
        # BOCPD
        "30": cmd_bocpd_calibrate_all, "31": cmd_bocpd_calibrate_single,
        # Ensemble
        "40": cmd_train_all_models, "41": cmd_test_ensemble,
        # Management
        "50": cmd_view_report, "51": cmd_evaluate,
        "52": cmd_config, "53": cmd_export, "54": cmd_cleanup,
    }

    while True:
        menu()
        try:
            choice = input(f"{C.CYAN}regime>{C.RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}Bye!{C.RESET}")
            break

        if choice in ("q", "quit", "exit"):
            print(f"{C.DIM}Bye!{C.RESET}")
            break
        elif choice in commands:
            try:
                commands[choice]()
            except KeyboardInterrupt:
                print(f"\n{C.YELLOW}Interrupted{C.RESET}")
            except Exception as e:
                print(f"\n{C.RED}Error: {e}{C.RESET}")
                import traceback
                traceback.print_exc()
        else:
            print(f"{C.DIM}Unknown command: {choice}{C.RESET}")


if __name__ == "__main__":
    main()
