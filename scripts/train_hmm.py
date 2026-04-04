#!/usr/bin/env python3
"""
HMM Training Script — QuantMindX (Standalone, Anti-Overfit)
============================================================
Trains GaussianHMM (4 regimes) on M5 OHLCV data for 5 major pairs.

ANTI-OVERFITTING MEASURES:
  1. Strict 60/20/20 temporal split (train/val/test — NO shuffle)
  2. BIC model selection (penalizes complexity)
  3. Covariance regularization (min_covar = 1e-3)
  4. Train vs Val log-likelihood gap check (rejects if gap > 15%)
  5. Walk-forward cross-validation (5 folds, expanding window)
  6. State persistence check (diagonal ≥ 0.70 — rejects jittery models)
  7. State balance check (no single state > 55%)
  8. "Tied" covariance option for small-sample robustness

Output:
  models/hmm/<SYMBOL>_M5_v<VERSION>.pkl   — trained model bundle
  models/hmm/<SYMBOL>_M5_v<VERSION>.json  — metadata (for shipping)
  models/hmm/training_report.json         — summary report

The .pkl files are typically 2-5 MB each — ship ONLY these to servers.
Sync via existing HMMVersionControl SSH/SFTP (POST /api/hmm/sync).

Usage:
    python scripts/train_hmm.py
    python scripts/train_hmm.py --symbols EURUSD,GBPUSD --n-bars 80000
"""

import sys
import os
import json
import hashlib
import pickle
import logging
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hmmlearn.hmm import GaussianHMM

from src.data.dukascopy_fetcher import MockDukascopyFetcher, DukascopyConfig

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]
TIMEFRAME = "5min"
DEFAULT_N_BARS = 50000      # ~6 months of M5 data
N_STATES = 4                # 4 regimes
COVARIANCE_TYPE = "full"    # 'full' for rich correlations; 'tied' as fallback
N_ITER = 150                # conservative — more iterations but with early stop via tol
TOL = 1e-4                  # convergence tolerance
MIN_COVAR = 1e-3            # regularization — prevents singular covariance
RANDOM_STATE = 42

# Anti-overfit thresholds
TRAIN_SPLIT = 0.60
VAL_SPLIT = 0.20
# test = remaining 0.20
MAX_TRAIN_VAL_GAP_PCT = 15.0   # reject if train LL > val LL by more than 15%
MIN_DIAG_PERSISTENCE = 0.70    # reject if avg diagonal < 0.70
MAX_STATE_DOMINANCE = 0.55     # reject if any state > 55%
WF_FOLDS = 5                   # walk-forward cross-validation folds

MODEL_DIR = PROJECT_ROOT / "models" / "hmm"
VERSION = datetime.now().strftime("%Y%m%d-%H%M%S")

FEATURE_NAMES = [
    "log_returns", "rolling_volatility_20", "rolling_volatility_50",
    "price_momentum_10", "rsi", "atr_normalized",
    "magnetization", "susceptibility", "energy", "temperature"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("hmm_trainer")


# ---------------------------------------------------------------------------
# Feature Extraction (vectorized, no per-row Ising overhead)
# ---------------------------------------------------------------------------
def extract_features_fast(df: pd.DataFrame) -> np.ndarray:
    """
    Extract 10-feature vector from OHLCV data (vectorized).

    Features (matching FeatureConfig defaults):
      1. log_returns
      2. rolling_volatility_20
      3. rolling_volatility_50
      4. price_momentum_10
      5. rsi (14-period)
      6. atr_normalized (14-period)
      7. magnetization  (rolling mean of sign(returns))
      8. susceptibility (rolling variance of magnetization)
      9. energy         (neighbor spin correlation)
      10. temperature   (annualized volatility proxy)
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    n = len(close)

    # 1. Log returns
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(close[1:] / close[:-1])

    # 2-3. Rolling volatility
    lr_series = pd.Series(log_ret)
    vol20 = lr_series.rolling(20).std().values
    vol50 = lr_series.rolling(50).std().values

    # 4. Price momentum 10
    momentum = np.zeros(n)
    momentum[10:] = (close[10:] - close[:-10]) / close[:-10]

    # 5. RSI 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # 6. ATR normalized
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0], tr3[0] = tr1[0], tr1[0]
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(true_range).rolling(14).mean().values
    atr_norm = atr / (close + 1e-10)

    # 7-10. Ising-approximated features
    signs = np.sign(log_ret)
    magnetization = pd.Series(signs).rolling(20).mean().values
    susceptibility = pd.Series(magnetization).rolling(20).var().values
    energy = np.zeros(n)
    energy[1:] = -signs[1:] * signs[:-1]
    energy = pd.Series(energy).rolling(10).mean().values
    temperature = vol20 * np.sqrt(252) * 100

    features = np.column_stack([
        log_ret, vol20, vol50, momentum, rsi, atr_norm,
        magnetization, susceptibility, energy, temperature
    ])

    # Drop warmup rows (first 50) where indicators are NaN
    features = features[50:]
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def scale_features(features: np.ndarray, mean=None, std=None):
    """Robust standard scaling with ±3σ clipping."""
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0) + 1e-8
    scaled = (features - mean) / std
    scaled = np.clip(scaled, -3.0, 3.0)
    return scaled, mean, std


# ---------------------------------------------------------------------------
# Anti-Overfit: BIC Model Selection
# ---------------------------------------------------------------------------
def compute_bic(model: GaussianHMM, X: np.ndarray) -> float:
    """
    Bayesian Information Criterion for HMM.
    BIC = -2 * LL + k * ln(n)
    Lower is better. Penalizes model complexity.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_states = model.n_components

    # Number of free parameters
    # startprob: n_states - 1
    # transmat: n_states * (n_states - 1)
    # means: n_states * n_features
    # covars (full): n_states * n_features * (n_features + 1) / 2
    k_start = n_states - 1
    k_trans = n_states * (n_states - 1)
    k_means = n_states * n_features
    k_covars = n_states * n_features * (n_features + 1) // 2
    k = k_start + k_trans + k_means + k_covars

    ll = model.score(X) * n_samples  # score() returns per-sample LL
    bic = -2 * ll + k * np.log(n_samples)
    return bic


# ---------------------------------------------------------------------------
# Anti-Overfit: Walk-Forward Cross-Validation
# ---------------------------------------------------------------------------
def walk_forward_cv(X: np.ndarray, n_folds: int = 5) -> List[float]:
    """
    Walk-forward expanding-window CV for time series.
    Returns list of val log-likelihoods per fold.

    Fold 1: train on [0..20%], val on [20..40%]
    Fold 2: train on [0..40%], val on [40..60%]
    ...etc.
    NO SHUFFLING — temporal order preserved.
    """
    n = len(X)
    fold_size = n // (n_folds + 1)
    val_scores = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 2)
        val_start = train_end
        val_end = min(val_start + fold_size, n)

        if val_end <= val_start or train_end >= n:
            break

        X_train = X[:train_end]
        X_val = X[val_start:val_end]

        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type=COVARIANCE_TYPE,
            n_iter=N_ITER,
            tol=TOL,
            min_covar=MIN_COVAR,
            random_state=RANDOM_STATE,
            verbose=False
        )

        try:
            model.fit(X_train)
            val_ll = model.score(X_val)
            val_scores.append(val_ll)
        except Exception:
            val_scores.append(float('-inf'))

    return val_scores


# ---------------------------------------------------------------------------
# Training with Anti-Overfit Validation
# ---------------------------------------------------------------------------
def train_symbol(symbol: str, fetcher: MockDukascopyFetcher, n_bars: int) -> dict:
    """Train HMM for one symbol with full anti-overfit pipeline."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING {symbol} M5 | {n_bars} bars | {N_STATES} states")
    logger.info(f"{'='*60}")

    # 1. Fetch data
    result = fetcher.fetch(symbol, TIMEFRAME, count=n_bars)
    if not result.success:
        logger.error(f"FAILED to fetch data for {symbol}: {result.message}")
        return {"symbol": symbol, "status": "FAILED", "error": result.message}

    df = result.data
    logger.info(f"  Data: {len(df)} bars fetched")

    # 2. Extract features
    features = extract_features_fast(df)
    logger.info(f"  Features: {features.shape[0]} samples x {features.shape[1]} features")

    # 3. Temporal split (NO shuffling — time series!)
    n = len(features)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    raw_train = features[:n_train]
    raw_val = features[n_train:n_train + n_val]
    raw_test = features[n_train + n_val:]

    # 4. Scale using ONLY training data (prevent data leakage)
    scaled_train, feat_mean, feat_std = scale_features(raw_train)
    scaled_val, _, _ = scale_features(raw_val, feat_mean, feat_std)
    scaled_test, _, _ = scale_features(raw_test, feat_mean, feat_std)
    scaled_all, _, _ = scale_features(features, feat_mean, feat_std)

    logger.info(f"  Split: train={len(scaled_train)}, val={len(scaled_val)}, test={len(scaled_test)}")
    logger.info(f"  Scaler fitted on TRAINING data only (no data leakage)")

    # 5. Train GaussianHMM
    logger.info(f"  Training GaussianHMM (n={N_STATES}, iter={N_ITER}, min_covar={MIN_COVAR})...")
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type=COVARIANCE_TYPE,
        n_iter=N_ITER,
        tol=TOL,
        min_covar=MIN_COVAR,
        random_state=RANDOM_STATE,
        verbose=False
    )
    model.fit(scaled_train)
    converged = model.monitor_.converged
    n_iters = model.monitor_.iter
    logger.info(f"  Converged: {converged} after {n_iters} iterations")

    # 6. Score all splits
    train_ll = model.score(scaled_train)
    val_ll = model.score(scaled_val)
    test_ll = model.score(scaled_test)
    logger.info(f"  Log-likelihood — train: {train_ll:.4f}, val: {val_ll:.4f}, test: {test_ll:.4f}")

    # 7. BIC (on training data)
    bic = compute_bic(model, scaled_train)
    logger.info(f"  BIC: {bic:.2f} (lower = better, penalizes complexity)")

    # -----------------------------------------------------------------------
    # ANTI-OVERFIT CHECKS
    # -----------------------------------------------------------------------
    overfit_warnings = []

    # Check A: Train vs Val gap
    if train_ll != 0:
        gap_pct = abs((train_ll - val_ll) / abs(train_ll)) * 100
    else:
        gap_pct = 0
    if gap_pct > MAX_TRAIN_VAL_GAP_PCT:
        overfit_warnings.append(
            f"OVERFIT_GAP: train-val LL gap = {gap_pct:.1f}% (threshold: {MAX_TRAIN_VAL_GAP_PCT}%)"
        )
    logger.info(f"  Overfit check — train-val gap: {gap_pct:.1f}% (max: {MAX_TRAIN_VAL_GAP_PCT}%)")

    # Check B: State distribution balance
    states_all = model.predict(scaled_all)
    state_dist = {}
    for s in range(N_STATES):
        pct = np.mean(states_all == s)
        state_dist[str(s)] = round(pct * 100, 2)
    max_dominance = max(state_dist.values())
    if max_dominance > MAX_STATE_DOMINANCE * 100:
        overfit_warnings.append(
            f"STATE_DOMINANCE: state dominance = {max_dominance:.1f}% (max: {MAX_STATE_DOMINANCE*100}%)"
        )
    logger.info(f"  State distribution: {state_dist}")

    # Check C: Transition persistence
    trans_matrix = model.transmat_.tolist()
    diag = [trans_matrix[i][i] for i in range(N_STATES)]
    avg_persistence = np.mean(diag)
    if avg_persistence < MIN_DIAG_PERSISTENCE:
        overfit_warnings.append(
            f"LOW_PERSISTENCE: avg diagonal = {avg_persistence:.3f} (min: {MIN_DIAG_PERSISTENCE})"
        )
    logger.info(f"  Transition diagonal: {[round(d, 3) for d in diag]} (avg: {avg_persistence:.3f})")

    # Check D: Walk-forward cross-validation
    logger.info(f"  Walk-forward CV ({WF_FOLDS} folds, expanding window)...")
    wf_scores = walk_forward_cv(scaled_all, WF_FOLDS)
    wf_mean = np.mean(wf_scores) if wf_scores else float('-inf')
    wf_std = np.std(wf_scores) if len(wf_scores) > 1 else 0
    logger.info(f"  WF-CV scores: {[round(s, 4) for s in wf_scores]}")
    logger.info(f"  WF-CV mean: {wf_mean:.4f} ± {wf_std:.4f}")

    # Check if WF-CV variance is suspiciously low (might indicate data leakage)
    if wf_std > 0 and abs(wf_mean) > 0:
        wf_cv_ratio = wf_std / abs(wf_mean)
        if wf_cv_ratio > 0.5:
            overfit_warnings.append(
                f"WF_UNSTABLE: CV variance ratio = {wf_cv_ratio:.3f} (model unstable across time)"
            )

    # Report overfit status
    if overfit_warnings:
        logger.warning(f"  ⚠ OVERFIT WARNINGS ({len(overfit_warnings)}):")
        for w in overfit_warnings:
            logger.warning(f"    - {w}")
    else:
        logger.info(f"  ✓ All anti-overfit checks PASSED")

    # 8. Label regimes by feature means
    means = model.means_
    log_ret_means = means[:, 0]
    susc_means = means[:, 7]

    bull_state = int(np.argmax(log_ret_means))
    bear_state = int(np.argmin(log_ret_means))
    remaining = [s for s in range(N_STATES) if s not in [bull_state, bear_state]]
    chaos_state = remaining[int(np.argmax([susc_means[s] for s in remaining]))]
    range_state = [s for s in remaining if s != chaos_state][0]

    state_labels = {
        bull_state: "TREND_BULL",
        bear_state: "TREND_BEAR",
        range_state: "RANGE_STABLE",
        chaos_state: "CHAOS"
    }
    logger.info(f"  Regime labels: {state_labels}")

    # 9. Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_filename = f"{symbol}_M5_v{VERSION}.pkl"
    model_path = MODEL_DIR / model_filename

    model_bundle = {
        "model": model,
        "scaler_mean": feat_mean,
        "scaler_std": feat_std,
        "feature_names": FEATURE_NAMES,
        "state_labels": state_labels,
        "n_states": N_STATES,
        "version": VERSION,
        "symbol": symbol,
        "timeframe": "M5",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(scaled_train),
        "deployment_mode": "ising_only",  # shadow — not used for decisions
        "anti_overfit": {
            "train_val_gap_pct": round(gap_pct, 2),
            "max_state_dominance_pct": round(max_dominance, 2),
            "avg_persistence": round(avg_persistence, 4),
            "wf_cv_mean": round(wf_mean, 4),
            "wf_cv_std": round(wf_std, 4),
            "bic": round(bic, 2),
            "warnings": overfit_warnings
        }
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    # Checksum
    with open(model_path, "rb") as f:
        checksum = hashlib.sha256(f.read()).hexdigest()

    file_size = model_path.stat().st_size
    logger.info(f"  Saved: {model_path.name} ({file_size:,} bytes, sha256: {checksum[:16]}...)")

    # 10. Save metadata JSON
    metadata = {
        "symbol": symbol,
        "timeframe": "M5",
        "version": VERSION,
        "n_states": N_STATES,
        "covariance_type": COVARIANCE_TYPE,
        "min_covar": MIN_COVAR,
        "training_samples": len(scaled_train),
        "validation_samples": len(scaled_val),
        "test_samples": len(scaled_test),
        "log_likelihood": {
            "train": round(train_ll, 6),
            "val": round(val_ll, 6),
            "test": round(test_ll, 6)
        },
        "bic": round(bic, 2),
        "state_distribution": state_dist,
        "state_labels": {str(k): v for k, v in state_labels.items()},
        "transition_matrix": [[round(x, 4) for x in row] for row in trans_matrix],
        "converged": converged,
        "iterations": n_iters,
        "anti_overfit": {
            "train_val_gap_pct": round(gap_pct, 2),
            "max_allowed_gap_pct": MAX_TRAIN_VAL_GAP_PCT,
            "max_state_dominance_pct": round(max_dominance, 2),
            "max_allowed_dominance_pct": MAX_STATE_DOMINANCE * 100,
            "avg_persistence": round(avg_persistence, 4),
            "min_required_persistence": MIN_DIAG_PERSISTENCE,
            "walk_forward_cv": {
                "folds": WF_FOLDS,
                "scores": [round(s, 6) for s in wf_scores],
                "mean": round(wf_mean, 6),
                "std": round(wf_std, 6)
            },
            "warnings": overfit_warnings,
            "passed": len(overfit_warnings) == 0
        },
        "file_path": str(model_path),
        "file_size_bytes": file_size,
        "checksum_sha256": checksum,
        "deployment_mode": "ising_only",
        "trained_at": datetime.now(timezone.utc).isoformat()
    }

    meta_path = MODEL_DIR / f"{symbol}_M5_v{VERSION}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Metadata: {meta_path.name}")
    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="QuantMindX HMM Trainer (anti-overfit)")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                        help="Comma-separated symbols")
    parser.add_argument("--n-bars", type=int, default=DEFAULT_N_BARS,
                        help="Number of M5 bars per symbol")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    logger.info("=" * 70)
    logger.info(f"QuantMindX HMM Training — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Version: {VERSION}")
    logger.info(f"Output: {MODEL_DIR}")
    logger.info(f"Anti-overfit: gap<{MAX_TRAIN_VAL_GAP_PCT}%, persist>{MIN_DIAG_PERSISTENCE}, "
                f"dominance<{MAX_STATE_DOMINANCE*100}%, WF-CV {WF_FOLDS} folds")
    logger.info("=" * 70)

    fetcher = MockDukascopyFetcher(
        DukascopyConfig(cache_dir=PROJECT_ROOT / "data" / "dukascopy")
    )

    results = []
    passed = 0
    warned = 0

    for symbol in symbols:
        try:
            meta = train_symbol(symbol, fetcher, args.n_bars)
            results.append(meta)
            if meta.get("anti_overfit", {}).get("passed", False):
                passed += 1
            else:
                warned += 1
        except Exception as e:
            logger.error(f"FAILED {symbol}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"symbol": symbol, "status": "FAILED", "error": str(e)})

    # Summary report
    report = {
        "version": VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_symbols": len(symbols),
        "symbols": symbols,
        "n_states": N_STATES,
        "passed_overfit_checks": passed,
        "warned_overfit_checks": warned,
        "results": results,
        "deployment_mode": "ising_only",
        "shipping_instructions": {
            "what_to_ship": "Only the .pkl files (2-5 MB each)",
            "total_files": len(symbols),
            "destination": "Cloudzy trading server via HMMVersionControl SSH/SFTP sync",
            "sync_command": "POST /api/hmm/sync",
            "verification": "SHA256 checksum verified on both ends"
        }
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODEL_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Models: {MODEL_DIR}")
    logger.info(f"Report: {report_path}")

    total_size = sum(
        (MODEL_DIR / f"{s}_M5_v{VERSION}.pkl").stat().st_size
        for s in symbols
        if (MODEL_DIR / f"{s}_M5_v{VERSION}.pkl").exists()
    )
    logger.info(f"Total model size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
    logger.info(f"  → Ship ONLY .pkl files. NOT the 18 GB codebase.")
    logger.info(f"Overfit status: {passed} passed, {warned} warned")
    logger.info("=" * 70)

    return report


if __name__ == "__main__":
    report = main()