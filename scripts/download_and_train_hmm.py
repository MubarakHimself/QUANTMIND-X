#!/usr/bin/env python3
"""
QuantMindX — Download Real Forex Data + Train HMM (Standalone)
================================================================
Run this from your laptop. It will:
  1. Download M1 data from histdata.com (free, real Dukascopy source)
  2. Resample M1 → M5
  3. Train GaussianHMM (4 regimes) with anti-overfit checks
  4. Save .pkl models ready to ship to Contabo/Cloudzy

Requirements:
    pip install hmmlearn histdata yfinance pandas numpy pyarrow

Usage:
    python scripts/download_and_train_hmm.py
    python scripts/download_and_train_hmm.py --years 2015,2016,2017,2018,2019,2020,2021,2022,2023,2024
    python scripts/download_and_train_hmm.py --pairs EURUSD,GBPUSD,USDJPY --years 2020,2021,2022,2023,2024
    python scripts/download_and_train_hmm.py --skip-download  # use cached data

Data sources:
    - histdata.com (M1 → resample M5) for EURUSD, GBPUSD, USDJPY, AUDUSD, EURGBP, EURJPY
    - yfinance H1 for XAUUSD (gold) — histdata doesn't carry it
"""

import sys
import os
import json
import hashlib
import pickle
import logging
import warnings
import argparse
import time
import zipfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]
DEFAULT_YEARS = ["2020", "2021", "2022", "2023", "2024"]
HISTDATA_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "EURGBP", "EURJPY",
                  "GBPJPY", "USDCHF", "USDCAD", "NZDUSD", "CADJPY", "AUDJPY"]

N_STATES_CANDIDATES = [3, 4, 5]   # Try multiple state counts, pick best
N_ITER = 200
TOL = 1e-4
COVAR_TYPES = ["full", "tied", "diag"]  # Try multiple, pick least overfit
MIN_COVAR_OPTIONS = [1e-3, 1e-2, 5e-2]  # Escalating regularization
N_SEEDS = 3  # Multiple random seeds per config
RANDOM_SEEDS = [42, 123, 7]
TRAIN_SPLIT = 0.60
VAL_SPLIT = 0.20
MAX_GAP_PCT = 30.0   # Per-sample train-val gap threshold
MAX_DOMINANCE = 85.0  # State dominance threshold (relaxed for forex)
MIN_PERSISTENCE = 0.65  # Minimum avg diagonal persistence

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "hmm"

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
logger = logging.getLogger("hmm")


# ===========================================================================
# STEP 1: DOWNLOAD DATA
# ===========================================================================
def download_histdata(pairs, years):
    """Download M1 data from histdata.com and resample to M5."""
    from histdata import download_hist_data
    from histdata.api import Platform as P, TimeFrame as TF

    m1_dir = DATA_DIR / "histdata_m1"
    m5_dir = DATA_DIR / "real_m5_multi_year"
    m1_dir.mkdir(parents=True, exist_ok=True)
    m5_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for pair in pairs:
        pair_upper = pair.upper()

        # XAUUSD not on histdata — use yfinance
        if pair_upper == "XAUUSD":
            logger.info(f"  {pair_upper}: Using yfinance (not on histdata)")
            try:
                import yfinance as yf
                data = yf.download("GC=F", period="730d", interval="1h", progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                df = pd.DataFrame({
                    "time": data.index, "open": data["Open"].values,
                    "high": data["High"].values, "low": data["Low"].values,
                    "close": data["Close"].values,
                    "tick_volume": data.get("Volume", pd.Series(np.zeros(len(data)))).values
                }).dropna(subset=["close"])
                save_path = m5_dir / f"{pair_upper}_H1.parquet"
                df.to_parquet(save_path, index=False)
                results[pair_upper] = {"rows": len(df), "source": "yfinance_H1", "path": str(save_path)}
                logger.info(f"  {pair_upper}: {len(df):,} H1 bars from yfinance")
            except Exception as e:
                logger.error(f"  {pair_upper} yfinance failed: {e}")
            continue

        if pair_upper not in HISTDATA_PAIRS:
            logger.warning(f"  {pair_upper}: Not available on histdata, skipping")
            continue

        all_dfs = []
        for year in years:
            zip_name = f"DAT_ASCII_{pair_upper}_M1_{year}.zip"
            zip_path = m1_dir / zip_name

            if not zip_path.exists():
                try:
                    logger.info(f"  Downloading {pair_upper} {year}...")
                    download_hist_data(
                        year=year, pair=pair.lower(),
                        platform=P.GENERIC_ASCII, time_frame=TF.ONE_MINUTE,
                        output_directory=str(m1_dir)
                    )
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"  {pair_upper} {year}: {e}")
                    continue
            else:
                logger.info(f"  {pair_upper} {year}: cached")

            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f, sep=';', header=None,
                                         names=['time', 'open', 'high', 'low', 'close', 'volume'])
                        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d %H%M%S')
                        all_dfs.append(df)
                        logger.info(f"  {pair_upper} {year}: {len(df):,} M1 bars")
            except Exception as e:
                logger.error(f"  Parse error {pair_upper} {year}: {e}")

        if not all_dfs:
            continue

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('time').drop_duplicates(subset='time').reset_index(drop=True)

        # Resample M1 → M5
        combined = combined.set_index('time')
        m5 = combined.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
        m5.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']

        save_path = m5_dir / f"{pair_upper}_M5.parquet"
        m5.to_parquet(save_path, index=False)
        results[pair_upper] = {
            "rows": len(m5), "m1_rows": len(combined),
            "source": "histdata.com", "years": years,
            "date_range": f"{m5['time'].iloc[0].date()} → {m5['time'].iloc[-1].date()}",
            "path": str(save_path)
        }
        logger.info(f"  {pair_upper}: {len(m5):,} M5 bars ({m5['time'].iloc[0].date()} → {m5['time'].iloc[-1].date()})")

    return results


# ===========================================================================
# STEP 2: FEATURE EXTRACTION
# ===========================================================================
def extract_features_fast(df):
    """Extract 10-feature vector from OHLCV data (vectorized)."""
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    n = len(close)

    log_ret = np.zeros(n)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    lr = pd.Series(log_ret)
    vol20 = lr.rolling(20).std().values
    vol50 = lr.rolling(50).std().values

    mom = np.zeros(n)
    mom[10:] = (close[10:] - close[:-10]) / close[:-10]

    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = pd.Series(gain).rolling(14).mean().values
    al = pd.Series(loss).rolling(14).mean().values
    rsi = 100 - (100 / (1 + ag / (al + 1e-10)))

    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0], tr3[0] = tr1[0], tr1[0]
    atr = pd.Series(np.maximum(tr1, np.maximum(tr2, tr3))).rolling(14).mean().values
    atr_n = atr / (close + 1e-10)

    signs = np.sign(log_ret)
    mag = pd.Series(signs).rolling(20).mean().values
    sus = pd.Series(mag).rolling(20).var().values
    eng = np.zeros(n)
    eng[1:] = -signs[1:] * signs[:-1]
    eng = pd.Series(eng).rolling(10).mean().values
    temp = vol20 * np.sqrt(252) * 100

    features = np.column_stack([log_ret, vol20, vol50, mom, rsi, atr_n, mag, sus, eng, temp])
    features = features[50:]
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def scale_features(features, mean=None, std=None):
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0) + 1e-8
    return np.clip((features - mean) / std, -3.0, 3.0), mean, std


# ===========================================================================
# STEP 3: TRAIN
# ===========================================================================
def _fit_candidate(sc_train, sc_val, n_states, cov_type, min_covar, seed):
    """Fit a single HMM candidate and return its metrics. Returns None on failure."""
    from hmmlearn.hmm import GaussianHMM
    try:
        model = GaussianHMM(
            n_components=n_states, covariance_type=cov_type,
            n_iter=N_ITER, tol=TOL, min_covar=min_covar,
            random_state=seed, verbose=False
        )
        model.fit(sc_train)
        ps_train = model.score(sc_train) / len(sc_train)
        ps_val = model.score(sc_val) / len(sc_val)
        gap = abs((ps_train - ps_val) / abs(ps_train)) * 100 if abs(ps_train) > 1e-10 else 999
        # BIC on validation set for model selection
        n_params = n_states * (n_states - 1)  # transition
        n_feat = sc_train.shape[1]
        if cov_type == "full":
            n_params += n_states * (n_feat + n_feat * (n_feat + 1) // 2)
        elif cov_type == "diag":
            n_params += n_states * (n_feat + n_feat)
        elif cov_type == "tied":
            n_params += n_states * n_feat + n_feat * (n_feat + 1) // 2
        bic_val = -2 * model.score(sc_val) + n_params * np.log(len(sc_val))
        return {
            "model": model, "ps_train": ps_train, "ps_val": ps_val,
            "gap": gap, "bic_val": bic_val, "converged": model.monitor_.converged,
            "iters": model.monitor_.iter, "n_states": n_states,
            "cov_type": cov_type, "min_covar": min_covar, "seed": seed,
        }
    except Exception as e:
        logger.debug(f"    Candidate failed ({n_states}s/{cov_type}/mc={min_covar}/seed={seed}): {e}")
        return None


def _label_states(model, n_states):
    """Assign regime labels based on emission means."""
    m = model.means_
    bull = int(np.argmax(m[:, 0]))  # highest log_return mean
    bear = int(np.argmin(m[:, 0]))  # lowest log_return mean
    if bull == bear:
        bear = int(np.argsort(m[:, 0])[0]) if np.argmax(m[:, 0]) != 0 else 1
    rem = [s for s in range(n_states) if s not in [bull, bear]]
    if n_states == 3:
        # 3-state: bull, bear, range
        rng = rem[0] if rem else 0
        return {bull: "TREND_BULL", bear: "TREND_BEAR", rng: "RANGE_STABLE"}
    elif n_states >= 4:
        # 4+ states: bull, bear, range, chaos (highest susceptibility)
        chaos = rem[int(np.argmax([m[s, 7] for s in rem]))] if rem else 0
        rng_list = [s for s in rem if s != chaos]
        rng = rng_list[0] if rng_list else 0
        labels = {bull: "TREND_BULL", bear: "TREND_BEAR", rng: "RANGE_STABLE", chaos: "CHAOS"}
        for s in range(n_states):
            if s not in labels:
                labels[s] = f"EXTRA_{s}"
        return labels
    return {s: f"STATE_{s}" for s in range(n_states)}


def train_symbol(symbol, data_path, version):
    """Train HMM for one symbol with adaptive model selection.

    Tries multiple (n_states, covariance_type, min_covar, seed) combos.
    Picks the candidate with lowest validation BIC that passes anti-overfit.
    If none pass, picks the candidate with lowest gap.
    """
    df = pd.read_parquet(data_path)
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING {symbol} | {len(df):,} bars | adaptive search")
    logger.info(f"{'='*60}")

    features = extract_features_fast(df)
    n = len(features)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    sc_train, mean, std = scale_features(features[:n_train])
    sc_val, _, _ = scale_features(features[n_train:n_train + n_val], mean, std)
    sc_test, _, _ = scale_features(features[n_train + n_val:], mean, std)
    sc_all, _, _ = scale_features(features, mean, std)

    logger.info(f"  Split: train={len(sc_train):,} val={len(sc_val):,} test={len(sc_test):,}")

    # Grid search over configs
    candidates = []
    total_configs = len(N_STATES_CANDIDATES) * len(COVAR_TYPES) * len(MIN_COVAR_OPTIONS) * N_SEEDS
    logger.info(f"  Searching {total_configs} configs...")

    for ns in N_STATES_CANDIDATES:
        for cov in COVAR_TYPES:
            for mc in MIN_COVAR_OPTIONS:
                for seed in RANDOM_SEEDS:
                    c = _fit_candidate(sc_train, sc_val, ns, cov, mc, seed)
                    if c is not None:
                        candidates.append(c)

    if not candidates:
        logger.error(f"  {symbol}: ALL candidates failed!")
        return {"symbol": symbol, "passed": False, "warnings": ["ALL_FAILED"]}

    logger.info(f"  {len(candidates)} candidates fitted successfully")

    # Evaluate each candidate
    for c in candidates:
        states = c["model"].predict(sc_all)
        dist = {str(s): round(np.mean(states == s) * 100, 2) for s in range(c["n_states"])}
        diag = [c["model"].transmat_[i][i] for i in range(c["n_states"])]
        persist = float(np.mean(diag))
        dom = max(float(v) for v in dist.values())
        c["dist"] = dist
        c["diag"] = diag
        c["persist"] = persist
        c["dom"] = dom
        c["warns"] = []
        if c["gap"] > MAX_GAP_PCT:
            c["warns"].append(f"GAP:{c['gap']:.1f}%")
        if dom > MAX_DOMINANCE:
            c["warns"].append(f"DOM:{dom:.1f}%")
        if persist < MIN_PERSISTENCE:
            c["warns"].append(f"PERSIST:{persist:.3f}")
        c["passed"] = len(c["warns"]) == 0

    # Selection strategy:
    # 1. Among passing candidates, pick lowest BIC (best model fit vs complexity)
    # 2. If none pass, pick lowest gap (least overfit)
    passing = [c for c in candidates if c["passed"]]
    if passing:
        best = min(passing, key=lambda c: c["bic_val"])
        logger.info(f"  PASS: {len(passing)} candidates pass. Best: {best['n_states']}s/{best['cov_type']}/mc={best['min_covar']}")
    else:
        # Fallback: pick model with lowest gap that has reasonable persistence
        reasonable = [c for c in candidates if c["persist"] >= 0.50]
        pool = reasonable if reasonable else candidates
        best = min(pool, key=lambda c: c["gap"])
        logger.info(f"  WARN: No candidate passed. Best gap: {best['gap']:.1f}% ({best['n_states']}s/{best['cov_type']}/mc={best['min_covar']})")

    model = best["model"]
    n_states = best["n_states"]
    ps_train, ps_val = best["ps_train"], best["ps_val"]
    ps_test = model.score(sc_test) / len(sc_test)
    gap = best["gap"]
    dist = best["dist"]
    persist = best["persist"]
    warns = best["warns"]

    labels = _label_states(model, n_states)

    logger.info(f"  LL: train={ps_train:.4f} val={ps_val:.4f} test={ps_test:.4f} gap={gap:.1f}%")
    logger.info(f"  Config: {n_states} states, {best['cov_type']} cov, min_covar={best['min_covar']}")
    logger.info(f"  States: {dist} | Persist: {persist:.3f}")
    logger.info(f"  Regimes: {labels}")

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{symbol}_M5_v{version}.pkl"
    mpath = MODEL_DIR / fname

    bundle = {
        "model": model, "scaler_mean": mean, "scaler_std": std,
        "feature_names": FEATURE_NAMES, "state_labels": labels,
        "n_states": n_states, "version": version, "symbol": symbol,
        "timeframe": "M5", "data_source": "histdata.com",
        "data_rows": len(df), "training_samples": len(sc_train),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "deployment_mode": "ising_only",
        "covariance_type": best["cov_type"],
        "min_covar": best["min_covar"],
        "search_stats": {
            "total_candidates": len(candidates),
            "passing_candidates": len(passing) if passing else 0,
        }
    }
    with open(mpath, "wb") as f:
        pickle.dump(bundle, f)
    with open(mpath, "rb") as f:
        cksum = hashlib.sha256(f.read()).hexdigest()

    logger.info(f"  Saved: {fname} ({mpath.stat().st_size:,} bytes)")

    meta = {
        "symbol": symbol, "version": version, "data_rows": len(df),
        "training_samples": len(sc_train),
        "n_states": n_states,
        "covariance_type": best["cov_type"],
        "min_covar": best["min_covar"],
        "per_sample_ll": {"train": round(ps_train, 6), "val": round(ps_val, 6), "test": round(ps_test, 6)},
        "gap_pct": round(gap, 2), "avg_persistence": round(persist, 4),
        "state_distribution": dist,
        "state_labels": {str(k): v for k, v in labels.items()},
        "transition_diagonal": [round(d, 4) for d in (best["diag"])],
        "warnings": warns, "passed": len(warns) == 0,
        "checksum_sha256": cksum,
        "file_size_bytes": mpath.stat().st_size,
        "search_stats": {
            "total_candidates": len(candidates),
            "passing_candidates": len(passing) if passing else 0,
        }
    }
    with open(MODEL_DIR / f"{symbol}_M5_v{version}.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="QuantMindX: Download data + Train HMM")
    parser.add_argument("--pairs", type=str, default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--years", type=str, default=",".join(DEFAULT_YEARS),
                        help="Years to download (e.g. 2015,2016,...,2024)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use cached data")
    args = parser.parse_args()

    pairs = [p.strip().upper() for p in args.pairs.split(",")]
    years = [y.strip() for y in args.years.split(",")]
    version = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.info("=" * 70)
    logger.info(f"QuantMindX HMM Download + Train")
    logger.info(f"Pairs: {pairs}")
    logger.info(f"Years: {years}")
    logger.info(f"Version: {version}")
    logger.info("=" * 70)

    # Step 1: Download
    if not args.skip_download:
        logger.info("\n--- STEP 1: DOWNLOADING DATA ---")
        download_histdata(pairs, years)
    else:
        logger.info("\n--- STEP 1: SKIPPED (using cached data) ---")

    # Step 2+3: Train
    logger.info("\n--- STEP 2: TRAINING ---")
    m5_dir = DATA_DIR / "real_m5_multi_year"
    results = []

    for pair in pairs:
        # Try M5 first, then H1
        m5_path = m5_dir / f"{pair}_M5.parquet"
        h1_path = m5_dir / f"{pair}_H1.parquet"

        if m5_path.exists():
            meta = train_symbol(pair, m5_path, version)
            results.append(meta)
        elif h1_path.exists():
            meta = train_symbol(pair, h1_path, version)
            results.append(meta)
        else:
            logger.warning(f"  No data found for {pair}, skipping")

    # Report
    passed = sum(1 for r in results if r.get("passed"))
    report = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "pairs": pairs, "years": years,
        "passed": passed, "total": len(results),
        "results": results
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info(f"Models: {MODEL_DIR}")
    total_size = sum(r.get("file_size_bytes", 0) for r in results)
    logger.info(f"Total size: {total_size:,} bytes ({total_size/1024:.1f} KB) — ship ONLY .pkl files")
    logger.info(f"Passed: {passed}/{len(results)}")
    for r in results:
        s = r["symbol"]
        g = r["gap_pct"]
        p = r["avg_persistence"]
        ok = "✓" if r["passed"] else "⚠"
        logger.info(f"  {ok} {s}: gap={g}% persist={p} samples={r['training_samples']:,}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
