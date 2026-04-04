"""
HMM Trainer Module — PRODUCTION
================================

Real GaussianHMM training with adaptive model selection.
Replaces the placeholder that just wrote text files.

Integrates with:
    - HMMFeatureExtractor (src/risk/physics/hmm/models.py)
    - FeatureConfig / FeatureScaler
    - config/hmm_config.json
    - Database models (HMMModel, HMMDeployment)

Usage:
    from src.risk.physics.hmm.trainer import HMMTrainer
    trainer = HMMTrainer()
    result = trainer.train_per_symbol_timeframe("EURUSD", "M5", data_df)
"""

import logging
import hashlib
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, List, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — overridden by config/hmm_config.json when loaded
# ---------------------------------------------------------------------------
# Full grid: 3×3×3×3 = 81 configs
DEFAULT_N_STATES = [3, 4, 5]
DEFAULT_COV_TYPES = ["full", "tied", "diag"]
DEFAULT_MIN_COVARS = [1e-3, 1e-2, 5e-2]
DEFAULT_SEEDS = [42, 123, 7]
DEFAULT_N_ITER = 200
DEFAULT_TOL = 1e-4
DEFAULT_TRAIN_SPLIT = 0.60
DEFAULT_VAL_SPLIT = 0.20

# Thresholds (relaxed for real forex data)
MAX_GAP_PCT = 30.0
MAX_DOMINANCE = 85.0
MIN_PERSISTENCE = 0.60

FEATURE_NAMES = [
    "log_returns", "rolling_volatility_20", "rolling_volatility_50",
    "price_momentum_10", "rsi", "atr_normalized",
    "magnetization", "susceptibility", "energy", "temperature"
]


def extract_features_vectorized(df: pd.DataFrame) -> np.ndarray:
    """Extract 10-feature vector from OHLCV data (fully vectorized, fast).

    Features: log_returns, vol20, vol50, momentum10, rsi14, atr_norm,
              magnetization, susceptibility, energy, temperature.
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    n = len(close)

    # Log returns
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    lr = pd.Series(log_ret)

    # Rolling volatility
    vol20 = lr.rolling(20).std().values
    vol50 = lr.rolling(50).std().values

    # Price momentum
    mom = np.zeros(n)
    mom[10:] = (close[10:] - close[:-10]) / close[:-10]

    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = pd.Series(gain).rolling(14).mean().values
    al = pd.Series(loss).rolling(14).mean().values
    rsi = 100 - (100 / (1 + ag / (al + 1e-10)))

    # ATR normalized
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0], tr3[0] = tr1[0], tr1[0]
    atr = pd.Series(np.maximum(tr1, np.maximum(tr2, tr3))).rolling(14).mean().values
    atr_n = atr / (close + 1e-10)

    # Ising-inspired features
    signs = np.sign(log_ret)
    mag = pd.Series(signs).rolling(20).mean().values
    sus = pd.Series(mag).rolling(20).var().values
    eng = np.zeros(n)
    eng[1:] = -signs[1:] * signs[:-1]
    eng = pd.Series(eng).rolling(10).mean().values
    temp = vol20 * np.sqrt(252) * 100

    features = np.column_stack([log_ret, vol20, vol50, mom, rsi, atr_n,
                                mag, sus, eng, temp])
    features = features[50:]  # Skip warmup
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _scale(features, mean=None, std=None):
    """Standard scale with clip at ±3σ."""
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0) + 1e-8
    return np.clip((features - mean) / std, -3.0, 3.0), mean, std


def _label_states(model_means: np.ndarray, n_states: int) -> Dict[int, str]:
    """Assign regime labels matching RegimeType enum in src/events/regime.py.

    Maps HMM hidden states to the 8 system regimes:
        TREND_BULL, TREND_BEAR, TREND_STABLE, RANGE_STABLE,
        RANGE_VOLATILE, BREAKOUT_UP, BREAKOUT_DOWN, CHAOS

    Strategy:
        - Feature 0 (log_returns): direction → BULL vs BEAR
        - Feature 1 (vol_20): volatility level → STABLE vs VOLATILE
        - Feature 7 (susceptibility): phase transition proximity → BREAKOUT/CHAOS
        - Feature 3 (momentum): trend strength → TREND vs RANGE
    """
    m = model_means
    # Sort states by return direction
    ret_order = np.argsort(m[:, 0])  # ascending: most bearish first
    bear_idx = int(ret_order[0])
    bull_idx = int(ret_order[-1])

    rem = [int(s) for s in range(n_states) if s not in [bull_idx, bear_idx]]

    if n_states == 3:
        # 3 states: bull, bear, and one middle state
        mid = rem[0] if rem else 0
        mid_vol = m[mid, 1]   # vol_20
        mid_sus = m[mid, 7]   # susceptibility
        # High susceptibility → CHAOS, low vol → RANGE_STABLE, else TREND_STABLE
        if mid_sus > np.mean(m[:, 7]):
            return {bull_idx: "TREND_BULL", bear_idx: "TREND_BEAR", mid: "CHAOS"}
        elif mid_vol < np.median(m[:, 1]):
            return {bull_idx: "TREND_BULL", bear_idx: "TREND_BEAR", mid: "RANGE_STABLE"}
        else:
            return {bull_idx: "TREND_BULL", bear_idx: "TREND_BEAR", mid: "TREND_STABLE"}

    elif n_states == 4:
        # 4 states: bull, bear, + 2 remaining
        # Classify remaining by volatility and susceptibility
        if len(rem) >= 2:
            # Higher susceptibility → CHAOS, lower → RANGE_STABLE
            sus_order = sorted(rem, key=lambda s: m[s, 7])
            low_sus, high_sus = sus_order[0], sus_order[-1]
            vol_low = m[low_sus, 1]
            return {
                bull_idx: "TREND_BULL", bear_idx: "TREND_BEAR",
                low_sus: "RANGE_STABLE" if vol_low < np.median(m[:, 1]) else "RANGE_VOLATILE",
                high_sus: "CHAOS"
            }
        else:
            mid = rem[0] if rem else 0
            return {bull_idx: "TREND_BULL", bear_idx: "TREND_BEAR",
                    mid: "RANGE_STABLE", 0: "CHAOS"}

    elif n_states == 5:
        # 5 states: bull, bear, + 3 remaining
        # Sort remaining by susceptibility (proxy for regime instability)
        if len(rem) >= 3:
            sus_order = sorted(rem, key=lambda s: m[s, 7])
            stable = sus_order[0]      # lowest susceptibility
            volatile = sus_order[1]    # medium susceptibility
            chaotic = sus_order[2]     # highest susceptibility
            # Classify stable/volatile by momentum
            stable_label = "RANGE_STABLE" if abs(m[stable, 3]) < abs(m[volatile, 3]) else "TREND_STABLE"
            volatile_label = "RANGE_VOLATILE" if stable_label == "RANGE_STABLE" else "RANGE_STABLE"
            return {
                bull_idx: "TREND_BULL", bear_idx: "TREND_BEAR",
                stable: stable_label, volatile: volatile_label,
                chaotic: "CHAOS"
            }
        # Fallback
        labels = {bull_idx: "TREND_BULL", bear_idx: "TREND_BEAR"}
        for i, s in enumerate(rem):
            labels[s] = ["RANGE_STABLE", "RANGE_VOLATILE", "CHAOS"][i % 3]
        return labels

    # n_states > 5: label first 5, rest as EXTRA
    labels = _label_states(m[:5], 5) if n_states > 5 else {}
    for s in range(n_states):
        if s not in labels:
            labels[s] = f"EXTRA_{s}"
    return labels


# ---------------------------------------------------------------------------
# Session-aware data segmentation
# ---------------------------------------------------------------------------
# Canonical session windows (UTC hours) from src/router/sessions.py
SESSION_WINDOWS = {
    "SYDNEY_OPEN":           (21, 23),
    "SYDNEY_TOKYO_OVERLAP":  (23, 0),
    "TOKYO_OPEN":            (0, 3),
    "TOKYO_LONDON_OVERLAP":  (7, 9),     # PREMIUM
    "LONDON_OPEN":           (8, 10),     # PREMIUM (overlaps TLO slightly)
    "LONDON_MID":            (10, 12),
    "INTER_SESSION_COOLDOWN":(12, 13),
    "LONDON_NY_OVERLAP":     (13, 16),    # PREMIUM
    "NY_WIND_DOWN":          (16, 20),
    "DEAD_ZONE":             (20, 21),
}

PREMIUM_SESSIONS = {"TOKYO_LONDON_OVERLAP", "LONDON_OPEN", "LONDON_NY_OVERLAP"}

# Bot type → suitable regimes (from SESSION_BOT_MIX + REGIME_STRATEGY_MAP)
STRATEGY_REGIME_SUITABILITY = {
    "MR":  ["TREND_STABLE", "RANGE_STABLE"],                          # Mean Reversion
    "MOM": ["TREND_BULL", "TREND_BEAR", "BREAKOUT_UP", "BREAKOUT_DOWN"],  # Momentum
    "ORB": ["BREAKOUT_UP", "BREAKOUT_DOWN", "TREND_BULL", "TREND_BEAR"],  # Order Block
    "TC":  ["TREND_BULL", "TREND_BEAR", "TREND_STABLE"],              # Trend Continuation
    "scalp":      ["RANGE_STABLE", "RANGE_VOLATILE"],
    "short_term": ["RANGE_VOLATILE"],
    "range_trade":["TREND_STABLE", "RANGE_STABLE"],
}


def segment_by_session(df: pd.DataFrame, session_name: str) -> pd.DataFrame:
    """Filter OHLCV data to only include bars within a session window.

    Args:
        df: OHLCV DataFrame with 'time' column (datetime)
        session_name: Key from SESSION_WINDOWS

    Returns:
        Filtered DataFrame (only bars during the session hours)
    """
    if session_name not in SESSION_WINDOWS:
        raise ValueError(f"Unknown session: {session_name}. Valid: {list(SESSION_WINDOWS.keys())}")

    start_h, end_h = SESSION_WINDOWS[session_name]
    times = pd.to_datetime(df["time"])
    hours = times.dt.hour

    if start_h < end_h:
        mask = (hours >= start_h) & (hours < end_h)
    else:
        # Wraps midnight (e.g., 23:00 → 00:00)
        mask = (hours >= start_h) | (hours < end_h)

    filtered = df[mask].copy()
    logger.info(f"  Session {session_name}: {len(filtered):,}/{len(df):,} bars "
                f"({len(filtered)/max(len(df),1)*100:.1f}%)")
    return filtered


def segment_premium_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Filter data to only premium session hours (combined)."""
    times = pd.to_datetime(df["time"])
    hours = times.dt.hour
    # Premium: 07-10 (TLO+LO) + 13-16 (LNO)
    mask = ((hours >= 7) & (hours < 10)) | ((hours >= 13) & (hours < 16))
    filtered = df[mask].copy()
    logger.info(f"  Premium sessions: {len(filtered):,}/{len(df):,} bars "
                f"({len(filtered)/max(len(df),1)*100:.1f}%)")
    return filtered


class HMMTrainer:
    """Production HMM trainer with adaptive model selection.

    Tries multiple (n_states, cov_type, min_covar, seed) combinations.
    Picks the best candidate by validation BIC that passes anti-overfit.
    Falls back to lowest-gap candidate if none pass.
    """

    def __init__(self, model_dir: Optional[Path] = None,
                 config_path: Optional[str] = None):
        self.model_dir = model_dir or Path("./models/hmm")
        self.config = self._load_config(config_path) if config_path else {}
        self._last_result: Optional[Dict] = None

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    def _get_version(self) -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------
    def train(self, symbol: str, timeframe: str,
              df: pd.DataFrame,
              n_states_list: Optional[List[int]] = None,
              cov_types: Optional[List[str]] = None,
              min_covars: Optional[List[float]] = None,
              seeds: Optional[List[int]] = None,
              version: Optional[str] = None,
              ) -> Dict[str, Any]:
        """Train HMM with adaptive grid search.

        Args:
            symbol: e.g. "EURUSD"
            timeframe: e.g. "M5"
            df: OHLCV DataFrame with columns [time, open, high, low, close, ...]
            n_states_list: list of state counts to try
            cov_types: list of covariance types
            min_covars: list of min_covar values
            seeds: random seeds
            version: model version string

        Returns:
            Training result dict with model path, metrics, warnings, etc.
        """
        from hmmlearn.hmm import GaussianHMM

        version = version or self._get_version()
        n_states_list = n_states_list or DEFAULT_N_STATES
        cov_types = cov_types or DEFAULT_COV_TYPES
        min_covars = min_covars or DEFAULT_MIN_COVARS
        seeds = seeds or DEFAULT_SEEDS

        logger.info(f"{'='*60}")
        logger.info(f"TRAINING {symbol}_{timeframe} | {len(df):,} bars | adaptive")
        logger.info(f"{'='*60}")

        # Feature extraction
        features = extract_features_vectorized(df)
        n = len(features)
        n_train = int(n * DEFAULT_TRAIN_SPLIT)
        n_val = int(n * DEFAULT_VAL_SPLIT)

        sc_train, mean, std = _scale(features[:n_train])
        sc_val, _, _ = _scale(features[n_train:n_train + n_val], mean, std)
        sc_test, _, _ = _scale(features[n_train + n_val:], mean, std)
        sc_all, _, _ = _scale(features, mean, std)

        logger.info(f"  train={len(sc_train):,} val={len(sc_val):,} test={len(sc_test):,}")

        # Grid search — suppress hmmlearn convergence warnings
        candidates = []
        total = len(n_states_list) * len(cov_types) * len(min_covars) * len(seeds)

        import warnings as _w
        import time as _time
        import sys as _sys
        _hmm_logger = logging.getLogger("hmmlearn")
        _prev_level = _hmm_logger.level
        _hmm_logger.setLevel(logging.CRITICAL)

        done = 0
        _t0 = _time.time()
        # Show initial ETA line
        _sys.stderr.write(f"\r  [0/{total}] starting grid search...\n")
        _sys.stderr.flush()

        for ns in n_states_list:
            for cov in cov_types:
                for mc in min_covars:
                    for seed in seeds:
                        done += 1
                        try:
                            with _w.catch_warnings():
                                _w.simplefilter("ignore")
                                mdl = GaussianHMM(
                                    n_components=ns, covariance_type=cov,
                                    n_iter=DEFAULT_N_ITER, tol=DEFAULT_TOL,
                                    min_covar=mc, random_state=seed, verbose=False
                                )
                                mdl.fit(sc_train)

                            # Live ETA on stderr
                            elapsed = _time.time() - _t0
                            avg_per = elapsed / done
                            remaining = avg_per * (total - done)
                            eta_min = remaining / 60
                            _sys.stderr.write(
                                f"\r  [{done}/{total}] {len(candidates)} ok | "
                                f"elapsed {elapsed:.0f}s | ~{eta_min:.1f}m left   "
                            )
                            _sys.stderr.flush()

                            ps_tr = mdl.score(sc_train) / len(sc_train)
                            ps_va = mdl.score(sc_val) / len(sc_val)
                            gap = abs((ps_tr - ps_va) / abs(ps_tr)) * 100 if abs(ps_tr) > 1e-10 else 999

                            n_feat = sc_train.shape[1]
                            n_params = ns * (ns - 1)
                            if cov == "full":
                                n_params += ns * (n_feat + n_feat * (n_feat + 1) // 2)
                            elif cov == "diag":
                                n_params += ns * n_feat * 2
                            elif cov == "tied":
                                n_params += ns * n_feat + n_feat * (n_feat + 1) // 2
                            bic = -2 * mdl.score(sc_val) + n_params * np.log(len(sc_val))

                            candidates.append({
                                "model": mdl, "ps_train": ps_tr, "ps_val": ps_va,
                                "gap": gap, "bic": bic, "n_states": ns,
                                "cov_type": cov, "min_covar": mc, "seed": seed,
                                "converged": mdl.monitor_.converged,
                            })
                        except Exception:
                            # Still update progress for failed candidates
                            elapsed = _time.time() - _t0
                            avg_per = elapsed / done
                            remaining = avg_per * (total - done)
                            eta_min = remaining / 60
                            _sys.stderr.write(
                                f"\r  [{done}/{total}] {len(candidates)} ok | "
                                f"elapsed {elapsed:.0f}s | ~{eta_min:.1f}m left   "
                            )
                            _sys.stderr.flush()

        total_elapsed = _time.time() - _t0
        _sys.stderr.write(f"\r  Grid complete: {len(candidates)}/{total} fitted "
                          f"in {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)       \n")
        _sys.stderr.flush()
        _hmm_logger.setLevel(_prev_level)

        if not candidates:
            logger.error(f"  {symbol}: ALL {total} candidates failed!")
            return {"symbol": symbol, "timeframe": timeframe, "passed": False,
                    "warnings": ["ALL_CANDIDATES_FAILED"], "version": version}

        logger.info(f"  {len(candidates)}/{total} candidates fitted")

        # Evaluate
        for c in candidates:
            states = c["model"].predict(sc_all)
            ns = c["n_states"]
            dist = {str(s): round(float(np.mean(states == s)) * 100, 2)
                    for s in range(ns)}
            diag = [float(c["model"].transmat_[i][i]) for i in range(ns)]
            persist = float(np.mean(diag))
            dom = max(float(v) for v in dist.values())
            c.update({"dist": dist, "diag": diag, "persist": persist, "dom": dom})
            c["warns"] = []
            if c["gap"] > MAX_GAP_PCT:
                c["warns"].append(f"GAP:{c['gap']:.1f}%")
            if dom > MAX_DOMINANCE:
                c["warns"].append(f"DOM:{dom:.1f}%")
            if persist < MIN_PERSISTENCE:
                c["warns"].append(f"PERSIST:{persist:.3f}")
            c["passed"] = len(c["warns"]) == 0

        passing = [c for c in candidates if c["passed"]]
        if passing:
            best = min(passing, key=lambda c: c["bic"])
            logger.info(f"  PASS: {len(passing)} candidates. "
                        f"Best={best['n_states']}s/{best['cov_type']}/mc={best['min_covar']}")
        else:
            pool = [c for c in candidates if c["persist"] >= 0.50] or candidates
            best = min(pool, key=lambda c: c["gap"])
            logger.info(f"  WARN: 0 passed. Best gap={best['gap']:.1f}% "
                        f"({best['n_states']}s/{best['cov_type']}/mc={best['min_covar']})")

        model = best["model"]
        ns = best["n_states"]
        ps_test = model.score(sc_test) / len(sc_test)
        labels = _label_states(model.means_, ns)

        logger.info(f"  LL: train={best['ps_train']:.4f} val={best['ps_val']:.4f} "
                    f"test={ps_test:.4f} gap={best['gap']:.1f}%")
        logger.info(f"  States: {best['dist']} | Persist: {best['persist']:.3f}")
        logger.info(f"  Labels: {labels}")

        # Save .pkl bundle
        self.model_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{symbol}_{timeframe}_v{version}.pkl"
        mpath = self.model_dir / fname

        bundle = {
            "model": model, "scaler_mean": mean, "scaler_std": std,
            "feature_names": FEATURE_NAMES, "state_labels": labels,
            "n_states": ns, "version": version, "symbol": symbol,
            "timeframe": timeframe, "data_rows": len(df),
            "training_samples": len(sc_train),
            "covariance_type": best["cov_type"],
            "min_covar": best["min_covar"],
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "deployment_mode": "ising_only",
        }
        with open(mpath, "wb") as f:
            pickle.dump(bundle, f)
        with open(mpath, "rb") as f:
            cksum = hashlib.sha256(f.read()).hexdigest()

        logger.info(f"  Saved: {fname} ({mpath.stat().st_size:,} bytes)")

        # Save .json metadata
        meta = {
            "symbol": symbol, "timeframe": timeframe, "version": version,
            "data_rows": len(df), "training_samples": len(sc_train),
            "validation_samples": len(sc_val), "test_samples": len(sc_test),
            "n_states": ns, "covariance_type": best["cov_type"],
            "min_covar": best["min_covar"],
            "per_sample_ll": {
                "train": round(best["ps_train"], 6),
                "val": round(best["ps_val"], 6),
                "test": round(ps_test, 6)
            },
            "per_sample_gap_pct": round(best["gap"], 2),
            "bic": round(best["bic"], 2),
            "state_distribution": best["dist"],
            "state_labels": {str(k): v for k, v in labels.items()},
            "transition_matrix": [[round(float(model.transmat_[i][j]), 4)
                                   for j in range(ns)] for i in range(ns)],
            "avg_persistence": round(best["persist"], 4),
            "converged": best["converged"],
            "warnings": best["warns"],
            "anti_overfit_passed": best["passed"],
            "file_size_bytes": mpath.stat().st_size,
            "checksum_sha256": cksum,
            "deployment_mode": "ising_only",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "search_stats": {
                "total_candidates": len(candidates),
                "passing_candidates": len(passing) if passing else 0,
            },
        }
        with open(self.model_dir / f"{symbol}_{timeframe}_v{version}.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._last_result = meta
        return meta

    # ------------------------------------------------------------------
    # Convenience wrappers matching old interface
    # ------------------------------------------------------------------
    def train_universal(self, data: Optional[Dict[str, pd.DataFrame]] = None
                        ) -> Tuple[Path, str]:
        """Train universal model (placeholder-compatible interface)."""
        version = self._get_version()
        logger.warning("train_universal: Provide per-symbol data via train() instead")
        return self.model_dir / f"universal_v{version}.pkl", version

    def train_per_symbol(self, symbol: str,
                         df: Optional[pd.DataFrame] = None,
                         timeframe: str = "M5") -> Tuple[Path, str]:
        """Train per-symbol model."""
        if df is None:
            logger.error(f"No data provided for {symbol}")
            version = self._get_version()
            return self.model_dir / f"{symbol}_v{version}.pkl", version
        result = self.train(symbol, timeframe, df)
        version = result["version"]
        return self.model_dir / f"{symbol}_{timeframe}_v{version}.pkl", version

    def train_per_symbol_timeframe(self, symbol: str, timeframe: str,
                                   df: Optional[pd.DataFrame] = None
                                   ) -> Tuple[Path, str]:
        """Train per-symbol-timeframe model."""
        if df is None:
            logger.error(f"No data provided for {symbol}_{timeframe}")
            version = self._get_version()
            return self.model_dir / f"{symbol}_{timeframe}_v{version}.pkl", version
        result = self.train(symbol, timeframe, df)
        version = result["version"]
        return self.model_dir / f"{symbol}_{timeframe}_v{version}.pkl", version

    # ------------------------------------------------------------------
    # Session-aware training
    # ------------------------------------------------------------------
    def train_session(self, symbol: str, timeframe: str,
                      df: pd.DataFrame, session_name: str,
                      version: Optional[str] = None) -> Dict[str, Any]:
        """Train HMM on data filtered to a specific session window.

        This produces models optimized for the volatility/regime patterns
        of that session (e.g., London Open has different dynamics than Tokyo).

        Args:
            symbol: e.g. "EURUSD"
            timeframe: e.g. "M5"
            df: Full OHLCV DataFrame with 'time' column
            session_name: Key from SESSION_WINDOWS
            version: model version string

        Returns:
            Training result dict
        """
        logger.info(f"\n  Segmenting for session: {session_name}")
        session_df = segment_by_session(df, session_name)

        if len(session_df) < 2000:
            logger.warning(f"  {session_name}: Only {len(session_df)} bars, "
                           f"need 2000+. Skipping.")
            return {"symbol": symbol, "timeframe": timeframe,
                    "session": session_name, "passed": False,
                    "warnings": [f"INSUFFICIENT_DATA:{len(session_df)}"]}

        # Use session name in the model filename
        version = version or self._get_version()
        old_dir = self.model_dir
        self.model_dir = self.model_dir / "sessions"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        result = self.train(symbol, timeframe, session_df, version=version)
        result["session"] = session_name
        result["session_bars"] = len(session_df)
        result["full_data_bars"] = len(df)

        # Rename model file to include session
        src_pkl = self.model_dir / f"{symbol}_{timeframe}_v{version}.pkl"
        dst_pkl = self.model_dir / f"{symbol}_{timeframe}_{session_name}_v{version}.pkl"
        src_json = self.model_dir / f"{symbol}_{timeframe}_v{version}.json"
        dst_json = self.model_dir / f"{symbol}_{timeframe}_{session_name}_v{version}.json"
        if src_pkl.exists():
            src_pkl.rename(dst_pkl)
        if src_json.exists():
            src_json.rename(dst_json)

        self.model_dir = old_dir
        return result

    def train_premium_sessions(self, symbol: str, timeframe: str,
                               df: pd.DataFrame,
                               version: Optional[str] = None) -> Dict[str, Any]:
        """Train HMM on combined premium session data only.

        Premium sessions: TOKYO_LONDON_OVERLAP, LONDON_OPEN, LONDON_NY_OVERLAP

        Args:
            symbol: e.g. "EURUSD"
            timeframe: e.g. "M5"
            df: Full OHLCV DataFrame
            version: model version string

        Returns:
            Training result dict
        """
        logger.info(f"\n  Training premium sessions model for {symbol}")
        premium_df = segment_premium_sessions(df)

        if len(premium_df) < 2000:
            logger.warning(f"  Premium: Only {len(premium_df)} bars, need 2000+")
            return {"symbol": symbol, "timeframe": timeframe,
                    "session": "PREMIUM_COMBINED", "passed": False,
                    "warnings": [f"INSUFFICIENT_DATA:{len(premium_df)}"]}

        version = version or self._get_version()
        old_dir = self.model_dir
        self.model_dir = self.model_dir / "sessions"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        result = self.train(symbol, timeframe, premium_df, version=version)
        result["session"] = "PREMIUM_COMBINED"
        result["session_bars"] = len(premium_df)

        # Rename to include PREMIUM tag
        src_pkl = self.model_dir / f"{symbol}_{timeframe}_v{version}.pkl"
        dst_pkl = self.model_dir / f"{symbol}_{timeframe}_PREMIUM_v{version}.pkl"
        src_json = self.model_dir / f"{symbol}_{timeframe}_v{version}.json"
        dst_json = self.model_dir / f"{symbol}_{timeframe}_PREMIUM_v{version}.json"
        if src_pkl.exists():
            src_pkl.rename(dst_pkl)
        if src_json.exists():
            src_json.rename(dst_json)

        self.model_dir = old_dir
        return result

    def train_all_sessions(self, symbol: str, timeframe: str,
                           df: pd.DataFrame,
                           sessions: Optional[List[str]] = None,
                           version: Optional[str] = None
                           ) -> List[Dict[str, Any]]:
        """Train session-specific models for multiple sessions.

        Args:
            symbol: e.g. "EURUSD"
            timeframe: e.g. "M5"
            df: Full OHLCV DataFrame
            sessions: List of session names (defaults to premium sessions)
            version: model version string

        Returns:
            List of training result dicts
        """
        if sessions is None:
            sessions = list(PREMIUM_SESSIONS) + ["UNIVERSAL"]

        version = version or self._get_version()
        results = []

        for session in sessions:
            if session == "UNIVERSAL":
                # Train on all data
                result = self.train(symbol, timeframe, df, version=version)
                result["session"] = "UNIVERSAL"
                results.append(result)
            elif session == "PREMIUM_COMBINED":
                result = self.train_premium_sessions(symbol, timeframe, df,
                                                      version=version)
                results.append(result)
            else:
                result = self.train_session(symbol, timeframe, df,
                                            session, version=version)
                results.append(result)

        return results

    @property
    def last_result(self) -> Optional[Dict]:
        return self._last_result


__all__ = ["HMMTrainer", "extract_features_vectorized",
           "segment_by_session", "segment_premium_sessions",
           "SESSION_WINDOWS", "PREMIUM_SESSIONS",
           "STRATEGY_REGIME_SUITABILITY"]
