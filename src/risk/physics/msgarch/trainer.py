"""MS-GARCH Trainer Module — Regime-Switching Volatility Model.

Trains GARCH(1,1) models and clusters conditional volatilities into
volatility regimes (LOW_VOL / MED_VOL / HIGH_VOL).  Complements HMM
by focusing on volatility dynamics rather than return dynamics.

Key design choices
------------------
* We save only the GARCH *parameters* (4 floats) + quantile boundaries,
  NOT the entire arch result object.  Model files are ~2-4 KB.
* Out-of-sample conditional vol is computed via the closed-form GARCH
  recursion  σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}  which is the same
  thing the arch library does internally.
* Anti-overfit is assessed by comparing the *regime distribution* between
  train / val / test splits (Jensen-Shannon divergence), not raw
  log-likelihood gaps which are meaningless across different data windows.

Integration points
------------------
- Uses extract_features_vectorized from HMM trainer
- Uses session segmentation from HMM trainer
- Outputs .pkl bundles compatible with MSGARCHSensor
"""

import logging
import pickle
import json
import hashlib
import sys as _sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, List, Any

logger = logging.getLogger(__name__)

# Try to import arch library
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch library not installed. MS-GARCH trainer will not function.")

# Import constants and utilities from HMM trainer
from src.risk.physics.hmm.trainer import (
    extract_features_vectorized,
    segment_by_session,
    segment_premium_sessions,
    SESSION_WINDOWS,
    PREMIUM_SESSIONS,
    FEATURE_NAMES,
)
from .utils import compute_vol_regimes, estimate_transition_matrix

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
DEFAULT_N_REGIMES = [2, 3, 4]
DEFAULT_TRAIN_SPLIT = 0.60
DEFAULT_VAL_SPLIT = 0.20

# Anti-overfit thresholds
MAX_GAP_PCT = 30.0       # Max regime-distribution divergence (%)
MAX_DOMINANCE = 85.0     # Max single regime share (%)
MIN_PERSISTENCE = 0.60   # Min avg diagonal of transition matrix


# ---------------------------------------------------------------------------
# GARCH recursion (pure NumPy, no arch needed at inference time)
# ---------------------------------------------------------------------------

def garch_conditional_vol(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    initial_var: float,
) -> np.ndarray:
    """Compute GARCH(1,1) conditional volatility for a return series.

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    Args:
        returns: 1-D array of log returns.
        omega, alpha, beta: GARCH(1,1) parameters.
        initial_var: Starting variance (σ²_0).

    Returns:
        1-D array of conditional standard deviations (same length as returns).
    """
    n = len(returns)
    var = np.empty(n, dtype=np.float64)
    var[0] = initial_var
    for t in range(1, n):
        var[t] = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
        # Floor to avoid numerical blow-up
        if var[t] < 1e-20:
            var[t] = 1e-20
    return np.sqrt(var)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two distributions (0-100 scale)."""
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm) * 100.0   # scale to %


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MSGARCHTrainer:
    """
    Production MS-GARCH trainer with adaptive regime selection.

    Fits GARCH(1,1) models and clusters conditional volatilities into
    regimes.  Uses BIC + anti-overfit checks for model selection.
    Saves lightweight param bundles (~2-4 KB).
    """

    def __init__(
        self, model_dir: Optional[Path] = None, config_path: Optional[str] = None
    ):
        self.model_dir = model_dir or Path("./models/msgarch")
        self.config = self._load_config(config_path) if config_path else {}
        self._last_result: Optional[Dict] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(path: str) -> Dict:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    @staticmethod
    def _get_version() -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def train(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        n_regimes_list: Optional[List[int]] = None,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train MS-GARCH model with adaptive regime selection.

        Returns a metadata dict (same shape as the .json sidecar).
        """
        if not ARCH_AVAILABLE:
            logger.error("arch library not available.")
            return {
                "symbol": symbol, "timeframe": timeframe, "passed": False,
                "warnings": ["ARCH_NOT_AVAILABLE"],
                "version": version or self._get_version(),
            }

        version = version or self._get_version()
        n_regimes_list = n_regimes_list or DEFAULT_N_REGIMES

        logger.info(f"{'=' * 60}")
        logger.info(f"TRAINING MS-GARCH {symbol}_{timeframe} | {len(df):,} bars")
        logger.info(f"{'=' * 60}")

        # --- Feature extraction + splits ---
        features = extract_features_vectorized(df)
        n = len(features)
        n_train = int(n * DEFAULT_TRAIN_SPLIT)
        n_val = int(n * DEFAULT_VAL_SPLIT)

        log_returns = features[:, 0]   # feature 0 = log returns
        lr_train = log_returns[:n_train]
        lr_val = log_returns[n_train:n_train + n_val]
        lr_test = log_returns[n_train + n_val:]

        scaler_mean = np.mean(features[:n_train], axis=0)
        scaler_std = np.std(features[:n_train], axis=0) + 1e-10

        logger.info(
            f"  train={len(lr_train):,}  val={len(lr_val):,}  "
            f"test={len(lr_test):,}"
        )

        # --- Fit GARCH(1,1) on training data ---
        _sys.stderr.write(f"  [MS-GARCH] Fitting GARCH(1,1) on {len(lr_train):,} bars...\n")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                am = arch_model(lr_train, vol="GARCH", p=1, q=1, rescale=False)
                result = am.fit(disp="off", show_warning=False)
            except Exception as e:
                logger.error(f"  GARCH fit failed: {e}")
                return {
                    "symbol": symbol, "timeframe": timeframe, "passed": False,
                    "warnings": [f"FIT_FAILED:{e}"], "version": version,
                }

        # Extract parameters
        params = dict(result.params)
        omega = float(params.get("omega", 1e-6))
        alpha = float(params.get("alpha[1]", 0.05))
        beta = float(params.get("beta[1]", 0.90))
        mu = float(params.get("mu", 0.0))
        bic_garch = float(result.bic)

        _sys.stderr.write(
            f"  [MS-GARCH] ω={omega:.2e}  α={alpha:.4f}  β={beta:.4f}  "
            f"BIC={bic_garch:.1f}\n"
        )

        # --- Compute conditional vol for ALL splits via recursion ---
        # Training: use arch's fitted conditional vol (more accurate)
        cv_obj = result.conditional_volatility
        cv_train = cv_obj.values if hasattr(cv_obj, "values") else np.asarray(cv_obj)
        initial_var = float(cv_train[-1]) ** 2

        # Validation + test: GARCH recursion from last training variance
        cv_val = garch_conditional_vol(lr_val, omega, alpha, beta, initial_var)
        last_val_var = float(cv_val[-1]) ** 2 if len(cv_val) > 0 else initial_var
        cv_test = garch_conditional_vol(lr_test, omega, alpha, beta, last_val_var)

        # --- Grid search over regime counts ---
        total = len(n_regimes_list)
        candidates = []

        for idx, n_reg in enumerate(n_regimes_list, 1):
            _sys.stderr.write(f"\r  [MS-GARCH] regime config {idx}/{total} (k={n_reg})   ")

            # Quantile boundaries from training conditional vol
            quantiles = np.linspace(0, 100, n_reg + 1)
            boundaries = np.percentile(cv_train, quantiles)

            # Assign regimes for each split using TRAINING boundaries
            reg_train = np.digitize(cv_train, boundaries[1:-1])
            reg_val = np.digitize(cv_val, boundaries[1:-1])
            reg_test = np.digitize(cv_test, boundaries[1:-1])

            # Regime distributions
            dist_train = np.array(
                [np.mean(reg_train == r) for r in range(n_reg)]
            )
            dist_val = np.array(
                [np.mean(reg_val == r) for r in range(n_reg)]
            )
            dist_test = np.array(
                [np.mean(reg_test == r) for r in range(n_reg)]
            )

            # Transition matrix on training data
            trans = estimate_transition_matrix(reg_train, n_reg)
            diag = [float(trans[i, i]) for i in range(n_reg)]
            persist = float(np.mean(diag))

            # Gap = max JS divergence between train-val and train-test
            gap_val = _js_divergence(dist_train, dist_val)
            gap_test = _js_divergence(dist_train, dist_test)
            gap = max(gap_val, gap_test)

            # Dominance
            dom = float(dist_train.max()) * 100.0

            # BIC-like score: penalise more regimes slightly
            n_params = 3 + n_reg  # GARCH params + regime boundaries
            score = bic_garch + n_params * np.log(len(lr_train))

            # Distribution as readable dict
            dist_dict = {
                str(s): round(float(dist_train[s]) * 100, 2) for s in range(n_reg)
            }

            # Anti-overfit warnings
            warns = []
            if gap > MAX_GAP_PCT:
                warns.append(f"GAP:{gap:.1f}%")
            if dom > MAX_DOMINANCE:
                warns.append(f"DOM:{dom:.1f}%")
            if persist < MIN_PERSISTENCE:
                warns.append(f"PERSIST:{persist:.3f}")

            candidates.append({
                "n_regimes": n_reg,
                "boundaries": boundaries.tolist(),
                "reg_train": reg_train,
                "reg_val": reg_val,
                "reg_test": reg_test,
                "trans_matrix": trans,
                "score": score,
                "gap": gap,
                "gap_val": gap_val,
                "gap_test": gap_test,
                "dist": dist_dict,
                "diag": diag,
                "persist": persist,
                "dom": dom,
                "warns": warns,
                "passed": len(warns) == 0,
            })

        _sys.stderr.write("\n")
        logger.info(f"  {len(candidates)} regime configurations evaluated")

        if not candidates:
            return {
                "symbol": symbol, "timeframe": timeframe, "passed": False,
                "warnings": ["ALL_CANDIDATES_FAILED"], "version": version,
            }

        # --- Select best ---
        passing = [c for c in candidates if c["passed"]]
        if passing:
            best = min(passing, key=lambda c: c["score"])
            logger.info(
                f"  PASS: {len(passing)} configs. "
                f"Best={best['n_regimes']}r (gap={best['gap']:.1f}%)"
            )
        else:
            pool = [c for c in candidates if c["persist"] >= 0.50] or candidates
            best = min(pool, key=lambda c: c["gap"])
            logger.info(
                f"  WARN: 0 passed. Best gap={best['gap']:.1f}% "
                f"({best['n_regimes']}r)"
            )

        n_regimes = best["n_regimes"]
        boundaries = best["boundaries"]

        # Create regime labels based on mean vol per regime
        mean_vol_per_regime = []
        for r in range(n_regimes):
            mask = best["reg_train"] == r
            mean_vol_per_regime.append(
                float(cv_train[mask].mean()) if mask.any() else 0.0
            )
        vol_order = np.argsort(mean_vol_per_regime)
        vol_regime_labels = {}
        vol_regime_labels[int(vol_order[0])] = "LOW_VOL"
        vol_regime_labels[int(vol_order[-1])] = "HIGH_VOL"
        for i in range(1, len(vol_order) - 1):
            vol_regime_labels[int(vol_order[i])] = "MED_VOL"

        logger.info(f"  Regime distribution: {best['dist']}")
        logger.info(f"  Persistence: {best['persist']:.3f}")
        logger.info(f"  Vol regimes: {vol_regime_labels}")

        # --- Save lightweight bundle ---
        self.model_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{symbol}_{timeframe}_v{version}.pkl"
        mpath = self.model_dir / fname

        bundle = {
            "symbol": symbol,
            "timeframe": timeframe,
            "n_regimes": n_regimes,
            # GARCH(1,1) params — this is ALL we need for inference
            "garch_params": {
                "omega": omega,
                "alpha": alpha,
                "beta": beta,
                "mu": mu,
            },
            "initial_var": initial_var,
            # Quantile boundaries for regime classification
            "boundaries": boundaries,
            # Scaler for feature normalisation
            "scaler_mean": scaler_mean.tolist(),
            "scaler_std": scaler_std.tolist(),
            # Regime labels
            "vol_regime_labels": vol_regime_labels,
            "transition_matrix": best["trans_matrix"].tolist(),
            # Meta
            "version": version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_rows": len(df),
            "training_samples": len(lr_train),
        }

        with open(mpath, "wb") as f:
            pickle.dump(bundle, f)

        with open(mpath, "rb") as f:
            cksum = hashlib.sha256(f.read()).hexdigest()

        fsize = mpath.stat().st_size
        logger.info(f"  Saved: {fname} ({fsize:,} bytes)")

        # --- Save metadata JSON ---
        meta = {
            "symbol": symbol,
            "timeframe": timeframe,
            "version": version,
            "data_rows": len(df),
            "training_samples": len(lr_train),
            "validation_samples": len(lr_val),
            "test_samples": len(lr_test),
            "n_regimes": n_regimes,
            "garch_bic": round(bic_garch, 2),
            "garch_params": {
                "omega": round(omega, 10),
                "alpha": round(alpha, 6),
                "beta": round(beta, 6),
                "mu": round(mu, 10),
            },
            "gap_pct": round(best["gap"], 2),
            "gap_val_pct": round(best["gap_val"], 2),
            "gap_test_pct": round(best["gap_test"], 2),
            "regime_distribution": best["dist"],
            "vol_regime_labels": vol_regime_labels,
            "transition_matrix": [
                [round(float(best["trans_matrix"][i, j]), 4) for j in range(n_regimes)]
                for i in range(n_regimes)
            ],
            "avg_persistence": round(best["persist"], 4),
            "dominance_pct": round(best["dom"], 2),
            "warnings": best["warns"],
            "anti_overfit_passed": best["passed"],
            "file_size_bytes": fsize,
            "checksum_sha256": cksum,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.model_dir / f"{symbol}_{timeframe}_v{version}.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._last_result = meta
        return meta

    # ------------------------------------------------------------------
    # Session-specific training
    # ------------------------------------------------------------------

    def train_session(
        self, symbol: str, timeframe: str, df: pd.DataFrame,
        session_name: str, version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train MS-GARCH on data filtered to a specific session window."""
        logger.info(f"\n  Segmenting for session: {session_name}")
        session_df = segment_by_session(df, session_name)

        if len(session_df) < 500:
            logger.warning(f"  {session_name}: Only {len(session_df)} bars, need 500+.")
            return {
                "symbol": symbol, "timeframe": timeframe,
                "session": session_name, "passed": False,
                "warnings": [f"INSUFFICIENT_DATA:{len(session_df)}"],
            }

        version = version or self._get_version()
        old_dir = self.model_dir
        self.model_dir = self.model_dir / "sessions"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        result = self.train(symbol, timeframe, session_df, version=version)
        result["session"] = session_name
        result["session_bars"] = len(session_df)

        # Rename model file to include session
        for ext in (".pkl", ".json"):
            src = self.model_dir / f"{symbol}_{timeframe}_v{version}{ext}"
            dst = self.model_dir / f"{symbol}_{timeframe}_{session_name}_v{version}{ext}"
            if src.exists():
                src.rename(dst)

        self.model_dir = old_dir
        return result

    def train_premium_sessions(
        self, symbol: str, timeframe: str, df: pd.DataFrame,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train MS-GARCH on combined premium session data."""
        logger.info(f"\n  Training premium sessions model for {symbol}")
        premium_df = segment_premium_sessions(df)

        if len(premium_df) < 500:
            logger.warning(f"  Premium: Only {len(premium_df)} bars, need 500+")
            return {
                "symbol": symbol, "timeframe": timeframe,
                "session": "PREMIUM_COMBINED", "passed": False,
                "warnings": [f"INSUFFICIENT_DATA:{len(premium_df)}"],
            }

        version = version or self._get_version()
        old_dir = self.model_dir
        self.model_dir = self.model_dir / "sessions"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        result = self.train(symbol, timeframe, premium_df, version=version)
        result["session"] = "PREMIUM_COMBINED"
        result["session_bars"] = len(premium_df)

        for ext in (".pkl", ".json"):
            src = self.model_dir / f"{symbol}_{timeframe}_v{version}{ext}"
            dst = self.model_dir / f"{symbol}_{timeframe}_PREMIUM_v{version}{ext}"
            if src.exists():
                src.rename(dst)

        self.model_dir = old_dir
        return result

    def train_all_sessions(
        self, symbol: str, timeframe: str, df: pd.DataFrame,
        sessions: Optional[List[str]] = None, version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Train session-specific MS-GARCH models."""
        if sessions is None:
            sessions = list(PREMIUM_SESSIONS) + ["UNIVERSAL"]

        version = version or self._get_version()
        results = []

        for session in sessions:
            if session == "UNIVERSAL":
                result = self.train(symbol, timeframe, df, version=version)
                result["session"] = "UNIVERSAL"
            elif session == "PREMIUM_COMBINED":
                result = self.train_premium_sessions(
                    symbol, timeframe, df, version=version
                )
            else:
                result = self.train_session(
                    symbol, timeframe, df, session, version=version
                )
            results.append(result)

        return results

    def train_per_symbol_timeframe(
        self, symbol: str, timeframe: str, df: Optional[pd.DataFrame] = None
    ) -> Tuple[Path, str]:
        """Train per-symbol-timeframe MS-GARCH model."""
        if df is None:
            logger.error(f"No data provided for {symbol}_{timeframe}")
            v = self._get_version()
            return self.model_dir / f"{symbol}_{timeframe}_v{v}.pkl", v

        result = self.train(symbol, timeframe, df)
        v = result["version"]
        return self.model_dir / f"{symbol}_{timeframe}_v{v}.pkl", v

    @property
    def last_result(self) -> Optional[Dict]:
        return self._last_result


__all__ = ["MSGARCHTrainer", "garch_conditional_vol"]
