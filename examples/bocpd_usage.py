"""BOCPD (Bayesian Online Changepoint Detection) Usage Examples.

Demonstrates:
    1. Basic online detection
    2. Calibration from historical data
    3. Integration with HMM features
    4. Save/load functionality
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.risk.physics.bocpd import (
    BOCPDDetector,
    ConstantHazard,
    LogisticHazard,
    StudentTObservation,
)


def example_basic_detection():
    """Example 1: Basic online detection."""
    print("=" * 60)
    print("Example 1: Basic Online Detection")
    print("=" * 60)

    # Create detector
    detector = BOCPDDetector(
        threshold=0.5,  # 50% probability threshold
        min_run_length=10,  # Require at least 10 bars before signaling
    )

    # Simulate data: normal market + sudden changepoint
    np.random.seed(42)
    data = np.concatenate([
        np.random.randn(100) * 0.01,  # Normal market, small returns
        np.random.randn(100) * 0.05,  # Volatility spike (changepoint at ~bar 100)
        np.random.randn(100) * 0.01,  # Back to normal
    ])

    changepoints = []
    for i, x in enumerate(data):
        result = detector.update(x)
        if result["is_changepoint"]:
            changepoints.append(i)
            print(f"  Changepoint detected at bar {i}, prob={result['changepoint_prob']:.4f}")

    print(f"\nTotal observations: {detector.n_observations}")
    print(f"Total changepoints detected: {detector.n_changepoints_detected}")
    print(f"Average run length: {detector.n_observations / max(detector.n_changepoints_detected, 1):.1f} bars")


def example_calibration():
    """Example 2: Calibration from historical data."""
    print("\n" + "=" * 60)
    print("Example 2: Calibration from Historical Data")
    print("=" * 60)

    # Simulate historical forex data (e.g., M5 EURUSD)
    np.random.seed(42)
    historical_data = np.random.randn(1000) * 0.01  # Realistic forex returns

    # Create detector
    detector = BOCPDDetector()

    # Calibrate: finds optimal lambda
    # Target: ~3 changepoints per 1000 bars (1-3 per trading day)
    calib = detector.calibrate(historical_data, symbol="EURUSD_M5")

    print(f"Calibration Results:")
    print(f"  Optimal lambda: {calib['optimal_lambda']:.1f}")
    print(f"    (Means average run length of {calib['optimal_lambda']:.0f} bars)")
    print(f"  Changepoints found: {calib['n_changepoints_found']}")
    print(f"  Average run length: {calib['avg_run_length']:.1f} bars")

    # Save calibration
    model_path = Path("/tmp/bocpd_eurusd_m5.json")
    detector.save(model_path)
    print(f"  Saved to {model_path}")


def example_hazard_functions():
    """Example 3: Different hazard functions."""
    print("\n" + "=" * 60)
    print("Example 3: Comparing Hazard Functions")
    print("=" * 60)

    h_const = ConstantHazard(lam=250.0)
    h_logistic = LogisticHazard(tau_0=200.0, k=0.02)

    print("Changepoint probability by run length:\n")
    print("Run Length | ConstantHazard | LogisticHazard")
    print("-" * 50)
    for t in [0, 50, 100, 200, 300, 400, 500]:
        p_const = h_const(t)
        p_logistic = h_logistic(t)
        print(f"  {t:4d}     |    {p_const:8.6f}     |   {p_logistic:8.6f}")

    print("\nInterpretation:")
    print("  - ConstantHazard: Memoryless, same probability at all run lengths")
    print("  - LogisticHazard: Increases with run length (longer runs → more likely to end)")


def example_predict_regime():
    """Example 4: Using predict_regime interface."""
    print("\n" + "=" * 60)
    print("Example 4: Using predict_regime Interface")
    print("=" * 60)

    detector = BOCPDDetector(threshold=0.4)

    # Simulate 10-feature vector from HMM trainer
    # Features: log_returns, vol20, vol50, momentum10, rsi, atr_norm,
    #           magnetization, susceptibility, energy, temperature
    print("Processing feature vectors:\n")

    for i in range(10):
        # Random feature vector (in practice, from extract_features_vectorized)
        features = np.random.randn(10)
        features[0] *= 0.01  # log_returns should be small

        result = detector.predict_regime(features)

        status = "TRANSITION" if result["is_changepoint"] else "STABLE"
        print(
            f"  Bar {i + 1}: {status:10s} "
            f"(cp_prob={result['changepoint_prob']:.4f}, "
            f"conf={result['confidence']:.4f})"
        )


def example_load_and_use():
    """Example 5: Load calibrated model and use for inference."""
    print("\n" + "=" * 60)
    print("Example 5: Load and Use Calibrated Model")
    print("=" * 60)

    # Create and save
    print("Step 1: Create and save detector")
    detector1 = BOCPDDetector(threshold=0.45)
    model_path = Path("/tmp/bocpd_model.json")
    detector1.save(model_path)
    print(f"  Saved to {model_path}")

    # Load
    print("\nStep 2: Load detector")
    detector2 = BOCPDDetector.load(model_path)
    print(f"  Loaded successfully")
    print(f"  Threshold: {detector2.threshold}")
    print(f"  Model loaded: {detector2.is_model_loaded()}")

    # Use for inference
    print("\nStep 3: Use for inference")
    data = np.random.randn(50) * 0.01
    for x in data:
        detector2.update(x)

    info = detector2.get_model_info()
    print(f"  Observations processed: {detector2.n_observations}")
    print(f"  Current run length: {info['current_run_length']}")
    print(f"  Model type: {info['model_type']}")


def example_changepoint_signal_strength():
    """Example 6: Analyzing changepoint signal strength."""
    print("\n" + "=" * 60)
    print("Example 6: Changepoint Signal Strength Analysis")
    print("=" * 60)

    detector = BOCPDDetector(threshold=0.3)

    # Gradual vs. sudden changepoint
    print("Scenario A: Gradual regime change (low signal)")
    for i in range(50):
        x = 0.01 + (i / 50) * 0.01  # Gradual trend
        result = detector.update(x)
        if i % 10 == 0:
            print(f"  Bar {i}: cp_prob={result['changepoint_prob']:.6f}")

    detector.reset()

    print("\nScenario B: Sudden regime change (high signal)")
    for i in range(25):
        detector.update(0.01)  # Normal

    for i in range(25):
        result = detector.update(0.1)  # Sudden jump
        if i < 5:
            print(f"  Bar {25 + i}: cp_prob={result['changepoint_prob']:.6f}")


if __name__ == "__main__":
    example_basic_detection()
    example_calibration()
    example_hazard_functions()
    example_predict_regime()
    example_load_and_use()
    example_changepoint_signal_strength()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
