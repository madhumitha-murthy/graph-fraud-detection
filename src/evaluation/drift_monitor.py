"""
Graph topology drift monitor — detects when fraud patterns shift over time.

Monitors two signals:
    1. Temporal fraud rate drift (KS-test across sliding windows of time steps)
    2. Anomaly score distribution drift (KS-test on GAE scores per time step)

If drift is detected:
    - Logs warning with affected time steps
    - Returns drift report (used by /drift/status API endpoint)
    - Flags for model retraining

Mirrors production fraud systems where adversaries adapt tactics:
    - Fake account networks change structure after detection
    - Spam patterns evolve to evade classifiers
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DRIFT_WINDOW, DRIFT_KS_ALPHA, DRIFT_ZSCORE_THRESH


@dataclass
class DriftReport:
    drifted:              bool
    affected_time_steps:  list[int]  = field(default_factory=list)
    ks_statistic:         float      = 0.0
    ks_pvalue:            float      = 1.0
    mean_shift_zscore:    float      = 0.0
    message:              str        = "No drift detected"
    recommendation:       str        = "No action required"


def detect_score_drift(
    anomaly_scores: np.ndarray,
    time_steps:     np.ndarray,
    baseline_window: tuple[int, int] = (1, 10),
) -> DriftReport:
    """
    KS-test between baseline anomaly score distribution and recent window.

    Args:
        anomaly_scores:  per-node GAE reconstruction errors
        time_steps:      per-node time step assignment
        baseline_window: (start, end) time steps used as baseline distribution
    """
    base_mask   = (time_steps >= baseline_window[0]) & (time_steps <= baseline_window[1])
    base_scores = anomaly_scores[base_mask]

    recent_start = int(time_steps.max()) - DRIFT_WINDOW + 1
    recent_mask  = time_steps >= recent_start
    recent_scores = anomaly_scores[recent_mask]

    if len(base_scores) < 10 or len(recent_scores) < 10:
        return DriftReport(drifted=False, message="Insufficient data for drift test")

    stat, pval = ks_2samp(base_scores, recent_scores)

    base_mean   = base_scores.mean()
    base_std    = base_scores.std() + 1e-8
    recent_mean = recent_scores.mean()
    zscore      = abs(recent_mean - base_mean) / base_std

    drifted = (pval < DRIFT_KS_ALPHA) or (zscore > DRIFT_ZSCORE_THRESH)
    affected = list(range(recent_start, int(time_steps.max()) + 1)) if drifted else []

    return DriftReport(
        drifted             = drifted,
        affected_time_steps = affected,
        ks_statistic        = float(stat),
        ks_pvalue           = float(pval),
        mean_shift_zscore   = float(zscore),
        message = (
            f"DRIFT DETECTED: KS p={pval:.4f} < α={DRIFT_KS_ALPHA}, "
            f"mean shift z={zscore:.2f}σ at time steps {affected}"
        ) if drifted else (
            f"Stable: KS p={pval:.4f}, mean shift z={zscore:.2f}σ"
        ),
        recommendation = (
            "Trigger GAE + GraphSAGE retraining on latest time steps"
        ) if drifted else "No action required",
    )


def detect_fraud_rate_drift(
    labels:     np.ndarray,
    time_steps: np.ndarray,
) -> dict:
    """
    Monitor per-time-step fraud rate for sudden spikes or drops.
    Returns dict of {time_step: fraud_rate} with drift flags.
    """
    result    = {}
    rates     = []
    ts_sorted = sorted(np.unique(time_steps))

    for t in ts_sorted:
        mask = (time_steps == t) & (labels >= 0)
        if mask.sum() < 5:
            continue
        rate = float((labels[mask] == 1).mean())
        rates.append(rate)
        result[int(t)] = {"fraud_rate": rate, "n": int(mask.sum())}

    if len(rates) < DRIFT_WINDOW + 1:
        return result

    # Flag time steps where fraud rate deviates > 3σ from rolling mean
    rates_arr = np.array(rates)
    roll_mean = np.convolve(rates_arr, np.ones(DRIFT_WINDOW) / DRIFT_WINDOW, mode="valid")
    roll_std  = np.array([
        rates_arr[i : i + DRIFT_WINDOW].std()
        for i in range(len(rates_arr) - DRIFT_WINDOW + 1)
    ])

    offset = DRIFT_WINDOW - 1
    for i, t in enumerate(ts_sorted[offset:]):
        if t not in result:
            continue
        z = abs(rates_arr[i + offset] - roll_mean[i]) / (roll_std[i] + 1e-8)
        result[t]["drift_zscore"] = float(z)
        result[t]["drift_flag"]   = bool(z > DRIFT_ZSCORE_THRESH)

    return result
