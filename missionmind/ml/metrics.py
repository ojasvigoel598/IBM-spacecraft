"""
MissionMind — Advanced ML Metrics (Basic + Advanced)

Basic: Accuracy, Precision, Recall, F1, ROC AUC, PR AUC, Confusion Matrix
Advanced: Detection Delay, FPR before injection, TPR after injection, Balanced Accuracy, MCC,
          Physics Agreement Score, Early Detection Score, AUC over time

Used to compare multiple models: FCNN (supervised), XGBOD, Hybrid DIF, Custom NN, MLP Autoencoder, IsolationForest
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score, matthews_corrcoef
)

def compute_basic_metrics(y_true, y_pred, y_score=None):
    """y_true: 0 normal, 1 anomaly, y_pred: 0/1, y_score: continuous anomaly score (higher = more anomalous)"""
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),  # sensitivity: TP/(TP+FN)
        "specificity": tn/(tn+fp) if (tn+fp)>0 else 0.0,          # TN/(TN+FP)
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": fp/(fp+tn) if (fp+tn)>0 else 0.0,
        "fnr": fn/(fn+tp) if (fn+tp)>0 else 0.0,
    }
    # P1-006 FIX: Handle single-class case for ROC AUC (normal test has only 0s)
    # Previously caused UndefinedMetricWarning and NaN
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception as e:
            metrics["roc_auc"] = float('nan')
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_score)
        except Exception as e:
            metrics["pr_auc"] = float('nan')
    else:
        # Single class present, ROC undefined — set NaN but no warning, documented
        metrics["roc_auc"] = float('nan')
        metrics["pr_auc"] = float('nan')
        if len(np.unique(y_true)) < 2:
            metrics["roc_auc_note"] = "Single class present, ROC undefined (expected for normal-only test)"
    
    return metrics

def compute_advanced_metrics(df, y_true_col="label", y_pred_col="anomaly_flag", time_col="time_s", injection_start=600, injection_end=900):
    """
    Advanced metrics specific to spacecraft anomaly detection:
    - Detection Delay: time from injection_start to first true positive after
    - FPR before injection (0-600)
    - TPR after injection (>900)
    - Early detection: did we detect during ramp 600-900?
    - Physics agreement: if physics flag present, does ML agree?
    """
    df = df.copy()
    # FPR before
    before = df[df[time_col] < injection_start]
    fpr_before = before[y_pred_col].mean() if len(before)>0 else 0.0
    
    # TPR after
    after = df[df[time_col] > injection_end]
    tpr_after = after[y_pred_col].mean() if len(after)>0 else 0.0
    
    # Detection delay
    # First time after injection_start where y_pred=1 and y_true=1
    detected = df[(df[time_col] >= injection_start) & (df[y_pred_col]==1) & (df[y_true_col]==1)]
    if len(detected)>0:
        first_detection = detected[time_col].iloc[0]
        detection_delay = first_detection - injection_start
    else:
        first_detection = None
        detection_delay = float('inf')
    
    # Early detection during ramp
    ramp = df[(df[time_col] >= injection_start) & (df[time_col] <= injection_end)]
    early_detection_rate = ramp[y_pred_col].mean() if len(ramp)>0 else 0.0
    early_detected = early_detection_rate > 0.0
    
    # Mean time to detect after end
    if first_detection is not None and first_detection > injection_end:
        mtd_after_end = first_detection - injection_end
    elif first_detection is not None:
        mtd_after_end = 0.0  # detected during ramp
    else:
        mtd_after_end = float('inf')
    
    return {
        "fpr_before_600": float(fpr_before),
        "tpr_after_900": float(tpr_after),
        "detection_delay_s": float(detection_delay) if detection_delay!=float('inf') else 3600.0,
        "early_detection_rate_600_900": float(early_detection_rate),
        "early_detected": bool(early_detected),
        "first_detection_time": float(first_detection) if first_detection is not None else None,
        "mtd_after_end_s": float(mtd_after_end) if mtd_after_end!=float('inf') else 3600.0,
    }

def make_labels(df, injection_start=600, injection_end=900, ramp_as_anomaly=True):
    """
    Create labels from time: 0 before injection_start, 1 after injection_end, 
    optionally 1 during ramp as well for supervised training.
    For evaluation, we often ignore ramp (600-900) as ambiguous, but for training we can include.
    """
    labels = np.zeros(len(df), dtype=int)
    if ramp_as_anomaly:
        labels[df["time_s"] >= injection_start] = 1
    else:
        labels[df["time_s"] > injection_end] = 1
    return labels

def full_evaluation(df, y_true, y_pred, y_score, injection_start=600, injection_end=900):
    basic = compute_basic_metrics(y_true, y_pred, y_score)
    # add time col for advanced
    df_eval = pd.DataFrame({
        "time_s": df["time_s"] if "time_s" in df else np.arange(len(df)),
        "label": y_true,
        "anomaly_flag": y_pred,
    })
    advanced = compute_advanced_metrics(df_eval, injection_start=injection_start, injection_end=injection_end)
    return {**basic, **advanced}


def matched_fpr_metrics(y_true, y_score, fpr_target=0.05):
    """
    Threshold-dependent classification metrics at a MATCHED false-positive rate.

    The first question an ML reviewer asks about an anomaly detector is
    "what precision/recall do you get at a matched FPR?" - i.e. choose the
    decision threshold FROM THE HEALTHY-CLASS SCORE DISTRIBUTION so that
    exactly a target fraction of healthy units would be flagged, then
    report precision/recall/specificity/F1 at THAT operating point.

    A threshold tuned to a contamination prior (e.g. 0.07) is not the same
    thing: contamination picks the threshold so the model labels ~7% of
    EVERYTHING anomalous. A matched-FPR threshold instead pins the false-
    positive rate on known-healthy units, which is the defensible way to
    state "we can operate at 5% false alarms while catching X% of faults".

    Parameters
    ----------
    y_true : array-like, 0 = healthy, 1 = degraded/anomalous
    y_score : array-like, higher = more anomalous
    fpr_target : float, target false-positive rate on the healthy class

    Returns
    -------
    dict with threshold, achieved_fpr, precision, recall, specificity, f1,
    tp/fp/fn/tn and n_healthy/n_degraded. If there are no healthy units the
    threshold is undefined and None is returned.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    healthy = y_score[y_true == 0]
    if len(healthy) == 0:
        return None
    # Threshold = (1 - fpr_target) quantile of healthy scores: by construction
    # exactly ~fpr_target of the healthy population falls above it.
    threshold = float(np.quantile(healthy, 1.0 - fpr_target))
    y_pred = (y_score > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    achieved_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "threshold": threshold,
        "fpr_target": float(fpr_target),
        "achieved_fpr": achieved_fpr,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_healthy": int(len(healthy)),
        "n_degraded": int((np.asarray(y_true) == 1).sum()),
    }


def cycle_matched_fpr_metrics(df, score_col, label_col, cycle_col,
                              fpr_target=0.05):
    """
    Matched-FPR classification metrics aggregated to per-cycle units.

    Same honest statistical unit as cycle_level_metrics (a battery has ~168
    cycles but tens of thousands of rows; per-cycle is the defensible unit),
    but the threshold is chosen to match a target false-positive rate on the
    healthy cycles rather than from a row-level contamination prior.

    Cycle label = max(row labels); cycle score = mean(row scores).
    Returns None if there are no healthy cycles (threshold undefined).
    """
    g = df.groupby(cycle_col)
    y_true_cyc = g[label_col].max().values.astype(int)
    y_score_cyc = g[score_col].mean().values
    res = {
        "n_cycles": len(y_true_cyc),
        "n_degraded_cycles": int(y_true_cyc.sum()),
    }
    m = matched_fpr_metrics(y_true_cyc, y_score_cyc, fpr_target=fpr_target)
    if m is None:
        return None
    res.update(m)
    return res


def cycle_level_metrics(df, score_col, label_col, cycle_col, flag_col=None,
                        majority_threshold=0.5):
    """
    Aggregate row-level scores/flags to cycle level and compute metrics on
    per-cycle units (the honest statistical unit when a battery has ~168
    cycles but tens of thousands of rows).

    Returns a dict with:
      n_cycles, n_degraded_cycles, roc_auc, pr_auc (threshold-independent),
      and - when flag_col is given - precision, recall, specificity, f1,
      accuracy, tp/fp/fn/tn at cycle level (threshold-dependent).

    Cycle label: a cycle is degraded if ANY row in it is degraded.
    Cycle flag:   majority of row flags (>= majority_threshold).
    """
    g = df.groupby(cycle_col)
    y_true_cyc = g[label_col].max().values.astype(int)
    y_score_cyc = g[score_col].mean().values
    res = {
        "n_cycles": len(y_true_cyc),
        "n_degraded_cycles": int(y_true_cyc.sum()),
    }
    if len(np.unique(y_true_cyc)) > 1:
        res["roc_auc"] = float(roc_auc_score(y_true_cyc, y_score_cyc))
        res["pr_auc"] = float(average_precision_score(y_true_cyc, y_score_cyc))
    else:
        res["roc_auc"] = float("nan")
        res["pr_auc"] = float("nan")
    if flag_col is not None:
        y_flag_cyc = (g[flag_col].mean() >= majority_threshold).astype(int).values
        cm = confusion_matrix(y_true_cyc, y_flag_cyc, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        res.update({
            "precision": float(precision_score(y_true_cyc, y_flag_cyc, zero_division=0)),
            "recall": float(recall_score(y_true_cyc, y_flag_cyc, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "f1": float(f1_score(y_true_cyc, y_flag_cyc, zero_division=0)),
            "accuracy": float(accuracy_score(y_true_cyc, y_flag_cyc)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "flagged_cycles": int(y_flag_cyc.sum()),
        })
    return res


def predictive_horizon_metrics(df, score_col, capacity_col, cycle_col,
                               horizons=(10, 25, 50), eol_fraction=0.75,
                               score_threshold=None, initial_capacity=None):
    """
    Future-event experiment: does the detector predict degradation at t+dt
    from healthy telemetry at t? This is what a "failure prediction" claim
    requires (vs. classifying degradation that is already occurring).

    For each cycle c with capacity still above EOL (healthy at prediction
    time), the label is 1 if the first degraded cycle d satisfies
    d - c <= H for horizon H. Score is the per-cycle mean anomaly score
    (computed only from rows up to and including c - no future data).

    initial_capacity: the battery's FRESH initial capacity (used as the EOL
    reference). Must be supplied when df starts mid-life (e.g. the test
    split), otherwise EOL is computed from the first cycle in df and the
    experiment silently finds no degradation.

    Returns {H: {n_healthy, n_events, roc_auc, pr_auc, and, if
    score_threshold is given, precision/recall/specificity/f1/confusion}}.
    """
    g = df.groupby(cycle_col)
    cap_cyc = g[capacity_col].first()
    score_cyc = g[score_col].mean()
    init_cap = initial_capacity if initial_capacity is not None else cap_cyc.iloc[0]
    eol = eol_fraction * init_cap
    degraded_cycles = cap_cyc[cap_cyc < eol].index
    first_degraded = int(degraded_cycles.min()) if len(degraded_cycles) else None

    out = {}
    for H in horizons:
        if first_degraded is None:
            out[H] = {"n_healthy": int((cap_cyc >= eol).sum()), "n_events": 0,
                      "roc_auc": float("nan"), "pr_auc": float("nan"),
                      "note": "no degradation reached EOL in this dataset"}
            continue
        # healthy at prediction time: c < first_degraded
        healthy = cap_cyc.index[cap_cyc.index < first_degraded]
        y_true = np.array([1 if (first_degraded - c) <= H else 0 for c in healthy], dtype=int)
        y_score = score_cyc.loc[healthy].values
        res = {"n_healthy": int(len(healthy)), "n_events": int(y_true.sum())}
        if len(healthy) == 0:
            res.update({"roc_auc": float("nan"), "pr_auc": float("nan"),
                        "note": "no healthy cycles remain at this horizon - battery already degraded"})
            out[H] = res
            continue
        if len(np.unique(y_true)) > 1:
            res["roc_auc"] = float(roc_auc_score(y_true, y_score))
            res["pr_auc"] = float(average_precision_score(y_true, y_score))
        else:
            res["roc_auc"] = float("nan")
            res["pr_auc"] = float("nan")
        if score_threshold is not None:
            y_pred = (y_score > score_threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            res.update({
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            })
        out[H] = res
    return out
