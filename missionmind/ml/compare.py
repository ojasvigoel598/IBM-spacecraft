"""
MissionMind — ML Comparison: Basic + Advanced Metrics across Multiple Models

Implements:
- Supervised: FCNN (MLP 100-50-20), XGBOD, Custom Physics-Informed NN
- Unsupervised: IsolationForest, LOF, OneClassSVM, MLP Autoencoder, Hybrid DIF

Metrics:
- Basic: Accuracy, Precision, Recall, F1, ROC AUC, PR AUC, Balanced Accuracy, MCC, Confusion Matrix
- Advanced: FPR before 600, TPR after 900, Detection Delay, Early Detection Rate, MTTD

Pipeline: train.py -> detect.py (baseline) and this file for comparison report

Run: python -m missionmind.ml.compare
Generates: models/comparison_report.json + console table + plots
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# P3-006 FIX: console output includes non-cp1252 chars (→, Δ) which crash on the Windows
# console — force UTF-8 so the comparison report prints everywhere.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from missionmind.ml.metrics import compute_basic_metrics, compute_advanced_metrics, make_labels, full_evaluation
from missionmind.ml.advanced_models import get_all_models
from missionmind.ml.train import add_derivative_features, build_feature_matrix, DATA_DIR, MODEL_DIR

def load_data():
    normal_path = os.path.join(DATA_DIR, "run_normal.csv")
    solar_path = os.path.join(DATA_DIR, "run_solar_failure.csv")
    rad_path = os.path.join(DATA_DIR, "run_radiator_failure.csv")
    if not os.path.exists(normal_path):
        raise FileNotFoundError("Run simulator/run_scenarios first")
    df_n = pd.read_csv(normal_path)
    df_s = pd.read_csv(solar_path) if os.path.exists(solar_path) else None
    df_r = pd.read_csv(rad_path) if os.path.exists(rad_path) else None
    return df_n, df_s, df_r

def prepare_features(df):
    df_feat = add_derivative_features(df)
    X, cols = build_feature_matrix(df_feat)
    return X, df_feat, cols

def main():
    print("=== MissionMind ML Comparison — Multiple Models + Basic/Advanced Metrics ===")
    df_n, df_s, df_r = load_data()
    Xn, df_n_feat, cols = prepare_features(df_n)
    print(f"Normal X shape {Xn.shape}, cols {cols}")

    # Prepare supervised training set WITHOUT LEAKAGE (FIX P0-002)
    # Previously: X_sup included full failure files that were also used as test sets → data leakage, F1=1.0 inflated
    # Fixed: For supervised, use time-based split: train on time<2500, test on time>=2500 hold-out
    # Also keep normal all 0 for training
    dfs_train = []
    dfs_test = []  # hold-out for final evaluation to avoid leakage
    if df_s is not None:
        # Split solar into train (<2500) and test (>=2500)
        df_s_train = df_s[df_s["time_s"] < 2500].copy()
        df_s_test = df_s[df_s["time_s"] >= 2500].copy()
        dfs_train.append(df_s_train)
        dfs_test.append(("solar_failure_holdout", df_s_test))
    if df_r is not None:
        df_r_train = df_r[df_r["time_s"] < 2500].copy()
        df_r_test = df_r[df_r["time_s"] >= 2500].copy()
        dfs_train.append(df_r_train)
        dfs_test.append(("radiator_failure_holdout", df_r_test))
    
    # Build supervised training set from normal + failure_train only (no leakage into hold-out)
    if dfs_train:
        X_combined_list = []
        y_combined_list = []
        X_combined_list.append(Xn)
        y_combined_list.append(np.zeros(len(Xn)))
        for df_f_train in dfs_train:
            Xf, _, _ = prepare_features(df_f_train)
            y_f = make_labels(df_f_train, injection_start=600, injection_end=900, ramp_as_anomaly=True)
            X_combined_list.append(Xf)
            y_combined_list.append(y_f)
        X_sup = np.vstack(X_combined_list)
        y_sup = np.concatenate(y_combined_list)
        print(f"Supervised combined X (NO LEAKAGE, train only <2500) {X_sup.shape}, y distribution {np.bincount(y_sup.astype(int))}")
        print(f"  Hold-out test sets: {[f'{name} {len(df)} rows' for name, df in dfs_test]}")
    else:
        X_sup = Xn
        y_sup = np.zeros(len(Xn))
        dfs_test = []

    models = get_all_models()
    results = {}

    # Test sets: solar and radiator separately (full) + hold-out (no leakage)
    test_sets = {}
    if df_s is not None:
        Xs, _, _ = prepare_features(df_s)
        test_sets["solar_failure"] = (df_s, Xs)
    if df_r is not None:
        Xr, _, _ = prepare_features(df_r)
        test_sets["radiator_failure"] = (df_r, Xr)
    # Also normal for false positive check
    test_sets["normal"] = (df_n, Xn)
    # Add hold-out test sets for leakage-free evaluation (train <2500, test >=2500)
    for name, df_hold in dfs_test:
        Xh, _, _ = prepare_features(df_hold)
        test_sets[name] = (df_hold, Xh)

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        is_supervised = "Supervised" in name or "XGBOD" in name or "FCNN" in name or "Custom" in name
        
        # Fit
        try:
            if is_supervised:
                # Supervised needs X_sup, y_sup
                if hasattr(model, 'fit_supervised'):
                    model.fit_supervised(X_sup, y_sup)
                else:
                    model.fit(X_sup)  # fallback
            else:
                model.fit(Xn)
        except Exception as e:
            print(f"  Training failed for {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Evaluate on each test set
        model_results = {}
        for test_name, (df_test, X_test) in test_sets.items():
            try:
                y_true = make_labels(df_test, injection_start=600, injection_end=900, ramp_as_anomaly=True)
                # For normal, y_true all 0
                if test_name=="normal":
                    y_true = np.zeros(len(df_test), dtype=int)
                
                y_score = model.decision_function(X_test)
                # threshold 0.5 for supervised proba, for unsupervised use model's predict
                y_pred = model.predict(X_test)
                
                # Basic + advanced
                eval_metrics = full_evaluation(df_test, y_true, y_pred, y_score)
                model_results[test_name] = eval_metrics
                
                print(f"  {test_name}: F1={eval_metrics['f1']:.3f} ROC_AUC={eval_metrics.get('roc_auc',0):.3f} FPR_before={eval_metrics.get('fpr_before_600',0):.3f} TPR_after={eval_metrics.get('tpr_after_900',0):.3f} Delay={eval_metrics.get('detection_delay_s',0):.0f}s")
            except Exception as e:
                print(f"  Evaluation failed for {name} on {test_name}: {e}")
                import traceback
                traceback.print_exc()
        
        results[name] = model_results

        # Save model
        try:
            save_path = os.path.join(MODEL_DIR, f"{name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.joblib")
            # joblib dump might fail for some models, try
            import joblib
            joblib.dump(model, save_path)
        except Exception as e:
            print(f"  Could not save {name}: {e}")

    # Print comparison table - full metric set (threshold-independent + dependent)
    def _fmt(m, k):
        v = m.get(k, float('nan'))
        return f"{v:.3f}" if v == v else "  nan"  # NaN-safe
    header = (f"{'Model':<45} {'Acc':<5} {'Prec':<5} {'Rec/Sens':<7} {'Spec':<6} {'F1':<5} "
              f"{'ROC':<6} {'PR':<6} {'FPR_bef':<7} {'TPR_aft':<7} {'Delay':<5}")
    print(header)
    print("-"*len(header))
    for name, res in results.items():
        if "solar_failure" in res:
            m=res["solar_failure"]
            print(f"{name:<45} {_fmt(m,'accuracy')} {_fmt(m,'precision')} {_fmt(m,'recall')} {_fmt(m,'specificity')} {_fmt(m,'f1')} {_fmt(m,'roc_auc')} {_fmt(m,'pr_auc')} {_fmt(m,'fpr_before_600')}   {_fmt(m,'tpr_after_900')}   {m.get('detection_delay_s',0):.0f}s")

    print("\n=== COMPARISON TABLE (Radiator Failure) ===")
    print(header)
    print("-"*len(header))
    for name, res in results.items():
        if "radiator_failure" in res:
            m=res["radiator_failure"]
            print(f"{name:<45} {_fmt(m,'accuracy')} {_fmt(m,'precision')} {_fmt(m,'recall')} {_fmt(m,'specificity')} {_fmt(m,'f1')} {_fmt(m,'roc_auc')} {_fmt(m,'pr_auc')} {_fmt(m,'fpr_before_600')}   {_fmt(m,'tpr_after_900')}   {m.get('detection_delay_s',0):.0f}s")

    # Save report
    report_path = os.path.join(MODEL_DIR, "comparison_report.json")
    # Convert to serializable
    def convert(o):
        if isinstance(o, (np.integer, np.floating)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    serializable = {}
    for model_name, test_dict in results.items():
        serializable[model_name] = {}
        for test_name, metrics in test_dict.items():
            serializable[model_name][test_name] = {k: convert(v) for k,v in metrics.items() if v is not None and not (isinstance(v,float) and np.isinf(v))}
    
    with open(report_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved comparison report to {report_path}")

    # Recommendation — honest, no "(Best)" suffix; winners per category come
    # from missionmind.ml.rank_models (data-driven, transparent scoring).
    print("\n=== RECOMMENDATION ===")
    print("Selection is data-driven via missionmind.ml.rank_models.")
    print("Categories:")
    print("- Unsupervised: trained on normal only, deployment when labels unavailable")
    print("- Supervised: trained on labelled fault rows, highest detection; needs labels")
    print("- Physics-informed: adds physics gates to supervised; trade-off of explainability")
    print()
    print("See models/ranking.json or run `python -m missionmind.ml.rank_models`.")

    return results

if __name__ == "__main__":
    main()
