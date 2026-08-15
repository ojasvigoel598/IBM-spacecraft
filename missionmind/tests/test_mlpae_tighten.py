"""TDD test for the MLP-AE tightening (clean rewrite, RED->GREEN).

After the fix, MLPAutoencoderDetector must:
  (a) accept and use a `contamination` parameter (not hard-coded 93rd pct)
  (b) accept `per_feature_std=True/False` and produce different score scales
  (c) accept and forward an explicit `validation_fraction`
And on the 3 simulator CSVs:
  - FPR on the strict pre-600s window of run_normal.csv <= 0.10
  - F1 on the post-900s window of both failure CSVs >= 0.85
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

from missionmind.ml.advanced_models import MLPAutoencoderDetector
from missionmind.ml.train import add_derivative_features


DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
FEAT = ["battery_voltage_v","solar_power_w","load_power_w",
        "heat_in_w","heat_out_w","temperature_c","d_temp_dt","d_volt_dt"]


def _df(csv):
    return add_derivative_features(pd.read_csv(csv))


def _score(csv, **kwargs):
    df = _df(csv)
    Xall = df[FEAT].values
    Xtr  = Xall[df.time_s.values < 600]
    m = MLPAutoencoderDetector(random_state=42, **kwargs)
    m.fit(Xtr)
    pred = m.predict(Xall)
    out = df[["time_s"]].copy()
    out["anomaly_flag"] = pred
    out["anomaly_score"] = m.decision_function(Xall)
    return out, m


def test_fpr_strict_drops():
    """Strict pre-600s window of run_normal.csv must show FPR <= 0.10."""
    out, m = _score(os.path.join(DATA, "run_normal.csv"))
    strict = out[(out.time_s >= 100) & (out.time_s <= 600)]
    fpr = strict.anomaly_flag.mean()
    print(f"[FPR] run_normal pre-600: {fpr:.4f}  thresh={m.threshold:.4f}")
    assert fpr <= 0.10, f"FPR too high: {fpr:.4f}"


def test_post900_f1_holds():
    """Both failure scenarios: F1 on post-900 rows (y_true=1) must be >= 0.85."""
    for csv in ("run_solar_failure.csv", "run_radiator_failure.csv"):
        out, m = _score(os.path.join(DATA, csv))
        post = out[out.time_s > 900]
        yt = np.ones(len(post), dtype=int)
        f1 = f1_score(yt, post.anomaly_flag.values.astype(int)) if len(post) else 0.0
        print(f"[F1 post900] {csv}: {f1:.4f}")
        assert f1 >= 0.85, f"F1 dropped on {csv}: {f1:.4f}"


def test_threshold_is_contamination_consistent():
    """The constructor must accept `contamination`; threshold must NOT be 0.0."""
    m = MLPAutoencoderDetector(contamination=0.07)
    assert m.contamination == 0.07
    df = _df(os.path.join(DATA, "run_normal.csv"))
    Xtr = df[FEAT].values[df.time_s.values < 600]
    m.fit(Xtr)
    assert m.threshold is not None and np.isfinite(m.threshold) and m.threshold > 0
    print(f"[threshold] m.threshold={m.threshold:.4f}")


def test_validation_fraction_is_explicit():
    """Constructor must accept and store `validation_fraction`."""
    m = MLPAutoencoderDetector(validation_fraction=0.15)
    assert m.validation_fraction == 0.15
    # Also verify it propagates into the underlying MLPRegressor
    assert abs(m.model.validation_fraction - 0.15) < 1e-9


def test_per_feature_std_paths_both_work():
    """Both per_feature_std=True and False must produce finite, varying scores."""
    out_t, m_t = _score(os.path.join(DATA, "run_normal.csv"), per_feature_std=True)
    out_f, m_f = _score(os.path.join(DATA, "run_normal.csv"), per_feature_std=False)
    a, b = out_t.anomaly_score.values, out_f.anomaly_score.values
    print(f"[per-feature std] True: range=[{a.min():.4f},{a.max():.4f}]  "
          f"False: range=[{b.min():.4f},{b.max():.4f}]")
    assert np.isfinite(a).all() and np.isfinite(b).all()
    # The score scales should differ because per-feature std changes the denominator
    assert a.std() != b.std(), "per_feature_std should change the score variance"


if __name__ == "__main__":
    print("=" * 72)
    print("MLP-AE TIGHTENING TEST SUITE")
    print("=" * 72)
    test_fpr_strict_drops()
    test_post900_f1_holds()
    test_threshold_is_contamination_consistent()
    test_validation_fraction_is_explicit()
    test_per_feature_std_paths_both_work()
    print("ALL 5 ASSERTIONS PASS")
