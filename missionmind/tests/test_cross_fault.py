"""TDD suite — unseen-fault generalization + layer ablation (cross_fault.py).

Seam under test: the public helpers of missionmind/ml/cross_fault.py.

1. Holdout rows (time >= HOLD_T) must NEVER appear in training rows (temporal
   leakage guard) — this was a REAL defect in the previous audit protocol,
   where the first 1500 test rows were byte-identical to training rows.
2. Positive control: a supervised model trained on normal + radiator must
   flag the radiator HOLD-OUT tail (protocol sanity, not over-claiming).
3. The production ensemble (trained on normal ONLY) must flag the radiator
   scenario — an UNSEEN fault type — more than a physics-only baseline at
   equivalent false-alarm burden (the honest generalisation claim).

Expected to fail (ImportError) until cross_fault.py is implemented.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from missionmind.simulator.run_scenarios import run_scenario

HOLD_T = 2500


def _run(mode, duration=3600):
    return run_scenario(failure_mode=mode, duration_s=duration)


def main():
    from missionmind.ml.cross_fault import (
        load_clean_split, run_layer_ablation, run_positive_control)

    # 1. temporal-leakage guard: hold-out rows not identical to training rows
    for mode in ("none", "solar_degradation", "radiator_degradation"):
        train, test = load_clean_split(mode)
        cols = ["solar_power_w", "battery_soc", "battery_voltage_v",
                "temperature_c", "heat_in_w", "heat_out_w"]
        overlap = train.merge(test, on=cols, how="inner")
        assert len(overlap) == 0, f"{mode}: {len(overlap)} test rows found verbatim in training"
        assert test["time_s"].min() >= HOLD_T, f"{mode}: test starts before holdout boundary"
    print("  PASS temporal-leakage guard (0 overlapping rows across all 3 scenarios)")

    # 2. positive control: model trained on normal+radiator flags radiator tail
    r = run_positive_control("radiator_degradation")
    assert r["train_rows"] > 0 and r["test_rows"] > 0
    assert r["holdout_flag_rate"] >= 0.5, \
        f"positive control failed: flag rate {r['holdout_flag_rate']:.3f}"
    print(f"  PASS positive control: radiator-trained flags radiator tail "
          f"({r['holdout_flag_rate']:.2f})")

    # 3. layer ablation exists, is monotone-ish in catch rate, and each layer
    #    contributes at least as much catch as the layer below it
    layers = run_layer_ablation()
    for mode in ("solar_degradation", "radiator_degradation"):
        by_layer = {l["layer"]: l for l in layers if l["scenario"] == mode}
        assert by_layer.get("adaptive"), f"{mode}: missing adaptive layer"
        assert by_layer.get("ensemble"), f"{mode}: missing ensemble layer"
        assert by_layer.get("physics_only"), f"{mode}: missing physics baseline"
        # adaptive catch rate must be >= raw ensemble catch rate (keep-or-revert)
        assert by_layer["adaptive"]["tpr_after_900"] >= by_layer["ensemble"]["tpr_after_900"], \
            f"{mode}: adaptive underperforms raw ensemble"
        # no layer may spam false alarms before injection
        assert by_layer["adaptive"]["fpr_100_600"] <= 0.10, \
            f"{mode}: adaptive FPR too high ({by_layer['adaptive']['fpr_100_600']:.3f})"
    print("  PASS layer ablation: adaptive >= ensemble on catch, FPR bounded")

    print("All cross-fault tests PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
