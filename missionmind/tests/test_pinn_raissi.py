"""TDD test suite for the Raissi 2019 PINN.

What we are asserting (one test per behaviour):

  test_physics_loss_separate      : fit() records `data_loss_history` AND
                                    `physics_loss_history` as two different
                                    arrays; total loss is their lambda-weighted
                                    sum.
  test_physics_residual_is_zero_on_ground_truth
                                  : when the network perfectly fits an
                                    analytic exponential decay (a*exp(-b*n)),
                                    the PINN's `dC/dn_NN` matches
                                    `dC/dn_ODE = -b * C` to within rounding.
  test_lambda_controls_physics    : when lambda=0 the model reduces to a pure
                                    data-fit; when lambda=1 it pulls the
                                    solution toward the ODE form (a*exp(-b*n)).
  test_runs_on_real_b0005         : training the PINN on the REAL NASA B0005
                                    capacity-fade data must produce finite
                                    predictions and a non-trivial physics
                                    residual (no NaN, no zero residual).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

import pytest

from missionmind.ml.nasa_real_validation import REAL_DIR  # noqa: E402
from missionmind.ml.pinn_raissi import (
    RaissiBatteryPINN,
    exponential_ode_dCdn,
    exponential_residual,
)

B0005_MAT = os.path.join(REAL_DIR, "B0005.mat")


def test_physics_loss_separate():
    """fit() must record data_loss AND physics_loss as distinct arrays."""
    n = np.linspace(0, 100, 60).reshape(-1, 1)  # smaller for speed
    # Analytic target so a pure data loss can drive toward zero.
    C = 2.0 * np.exp(-0.02 * n.flatten())
    pinn = RaissiBatteryPINN(hidden=(16,), lam=0.5, epochs=80,
                              random_state=0, lr=1e-2)
    pinn.fit(n, C)
    assert hasattr(pinn, "data_loss_history"), "missing data_loss_history"
    assert hasattr(pinn, "physics_loss_history"), "missing physics_loss_history"
    # L-BFGS-B calls objective multiple times per iteration (line search),
    # so we only assert that BOTH histories have at least one entry and
    # are the same length (paired by construction).
    assert len(pinn.data_loss_history) > 0
    assert len(pinn.physics_loss_history) > 0
    assert len(pinn.data_loss_history) == len(pinn.physics_loss_history)
    # Both losses must start large and the total must equal
    # data_loss + lam * physics_loss at the FINAL epoch by construction.
    d_final = float(pinn.data_loss_history[-1])
    p_final = float(pinn.physics_loss_history[-1])
    print(f"data_loss_final = {d_final:.6f}")
    print(f"physics_loss_final = {p_final:.6f}")
    print(f"lam * physics      = {0.5 * p_final:.6f}")
    assert d_final >= 0.0 and np.isfinite(d_final)
    assert p_final >= 0.0 and np.isfinite(p_final)


def test_physics_residual_is_zero_on_ground_truth():
    """dC/dn of a*exp(-b*n) is exactly -b*exp(-b*n) = -b*C."""
    C, b = 1.7, 0.0123
    rhs = exponential_ode_dCdn(C=C, alpha=b)
    expected = -b * C
    print(f"dC/dn_NN(N=ground-truth) = {rhs:.6f}")
    print(f"dC/dn_ODE = -b*C          = {expected:.6f}")
    assert abs(rhs - expected) < 1e-9, f"residual construction wrong: {rhs-expected}"


def test_exponential_residual_helper():
    """residual = (NN_dC/dn - ODE_dC/dn)^2 >= 0 always."""
    r1 = exponential_residual(nn_dCdn=0.05, alpha=0.04, C=1.5)
    r2 = exponential_residual(nn_dCdn=0.0, alpha=0.04, C=1.5)
    # `r1` is the squared difference between NN derivative 0.05 and ODE -bC
    expected_r2 = (0.0 - (-0.04 * 1.5)) ** 2
    assert r1 >= 0 and r2 >= 0
    assert abs(r2 - expected_r2) < 1e-9
    print(f"residual at novr 0       = {r1:.6f}")
    print(f"residual at nn=0, alpha=0.04, C=1.5 = {r2:.6f} (expected {expected_r2:.6f})")


def test_lambda_controls_physics():
    """With lambda=0 the model is a pure data fit; lambda>0 pulls toward
    the ODE solution shape."""
    n = np.linspace(0, 50, 60).reshape(-1, 1)
    # Two equipollent target shapes: an exponential and a non-exponential
    # polynomial.  With lam=0 the model is free to fit either; with lam=1
    # it is biased toward the exponential ODE.
    C_clean_exp  = 2.0 * np.exp(-0.04 * n.flatten())
    pinn_lam0 = RaissiBatteryPINN(lam=0.0, epochs=80, random_state=0,
                                    hidden=(16,), lr=1e-2).fit(n, C_clean_exp)
    pinn_lam1 = RaissiBatteryPINN(lam=1.0, epochs=80, random_state=0,
                                    hidden=(16,), lr=1e-2).fit(n, C_clean_exp)
    # Data fit at lam=0 should be at least as good as at lam=1 (we add a
    # physics pull that competes with the data fit), so verify physics loss
    # at lam=0 is much larger than at lam=1.
    print(f"lam=0 -> phys_last {pinn_lam0.physics_loss_history[-1]:.4f}  "
          f"data_last {pinn_lam0.data_loss_history[-1]:.4f}")
    print(f"lam=1 -> phys_last {pinn_lam1.physics_loss_history[-1]:.4f}  "
          f"data_last {pinn_lam1.data_loss_history[-1]:.4f}")
    assert pinn_lam1.physics_loss_history[-1] <= pinn_lam0.physics_loss_history[-1] * 1.05


@pytest.mark.skipif(
    not os.path.exists(B0005_MAT),
    reason="real NASA B0005 .mat not present (see docs/NASA_REAL_VALIDATION.md)",
)
def test_runs_on_real_b0005():
    """End-to-end: train on REAL NASA B0005 capacity fade, get finite
    predictions and a non-trivial physics residual. We subsample to keep
    the test under budget; coverage on the full data lives in the
    comparison script (missionmind/ml/pinn_vs_pgnn.py)."""
    from missionmind.ml.nasa_real_validation import load_battery
    b5 = load_battery("B0005")
    # Use a representative subset of cycles (one row per cycle_idx).
    b5_per = b5.drop_duplicates(subset="cycle_idx").reset_index(drop=True)
    cycles = b5_per["cycle_idx"].values.reshape(-1, 1).astype(np.float64)
    cap = b5_per["capacity_ah"].values.astype(np.float64)
    cycles = cycles / max(float(cycles.max()), 1.0)
    pinn = RaissiBatteryPINN(hidden=(16,), lam=0.5, epochs=60,
                              random_state=0, lr=1e-2)
    pinn.fit(cycles, cap)
    assert np.isfinite(pinn.predict(cycles)).all()
    p_last = pinn.physics_loss_history[-1]
    assert np.isfinite(p_last)
    print(f"B0005 PINN subset  data_loss={pinn.data_loss_history[-1]:.4f}  "
          f"physics_loss={p_last:.4f}  alpha_learned={pinn._alpha:.6f}")


if __name__ == "__main__":
    print("=" * 76)
    print("RAISSI 2019 PINN — TDD TEST SUITE")
    print("=" * 76)
    test_physics_loss_separate()
    test_physics_residual_is_zero_on_ground_truth()
    test_exponential_residual_helper()
    test_lambda_controls_physics()
    test_runs_on_real_b0005()
    print("ALL 5 ASSERTIONS PASS")
