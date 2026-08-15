"""Raissi (2019) PINN re-implemented on PyTorch autograd backprop.

The numpy version in pinn_raissi.py optimises with scipy L-BFGS-B and
finite-difference Jacobians.  This module is the same network and the same
composite loss

    L = MSE(C_pred, C_obs)  +  lambda * mean( (dC/dn_NN - (-alpha*C))^2 )

but the derivative dC/dn_NN is computed by torch.autograd.grad instead of
central finite differences, and the whole objective (theta + alpha) is
optimised by Adam — i.e. genuine backpropagation.

Purpose (per the P4-xxx audit): answer the question "does real autograd
backprop change the B0005 result vs finite-difference L-BFGS-B?"  It is a
drop-in experimental replacement for RaissiBatteryPINN with the same
fit / predict / decision_function interface, so either implementation can be
swapped into pinn_vs_pgnn.py without touching the protocol.

Run:
    .venv/Scripts/python.exe -m missionmind.ml.pinn_torch
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Tuple

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn


def _torch_version() -> str:
    return torch.__version__


class _Net(nn.Module):
    """1 -> hidden... -> 1 tanh MLP (same topology as RaissiBatteryPINN)."""

    def __init__(self, hidden: Tuple[int, ...], seed: int):
        super().__init__()
        torch.manual_seed(seed)
        layers = []
        prev = 1
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class RaissiBatteryPINNTorch:
    """Drop-in torch-autograd twin of RaissiBatteryPINN (same API)."""

    def __init__(self,
                 hidden: Tuple[int, ...] = (32, 16),
                 lam: float = 0.5,
                 alpha_init: float = 1e-2,
                 epochs: int = 3000,
                 lr: float = 1e-2,
                 random_state: int = 0):
        self.hidden = tuple(hidden)
        self.lam = float(lam)
        self.alpha_init = float(alpha_init)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.random_state = int(random_state)
        self._net = _Net(self.hidden, seed=self.random_state)
        self._alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.data_loss_history = []
        self.physics_loss_history = []

    # ------------------------------------------------------------------ fit
    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float64).flatten()
        y = np.asarray(y, dtype=np.float64).flatten()
        x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        opt = torch.optim.Adam(
            list(self._net.parameters()) + [self._alpha], lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=self.lr * 0.05)

        self.data_loss_history = []
        self.physics_loss_history = []
        for step in range(self.epochs):
            opt.zero_grad()
            c_pred = self._net(x_t)
            # dC/dn via autograd (this is the "better backpropagation" claim).
            dCdn_nn = torch.autograd.grad(
                c_pred, x_t, grad_outputs=torch.ones_like(c_pred),
                create_graph=True)[0]
            dCdn_ode = -self._alpha * c_pred
            data_loss = torch.mean((c_pred - y_t) ** 2)
            physics_loss = torch.mean((dCdn_nn - dCdn_ode) ** 2)
            loss = data_loss + self.lam * physics_loss
            loss.backward()
            opt.step()
            sched.step()
            if step % max(1, self.epochs // 10) == 0 or step == self.epochs - 1:
                self.data_loss_history.append(float(data_loss.detach()))
                self.physics_loss_history.append(float(physics_loss.detach()))
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.tensor(np.asarray(x, dtype=np.float64).flatten(),
                           dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            return self._net(x_t).numpy().flatten()

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Signed physics residual dC/dn_NN - dC/dn_ODE (same semantics as
        RaissiBatteryPINN.decision_function)."""
        x_t = torch.tensor(np.asarray(x, dtype=np.float64).flatten(),
                           dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
        c_pred = self._net(x_t)
        dCdn_nn = torch.autograd.grad(
            c_pred, x_t, grad_outputs=torch.ones_like(c_pred))[0]
        dCdn_ode = -float(self._alpha.detach()) * c_pred
        return (dCdn_nn - dCdn_ode).detach().numpy().flatten()

    @property
    def _alpha_value(self) -> float:
        return float(self._alpha.detach())


# ---------------------------------------------------------------------------
# Head-to-head vs the numpy L-BFGS-B implementation on the SAME Arm-D split.
# ---------------------------------------------------------------------------
def _split(b5, train_rows=4000, seed=0):
    from missionmind.ml.nasa_real_validation import features, degraded_label
    import pandas as pd
    cycles = sorted(b5["cycle_idx"].unique())
    n = len(cycles)
    cut_h, cut_d = int(n * 0.15), int(n * 0.85)
    early, late, mid = cycles[:cut_h], cycles[cut_d:], cycles[cut_h:cut_d]
    tr_s = pd.concat([b5[b5["cycle_idx"].isin(early)],
                      b5[b5["cycle_idx"].isin(late)]])
    y_s = (tr_s["cycle_idx"] >= cut_d).astype(int).values
    te = b5[b5["cycle_idx"].isin(mid + late)]
    y_te = degraded_label(b5)[b5["cycle_idx"].isin(mid + late)]
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y_s == c)[0],
                                     train_rows // 2, replace=False)
                          for c in (0, 1)])
    return (features(tr_s)[idx], y_s[idx], features(te), te, y_te)


def _eval(ctor, kwargs, b5, seed=0):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    X_tr, y_tr, X_te, te, y_te = _split(b5, seed=seed)
    cycles = sorted(b5["cycle_idx"].unique())
    n = len(cycles)
    cut_h, cut_d = int(n * 0.15), int(n * 0.85)
    early, late, mid = cycles[:cut_h], cycles[cut_d:], cycles[cut_h:cut_d]
    train_pts = b5[b5["cycle_idx"].isin(early + late)]
    train_one = train_pts.drop_duplicates(subset="cycle_idx").reset_index(drop=True)
    n_arr = train_one["cycle_idx"].values.astype(np.float64)
    cap_arr = train_one["capacity_ah"].values.astype(np.float64)
    n_arr = n_arr / max(float(n_arr.max()), 1.0)
    m = ctor(**kwargs)
    m.fit(n_arr, cap_arr)
    test_cycles = mid + late
    test_pts = b5[b5["cycle_idx"].isin(test_cycles)]
    test_one = test_pts.drop_duplicates(subset="cycle_idx").reset_index(drop=True)
    n_te = test_one["cycle_idx"].values.astype(np.float64)
    cap_te = test_one["capacity_ah"].values.astype(np.float64)
    n_te_n = n_te / max(float(n_te.max()), 1.0)
    res = m.decision_function(n_te_n)
    threshold_cap = float(np.median(cap_te))
    y_bin = (cap_te <= threshold_cap).astype(int)
    if len(np.unique(y_bin)) < 2:
        return float("nan"), float("nan"), None
    auc = float(roc_auc_score(y_bin, np.abs(res)))
    sp = float(spearmanr(np.abs(res), cap_te).statistic)
    alpha = getattr(m, "_alpha_value", None)
    if alpha is None:
        alpha = getattr(m, "_alpha", None)
    return auc, sp, alpha


def main():
    from missionmind.ml.nasa_real_validation import load_battery, REAL_DIR
    if not os.path.exists(os.path.join(REAL_DIR, "B0005.mat")):
        raise SystemExit(f"Real NASA .mat files missing in {REAL_DIR}")
    b5 = load_battery("B0005")
    print("=" * 80)
    print(f"torch-autograd PINN vs numpy L-BFGS-B PINN on real B0005")
    print(f"torch {_torch_version()} | CUDA: {torch.cuda.is_available()}")
    print("=" * 80)

    # numpy reference (finite-difference L-BFGS-B)
    from missionmind.ml.pinn_raissi import RaissiBatteryPINN
    for lam in (0.3, 0.5):
        auc, sp, alpha = _eval(RaissiBatteryPINN,
                               {"hidden": (16,), "lam": lam, "epochs": 80},
                               b5)
        print(f"\nnumpy L-BFGS-B  lam={lam}: AUC={auc:.4f}  |Sp|={abs(sp) if sp==sp else 0:.4f}"
              f"  alpha={alpha:.4f}")

    # torch autograd (same hidden, more epochs to reach equivalent loss)
    for lam in (0.3, 0.5):
        auc, sp, alpha = _eval(RaissiBatteryPINNTorch,
                               {"hidden": (16,), "lam": lam, "epochs": 3000,
                                "lr": 1e-2},
                               b5)
        print(f"torch autograd    lam={lam}: AUC={auc:.4f}  |Sp|={abs(sp) if sp==sp else 0:.4f}"
              f"  alpha={alpha:.4f}")

    print("\nInterpretation: this module is a drop-in experimental twin; see")
    print("pinn_vs_pgnn.py for the PGNN-vs-PINN verdict (PGNN wins on this data).")


if __name__ == "__main__":
    main()
