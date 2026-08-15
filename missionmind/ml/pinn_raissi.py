"""Raissi (2019) Physics-Informed Neural Network for NASA battery capacity fade.

This is the *strict* PINN: the loss function has two terms

    L = L_data   +   lambda * L_physics
        = MSE(C_pred, C_obs)  +  lambda * mean( (dC/dn_NN - dC/dn_ODE)^2 )

where the ODE form for battery capacity fade is the exponential one used
across the NASA PCoE literature (Goebel, Saha, Saxena 2008; Saxena et al.
2017 prognostics challenge):

    dC/dn = -alpha * C(n)            (alpha > 0)

whose closed-form solution is C(n) = C_0 * exp(-alpha * n). The PINN
parameterises C(n; theta) and learns the residual physics alpha together
with the network weights theta.

Environment
-----------
torch / jax / autograd / tensorflow are NOT installed in this worktree
(verified); we deliberately use a pure numpy implementation with
finite-difference dC/dn to keep the dependency surface flat. This makes
the gradient w.r.t. n analytically traceable and the physics loss is
genuinely an extra term in the training objective — not a hand-fitted post-hoc
adjustment.

Reference
---------
Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed
neural networks: A deep learning framework for solving forward and inverse
problems involving nonlinear partial differential equations. Journal of
Computational Physics, 378, 686-707.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Tuple

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def exponential_ode_dCdn(C: np.ndarray | float,
                          alpha: float) -> np.ndarray | float:
    """ODE form for capacity fade: dC/dn = -alpha * C."""
    return -float(alpha) * np.asarray(C)


def exponential_residual(nn_dCdn: np.ndarray | float,
                          alpha: float,
                          C: np.ndarray | float) -> np.ndarray | float:
    """Squared residual  r^2 = (dC/dn_NN - dC/dn_ODE)^2
    (non-negative; identical to one row of the PINN physics loss)."""
    diff = np.asarray(nn_dCdn) - exponential_ode_dCdn(C, alpha)
    return diff ** 2


# ---------------------------------------------------------------------------
# The PINN module
# ---------------------------------------------------------------------------

class RaissiBatteryPINN:
    """Single-input MLP PINN with composite data + physics loss.

    Network
        C_pred(n; theta) = W3 * tanh(W2 * tanh(W1 * n + b1) + b2) + b3
    Default architecture: 1 -> 32 -> 16 -> 1 with tanh activations.
    Loss
        L(theta, alpha) =
            MSE(C_pred, C_obs)
          + lambda * mean((dC/dn_NN - (-alpha*C_pred))^2)
    Optimiser
        scipy L-BFGS-B with finite-difference Jacobians; this is the
        standard choice when the residual NN compiler is unavailable.
    """

    def __init__(self,
                 hidden: Tuple[int, ...] = (32, 16),
                 lam: float = 0.5,
                 alpha_init: float = 1e-2,
                 epochs: int = 800,
                 lr: float = 1e-2,
                 fd_eps: float = 1e-4,
                 random_state: int = 0):
        self.hidden = tuple(hidden)
        self.lam = float(lam)
        self.alpha_init = float(alpha_init)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.fd_eps = float(fd_eps)
        self.random_state = int(random_state)
        self._shapes = None
        self._sizes = None
        self._theta = None
        self._alpha = float(alpha_init)
        self.data_loss_history = []
        self.physics_loss_history = []

    # ---- network forward -----------------------------------------------------
    def _unpack(self, theta_vec: np.ndarray):
        params = {}
        i = 0
        for li in range(len(self._sizes)):
            out_d = int(self._sizes[li])
            in_d  = int(self._shapes[li][0])  # first entry = input dim
            n_w = in_d * out_d
            n_b = out_d
            params[f"W{li}"] = theta_vec[i:i + n_w].reshape(in_d, out_d)
            i += n_w
            params[f"b{li}"] = theta_vec[i:i + n_b]
            i += n_b
        # last layer stays linear (no activation stored)
        return params

    def _forward(self, x: np.ndarray, theta_vec: np.ndarray):
        params = self._unpack(theta_vec)
        a = x.reshape(-1, 1) if x.ndim == 1 else x
        for li in range(len(self.hidden)):
            z = a @ params[f"W{li}"] + params[f"b{li}"]
            a = np.tanh(z)
        out = a @ params[f"W{len(self.hidden)}"] + params[f"b{len(self.hidden)}"]
        return out.flatten()

    def _predict_full(self, x: np.ndarray):
        return self._forward(x, self._theta)

    # ---- dC/dn via central finite differences ---------------------------------
    def _dCdn_via_fd(self, x: np.ndarray) -> np.ndarray:
        e = self.fd_eps
        return (self._forward(x + e, self._theta)
                - self._forward(x - e, self._theta)) / (2.0 * e)

    # ---- composite loss ------------------------------------------------------
    def _total_theta(self) -> int:
        return int(np.sum([s_in[0] * s_out + s_out
                            for s_in, s_out in zip(self._shapes, self._sizes)]))

    def _loss(self, z: np.ndarray, x: np.ndarray, y: np.ndarray):
        # Unpack alpha + theta from the optimisation vector z = [theta..., alpha].
        total_theta = self._total_theta()
        theta = z[:total_theta]
        alpha = float(z[total_theta])
        c_pred = self._forward(x, theta)
        dCdn_nn = (self._forward(x + self.fd_eps, theta)
                    - self._forward(x - self.fd_eps, theta)) / (2.0 * self.fd_eps)
        dCdn_ode = -alpha * c_pred
        data_loss = float(np.mean((c_pred - y) ** 2))
        physics_loss = float(np.mean((dCdn_nn - dCdn_ode) ** 2))
        total = data_loss + self.lam * physics_loss
        return total, data_loss, physics_loss

    # ---- callback for scipy.optimize.minimize -------------------------------
    def _objective(self, z, x, y, history):
        total, dl, pl = self._loss(z, x, y)
        history.append((dl, pl))
        return total

    # ---- public API ----------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float64).flatten()
        y = np.asarray(y, dtype=np.float64).flatten()
        rng = np.random.default_rng(self.random_state)
        # Build network shapes: input 1, hidden layers, output 1.
        prev = 1
        self._shapes = []
        self._sizes = []
        for h in self.hidden:
            self._shapes.append((prev, h))
            self._sizes.append(h)
            prev = h
        # output layer (linear)
        self._shapes.append((prev, 1))
        self._sizes.append(1)
        # Initialise all parameters with small random values + alpha at the end.
        theta0_parts = []
        for shape_in_out, out_d in zip(self._shapes, self._sizes):
            in_dim = shape_in_out[0]
            w = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, out_d))
            b = np.zeros(out_d)
            theta0_parts.extend([w.reshape(-1), b])
        theta0 = np.concatenate(theta0_parts + [np.array([self.alpha_init])])
        # Track histories.
        self.data_loss_history = []
        self.physics_loss_history = []
        history = []
        result = minimize(
            fun=self._objective,
            x0=theta0,
            args=(x, y, history),
            method="L-BFGS-B",
            jac="2-point",
            options={"maxiter": self.epochs, "ftol": 1e-9, "gtol": 1e-7},
        )
        # Extract best.
        z = result.x
        total_theta = len(theta0) - 1
        self._theta = z[:total_theta]
        self._alpha = float(z[total_theta])
        # Save the histories collected from the callback.
        if history:
            self.data_loss_history = [h[0] for h in history]
            self.physics_loss_history = [h[1] for h in history]
        # If L-BFGS-B did not iterate enough times, fall back to FD SGD with lr.
        if not self.data_loss_history or len(self.data_loss_history) < 4:
            # Fallback path: simple FD-SGD on the composite loss.
            z = theta0.copy()
            for _ in range(self.epochs):
                grad = np.zeros_like(z)
                for j in range(len(z)):
                    zp = z.copy(); zp[j] += self.fd_eps
                    zm = z.copy(); zm[j] -= self.fd_eps
                    grad[j] = (self._objective(zp, x, y, [])[0]
                                - self._objective(zm, x, y, [])[0]) / (2.0 * self.fd_eps)
                z = z - self.lr * grad
            total_theta = len(z) - 1
            self._theta = z[:total_theta]
            self._alpha = float(z[total_theta])
            self._obj_history_sgd = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._predict_full(np.asarray(x, dtype=np.float64).flatten())

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Higher = more anomalous vs the ODE form.

        We return the per-row physics residual magnitude (signed), so a positive
        score means the NN predicted a derivative more 'positive' (i.e. less
        decay) than the ODE expects — i.e. less degraded than physics says it
        should be.  A negative score means more decay than expected.
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        c_pred = self._forward(x, self._theta)
        dCdn_nn = self._dCdn_via_fd(x)
        dCdn_ode = -self._alpha * c_pred
        residual = dCdn_nn - dCdn_ode
        return residual
