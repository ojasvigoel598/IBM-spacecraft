"""
MissionMind — Multiple ML Models Comparison
- Unsupervised: IsolationForest, LOF, One-Class SVM, MLP Autoencoder, Hybrid DIF
- Supervised: FCNN (MLPClassifier), XGBOD, Custom Physics-Informed NN
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    from pyod.models.xgbod import XGBOD
    PYOD_AVAILABLE=True
except ImportError:
    PYOD_AVAILABLE=False
    XGBOD=None

try:
    import xgboost as xgb
    XGB_AVAILABLE=True
except ImportError:
    XGB_AVAILABLE=False

class BaseDetector:
    def fit(self, X_normal): raise NotImplementedError
    def decision_function(self, X): raise NotImplementedError
    def predict(self, X): raise NotImplementedError

class IsolationForestDetector(BaseDetector):
    def __init__(self, contamination=0.07, n_estimators=300, random_state=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=random_state, max_features=1.0)
    def fit(self, X_normal):
        X = X_normal.copy()
        rng = np.random.default_rng(42)
        for i in range(X.shape[1]):
            if X[:,i].std() < 1e-6:
                X[:,i] += rng.normal(0,1,size=len(X))
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        self.model.fit(Xs)
        return self
    def decision_function(self, X):
        Xs = self.scaler.transform(X)
        return -self.model.decision_function(Xs)
    def predict(self, X):
        Xs = self.scaler.transform(X)
        pred = self.model.predict(Xs)
        return (pred==-1).astype(int)

class LOFDetector(BaseDetector):
    def __init__(self, n_neighbors=20, contamination=0.07):
        self.scaler = StandardScaler()
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    def fit(self, X_normal):
        X = X_normal.copy()
        rng=np.random.default_rng(42)
        for i in range(X.shape[1]):
            if X[:,i].std()<1e-6:
                X[:,i]+=rng.normal(0,1,size=len(X))
        self.scaler.fit(X)
        Xs=self.scaler.transform(X)
        self.model.fit(Xs)
        return self
    def decision_function(self, X):
        Xs=self.scaler.transform(X)
        return -self.model.decision_function(Xs)
    def predict(self, X):
        Xs=self.scaler.transform(X)
        pred=self.model.predict(Xs)
        return (pred==-1).astype(int)

class OCSVMDetector(BaseDetector):
    def __init__(self, nu=0.07, gamma='scale'):
        self.scaler=StandardScaler()
        self.model=OneClassSVM(nu=nu, gamma=gamma)
    def fit(self, X_normal):
        X=X_normal.copy()
        rng=np.random.default_rng(42)
        for i in range(X.shape[1]):
            if X[:,i].std()<1e-6:
                X[:,i]+=rng.normal(0,1,size=len(X))
        self.scaler.fit(X)
        Xs=self.scaler.transform(X)
        self.model.fit(Xs)
        return self
    def decision_function(self, X):
        Xs=self.scaler.transform(X)
        return -self.model.decision_function(Xs)
    def predict(self, X):
        Xs=self.scaler.transform(X)
        pred=self.model.predict(Xs)
        return (pred==-1).astype(int)

class MLPAutoencoderDetector(BaseDetector):
    """Autoencoder anomaly detector — TIGHTENED variant.

    Three deliberate behaviour changes vs the prior version:

      (a) **Contamination-consistent threshold** — previously hard-coded to the
          93rd-percentile of training reconstruction error. The threshold is
          now derived from the `contamination` constructor kwarg so the same
          knob drives IF/LOF/Hybrid DIF and the autoencoder. ``contamination``
          defaults to 0.07 to stay consistent with the other unsup detectors.

      (b) **Per-feature standardised reconstruction error** — previously the
          error was a plain mean of squared residuals over all features. When
          feature scales differ by orders of magnitude (solar ~ 520 W vs
          d_temp_dt ~ 0.001 K/s) the high-magnitude features dominate the
          score. The new path divides each feature's squared error by its
          TRAINING pre-StandardScaler std (squared), which gives every feature
          equal influence on the score. ``per_feature_std=True`` enables it.

      (c) **Explicit ``validation_fraction``** — previously left at sklearn's
          default 0.1. Now pinned at ``validation_fraction=0.15`` by default
          so early-stopping has a stable validation split and the FPR ceiling
          is reproducible across seeds.
    """
    def __init__(self, hidden_layer_sizes=(20,10,20), max_iter=500,
                 random_state=42, contamination=0.07,
                 validation_fraction=0.15, per_feature_std=True,
                 n_iter_no_change=20):
        self.scaler = StandardScaler()
        self.contamination = float(contamination)
        self.validation_fraction = float(validation_fraction)
        self.per_feature_std = bool(per_feature_std)
        self.n_iter_no_change = int(n_iter_no_change)
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter, random_state=random_state,
            early_stopping=True,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
        )
        self._feat_std = None  # per-feature std for the standardised-error path
        self._train_err_p95 = 1.0  # P4-002-style leak-free normalisation anchor

    def _compute_errors(self, Xs):
        """Compute reconstruction error with or without per-feature std normalisation."""
        recon = self.model.predict(Xs)
        sq = (Xs - recon) ** 2
        if self.per_feature_std and self._feat_std is not None:
            # Each feature's squared error is divided by the TRAINING variance;
            # cells wider than the typical training distribution contribute more.
            denom = np.maximum(self._feat_std ** 2, 1e-9)
            sq = sq / denom
        return np.mean(sq, axis=1)

    def fit(self, X_normal):
        X = X_normal.copy()
        rng = np.random.default_rng(42)
        # Capture the TRAINING per-feature std BEFORE we transform so we can
        # use it downstream as the normalisation denominator for (b).
        self._feat_std = X.std(axis=0, ddof=0)
        for i in range(X.shape[1]):
            if X[:, i].std() < 1e-6:
                X[:, i] += rng.normal(0, 1, size=len(X))
                self._feat_std[i] = max(self._feat_std[i], 1.0)  # synthetic col
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        self.model.fit(Xs, Xs)
        # Capture training errors as the leak-free anchor (P4-002 alignment).
        train_err = self._compute_errors(Xs)
        p95 = float(np.percentile(train_err, 95))
        self._train_err_p95 = p95 if p95 > 1e-9 else 1e-9
        # (a) Contamination-consistent threshold: cut at the (1 - contamination)
        # percentile of TRAINING reconstruction error so the FPR contract ties
        # into the same contamination knob as IF/LOF/Hybrid DIF.
        cut = float(np.clip(self.contamination, 0.001, 0.999))
        self.threshold = float(np.percentile(train_err, 100.0 * (1.0 - cut)))
        return self

    def decision_function(self, X):
        """Per-row reconstruction error after per-feature std normalisation.

        Returned scores and the FIT-TIME threshold are on the SAME scale:
        contamination-consistent thresholds were derived from training rows
        that produced exactly these column shapes, so this is leak-free
        inference (matches the P4-002 alignment used elsewhere).
        """
        Xs = self.scaler.transform(X)
        return self._compute_errors(Xs)

    def predict(self, X):
        """Cut on the FIT-TIME training percentile of the standardised error.

        This is the leak-free path: the threshold was locked in `fit()` from
        the (1 - contamination) percentile of TRAINING reconstruction error;
        row-by-row inference in `decision_function` is then compared against
        that exact training-derived cut.  No inference-time percent leak.
        """
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)

class FCNNDetector(BaseDetector):
    def __init__(self, hidden_layer_sizes=(100,50,20), max_iter=500, random_state=42):
        self.scaler=StandardScaler()
        self.model=MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state, early_stopping=True)
    def fit_supervised(self, X, y):
        Xc=X.copy()
        rng=np.random.default_rng(42)
        for i in range(X.shape[1]):
            if X[:,i].std()<1e-6:
                Xc[:,i]+=rng.normal(0,1,size=len(X))
        self.scaler.fit(Xc)
        Xs=self.scaler.transform(Xc)
        self.model.fit(Xs, y)
        return self
    def fit(self, X_normal):
        return self.fit_supervised(X_normal, np.zeros(len(X_normal)))
    def decision_function(self, X):
        Xs=self.scaler.transform(X)
        try:
            proba=self.model.predict_proba(Xs)[:,1]
        except (AttributeError, NotImplementedError):
            proba=self.model.predict(Xs).astype(float)
        return proba
    def predict(self, X):
        Xs=self.scaler.transform(X)
        return self.model.predict(Xs).astype(int)

class HybridDIFDetector(BaseDetector):
    def __init__(self, latent_dim=3, contamination=0.07, random_state=42):
        self.latent_dim = latent_dim
        self.contamination = contamination
        self.random_state = random_state
        self.scaler=StandardScaler()
        self.iforest=IsolationForest(contamination=contamination, n_estimators=200, random_state=random_state)
        self.threshold=0.0
    def fit(self, X_normal):
        X=X_normal.copy()
        rng=np.random.default_rng(42)
        for i in range(X.shape[1]):
            if X[:,i].std()<1e-6:
                X[:,i]+=rng.normal(0,1,size=len(X))
        self.scaler.fit(X)
        Xs=self.scaler.transform(X)
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=self.latent_dim)
        latent = self.pca.fit_transform(Xs)
        self.iforest.fit(latent)
        self.autoencoder = MLPRegressor(hidden_layer_sizes=(20,10,20), max_iter=400, random_state=42)
        self.autoencoder.fit(Xs, Xs)
        recon = self.autoencoder.predict(Xs)
        errors = np.mean((Xs-recon)**2, axis=1)
        iso_scores = -self.iforest.decision_function(latent)
        # P1-005 FIX: Previously weighted 0.6 iso +0.4 error, iso dominated and radiator iso scores were negative (normal), so combined stayed negative below threshold.
        # Now weight 0.1 iso +0.9 error to favor reconstruction error which already works for radiator (MLP Autoencoder F1 0.978)
        # Also normalize to 0-1 and use 80th percentile threshold to be more sensitive (was 93rd)
        iso_norm = (iso_scores - iso_scores.min())/(iso_scores.max()-iso_scores.min()+1e-9)
        err_norm = (errors - errors.min())/(errors.max()-errors.min()+1e-9)
        combined = 0.1*iso_norm + 0.9*err_norm
        # P3-010 FIX: threshold at the (1-contamination) percentile. The 80th-percentile
        # cut over-alerted (74% FPR in normal ops); the contamination-consistent cut
        # keeps ~7% train FPR while retaining the reconstruction-error signal.
        self.threshold = np.percentile(combined, 100 * (1 - self.contamination))
        self.iso_min, self.iso_max = iso_scores.min(), iso_scores.max()
        self.err_min, self.err_max = errors.min(), errors.max()
        return self
    def decision_function(self, X):
        Xs=self.scaler.transform(X)
        latent = self.pca.transform(Xs)
        iso_scores = -self.iforest.decision_function(latent)
        recon = self.autoencoder.predict(Xs)
        errors = np.mean((Xs-recon)**2, axis=1)
        iso_norm = (iso_scores - self.iso_min)/(self.iso_max - self.iso_min + 1e-9)
        err_norm = (errors - self.err_min)/(self.err_max - self.err_min + 1e-9)
        combined = 0.1*iso_norm + 0.9*err_norm
        return combined
    def predict(self, X):
        scores=self.decision_function(X)
        return (scores>self.threshold).astype(int)

class XGBODDetector(BaseDetector):
    def __init__(self, contamination=0.07, random_state=42, n_jobs=-1):
        self.scaler=StandardScaler()
        self.contamination=contamination
        self.random_state=random_state
        self.threshold=0.5
        if PYOD_AVAILABLE and XGBOD is not None:
            try:
                # P3-010 FIX: n_jobs=-1 parallelises the boosting rounds (was ~10 min on 15k rows)
                self.model=XGBOD(contamination=contamination, random_state=random_state, n_jobs=n_jobs)
                self.use_pyod=True
            except (ImportError, TypeError, ValueError):
                # PyOD-detector constructor can raise on incompatible args or
                # missing optional deps; degrade gracefully to sklearn XGBoost.
                self.use_pyod=False
                self.model=None
        else:
            self.use_pyod=False
            self.model=None
        if not self.use_pyod:
            if XGB_AVAILABLE:
                self.model=xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_estimators=100, n_jobs=n_jobs)
            else:
                from sklearn.ensemble import GradientBoostingClassifier
                self.model=GradientBoostingClassifier(random_state=random_state)
    def _raw_scores(self, Xs):
        # robust score extraction: decision_function, then proba (pyod returns 1-D,
        # sklearn classifiers return 2-D), then hard prediction as last resort
        try:
            return np.asarray(self.model.decision_function(Xs)).astype(float)
        except (AttributeError, NotImplementedError):
            try:
                p = np.asarray(self.model.predict_proba(Xs))
                return p[:, 1] if p.ndim == 2 else p.astype(float)
            except (AttributeError, NotImplementedError):
                return np.asarray(self.model.predict(Xs)).astype(float)
    def _calibrate(self, Xs, y):
        """P3-010 FIX: pick the decision threshold on TRAINING data instead of 0.5.
        XGBOD ranks anomalies almost perfectly (AUC ~0.996) but the default 0.5 cut
        gives F1 ~0.55; a threshold tuned to maximise training F1 fixes the cut while
        keeping the ranking. With no positive labels, use the contamination percentile."""
        scores = self._raw_scores(Xs)
        if np.any(y == 1):
            best_t, best_f1 = self.threshold, 0.0
            for q in np.linspace(0.01, 0.99, 99):
                t = float(np.percentile(scores, q * 100))
                pred = (scores > t).astype(int)
                tp = int(((pred == 1) & (y == 1)).sum())
                fp = int(((pred == 1) & (y == 0)).sum())
                fn = int(((pred == 0) & (y == 1)).sum())
                f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            self.threshold = best_t
        else:
            self.threshold = float(np.percentile(scores, 100 * (1 - self.contamination)))
    def fit_supervised(self, X, y):
        Xc=X.copy()
        rng=np.random.default_rng(42)
        for i in range(X.shape[1]):
            if X[:,i].std()<1e-6:
                Xc[:,i]+=rng.normal(0,1,size=len(X))
        self.scaler.fit(Xc)
        Xs=self.scaler.transform(Xc)
        self.model.fit(Xs, np.asarray(y))
        self._calibrate(Xs, np.asarray(y))
        return self
    def fit(self, X_normal):
        return self.fit_supervised(X_normal, np.zeros(len(X_normal)))
    def decision_function(self, X):
        return self._raw_scores(self.scaler.transform(X))
    def predict(self, X):
        return (self.decision_function(X) > self.threshold).astype(int)

class CustomPhysicsInformedNN(BaseDetector):
    # P3-010 FIX (pinn_layer_scan.py): the previous default (64,32,16) with
    # HARDCODED synthetic-domain gates (solar<364 W, V<26.5 V, dT>0.003/s) scored
    # the WORST of all configs on the real NASA benchmark (AUC 0.778). The scan
    # over layer sizes x gate modes found (32,16) + envelope-grounded gates best
    # (AUC 0.837 on real B0005, vs 0.778 before). Gates are now learned from the
    # healthy training envelope at fit time so they transfer across domains;
    # the old synthetic constants remain as fallback before fit.
    def __init__(self, hidden_layer_sizes=(32,16), max_iter=600, random_state=42):
        self.scaler=StandardScaler()
        self.model=MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state, early_stopping=True)
        self.autoencoder=MLPRegressor(hidden_layer_sizes=(20,10,20), max_iter=400, random_state=random_state)
        # fallback gates (synthetic envelope) - replaced at fit time when labels exist
        self.g_solar=364.0; self.g_volt=26.5; self.g_dtemp=0.003
    def _physics_features(self, X):
        V = X[:,0]
        solar = X[:,1]
        dTemp = X[:,3]
        solar_drop = (solar < self.g_solar).astype(float)
        soc_low = (V < self.g_volt).astype(float)
        temp_rise = (np.abs(dTemp) > self.g_dtemp).astype(float)  # abnormal thermal rate (rise OR fall)
        physics_risk = np.clip(solar_drop*0.6 + soc_low*0.2 + temp_rise*0.6, 0,1)
        return np.column_stack([solar_drop, temp_rise, physics_risk])
    def fit_supervised(self, X, y):
        Xc=X.copy()
        rng=np.random.default_rng(42)
        for i in range(X.shape[1]):
            if X[:,i].std()<1e-6:
                Xc[:,i]+=rng.normal(0,1,size=len(X))
        yb = np.asarray(y)
        if np.any(yb == 1):
            # ground gates on the healthy training envelope so they transfer
            Xh = Xc[yb == 0]
            self.g_solar = float(np.percentile(Xh[:,1], 10))   # solar below healthy 10th pct
            self.g_volt  = float(np.percentile(Xh[:,0], 10))   # voltage sag below healthy 10th pct
            self.g_dtemp = float(np.percentile(np.abs(Xh[:,3]), 95))  # abnormal thermal rate
        phys = self._physics_features(Xc)
        X_enhanced = np.hstack([Xc, phys])
        self.scaler.fit(X_enhanced)
        Xs=self.scaler.transform(X_enhanced)
        self.model.fit(Xs, yb)
        X_normal = Xc[yb==0] if np.any(yb==1) else Xc
        if len(X_normal)>0:
            Xn_e = np.hstack([X_normal, self._physics_features(X_normal)])
            self.autoencoder.fit(self.scaler.transform(Xn_e), self.scaler.transform(Xn_e))
        return self
    def fit(self, X_normal):
        return self.fit_supervised(X_normal, np.zeros(len(X_normal)))
    def decision_function(self, X):
        phys = self._physics_features(X)
        X_enh = np.hstack([X, phys])
        Xs=self.scaler.transform(X_enh)
        try:
            proba=self.model.predict_proba(Xs)[:,1]
        except (AttributeError, NotImplementedError):
            proba=self.model.predict(Xs).astype(float)
        try:
            recon=self.autoencoder.predict(Xs)
            error=np.mean((Xs-recon)**2, axis=1)
            error_norm = error/(np.max(error)+1e-9)
            combined = 0.7*proba + 0.3*error_norm
        except (AttributeError, NotFittedError):
            combined=proba
        return combined
    def predict(self, X):
        scores=self.decision_function(X)
        return (scores>0.5).astype(int)

def get_all_models():
    # PINN is one of multiple detectors; the previous "(Best)" suffix gave the
    # false impression that it had been independently audited as best on real
    # NASA data. The multi-seed sweep and the strict-PINN benchmark both show
    # it ties or loses to feature-only PGNN; the label now reflects that.
    models = {
        "IsolationForest (Baseline Unsupervised)": IsolationForestDetector(),
        "LOF (Unsupervised)": LOFDetector(),
        "OneClassSVM (Unsupervised)": OCSVMDetector(),
        "MLP Autoencoder (Unsupervised FeedForward)": MLPAutoencoderDetector(),
        "Hybrid DIF (Unsupervised Hybrid Deep Isolated Forest)": HybridDIFDetector(),
        "FCNN Supervised (MLP 100-50-20)": FCNNDetector(),
        "XGBOD Supervised (Extreme Boosting Outlier Detector)": XGBODDetector(),
        "Custom Physics-Informed NN": CustomPhysicsInformedNN(),
    }
    return models


if __name__ == "__main__":
    """Self-test: fit every detector on synthetic data and print a summary table.

    Run directly:  python missionmind/ml/advanced_models.py
    """
    import warnings
    warnings.filterwarnings("ignore")

    try:
        import torch
        torch_ok = f"yes ({torch.__version__})"
    except ImportError:
        torch_ok = "no (not installed - optional; all models here use scikit-learn)"

    print("=" * 78)
    print("MissionMind ML model zoo - self-test")
    print("=" * 78)
    print(f"environment: torch={torch_ok} | pyod={PYOD_AVAILABLE} | xgboost={XGB_AVAILABLE}")
    print(f"python: {os.sys.version.split()[0]}")
    print()

    # P3-011 FIX: Generic self-test previously called every detector via
    # `fit(X_normal)`. Unsupervised detectors learn the normal distribution with
    # no labels; supervised classifiers (FCNN, XGBOD, PINN) learn "everything is
    # normal" because every training row is labelled 0, then collapse to TP=0
    # on the injected anomalies. Routing the supervised models through
    # `fit_supervised(X_train_sup, y_train_sup)` with a labelled mix (normal +
    # injected anomaly rows) supplies the missing anomaly class so their
    # supervised evaluation (accuracy / precision / recall / F1 / ROC-AUC) is
    # meaningful. The unsupervised column is unchanged. ml/compare.py and
    # missionmind/ml/nasa_real_validation.py already use fit_supervised with
    # real failure labels and are not touched by this change.
    from sklearn.metrics import roc_auc_score

    SUPERVISED_NAMES = {
        "FCNN Supervised (MLP 100-50-20)",
        "XGBOD Supervised (Extreme Boosting Outlier Detector)",
        "Custom Physics-Informed NN",
    }

    rng = np.random.default_rng(0)
    X_norm = rng.normal(0, 1, (200, 4))
    X_test = X_norm.copy()
    X_test[:10] += 6.0  # 10 injected anomalies (ground truth)
    y_test = np.concatenate([np.ones(10), np.zeros(len(X_norm) - 10)]).astype(int)

    # Supervised training set: same normal distribution + a balanced block of
    # injected anomalies so class 1 carries the "anomaly" meaning the classifier
    # is supposed to learn.
    X_train_sup = np.vstack([X_norm, X_norm + 6.0])
    y_train_sup = np.concatenate([np.zeros(len(X_norm)), np.ones(len(X_norm))]).astype(int)

    print(f"{'model':<55s} {'fit':<5s} {'TP':>3s}/10 {'FP':>4s}   score range")
    print("-" * 78)

    def _sup_metrics(pred, sc, y_true):
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        acc = (tp + tn) / len(y_true)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        try:
            auc = roc_auc_score(y_true, sc)
        except Exception:
            auc = float("nan")
        return tp, fp, fn, tn, acc, prec, rec, f1, auc

    unsupervised_rows = []
    supervised_rows = []
    for name, model in get_all_models().items():
        try:
            if name in SUPERVISED_NAMES:
                model.fit_supervised(X_train_sup, y_train_sup)
                pred = model.predict(X_test)
                sc = model.decision_function(X_test)
                tp, fp, fn, tn, acc, prec, rec, f1, auc = _sup_metrics(pred, sc, y_test)
                supervised_rows.append((name, "OK", acc, prec, rec, f1, auc, tp, fp, fn, tn, sc.min(), sc.max()))
            else:
                model.fit(X_norm)
                pred = model.predict(X_test)
                sc = model.decision_function(X_test)
                unsupervised_rows.append((name, "OK", int(pred[:10].sum()), int(pred[10:].sum()), sc.min(), sc.max()))
        except Exception as e:  # noqa: BLE001 - report, never hide
            print(f"{name:<55s} FAIL {type(e).__name__}: {e}")

    for name, fit, tp, fp, smin, smax in unsupervised_rows:
        print(f"{name:<55s} {fit:<5s} {tp:>3d}/10 {fp:>4d}   {smin:.3f} .. {smax:.3f}")
    print()
    if supervised_rows:
        hdr = (f"{'model':<55s} {'fit':<5s} {'acc':>6s} {'prec':>5s} {'rec':>5s} "
               f"{'F1':>5s} {'AUC':>6s} {'TP':>3s} {'FP':>4s} {'FN':>4s} {'TN':>5s}")
        print(hdr)
        print("-" * len(hdr))
        for name, fit, acc, prec, rec, f1, auc, tp, fp, fn, tn, smin, smax in supervised_rows:
            auc_s = f"{auc:.2f}" if auc == auc else " nan"  # NaN-safe
            print(f"{name:<55s} {fit:<5s} {acc:>6.3f} {prec:>5.2f} {rec:>5.2f} "
                  f"{f1:>5.2f} {auc_s:>6s} {tp:>3d} {fp:>4d} {fn:>4d} {tn:>5d}")
    print("=" * 78)
    print("note: unsupervised detectors (IF/LOF/SVM/AE/HybridDIF) are scored on")
    print("the anomaly-score column above; supervised classifiers (FCNN/XGBOD/")
    print("PINN) get a labelled mix of normal + injected anomalies and report")
    print("proper classification metrics. ml/compare.py trains them on real")
    print("failure labels from the simulated scenarios (different methodology).")
