from typing import Optional
import numpy as np

class GaussianMixtureModel:
    def __init__(
        self,
        n_components: int = 4,
        n_iter      : int = 100,
        tol         : float = 1e-3,
        reg_covar   : float = 1e-6,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.n_iter       = n_iter
        self.tol          = tol
        self.reg_covar    = reg_covar
        self.random_state = random_state

        self.weights_    : Optional[np.ndarray] = None   # (K,)
        self.means_      : Optional[np.ndarray] = None   # (K, D)
        self.covariances_: Optional[np.ndarray] = None   # (K, D) diagonal
        self.converged_  : bool  = False
        self.n_iter_done_: int   = 0
        self.log_likelihood_history_: list = []

    def _init_params(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        T, D = X.shape
        K    = self.n_components
        idx  = rng.choice(T, size=K, replace=False)
        self.means_       = X[idx].copy()
        self.weights_     = np.full(K, 1.0 / K)
        self.covariances_ = np.tile(np.var(X, axis=0) + self.reg_covar, (K, 1))

    def _log_gaussian(self, X: np.ndarray, k: int) -> np.ndarray:
        """log N(x; μ_k, Σ_k) untuk semua frame — diagonal covariance."""
        D   = X.shape[1]
        mu  = self.means_[k]
        var = self.covariances_[k]
        log_det  = np.sum(np.log(var))
        diff     = X - mu
        maha     = np.sum(diff ** 2 / var, axis=1)
        return -0.5 * (D * np.log(2 * np.pi) + log_det + maha)

    def _e_step(self, X: np.ndarray):
        T, K = X.shape[0], self.n_components
        log_w = np.zeros((T, K))
        for k in range(K):
            log_w[:, k] = np.log(self.weights_[k] + 1e-300) + self._log_gaussian(X, k)
        # log-sum-exp trick
        lse      = np.max(log_w, axis=1, keepdims=True)
        log_sum  = lse.squeeze() + np.log(np.sum(np.exp(log_w - lse), axis=1))
        log_resp = log_w - log_sum[:, None]
        return log_resp, log_sum

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray):
        T, D = X.shape
        K    = self.n_components
        resp = np.exp(log_resp)
        Nk   = resp.sum(axis=0) + 1e-300

        self.weights_     = Nk / T
        self.means_       = (resp.T @ X) / Nk[:, None]

        new_cov = np.zeros((K, D))
        for k in range(K):
            diff       = X - self.means_[k]
            new_cov[k] = (resp[:, k, None] * diff ** 2).sum(axis=0) / Nk[k]
        self.covariances_ = new_cov + self.reg_covar

    def fit(self, X: np.ndarray) -> "GaussianMixtureModel":
        """Latih GMM dengan EM Algorithm."""
        assert X.ndim == 2, "X harus shape (T, D)"
        self.n_components = min(self.n_components, X.shape[0])
        self._init_params(X)
        self.log_likelihood_history_ = []
        prev_ll = -np.inf

        for it in range(self.n_iter):
            log_resp, log_sum = self._e_step(X)
            ll = float(log_sum.mean())
            self.log_likelihood_history_.append(ll)

            if abs(ll - prev_ll) < self.tol:
                self.converged_    = True
                self.n_iter_done_  = it + 1
                break
            prev_ll = ll
            self._m_step(X, log_resp)
        else:
            self.converged_    = False
            self.n_iter_done_  = self.n_iter

        return self

    def score(self, X: np.ndarray) -> float:
        """Rata-rata log p(X) untuk emission probability."""
        assert X.ndim == 2, "X harus shape (T, D)"
        K = self.n_components
        T = X.shape[0]
        log_w = np.zeros((T, K))
        for k in range(K):
            log_w[:, k] = np.log(self.weights_[k] + 1e-300) + self._log_gaussian(X, k)
        lse     = np.max(log_w, axis=1, keepdims=True)
        log_sum = lse.squeeze() + np.log(np.sum(np.exp(log_w - lse), axis=1))
        return float(log_sum.mean())