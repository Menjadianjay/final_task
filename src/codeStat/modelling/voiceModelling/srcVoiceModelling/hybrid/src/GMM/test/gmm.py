import sys
sys.path.insert(0, r"D:\doc\2026\FINAL_CODE\src") 
from codeStat.modelimports.modelFunction import *

class GaussianMixtureModel:
    def __init__(
        self,
        n_components: int = 4,
        n_iter: int = 100,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        random_state: Optional[int] = None,
    ):
        self.n_components  = n_components
        self.n_iter        = n_iter
        self.tol           = tol
        self.reg_covar     = reg_covar
        self.random_state  = random_state

        # Hasil setelah fit()
        self.weights_   : np.ndarray = None   # (K,)
        self.means_     : np.ndarray = None   # (K, D)
        self.covariances_: np.ndarray = None  # (K, D)
        self.converged_ : bool = False
        self.n_iter_done_: int = 0
        self.log_likelihood_history_: list = []

    # Inisialisasi parameter GMM
    def _initialize(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        T, D = X.shape
        K    = self.n_components
        
        idx            = rng.choice(T, size=K, replace=False)
        self.means_    = X[idx].copy()                        # (K, D)
        self.weights_  = np.full(K, 1.0 / K)                 # (K,)
        self.covariances_ = np.tile(np.var(X, axis=0) + self.reg_covar, (K, 1))  # (K, D)
        
    # Log-probability satu komponen Gaussian (diagonal covariance)
    def _log_gaussian(self, X: np.ndarray, k: int) -> np.ndarray:
        D      = X.shape[1]
        mu     = self.means_[k]           # (D,)
        var    = self.covariances_[k]     # (D,) diagonal

        # log det(Σ) = sum(log(var_d))
        log_det  = np.sum(np.log(var))

        # Mahalanobis distance: (x-μ)² / σ²  → shape (T,)
        diff     = X - mu                 # (T, D)
        maha     = np.sum(diff ** 2 / var, axis=1)   # (T,)

        log_prob = -0.5 * (D * np.log(2 * np.pi) + log_det + maha)
        return log_prob   # (T,)
    # E-step: responsibility γ_k(x_t)
    def _e_step(self, X: np.ndarray):
        T = X.shape[0]
        K = self.n_components

        # Kumpulkan log(w_k) + log N(x; μ_k, Σ_k)  →  (T, K)
        log_weighted = np.zeros((T, K))
        for k in range(K):
            log_weighted[:, k] = np.log(self.weights_[k] + 1e-300) + self._log_gaussian(X, k)

        # Log-sum-exp trick untuk numerical stability
        log_norm     = np.max(log_weighted, axis=1, keepdims=True)        # (T, 1)
        log_sum      = log_norm.squeeze() + np.log(
            np.sum(np.exp(log_weighted - log_norm), axis=1))              # (T,)

        log_resp     = log_weighted - log_sum[:, np.newaxis]              # (T, K)
        return log_resp, log_sum

    # M-step: update parameter GMM
    def _m_step(self, X: np.ndarray, log_resp: np.ndarray):
        T, D = X.shape
        K    = self.n_components

        resp = np.exp(log_resp)                 # (T, K) — γ_k(x_t)
        Nk   = resp.sum(axis=0)                 # (K,)   — Σ_t γ_k(x_t)

        self.weights_     = Nk / T
        self.means_       = (resp.T @ X) / Nk[:, np.newaxis]   
        
        # Update covariance (diagonal): σ²_k = (Σ_t γ_k(x_t)(x_t-μ_k)²) / N_k
        new_cov = np.zeros((K, D))
        for k in range(K):
            diff       = X - self.means_[k]                          # (T, D)
            new_cov[k] = (resp[:, k, np.newaxis] * diff ** 2).sum(axis=0) / Nk[k]

        self.covariances_ = new_cov + self.reg_covar  
    # Fit GMM ke data X menggunakan EM Algorithm
    def fit(self, X: np.ndarray) -> "GaussianMixtureModel":
        assert X.ndim == 2, "X harus 2D array (T, D)"
        T, D = X.shape
        self.n_components = min(self.n_components, T)

        self._initialize(X)
        self.log_likelihood_history_ = []
        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            # E-step
            log_resp, log_sum = self._e_step(X)
            ll = log_sum.mean()
            self.log_likelihood_history_.append(ll)

            # Cek konvergensi
            if abs(ll - prev_ll) < self.tol:
                self.converged_   = True
                self.n_iter_done_ = iteration + 1
                break

            prev_ll = ll

            # M-step
            self._m_step(X, log_resp)
        else:
            self.converged_   = False
            self.n_iter_done_ = self.n_iter

        return self


    def score(self, X: np.ndarray) -> float:
        assert X.ndim == 2, "X harus 2D array (T, D)"
        K = self.n_components
        T = X.shape[0]

        log_weighted = np.zeros((T, K))
        for k in range(K):
            log_weighted[:, k] = np.log(self.weights_[k] + 1e-300) + self._log_gaussian(X, k)

        # Log-sum-exp
        log_norm = np.max(log_weighted, axis=1, keepdims=True)
        log_sum  = log_norm.squeeze() + np.log(
            np.sum(np.exp(log_weighted - log_norm), axis=1))

        return float(log_sum.mean())

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        K = self.n_components
        T = X.shape[0]

        log_weighted = np.zeros((T, K))
        for k in range(K):
            log_weighted[:, k] = np.log(self.weights_[k] + 1e-300) + self._log_gaussian(X, k)

        log_norm = np.max(log_weighted, axis=1, keepdims=True)
        log_sum  = log_norm.squeeze() + np.log(
            np.sum(np.exp(log_weighted - log_norm), axis=1))
        return log_sum