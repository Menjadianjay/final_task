"""
Implementasi Gaussian Mixture Model (GMM)
==========================================
Berdasarkan: "Perbandingkan Kinerja Algoritma GMM dan TDNN dalam
              Pengenalan Wicara Bahasa Jawa"
Penulis     : Joseph Seri Aprillio De Ornay (225314118)
Universitas : Sanata Dharma, Yogyakarta

Sub-bab     : 2.2.10 — Gaussian Mixture Model
Formula     : P(x|θ) = Σ w_k × b(x | μ_k, ρ_k)   ...(2.8)

Modul:
  1. GaussianMixtureModel  — model inti GMM dengan algoritma EM
  2. GMMClassifier         — klasifikasi fonem menggunakan GMM per kelas
  3. compute_wer           — evaluasi Word Error Rate
"""

import math
import os
import numpy as np
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# GAUSSIAN MIXTURE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class GaussianMixtureModel:
    """
    Gaussian Mixture Model sesuai sub-bab 2.2.10.

    Formula GMM (2.8):
        P(x | θ) = Σ_{k=1}^{N} w_k × b(x | μ_k, ρ_k)

    Keterangan:
        x          : Vektor fitur (MFCC)
        θ          : Parameter GMM (w_k, μ_k, ρ_k)
        N          : Jumlah komponen Gaussian
        w_k        : Bobot komponen ke-k  (Σ w_k = 1)
        b(x|μ_k,ρ_k): Fungsi densitas Gaussian Multivariat

    Algoritma pelatihan: Expectation-Maximization (EM)
        - E-step : Hitung responsibility r_{nk}
        - M-step : Update w_k, μ_k, ρ_k dari responsibilities
    """

    def __init__(self,
                 n_components: int = 16,
                 n_iter: int = 100,
                 tol: float = 1e-4,
                 reg_covar: float = 1e-6,
                 random_state: Optional[int] = None):
        """
        Args:
            n_components : Jumlah komponen Gaussian (N dalam formula 2.8)
            n_iter       : Maksimum iterasi EM
            tol          : Toleransi konvergensi log-likelihood
            reg_covar    : Regularisasi kovarians untuk stabilitas numerik
            random_state : Seed untuk reproduksibilitas
        """
        self.n_components  = n_components
        self.n_iter        = n_iter
        self.tol           = tol
        self.reg_covar     = reg_covar
        self.random_state  = random_state

        # Parameter GMM yang akan diestimasi
        self.weights_     : Optional[np.ndarray] = None   # w_k    — (K,)
        self.means_       : Optional[np.ndarray] = None   # μ_k    — (K, D)
        self.covariances_ : Optional[np.ndarray] = None   # ρ_k    — (K, D) diagonal
        self.converged_   : bool = False
        self.n_iter_done_ : int  = 0
        self.log_likelihood_history_: List[float] = []

    # ── Inisialisasi Parameter ────────────────────────────────────────────────

    def _init_params(self, X: np.ndarray):
        """
        Inisialisasi parameter GMM:
          - w_k   : seragam = 1/K
          - μ_k   : dipilih secara acak dari data (K-Means++)
          - ρ_k   : variansi keseluruhan data
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        N, D = X.shape
        K    = self.n_components

        # Bobot awal: seragam
        self.weights_ = np.full(K, 1.0 / K)

        # Rata-rata: inisialisasi K-Means++ untuk konvergensi lebih cepat
        self.means_ = self._kmeans_plus_plus_init(X, K)

        # Kovarians: variansi keseluruhan data
        global_var        = np.var(X, axis=0) + self.reg_covar
        self.covariances_ = np.tile(global_var, (K, 1))   # (K, D)

    def _kmeans_plus_plus_init(self, X: np.ndarray, K: int) -> np.ndarray:
        """
        Inisialisasi pusat K-Means++ untuk meminimalkan iterasi EM.
        Memilih K titik dari data dengan probabilitas proporsional
        terhadap jarak ke pusat yang sudah dipilih.
        """
        N = X.shape[0]
        centers = [X[np.random.randint(N)]]
        for _ in range(1, K):
            # Jarak minimum ke semua pusat yang sudah ada
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centers)
                for x in X
            ])
            probs = dists / dists.sum()
            idx   = np.random.choice(N, p=probs)
            centers.append(X[idx])
        return np.array(centers)

    # ── Fungsi Densitas Gaussian ──────────────────────────────────────────────

    def _log_gaussian_pdf(self, X: np.ndarray,
                           mean: np.ndarray,
                           cov: np.ndarray) -> np.ndarray:
        """
        Log-densitas Gaussian Multivariat (kovarians diagonal):
            log b(x | μ, ρ) = -0.5 × [D·ln(2π) + Σ ln(ρ_d) + Σ (x_d-μ_d)²/ρ_d]

        Args:
            X    : (N, D) — matriks fitur
            mean : (D,)   — vektor rata-rata μ_k
            cov  : (D,)   — vektor varians diagonal ρ_k

        Returns:
            (N,) — log-probabilitas setiap sampel
        """
        D        = X.shape[1]
        cov_safe = cov + self.reg_covar
        diff     = X - mean                                   # (N, D)
        log_det  = np.sum(np.log(cov_safe))                   # skalar
        quad     = np.sum((diff ** 2) / cov_safe, axis=1)    # (N,)
        return -0.5 * (D * np.log(2 * np.pi) + log_det + quad)

    def _log_weighted_gaussian(self, X: np.ndarray) -> np.ndarray:
        """
        Hitung log(w_k × b(x | μ_k, ρ_k)) untuk semua komponen k.

        Returns:
            (N, K) — log-probabilitas tertimbang setiap sampel per komponen
        """
        N, K = X.shape[0], self.n_components
        log_weighted = np.zeros((N, K))
        for k in range(K):
            log_w_k           = math.log(self.weights_[k] + 1e-300)
            log_weighted[:, k] = log_w_k + self._log_gaussian_pdf(
                X, self.means_[k], self.covariances_[k])
        return log_weighted

    # ── E-Step ───────────────────────────────────────────────────────────────

    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        E-Step: Hitung responsibility r_{nk} dan log-likelihood.

        r_{nk} = w_k × b(x_n | μ_k, ρ_k)
                 ─────────────────────────────
                 Σ_j w_j × b(x_n | μ_j, ρ_j)

        Returns:
            resp           : (N, K) — matriks responsibility
            log_likelihood : float  — rata-rata log P(x|θ)
        """
        log_weighted = self._log_weighted_gaussian(X)          # (N, K)

        # Log-sum-exp untuk stabilitas numerik
        log_norm     = self._log_sum_exp(log_weighted)         # (N,)
        log_resp     = log_weighted - log_norm[:, None]        # (N, K)
        resp         = np.exp(log_resp)                        # (N, K)

        log_likelihood = float(log_norm.mean())
        return resp, log_likelihood

    @staticmethod
    def _log_sum_exp(log_probs: np.ndarray) -> np.ndarray:
        """
        Stabilisasi numerik: log Σ exp(a_k) = max + log Σ exp(a_k - max)
        """
        max_log = log_probs.max(axis=1, keepdims=True)
        return (np.log(np.exp(log_probs - max_log).sum(axis=1))
                + max_log.squeeze(axis=1))

    # ── M-Step ───────────────────────────────────────────────────────────────

    def _m_step(self, X: np.ndarray, resp: np.ndarray):
        """
        M-Step: Update parameter GMM dari responsibilities.

            N_k  = Σ_n r_{nk}
            w_k  = N_k / N
            μ_k  = (Σ_n r_{nk} x_n) / N_k
            ρ_k  = (Σ_n r_{nk} (x_n - μ_k)²) / N_k

        Args:
            X    : (N, D) — matriks fitur
            resp : (N, K) — matriks responsibility dari E-step
        """
        N, D = X.shape
        K    = self.n_components

        # N_k — jumlah efektif sampel per komponen
        Nk = resp.sum(axis=0) + 1e-10                         # (K,)

        # Update bobot w_k
        self.weights_ = Nk / N                                 # (K,)

        # Update rata-rata μ_k
        self.means_ = (resp.T @ X) / Nk[:, None]              # (K, D)

        # Update kovarians ρ_k (diagonal)
        new_cov = np.zeros((K, D))
        for k in range(K):
            diff          = X - self.means_[k]                # (N, D)
            weighted_sq   = (resp[:, k:k+1] * diff ** 2)      # (N, D)
            new_cov[k]    = weighted_sq.sum(axis=0) / Nk[k] + self.reg_covar
        self.covariances_ = new_cov

    # ── Pelatihan EM ─────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "GaussianMixtureModel":
        """
        Melatih GMM menggunakan algoritma Expectation-Maximization (EM).

        Args:
            X : (N, D) — matriks fitur MFCC pelatihan

        Returns:
            self (untuk method chaining)
        """
        self._init_params(X)
        self.log_likelihood_history_ = []
        prev_ll = -np.inf

        for iteration in range(1, self.n_iter + 1):
            # E-step
            resp, log_likelihood = self._e_step(X)

            # M-step
            self._m_step(X, resp)

            self.log_likelihood_history_.append(log_likelihood)
            delta = log_likelihood - prev_ll

            if iteration % 10 == 0 or iteration == 1:
                print(f"    Iterasi {iteration:3d}: log-likelihood = {log_likelihood:.6f}  "
                      f"Δ = {delta:.2e}")

            if delta < self.tol and iteration > 1:
                self.converged_   = True
                self.n_iter_done_ = iteration
                print(f"    ✓ Konvergen pada iterasi {iteration}")
                break

            prev_ll = log_likelihood

        if not self.converged_:
            self.n_iter_done_ = self.n_iter
            print(f"    ⚠ Belum konvergen setelah {self.n_iter} iterasi")

        return self

    # ── Inferensi ─────────────────────────────────────────────────────────────

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Hitung log P(x|θ) = log Σ_k w_k × b(x | μ_k, ρ_k) untuk setiap sampel.

        Returns:
            (N,) — log-likelihood setiap bingkai
        """
        log_weighted = self._log_weighted_gaussian(X)
        return self._log_sum_exp(log_weighted)

    def score(self, X: np.ndarray) -> float:
        """Rata-rata log-likelihood pada data X."""
        return float(self.score_samples(X).mean())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Hitung probabilitas posterior setiap komponen (responsibility).

        Returns:
            (N, K) — probabilitas P(k | x_n)
        """
        resp, _ = self._e_step(X)
        return resp

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Tentukan komponen Gaussian yang paling mungkin untuk setiap sampel.

        Returns:
            (N,) — indeks komponen terbaik per sampel
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def summary(self) -> str:
        """Ringkasan parameter GMM yang telah dilatih."""
        if self.weights_ is None:
            return "GMM belum dilatih."
        lines = [
            f"GMM Summary",
            f"  Komponen (K)      : {self.n_components}",
            f"  Dimensi fitur (D) : {self.means_.shape[1]}",
            f"  Konvergen         : {self.converged_}",
            f"  Iterasi EM        : {self.n_iter_done_}",
            f"  Log-likelihood akhir: {self.log_likelihood_history_[-1]:.6f}",
            f"",
            f"  Bobot w_k:",
        ]
        for k in range(self.n_components):
            lines.append(f"    k={k:2d}: w={self.weights_[k]:.4f}  "
                         f"μ[0:3]={self.means_[k,:3].round(3)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# GMM CLASSIFIER — Klasifikasi Fonem
# ─────────────────────────────────────────────────────────────────────────────

class GMMClassifier:
    """
    Klasifikasi fonem menggunakan satu GMM per kelas (fonem).

    Strategi:
        Untuk setiap fonem c, latih GMM_c pada data fitur fonem tersebut.
        Saat inferensi, pilih fonem dengan log-likelihood tertinggi:
            ĉ = argmax_c  log P(x | GMM_c)
    """

    def __init__(self, n_components: int = 16, n_iter: int = 100,
                 tol: float = 1e-4, reg_covar: float = 1e-6):
        self.n_components = n_components
        self.n_iter       = n_iter
        self.tol          = tol
        self.reg_covar    = reg_covar
        self.classes_     : List[str]                    = []
        self.gmm_models_  : Dict[str, GaussianMixtureModel] = {}

    def fit(self, X_per_class: Dict[str, np.ndarray]) -> "GMMClassifier":
        """
        Melatih satu GMM untuk setiap kelas fonem.

        Args:
            X_per_class : {nama_fonem: matriks_fitur (N, D)}
        """
        self.classes_ = sorted(X_per_class.keys())
        for fonem, X in X_per_class.items():
            print(f"\n  ── Melatih GMM untuk fonem '{fonem}' "
                  f"[{X.shape[0]} bingkai × {X.shape[1]} fitur]")
            gmm = GaussianMixtureModel(
                n_components=self.n_components,
                n_iter=self.n_iter,
                tol=self.tol,
                reg_covar=self.reg_covar,
                random_state=hash(fonem) % 10000
            )
            gmm.fit(X)
            self.gmm_models_[fonem] = gmm
        return self

    def predict_frame(self, x: np.ndarray) -> str:
        """
        Klasifikasi satu bingkai fitur ke fonem terbaik.

        Args:
            x : (D,) atau (1, D)

        Returns:
            nama fonem dengan log-likelihood tertinggi
        """
        if x.ndim == 1:
            x = x[None, :]
        scores = {
            fonem: float(gmm.score_samples(x)[0])
            for fonem, gmm in self.gmm_models_.items()
        }
        return max(scores, key=scores.get)

    def predict_sequence(self, X: np.ndarray) -> List[str]:
        """
        Klasifikasi setiap bingkai dalam matriks fitur.

        Args:
            X : (T, D) — urutan bingkai MFCC

        Returns:
            List[str] — urutan fonem hasil dekoding
        """
        return [self.predict_frame(X[t]) for t in range(X.shape[0])]

    def score_all_classes(self, X: np.ndarray) -> Dict[str, float]:
        """
        Hitung log-likelihood rata-rata dari setiap GMM terhadap data X.

        Returns:
            {fonem: rata_rata_log_likelihood}
        """
        return {
            fonem: float(gmm.score(X))
            for fonem, gmm in self.gmm_models_.items()
        }