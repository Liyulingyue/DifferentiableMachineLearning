"""ClassicalML submodule: classical ML models re-implemented as
``torch.nn.Module`` so they can be composed with deep networks.

Currently hosts:

* :class:`KMeans`            — centroids as learnable parameters; soft or
  hard assignment; standard Lloyd updates during ``fit``.
* :class:`KNN`               — non-parametric memory layer; top-k
  retrieval and average-pool reconstruction.
* :class:`PCA`               — orthonormal basis as a learnable
  parameter; project / reconstruct; SVD-based fitting.
* :class:`KernelRidge`       — dual-form linear-in-memory model with
  RBF / linear / polynomial kernels.
* :class:`GMM`               — Gaussian mixture with learnable means,
  log-variances, and weights; EM fitting; differentiable soft
  responsibilities.
* :class:`LDA`               — Fisher linear discriminant analysis with
  closed-form eigen solve.
* :class:`GaussianNB`        — per-class Gaussian Naive Bayes with
  MLE-estimated mean / variance.
* :class:`MultinomialNB`     — multinomial Naive Bayes for count
  features with Laplace smoothing.
* :class:`ICA`               — FastICA fixed-point source separation
  with whitening.
* :class:`SoftDecisionTree`  — Frosst 2017 differentiable decision
  tree with sigmoid inner nodes and per-leaf class distributions.
* :class:`BayesianRidge`     — Bayesian linear regression with
  marginal-likelihood maximisation; full predictive distribution.
* :class:`SVM`               — LS-SVM with kernelised RBF / linear /
  polynomial kernels and one-vs-rest multi-class extension.
* :class:`GaussianProcess`   — GP regression with RBF / Matérn-3/2 /
  Matérn-5/2 / linear / polynomial kernels and learnable
  hyperparameters.
* :class:`GaussianHMM`       — categorical-emission hidden Markov
  model with closed-form EM (Baum-Welch).
* :class:`KalmanFilter`      — Linear Dynamical System with
  Kalman filtering / RTS smoothing and closed-form EM.
* :class:`LinearChainCRF`    — linear-chain CRF with
  forward-backward training and Viterbi decode.
* :class:`tSNE`              — Student-t SNE embedding with
  early-exaggeration and Adam-based KL minimisation.
* :class:`SparseCoding`     — dictionary learning with a learnable
  ``[n_atoms, n_features]`` matrix and ISTA / FISTA encoder.
* :class:`BayesianLinearVI` — Bayesian linear regression with
  mean-field variational inference over the weights, reparameterised
  samples, and an analytical KL term.
* :class:`tICA`              — Time-lagged Independent Component
  Analysis (slow-modes projection) with closed-form generalised
  eigen solve.
* :class:`NMF`               — Non-negative matrix factorisation
  with Lee-Seung multiplicative updates and NNDSVD init.
* :class:`GMMHMM`            — Gaussian-Mixture-emission HMM
  (continuous observations, Baum-Welch EM).
* :class:`ProbabilisticPCAMixture` — mixture of probabilistic PCA
  with closed-form EM per cluster (Tipping & Bishop 1999).
* :class:`SparseGP`            — variational sparse Gaussian Process
  with Titsias 2009 ELBO and learnable inducing points.
* :class:`SpectralClustering`  — Ng-Jordan-Weiss spectral clustering
  with RBF affinity and symmetric normalised graph Laplacian.
* :class:`RBM`                 — Bernoulli Restricted Boltzmann Machine
  with CD-k training and free-energy evaluation.
* :class:`IsingModel`          — 2-D Ising model with single-spin
  Gibbs sampling under periodic boundary conditions.
"""

from .KMeans import KMeans
from .KNN import KNN
from .PCA import PCA
from .KernelRidge import KernelRidge
from .GMM import GMM
from .LDA import LDA
from .NaiveBayes import GaussianNB, MultinomialNB
from .ICA import ICA
from .SoftDecisionTree import SoftDecisionTree
from .BayesianRidge import BayesianRidge
from .SVM import SVM
from .GaussianProcess import GaussianProcess
from .GaussianHMM import GaussianHMM
from .KalmanFilter import KalmanFilter
from .LinearChainCRF import LinearChainCRF
from .tSNE import tSNE
from .SparseCoding import SparseCoding
from .BayesianLinearVI import BayesianLinearVI
from .tICA import tICA
from .NMF import NMF
from .GMMHMM import GMMHMM
from .ProbabilisticPCAMixture import ProbabilisticPCAMixture
from .SparseGP import SparseGP
from .SpectralClustering import SpectralClustering
from .RBM import RBM
from .IsingModel import IsingModel

__all__ = [
    "KMeans", "KNN", "PCA", "KernelRidge", "GMM",
    "LDA", "GaussianNB", "MultinomialNB", "ICA", "SoftDecisionTree",
    "BayesianRidge", "SVM", "GaussianProcess", "GaussianHMM",
    "KalmanFilter", "LinearChainCRF", "tSNE",
    "SparseCoding", "BayesianLinearVI", "tICA", "NMF",
    "GMMHMM", "ProbabilisticPCAMixture", "SparseGP",
    "SpectralClustering", "RBM", "IsingModel",
]
