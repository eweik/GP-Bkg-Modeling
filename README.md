# Machine Learning for the ATLAS MET Trigger and Anomaly Detection: Background Modeling

This repository contains the background modeling and validation framework for the Machine Learning-based Anomaly Detection search within the ATLAS MET Trigger phase space. 

The primary objective of this codebase is to establish a mathematically robust, non-parametric background model capable of accurately estimating QCD backgrounds in phase spaces that have been kinematically sculpted by Autoencoders. This framework replaces rigid parametric models with an Advanced Gaussian Process (GP), enabling stable evaluation of local anomalies and the subsequent calculation of Global Significance ($Z_{global}$) and Look-Elsewhere Effect (LEE) survival curves.

## Table of Contents
1. [Physics Motivation](#physics-motivation)
2. [The Advanced Gaussian Process Architecture](#the-advanced-gaussian-process-architecture)
3. [Repository Structure](#repository-structure)
4. [Running the Validation Pipeline](#running-the-validation-pipeline)

---

## Physics Motivation

Standard parametric models (e.g., the 5-parameter dijet function) are highly optimized for pure QCD phase space. However, when data is filtered through an ML anomaly detection algorithm (like an Autoencoder), the resulting invariant mass spectrum can exhibit localized kinematic sculpting. 

When a rigid 5-parameter fit is forced onto this sculpted data, it fails to capture local structures, resulting in:
* **Heavy Tails:** Non-Gaussian pull distributions in the high-mass region.
* **Artificial Anomalies:** Wavy residuals that manifest as fake local significance spikes (spurious signals), severely degrading the discovery power of the search.

To resolve this, this repository implements a **Gaussian Process (GP)** to serve as a flexible, data-driven background model.

---

## The Advanced Gaussian Process Architecture

To prevent the GP from overfitting the sparse, high-mass tails (the "Floating Kernel Trap") while maintaining flexibility in the bulk, we utilize an **Advanced GP Architecture** with three core features:

1. **Log-X Transformation:** The mass variable is transformed to $X = \ln(m)$. This ensures that a stationary RBF length scale correctly mirrors the ATLAS detector's fractional mass resolution, which widens linearly at higher energies ($\frac{\sigma_m}{m} \approx \text{const}$).
2. **Parametric Mean Prior:** The GP fits the *log-ratio* of the data to the legacy 5-parameter fit. In completely empty bins at the kinematic limit, the GP safely decays back to the physical 5-parameter QCD prediction.
3. **Locked Kernel Extraction:** To prevent the GP from absorbing genuine new physics during signal extraction, the covariance matrix (length scale and amplitude) is globally optimized on the background-only hypothesis and *frozen* prior to evaluating the Signal + Background hypothesis.

---

## Repository Structure

* `fits/` - JSON files containing baseline 5-parameter fit values.
* `plots/` - Generated diagnostic and validation plots.
  * `advanced_comparisons/` - Spectral overlays of data vs. models.
  * `efficiency_comparisons/` - Signal absorption/efficiency test results.
  * `pull_diagnostics/` - KS tests and Q-Q plots for normality.
  * `spurious_comparisons/` - Spurious signal test results.
* `python/` - Core Python execution scripts.
* `root/` - Input ROOT files (1% blinded data).
* `run/` - Bash wrapper scripts for pipeline execution.

---

## Running the Validation Pipeline

The pipeline is executed via bash wrappers in the `run/` directory. All scripts support standard arguments to isolate specific triggers and channels, or run the entire matrix.

**Global Arguments:**
* `-t, --trigger`: Specify a trigger (e.g., `t1`) or `all`.
* `-c, --channel`: Specify a channel (e.g., `jj`) or `all`.
* `-m, --min-len`: The minimum GP log-space length scale bound (default: `0.15`, enforcing a smoothing window of $\ge 15\%$ of the local mass).

### 1. Pull Diagnostics
Evaluates the normality of the residuals for both the 5-parameter and GP models. Calculates the Kolmogorov-Smirnov (KS) $p$-value and generates Q-Q plots to identify heavy tails.

`./run/run_advanced_diagnostics.sh -m 0.15`

### 2. Spectral Comparisons
Generates visual overlays of the 1% data, the legacy 5-parameter fit, and the Advanced GP fit, including pull distributions and empirical Gaussian fits of the errors.

`./run/run_advanced_comparisons.sh -m 0.15`

### 3. Signal Extraction Efficiency
Tests for **Signal Absorption** (Is the model too flexible?). Injects a $5\sigma$ Gaussian anomaly (3% mass resolution) into the background and measures the fraction of signal successfully extracted. Supports both perfect mathematical evaluation (`asimov`) and stability testing via Poisson fluctuations (`toys`).

**Fast Asimov Test (Mathematical Bias):**
`./run/run_efficiency_comparison.sh -t t1 -c jj -M asimov -m 0.15`

**Stability Test with Poisson Toys:**
`./run/run_efficiency_comparison.sh -t all -c all -M toys --toys 20 -m 0.15`

### 4. Spurious Signal Tests
Tests for **Model Rigidity** (Does the model hallucinate fake signals?). Fits the background-only data with the Signal + Background hypothesis to measure artificial signal extraction ($S_{spur}$). A successful model should maintain an average $\frac{|S_{spur}|}{\sigma_{stat}}$ well below 0.5.

**Evaluate Spurious Signal across all channels using 20 toys per mass point:**
`./run/run_spurious_comparison.sh -t all -c all -M toys --toys 20 -m 0.15`

---

## Dependencies
* Python 3.9+
* ROOT / `uproot`
* `numpy`, `scipy`, `matplotlib`
* `scikit-learn` (for `GaussianProcessRegressor`)
