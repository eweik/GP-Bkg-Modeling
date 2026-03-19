# Machine Learning for the Jet+X Background Modeling in ATLAS Anomaly Detection Searches

This repository contains the background modeling and validation framework for the Machine Learning-based Anomaly Detection search within the ATLAS experiment. 

The primary objective of this codebase is to establish a mathematically robust, non-parametric background model capable of accurately estimating QCD backgrounds in phase spaces that have been kinematically sculpted by Autoencoders. This framework replaces rigid parametric models with an Advanced Gaussian Process (GP), enabling stable evaluation of local anomalies and the subsequent calculation of Global Significance ($Z_{global}$) and Look-Elsewhere Effect (LEE) survival curves.

## Table of Contents
1. [Physics Motivation](#physics-motivation)
2. [The Advanced Gaussian Process Architecture](#the-advanced-gaussian-process-architecture)
3. [Repository Structure](#repository-structure)
4. [Script Directory & Execution Guide](#script-directory--execution-guide)
5. [Global Significance & Look-Elsewhere Effect (LEE)](#global-significance--look-elsewhere-effect-lee)

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
3. **Locked Kernel Extraction:** To prevent the GP from absorbing genuine new physics during signal extraction, the covariance matrix (length scale and amplitude) is globally optimized on the background-only hypothesis and *frozen* prior to evaluating the Signal + Background hypothesis. This allows toy pseudo-experiments to be evaluated via rapid linear algebra matrix multiplication rather than slow, iterative MINUIT fits.

---

## Repository Structure

* `fits/` - JSON files containing baseline 5-parameter fit values.
* `plots/` - Generated diagnostic and validation plots.
* `python/` - Core Python execution scripts.
* `root/` - Input ROOT files (1% blinded data).
* `results/` - Output `.npy` arrays containing global toy statistics.
* `run/` - Bash wrapper scripts for pipeline and HTCondor execution.
* `data/` - Contains `.npz` Copula matrices for correlated toy generation.

---

## Script Directory & Execution Guide

Below is a complete glossary of the execution scripts located in the `run/` (and `python/`) directories. They are grouped by their role in the analysis pipeline.

### Part 1: Background Validation & Diagnostics
* **`run_advanced_diagnostics.sh`**
  * **Function:** Evaluates the normality of residuals for both the 5-parameter and Advanced GP models. Computes Kolmogorov-Smirnov (KS) $p$-values and generates Q-Q plots to identify heavy tails.
  * **Usage:** `./run/run_advanced_diagnostics.sh -m 0.15`
* **`run_advanced_comparisons.sh`**
  * **Function:** Generates spectral overlays of the 1% data against the legacy 5-parameter fit and the Advanced GP fit, including pull distributions.
  * **Usage:** `./run/run_advanced_comparisons.sh -m 0.15`
* **`run_pull_diagnostics.sh` / `run_comparisons.sh`**
  * **Function:** *Legacy scripts.* Runs basic diagnostics and spectral comparisons solely on the 5-parameter model.

### Part 2: Signal Injection & Spurious Signals
* **`run_efficiency_comparison.sh`**
  * **Function:** Runs a head-to-head signal efficiency comparison between the floating 5-parameter model and the locked Advanced GP. Tests if the model is too flexible (Signal Absorption).
  * **Usage:** `./run/run_efficiency_comparison.sh -t [trigger] -c [channel] -M [asimov|toys] -m 0.15 --toys [N]`
* **`run_spurious_comparison.sh`**
  * **Function:** Fits the background-only data with a Signal+Background hypothesis to measure artificial signal extraction ($S_{spur}$). Tests if the model is too rigid (Spurious Signal).
  * **Usage:** `./run/run_spurious_comparison.sh -t [trigger] -c [channel] -M [asimov|toys] -m 0.15 --toys [N]`
* **`run_gp_efficiency.sh` / `run_injection_test.sh`**
  * **Function:** *Legacy/Debug scripts.* Evaluates raw GP efficiency at specific length scales, or tests basic signal absorption for the 5-parameter model. 

### Part 3: Toy Generation & LEE Mapping
* **`run_all_toys_gp.sh`**
  * **Function:** Runs local pseudo-experiments (toys) using the GP model to map the global significance, bypassing HTCondor. Excellent for fast, small-scale testing.
  * **Usage:** `./run/run_all_toys_gp.sh [trigger] [number_of_toys]`
* **`local_to_global_z_gp.py`**
  * **Function:** Post-processing script. Reads the generated `.npy` toy arrays, maps Local Significance ($Z_{local}$) to Global Significance ($Z_{global}$), and plots the Look-Elsewhere Effect (LEE) survival curve.
  * **Usage:** `python3 python/local_to_global_z_gp.py --trigger [trigger] --ExpectedLocalZvalue [Z]`

### Part 4: HTCondor Infrastructure
* **`submit_all_triggers_gp.sh`**
  * **Function:** The master submission script. Submits massive batch jobs to the HTCondor cluster for all 7 triggers and all 3 generation methods (`naive`, `linear`, `copula`), splitting the total toys into chunks.
  * **Usage:** `./run/submit_all_triggers_gp.sh [total_toys_per_method] [toys_per_job]`
* **`submit_one_trigger_gp.sh`**
  * **Function:** Submits HTCondor batch jobs for a single, specific trigger.
  * **Usage:** `./run/submit_one_trigger_gp.sh [trigger] [total_toys] [toys_per_job] [min_len]`
* **`submit_toys_gp.sub`**
  * **Function:** The HTCondor configuration file (ClassAd). Defines memory requests, job flavors, and file transfer rules. *Not executed directly.*
* **`condor_wrapper_gp.sh`**
  * **Function:** The executable run by the isolated Condor worker nodes. Sets up the LCG environment and executes the Python toy script. *Not executed directly.*

---

## Global Significance & Look-Elsewhere Effect (LEE)

Once the GP background model is validated, we map the Global Significance ($Z_{global}$) across the entire phase space by generating massive sets of pseudo-experiments (toys). This establishes the exact Local Significance ($Z_{local}$) threshold required to claim a discovery prior to unblinding the data.

Toys are generated using three methodologies:
1. `naive`: Standard independent Poisson fluctuations.
2. `linear`: Highly correlated channel fluctuations.
3. `copula`: Fully correlated multi-channel fluctuations derived from data-driven covariance matrices.

### Step 1: Local Testing
Before submitting to HTCondor, verify the pipeline locally by generating a small batch of toys. 

*Run 10 toys for trigger t1 directly in the terminal:*
`./run/run_all_toys_gp.sh t1 10`

### Step 2: HTCondor Batch Submission
Scale the toy generation across the CERN HTCondor cluster. 

*Submit 100,000 toys for trigger t1 (split into chunks of 1,000 per job):*
`./run/submit_one_trigger_gp.sh t1 100000 1000`

*Submit 100,000 toys for ALL 7 triggers simultaneously:*
`./run/submit_all_triggers_gp.sh 100000 1000`

Use `condor_q` to monitor the jobs. Output statistics are saved as numpy arrays in the `results/` directory.

### Step 3: Mapping the LEE Survival Curve
Once the Condor jobs complete, merge the output arrays and plot the LEE survival curve.

*Calculate what Global Z corresponds to an expected Local Z of 5.0:*
`python3 python/local_to_global_z_gp.py --trigger t1 --ExpectedLocalZvalue 5.0`

The resulting plot (`plots/LEE_Curve_GP_t1.png`) will display the survival curves for the Naive, Linear, and Copula methods against standard ATLAS discovery thresholds.

---

## Dependencies
* Python 3.9+
* ROOT / `uproot`
* `numpy`, `scipy`, `matplotlib`
* `scikit-learn` (for `GaussianProcessRegressor`)
