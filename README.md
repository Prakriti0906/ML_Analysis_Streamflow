# 🌊 ML Analysis of Streamflow: Uncertainty Quantification & TDS Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Models-TabNet%20%7C%20Treeffuser%20%7C%20XGBoost%20%7C%20CatBoost-orange?style=for-the-badge)
![Uncertainty Quantification](https://img.shields.io/badge/UQ-Conformal%20Prediction%20%7C%20Probabilistic%20Diffusion-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

A comprehensive machine learning framework designed to predict **Total Dissolved Solids (TDS)** in streamflow. This project moves beyond standard point predictions by implementing state-of-the-art **Uncertainty Quantification (UQ)** workflows, including **Generative Diffusion Models (Treeffuser)**, **Deep Learning (TabNet)**, and **Conformal Predictions**, ensuring that environmental monitoring decisions are backed by statistical confidence.

---

## 📑 Table of Contents (Navigation)

1. [📌 Project Overview](#-project-overview)
2. [📂 Repository Structure](#-repository-structure)
3. [📊 Dataset Details](#-dataset-details)
4. [🛠️ Workflow & Methodology](#-workflow--methodology)
    - [Phase 1: Hyperparameter Tuning](#phase-1-hyperparameter-tuning)
    - [Phase 2: Quantile Regression](#phase-2-quantile-regression)
    - [Phase 3: Probabilistic Distribution](#phase-3-probabilistic-distribution)
    - [Phase 4: Conformal Predictions](#phase-4-conformal-predictions)

---

## 📌 Project Overview

Accurate prediction of water quality parameters like TDS is vital for environmental management. This repository provides a robust pipeline for analyzing hydrometric data using advanced Machine Learning regressors. Key features include:

* **Deep Learning Integration**: Implementation of **TabNet** (Attentive Interpretable Tabular Learning).
* **Generative Uncertainty**: Utilization of **Treeffuser**, a score-based diffusion model for probabilistic prediction on tabular data.
* **Advanced Optimization**: Automated hyperparameter tuning using **Optuna** with various samplers.
* **Guaranteed Coverage**: **Conformal Prediction** (using MAPIE & PUNCC) to generate prediction intervals with mathematically guaranteed validity (e.g., 90% confidence).

---

## 📂 Repository Structure

The project is organized into modular directories representing the analysis pipeline.

### 1. 🗂️ [Data](./Data)
Contains the water quality dataset.
* **[`train.csv`](./Data/train.csv)**: Labeled data used for model training.
* **[`test.csv`](./Data/test.csv)**: Held-out data for final model evaluation.

### 2. 🎛️ [Hyperparameter Tuning](./Hyperparameter%20Tuning)
Before training UQ models, base regressors are optimized to minimize error.
* **[`Optuna_autosampler_TDS_tabnet.ipynb`](./Hyperparameter%20Tuning/Optuna_autosampler_TDS_tabnet.ipynb)**: Implements Bayesian Optimization via Optuna to find optimal parameters for TabNet, XGBoost, CatBoost, etc.
* **`test_results.xlsx`**: Stores the optimized parameter sets and comparative scores.

### 3. 📉 [Quantile Regression](./Quantile%20Regression)
Focuses on predicting conditional quantiles (e.g., $Q_{0.05}$ and $Q_{0.95}$) rather than just the mean.
* **[`Quantile_Regression.ipynb`](./Quantile%20Regression/Quantile_Regression.ipynb)**: Trains models to minimize Pinball Loss.
* **`Results/`**: Contains prediction files for **TabNet, XGBoost, LightGBM, CatBoost, HGBM, GPBoost,** and **PGBM**.

### 4. 📊 [Probabilistic Distribution](./Probabilistic%20Distribution)
Treats the target variable as a distribution or uses generative models to capture uncertainty.
* **[`Probabilistic__Distribution_TDS_TF.ipynb`](./Probabilistic%20Distribution/Probabilistic__Distribution_TDS_TF.ipynb)**: Implements **Treeffuser** (Diffusion Model), **NGBoost**, and **PGBM** to model the full conditional distribution $P(Y|X)$.
* **`Results/`**: Includes Matrix Evaluation metrics and PDF (Probability Density Function) plots.

### 5. 🛡️ [Conformal Predictions](./Conformal%20Predictions)
Applies rigorous statistical calibration to ensure prediction intervals are valid.
* **[`Conformal Predictions(MAPIE,PUNCC)_TDS_Tabnet.ipynb`](./Conformal%20Predictions/Conformal%20Predictions(MAPIE,PUNCC)_TDS_Tabnet.ipynb)**: Uses Split Conformal Prediction (SCP) and Cross-Validation (CV+) techniques.
* **`Results/`**: CSVs containing lower and upper bounds tailored to a specific $\alpha$ (error rate).

---

## 📊 Dataset Details

The analysis is based on experimental water quality data containing key physicochemical parameters:

| Feature | Description |
| :--- | :--- |
| **pH** | Measure of acidity or alkalinity of the water |
| **Salinity** | Concentration of dissolved salts |
| **Turbidity** | Measure of relative clarity of a liquid |
| **Water Temperature** | Temperature of the streamflow |
| **TDS** | **Target Variable**: Total Dissolved Solids (mg/L) |

---

## 🛠️ Workflow & Methodology

### Phase 1: Hyperparameter Tuning
We utilize **Optuna** with the Tree-structured Parzen Estimator (TPE) sampler.
1.  **Search Space**: Defined for Learning Rate, Max Depth, Regulators, and TabNet specific parameters (sparsity, steps).
2.  **Objective**: Minimize RMSE (Root Mean Squared Error) on cross-validation folds.
3.  **Outcome**: Best parameters are saved to `test_results.xlsx` and passed to the UQ modules.

### Phase 2: Quantile Regression
Standard regression predicts the conditional mean $E[Y|X]$. Quantile regression predicts $Q_\tau(Y|X)$.
* **Models**: TabNet, XGBoost, CatBoost, LightGBM, HGBM, GPBoost.
* **Application**: We predict the 5th and 95th percentiles to create a 90% prediction interval.
* **Metric**: Pinball Loss (Quantile Loss).

### Phase 3: Probabilistic Distribution
This approach assumes $Y|X \sim \mathcal{D}(\theta)$ or generates samples from the learned distribution.
* **Treeffuser**: A novel score-based generative model using diffusion processes for tabular data.
* **NGBoost**: Uses Natural Gradients to boost the parameters of a Gaussian distribution.
* **PGBM**: Probabilistic Gradient Boosting Machines optimizing CRPS.
* **Benefit**: Allows calculating the probability of TDS exceeding a safety threshold.

### Phase 4: Conformal Predictions
A wrapper method that calibrates any base model to provide valid intervals.
* **Libraries Used**: `MAPIE`, `PUNCC`.
* **Guarantee**: If we set confidence to 90%, the true TDS value is mathematically guaranteed to fall within the predicted range 90% of the time (under exchangeability assumptions).
* **Metric**: Mean Prediction Interval Width (MPIW) vs. Prediction Interval Coverage Probability (PICP).

---

*Analysis by []. Data Science for Environmental Good.* 🌍
