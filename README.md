# 🌊 Streamflow Quality Analysis: Predicting TDS with Machine Learning

Welcome to the **Streamflow Quality Analysis** repository! This project leverages advanced Machine Learning techniques to predict **Total Dissolved Solids (TDS)** in water bodies based on key water quality indicators.

We go beyond simple point predictions by exploring **Uncertainty Quantification** methods, ensuring robust and reliable insights for environmental monitoring.

---

## 🧭 Navigation

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodologies](#methodologies)
- [Models Evaluated](#models-evaluated)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Results](#results)

---

## 🔭 Project Overview

Water quality is a critical aspect of environmental health. **Total Dissolved Solids (TDS)** is a measure of the dissolved combined content of all inorganic and organic substances present in a liquid. High TDS levels can indicate pollution and affect aquatic life.

This project aims to:
1.  **Predict TDS levels** using features like pH, Salinity, Turbidity, and Water Temperature.
2.  **Optimize Model Performance** using state-of-the-art hyperparameter tuning.
3.  **Quantify Uncertainty** to provide confidence intervals alongside predictions, crucial for risk assessment.

---

## 📊 Dataset

The project uses a dataset containing water quality measurements.

| Column Name | Description | Type |
| :--- | :--- | :--- |
| **pH** | Measure of acidity or basicity of the water. | Feature |
| **Salinity** | Concentration of dissolved salts. | Feature |
| **Turbidity** | Measure of the degree to which water loses its transparency. | Feature |
| **Water Temperature** | Temperature of the water sample. | Feature |
| **TDS** | Total Dissolved Solids (Target Variable). | **Target** |

**Data Location:**
- `Data/train.csv`: Training dataset.
- `Data/test.csv`: Testing dataset.

---

## 🧪 Methodologies

We employ a three-pronged approach to modeling:

### 1. 🎛️ Hyperparameter Tuning
We utilize **Optuna**, an automatic hyperparameter optimization software framework, to find the best parameters for our models, ensuring peak performance.
- *Focus:* Autosampling and tuning for models like TabNet.

### 2. 📏 Conformal Predictions
A statistical framework that constructs prediction sets with valid coverage probabilities.
- *Libraries:* `MAPIE`, `PUNCC`.
- *Goal:* To generate rigorous uncertainty estimates that are guaranteed to cover the true value with a user-specified probability.

### 3. 🎲 Probabilistic Distribution & Quantile Regression
- **Probabilistic Distribution:** Modeling the output as a probability distribution (e.g., using TensorFlow Probability) to capture the full range of possible outcomes.
- **Quantile Regression:** Predicting specific percentiles (quantiles) of the target variable to understand the distribution of risk (e.g., 90th percentile of TDS).

---

## 🤖 Models Evaluated

We compare a wide array of powerful Gradient Boosting and Deep Learning models:

*   **CatBoost**
*   **XGBoost**
*   **LightGBM**
*   **Gradient Boosting** (sklearn)
*   **HGBM** (Histogram-based Gradient Boosting)
*   **NGBoost** (Natural Gradient Boosting)
*   **PGBM** (Probabilistic Gradient Boosting Machines)
*   **GPBoost** (Gaussian Process Boosting)
*   **TabNet** (Attentive Interpretable Tabular Learning)

---

## 📂 Repository Structure

```plaintext
.
├── Conformal Predictions/      # Experiments with MAPIE and PUNCC
│   ├── Results/                # Excel files with model metrics
│   └── ...ipynb                # Notebooks for Conformal Prediction
├── Data/                       # Train and Test CSV files
├── Hyperparameter Tuning/      # Optuna optimization scripts
├── Probabilistic Distribution/ # TensorFlow Probability experiments
├── Quantile Regression/        # Quantile Regression analysis
└── README.md                   # You are here!
```

---

## 🚀 Getting Started

### Prerequisites
To run the notebooks, you will need a Python environment (Google Colab is also a good option) with the following likely dependencies:
- Python 3.8+
- Jupyter Notebook
- Pandas, NumPy, Scikit-learn
- Optuna
- XGBoost, LightGBM, CatBoost
- TensorFlow (for Probabilistic Distribution)
- MAPIE, PUNCC (for Conformal Predictions)
- PyTorch (for TabNet)

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/ML_Analysis_Streamflow.git
    cd ML_Analysis_Streamflow
    ```
2.  Install dependencies (example):
    ```bash
    pip install pandas numpy scikit-learn optuna xgboost lightgbm catboost tensorflow torch pytorch-tabnet mapie puncc
    ```

### Running the Analysis
Navigate to the desired folder (e.g., `Hyperparameter Tuning`) and launch Jupyter Lab or Notebook:
```bash
jupyter lab
```

---

## 📈 Results

Detailed performance metrics for each model and methodology are stored in the **`Results/`** subdirectories within each technique folder. Check the `.xlsx` files for comprehensive comparisons.

---

*Analysis by [Your Name/Team]. Data Science for Environmental Good.* 🌍
