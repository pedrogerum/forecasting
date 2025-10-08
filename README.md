# Improving Demand Forecasts for Omnichannel Grocery Retail

## Overview
This repository implements the N-HiTS + MDN framework for high-frequency demand forecasting in grocery retail, as accepted in the *International Journal of Forecasting (2025)*. The method combines neural hierarchical interpolation with mixture density networks to improve forecast accuracy.

The main goal of the project is to provide the N-HiTS + MDN framework for high-frequency time series forecasting. This novel framework integrates deep learning models with a decoupled approach that separates structural demand modeling from short-term fluctuation prediction to enhance prediction accuracy. Our method combines Neural Hierarchical Interpolation for Time Series Forecasting (N-HiTS) and a Mixture Density Network (MDN) to capture short-term fluctuations and structural demand patterns, respectively.

---

## Citation

If you use this code or our findings in your research, please cite our paper:

```bibtex
@article{Gerum2025Improving,
  title={Improving High-Frequency Demand Forecasts for Omnichannel Grocery Retail},
  author={Gerum, Pedro Cesar Lopes and Herrero, Javier Rubio and Chung, Moonwon and Giaretti, Matteo},
  journal={International Journal of Forecasting},
  year={2025},
  publisher={Elsevier}
}
```

---

## Note on Reproducibility

Please be aware that due to multiple sources of randomness in deep learning experiments, the final metrics produced by this code may vary slightly from those reported in the paper. Sources of variation include:

- Stochastic elements in model training (e.g., weight initialization, data batching)
- Minor differences in software package versions
- Hardware (GPU) variations

The provided seeds should ensure that results are directionally consistent to the published findings.

---

## Repository Structure

- **`MDN_LGBM_NHITS_bike.ipynb`**: A Jupyter Notebook providing the code demonstration of the N-HiTS + MDN framework on the Bike Sharing dataset, as well as computing its benchmarks.
- **`MDN-LGBM-NHITS_Electricity.ipynb`**: The complete archival code to reproduce the experimental results from our paper using the Electricity Consumption dataset. Note: This is computationally intensive.
- **`requirements.txt`**: A list of Python packages for setting up a local environment.
- **`bike_raw.csv`**: The Bike Sharing dataset file.
- **`electricity_results.zip`**: Pre-computed forecasts and evaluation metrics for all models on the Electricity Consumption dataset. This allows for analysis of our results without re-running the lengthy experiments.

---

## Setup and Usage

We recommend using Google Colab with a GPU runtime for ease of setup.

### Step 1: Get the Code and Set Up Your Environment

#### Option A: Google Colab (Recommended)
 1. Open `MDN_LGBM_NHITS_bike.ipynb` or `MDN-LGBM-NHITS_Electricity.ipynb` in Google Colab.

 2. The first code cell in each notebook will install all necessary dependencies via pip. This process takes approximately 3-5 minutes.


#### Option B: Local Machine

To run the project locally, ensure you have Python 3.8+ and a CUDA-capable GPU.

```bash
# Clone this repository

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

### Step 2: Download and Position Datasets

This project uses two public datasets. You will need to place them in a location accessible by your notebook (e.g., Google Drive for Colab).

- **Bike Sharing Dataset**: The `bike_raw.csv` file is already included in this repository.
  - Place it in a known path. The default path in the notebook is `/content/drive/MyDrive/myproject/bike_raw.csv`.
- **Electricity Consumption Dataset**: This dataset is too large for the repository.
  - Download it from: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
  - Place it in a known path. The default path in the notebook is `/content/drive/MyDrive/myproject/electricity.csv`.


The **SuperGlovo Demand Data** used in the main part of the manuscript cannot be shared due to an NDA with data provider.


### Step 3: Configure Paths and Run the Notebooks

Before running, update the data paths in the configuration cell at the top of each notebook to match the location where you saved the datasets.

- For full replication using the smaller *Bike Sharing* dataset (faster, less storage required), run the cells in `MDN_LGBM_NHITS_bike.ipynb` sequentially.
- To reproduce results using the *Electricity Consumption* dataset (computationally intensive), run the cells in `MDN-LGBM-NHITS_Electricity.ipynb` sequentially (LGBM is commented out, so make sure to uncomment it if you want these results as well).

**Warning**: This notebook for the full experimental replication of the *Electricity Consumption* dataset is computationally intensive and may take several days to complete, requiring 150-200 GB of storage for its artifacts.

---

## Point vs. Probabilistic Forecasting

The framework is configured for point forecasting (using MAE) by default. However, it can be easily adapted for probabilistic forecasting to generate prediction intervals or full distributions.

### To generate quantile forecasts:

1. Import the QuantileLoss function: `from neuralforecast.losses.pytorch import QuantileLoss`.
2. In the N-HiTS model initialization, set the loss to `loss=QuantileLoss(quantiles=[0.05, 0.1, ..., 0.95])`.
3. Evaluate forecasts using the approximated CRPS computed using equation (14) from Section 4.3.1, or using the `properscoring` library's `crps_ensemble` function.

### To generate distributional forecasts (GMM/PMM):

1. Use the loss functions: `GaussianMixtureLoss(n_components=15)` or `PoissonMixtureLoss(n_components=15)`
2. Model will output distribution parameters to files (`test_forecasts_....csv`)
3. Evaluate using CRPS metric

Note: Pre-computed GMM/PMM results are not included due to file size. Run the notebooks with these loss functions to generate them.



