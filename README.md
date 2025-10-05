# Improving Demand Forecasts for Omnichannel Grocery Retail

## Overview
This repository implements the N-HiTS + MDN framework for high-frequency demand forecasting in grocery retail, as published in the *International Journal of Forecasting (2025)*. The method combines neural hierarchical interpolation with mixture density networks to improve forecast accuracy.

The main goal of the project is to provide the N-HiTS + MDN framework for high-frequency time series forecasting. This novel framework integrates deep learning models with a decoupled approach that separates structural demand modeling from short-term fluctuation prediction to enhance prediction accuracy. Our method combines Neural Hierarchical Interpolation for Time Series Forecasting (N-HiTS) and a Mixture Density Network (MDN) to capture short-term fluctuations and structural demand patterns, respectively.

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

## Note on Reproducibility

Please be aware that due to multiple sources of randomness in deep learning experiments, the final metrics produced by this code may vary slightly from those reported in the paper. Sources of variation include:

- Stochastic elements in model training (e.g., weight initialization, data batching)
- Minor differences in software package versions
- Hardware (GPU) variations

The provided seeds should ensure that results are directionally consistent and numerically close to the published findings.

## Repository Structure

- **`MDN-NHITS forecasting_bike.ipynb`**: A Jupyter Notebook providing a simplified and faster demonstration of the N-HiTS + MDN framework on the Bike Sharing dataset.
- **`MDN-LGBM-NHITS_Electricity.ipynb`**: A Jupyter Notebook containing the full, archival code to reproduce the experiments on the Electricity Load Diagrams dataset.
- **`requirements.txt`**: A list of Python packages for setting up a local environment.
- **`bike_raw.csv`**: Bike Dataset.
- **`electricity.csv`**: Electricity Dataset.

## Software Requirements
- Python 3.8+
- PyTorch 1.x+ (installed via Colab)
- neuralforecast (version specified in requirements.txt)
- Google Colab (recommended) OR local environment with CUDA-capable GPU

See `requirements.txt` for complete dependency list with pinned versions.

## Data Availability and Provenance

### Public Datasets (Included in Repository)
- **Bike Sharing Dataset**
  - Source: UCI Machine Learning Repository
  - URL: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
  - File: `bike_raw.csv`
  
- **Electricity Load Diagrams 2011-2014**
  - Source: UCI Machine Learning Repository  
  - URL: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
  - File: `electricity.csv`

### Proprietary Dataset (Not Available)
- **Glovo Demand Data**: Cannot be shared due to NDA with data provider

### Data Setup

Before running the notebooks, please upload the necessary data files to your Google Drive and ensure the paths in the CONFIG section of each notebook point to the correct locations. The default paths are:

- **Electricity Dataset**: `/content/drive/MyDrive/electricity.csv`
- **Bike Sharing Dataset**: `/content/drive/MyDrive/myproject/bike_raw.csv`

## Setup for Google Colab

This project is designed to be run in a **Google Colab environment with a GPU runtime enabled**.

### Steps:

We provide two notebooks to demonstrate our framework.

### 1. Lightweight implementation of N-HiTS + MDN framework (Bike Sharing Dataset)

The notebook **`MDN-NHITS forecasting_bike.ipynb`** serves as a lightweight example of the N-HiTS + MDN framework. It runs on the free tier of Google Colab.

1. Upload `bike_raw.csv` to your Google Drive at `/content/drive/MyDrive/myproject/`
2. Open `MDN-NHITS forecasting_bike.ipynb` in Google Colab
3. Run Cell 1 to install dependencies (~3-5 minutes)
4. Run Cell 2 to configure model settings (see Configuration Options below)
5. Execute remaining cells sequentially

This notebook is configured to run **one model variant at a time**. To test the different configurations explored in our paper, you will need to manually adjust the code in the second cell:

#### To Change the Loss Function

In the `CONFIG` dictionary, modify the `NHITS_LOSS` key. Valid options are:

- `"NHITS_LOSS": "PMM"` (Poisson Mixture Model)
- `"NHITS_LOSS": "GMM"` (Gaussian Mixture Model)
- `"NHITS_LOSS": "MAE"` (Mean Absolute Error for point forecasts)

#### To Enable/Disable MDN Covariates

In the NHITS model definition within the main function, comment or uncomment the `futr_exog_list` parameter:

- **With MDN**: `futr_exog_list=["total_mean", "total_variance"]`
- **Without MDN**: `# futr_exog_list=["total_mean", "total_variance"]`

#### For the proposed model N-HiTS + MDN, please set:
- `"NHITS_LOSS": "MAE"` (Mean Absolute Error for point forecasts)
- `futr_exog_list=["total_mean", "total_variance"]`


### 2. Full experimental replication with LightGBM baseline (Electricity Dataset)

The notebook **`MDN-LGBM-NHITS_Electricity.ipynb`** contains the complete code to replicate the results for the Electricity dataset.

**Important Note on Runtime**: This is an archival notebook and is **computationally intensive**. Training models may take **several days** to complete, even on a high-performance Google Colab instance.

To run the experiments, you must execute the cells sequentially. Note that some blocks are commented out by default to allow for modular execution.

For LGBM training inclusion, please uncomment *BLOCK 2: LGBM FORECASTING* to generate the LightGBM model predictions before running the subsequent evaluation cells for that model.

## Running Locally (Alternative to Colab)

If you prefer to run this project on your local machine, you can use the `requirements.txt` file to set up a Python environment.

## Probabilistic Forecasting

The code provided implements point forecasts with MAE loss. To reproduce probabilistic forecasting results:

### For Multiquantile (Pinball Loss):
1. Include `from neuralforecast.losses.pytorch import QuantileLoss`
1. In the N-HiTS model initialization, change loss function to:
   - `loss = QuantileLoss(q=[0.05, 0.10, ..., 0.95])`
2. Training remains identical, evaluation uses CRPS metric

### For GMM/PMM distributions:
1. Replace loss function with:
   - `loss = GaussianMixtureLoss(n_components=15)` or
   - `loss = PoissonMixtureLoss(n_components=15)`
2. Model outputs distribution parameters instead of point estimates
3. Evaluation uses CRPS metric

**Computational requirements:** Full replication requires several days of GPU time (T4 equivalent) per model variant.


