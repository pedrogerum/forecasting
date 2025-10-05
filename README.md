# Improving Demand Forecasts for Omnichannel Grocery Retail

This repository contains the code for experiments on public datasets presented in the paper **"Improving Demand Forecasts for Omnichannel Grocery Retail"**, published in the *International Journal of Forecasting*.

## Abstract

The rise of omnichannel grocery retail has introduced significant operational complexities, emphasizing the importance of high-frequency demand forecasting. In this context, achieving both accuracy and computational efficiency with standard forecasting tools remains a challenge. 

Addressing these limitations, we propose a novel framework that integrates deep learning models with a decoupled approach that separates structural demand modeling from short-term fluctuation prediction to enhance prediction accuracy. Our method combines **Neural Hierarchical Interpolation for Time Series Forecasting (N-HiTS)** and a **Mixture Density Network (MDN)** to capture short-term fluctuations and structural demand patterns, respectively. 

This framework is extended to probabilistic forecasting, comparing quantile-based and distributional models, both with and without the decoupling approach. Empirical validation using data from a leading on-demand delivery service demonstrates significant improvements in deep learning methods over traditional ARIMA methods and the industry-standard Gradient Boosting Machine (GBM), and validates the effectiveness of the decoupling approach. 

### Key Results

- Reduces **mean absolute percentage error (MAPE)** for point estimates from **23.00% to 14.31%** (a **37.78% reduction**)
- Reduces **continuous ranked probability score (CRPS)** from **10.85 to 2.34** (a **78.44% reduction**)

These findings can provide grocery e-commerce companies with valuable insights for optimizing inventory management, driver scheduling, and overall operational efficiency.

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

- **`MDN-LGBM-NHITS_Electricity.ipynb`**: A Jupyter Notebook containing the full, archival code to reproduce the experiments on the Electricity Load Diagrams dataset.
- **`MDN-NHITS forecasting_bike.ipynb`**: A Jupyter Notebook providing a simplified and faster demonstration of the N-HiTS + MDN framework on the Bike Sharing dataset.
- **`requirements.txt`**: A list of Python packages for setting up a local environment.

## Data

The primary dataset from Glovo used in our paper **cannot be shared** due to a non-disclosure agreement (NDA). However, the public datasets are included in the repository.

### Data Setup

Before running the notebooks, please upload the necessary data files to your Google Drive and ensure the paths in the CONFIG section of each notebook point to the correct locations. The default paths are:

- **Electricity Dataset**: `/content/drive/MyDrive/electricity.csv`
- **Bike Sharing Dataset**: `/content/drive/MyDrive/myproject/bike_raw.csv`

## Setup for Google Colab

This project is designed to be run in a **Google Colab environment with a GPU runtime enabled**.

### Steps:

1. Upload the notebooks and data files to your Google Drive
2. Open a notebook in Google Colab
3. Run the first code cell in the notebook (this cell contains commands to install the correct versions of all required packages directly within the Colab environment)

We provide two notebooks to demonstrate our framework.

### 1. Simplified Demonstration (Bike Sharing Dataset)

The notebook **`MDN-NHITS forecasting_bike.ipynb`** serves as a lightweight example of the N-HiTS + MDN framework. It runs on the free tier of Google Colab.

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

### 2. Full Replication (Electricity Dataset)

The notebook **`MDN-LGBM-NHITS_Electricity.ipynb`** contains the complete code to replicate the results for the Electricity dataset.

**Important Note on Runtime**: This is an archival notebook and is **computationally intensive**. Training all models may take **over 1.5 days** to complete, even on a high-performance Google Colab instance.

To run the experiments, you must execute the cells sequentially. Note that some blocks are commented out by default to allow for modular execution. For a full replication of the paper's results:

- You may need to uncomment and run **BLOCK 2: LGBM FORECASTING** to generate the LightGBM model predictions before running the subsequent evaluation cells for that model.

## Running Locally (Alternative to Colab)

If you prefer to run this project on your local machine, you can use the `requirements.txt` file to set up a Python environment.

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt