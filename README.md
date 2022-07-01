# Enhancing Anomaly Detection Methods for Energy Time Series Using Latent Space Data Representations

This repository contains the Python implementation of the approach to generally enhance anomaly detection methods for energy time series by taking advantage of their latent space representation. The approach is presented in the following paper:
>M. Turowski, B. Heidrich, K. Phipps, K. Schmieder, O. Neumann, R. Mikut, and V. Hagenmeyer, 2022, "Enhancing Anomaly Detection Methods for Energy Time Series Using Latent Space Data Representations," in The Thirteenth ACM International Conference on Future Energy Systems (e-Energy ’22). ACM, pp. 208–227. doi: [10.1145/3538637.3538851](https://doi.org/10.1145/3538637.3538851).


## Installation

Before anomaly detection methods can be enhanced using a [pyWATTS](https://github.com/KIT-IAI/pyWATTS) pipeline, you need to prepare a Python environment and download energy time series (if you have no data available).

### 1. Setup Python Environment

Set up a virtual environment using e.g. venv (`python -m venv venv`) or Anaconda (`conda create -n env_name`). Afterwards, install the dependencies with `pip install -r requirements.txt`. 

### 2. Download Data (optional)

If you do not have any data available, you can download exemplary data by executing `python download.py`. This script downloads and unpacks the [ElectricityLoadDiagrams20112014 Data Set](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) as CSV file.

### 3. Create Data with Synthetic Anomalies (optional)

Before applying the proposed method, you may want to create data with synthetic anomalies. You may generate anomalies of two groups: technical faults and unusual consumption.
- For generating synthetic technical faults, take a look at the repository of the article ["Modeling and Generating Synthetic Anomalies for Energy and Power Time Series"](https://github.com/KIT-IAI/GeneratingSyntheticEnergyPowerAnomalies)
- For generating unusual consumption, use the `generation_pipeline.py` in the folder unusual_behaviour_ts_generation.


## Enhancing Anomaly Detection Methods

Finally, you can enhance arbitrary anomaly detection methods for energy time series.

### Input

To start the pipeline, you can either use one of the scripts defined in the folder scripts or start `run_classifiers.py` or `run_unsupervised_methods.py` directly. To start them directly, consider to take a look at the needed command line arguments.

### Output

After running the command, the pipeline returns a folder called results/unusual or results/technical where the results are saved.


## Funding

This project is supported by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, by the Helmholtz Association under the Program “Energy System Design”, the Helmholtz Metadata Collaboration, and by the German Research Foundation (DFG) Research Training Group 2153 "Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation".


## License

This code is licensed under the [MIT License](LICENSE).
