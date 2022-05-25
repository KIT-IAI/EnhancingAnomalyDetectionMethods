# Enhancing Anomaly Detection Methods for Energy Time Series Using Latent Space Data Representations

This repository contains the Python implementation of the approach to generally enhance anomaly detection methods for energy time series by taking advantage of their latent space representation. The approach is presented in the following paper:
>M. Turowski, B. Heidrich, K. Phipps, K. Schmieder, O. Neumann, R. Mikut, and V. Hagenmeyer, 2022, "Enhancing Anomaly Detection Methods for Energy Time Series Using Latent Space Data Representations," in The Thirteenth ACM International Conference on Future Energy Systems (e-Energy '22), doi: [10.1145/3538637.3538851](https://doi.org/10.1145/3538637.3538851).


## Installation

Before anomaly detection methods can be enhanced using a [pyWATTS](https://github.com/KIT-IAI/pyWATTS) pipeline, you need to prepare a Python environment and download energy time series (if you have no data available).

### 1. Setup Python Environment

Set up a virtual environment using e.g. venv (`python -m venv venv`) or Anaconda (`conda create -n env_name`). Afterwards, install the dependencies with `pip install -r requirements.txt`. 

### 2. Download Data (optional)

If you do not have any data available, you can download exemplary data by executing `python download.py`. This script downloads and unpacks the [ElectricityLoadDiagrams20112014 Data Set](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) as CSV file.

## Enhancing Anomaly Detection Methods

Finally, you can enhance anomaly detection methods for energy time series.

### Input

To ...

### Output

After running the command, the pipeline returns 


## Funding

This project is supported by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, by the Helmholtz Association under the Program “Energy System Design”,  the Helmholtz Metadata Collaboration, and by the German Research Foundation (DFG) Research Training Group 2153 "Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation".


## License

This code is licensed under the [MIT License](LICENSE).
