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

### Input data
The pipeline requires the following two input files in the project repository (both in 15 minutes resolution):
* in_train_ID200.csv

| Column name | Description                                                                                                                                                                           |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| time        | Date and time of each measurement (used as index), starting with 2011-01-01 00:15:00                                                                                                  |
| y           | Measured power values (in kW) without anomalies (potentially replaced with realistic values (e.g., using the [Copy Paste Imputation](https://github.com/KIT-IAI/CopyPasteImputation)) |

* out_train_ID200_{number of type 1 anomalies}\_{number of type 2 anomalies}\_{number of type 3 anomalies}\_{number of type 4 anomalies}_small.csv

| Column name | Description                                                                                                                                                                           |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| time        | Date and time of each measurement (used as index), starting with 2011-01-01 00:15:00                                                                                                  |
| y           | Measured power values (in kW) with inserted synthetic or labeled anomalies |
| anomalies   | Labels for inserted synthetic or labeled anomalies: 0 = no anomaly; 1 = anomaly of type 1; 2 = anomaly of type 2; etc.                                                                                                                                                                                      |


### Execution
To start the pipeline, you can either use one of the scripts defined in the folder scripts or start `run_classifiers.py` or `run_unsupervised_methods.py` directly. To start them directly, consider using the following command line arguments:

```
# Available arguments
--anomalies
    Number of anomalies (default=20)
--generator-methods
    The chosen generator: "cvae", "cinn" (default=["cinn", "cvae"])
--base
    Used classifier: "knn", "lr", "mlp", "nb", "rf", "svc", "xgboost" (default="lr")
    or used unsupervised method: "iForest", "LOF", "Envelope", "AE", "VAE" (default="VAE")
--hyperparams
    Hyperparameters used for classifiers: "search", "default", "optimal_technical", "optimal_unusual" (default="optimal_unusual")
--contaminations 
    Contamination values used for unsupervised methods (default=[0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
--anomaly_types
    Considered types of anomalies: "1", "2", "3", "4", "all" (default="all")
--anomaly_group
    Considered group of anomalies: "technical", "unusual" (default="technical")
```

####Exemplary commands

*Supervised anomaly detection*

Run MLP with optimal hyperparameters on cINN and cVAE latent space with anomalies from group of technical faults (5 of each anomaly type): 

`python run_classifiers.py --hyperparams optimal_technical --anomalies 5 --base mlp --generator-method cvae cinn --type technical`

*Unsupervised anomaly detection*

Run iForest with contamination of 0.95 on cINN and cVAE latent space with anomalies from group of unusual consumption (10 of each type):

`python run_unsupervised_methods.py --anomalies 10 --base iForest --generator-method cvae cinn --contaminations 0.95  --type unusual`


### Output

After running the command, the pipeline returns a folder called results/unusual or results/technical where the respective results are saved.


## Funding

This project is supported by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, by the Helmholtz Association under the Program “Energy System Design”, the Helmholtz Metadata Collaboration, and by the German Research Foundation (DFG) Research Training Group 2153 "Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation".


## License

This code is licensed under the [MIT License](LICENSE).
