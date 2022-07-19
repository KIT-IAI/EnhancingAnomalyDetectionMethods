import functools
import inspect
import os
import sys

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from generative_models.inn_base_functions import AnomalyINN
from classification_utils import create_run_pipelines, train_sklearn_modules

from config import *


def get_trained_svcs(name, HORIZON, column, generator, scaler, data, gs=True, filter=lambda data: data >= 1):
    # Default parameters
    # svc_module_latent = SVC()
    # svc_module_data_unscaled = SVC()
    # svc_module_data_scaled = SVC()
    if gs == "search":
        svc_module = GridSearchCV(SVC(), param_grid={
            "C": [0.01, 0.1, 1, 10, 100],
            "kernel": ["linear", "sigmoid", "rbf"],
            "gamma": ["scale", "auto"],
        }, n_jobs=-1)
    elif gs == "default":
        svc_module = SVC()
    elif gs == "optimal_technical":
        if generator is None and scaler is None:
            svc_module = SVC(**{'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'})
        elif generator is None and scaler is not None:
            svc_module = SVC(**{'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'})
        elif "INN" in generator.name:
            svc_module = SVC(**{'C': 100, 'gamma': 'scale', 'kernel': 'rbf'})
        elif "VAE" in generator.name:
            svc_module = SVC(**{'C': 100, 'gamma': 'scale', 'kernel': 'rbf'})
    elif gs == "optimal_unusual":
        if generator is None and scaler is None:
            svc_module = SVC(**{'C': 10, 'gamma': 'scale', 'kernel': 'rbf'})
        elif generator is None and scaler is not None:
            svc_module = SVC(**{'C': 10, 'gamma': 'scale', 'kernel': 'rbf'})
        elif "INN" in generator.name:
            svc_module = SVC(**{'C': 10, 'gamma': 'auto', 'kernel': 'rbf'})
        elif "VAE" in generator.name:
            svc_module = SVC(**{'C': 1, 'gamma': 'scale', 'kernel': 'rbf'})

    # TRAIN SVC
    return train_sklearn_modules(HORIZON, column, data, generator, svc_module, name, scaler)


if __name__ == "__main__":
    for column, path, start_date, date_col, HORIZON, freq, SCALING, test_path, name in DATASETS:
        inn = functools.partial(AnomalyINN, horizon=HORIZON, cond_features=COND_FEATURES, n_layers_cond=10)

        data = pd.read_csv(path, index_col=date_col, parse_dates=[date_col],
                           infer_datetime_format=True)
        test_data = pd.read_csv(test_path, index_col=date_col, parse_dates=[date_col],
                                infer_datetime_format=True)
        from datetime import datetime
        custom_date_parser = lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M")
        test_data = pd.read_csv(test_path, index_col=date_col, parse_dates=[date_col],
                                infer_datetime_format=True, date_parser=custom_date_parser)

        create_run_pipelines(column, data, HORIZON, inn, test_data, "SVC" + name,get_sklearn_modules=get_trained_svcs)
        print("Finished main")
