import functools
import inspect
import os
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from generative_models.inn_base_functions import AnomalyINN
from classification_utils import train_sklearn_modules, create_run_pipelines

from config import *


def get_trained_lrs(name, HORIZON, column, generator, scaler, data, gs=True, filter=lambda data: data >= 1):
    # Parameter search
    if gs == "search":
        lr_module = GridSearchCV(LogisticRegression(n_jobs=-1), param_grid={
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        }, n_jobs=-1)
    elif gs == "default":
        lr_module = LogisticRegression(n_jobs=-1)
    elif gs == "optimal_technical":
        if generator is None and scaler is None:
            lr_module = LogisticRegression(n_jobs=-1, **{'C': 1, 'penalty': 'l2', 'solver': 'sag'})
        elif generator is None and scaler is not None:
            lr_module = LogisticRegression(n_jobs=-1, **{'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'})
        elif "INN" in generator.name:
            lr_module = LogisticRegression(n_jobs=-1, **{'C': 10, 'penalty': 'none', 'solver': 'sag'})
        elif "VAE" in generator.name:
            lr_module = LogisticRegression(n_jobs=-1, **{'C': 1, 'penalty': 'none', 'solver': 'newton-cg'})
            # TRAIN LR
    elif gs == "optimal_unusual":
        if generator is None and scaler is None:
            lr_module = LogisticRegression(n_jobs=-1, **{'C': 10, 'penalty': 'l2', 'solver': 'lbfgs'})
        elif generator is None and scaler is not None:
            lr_module = LogisticRegression(n_jobs=-1, **{'C': 0.01, 'penalty': 'l1', 'solver': 'liblinear'})
        elif "INN" in generator.name:
            lr_module = LogisticRegression(n_jobs=-1, **{'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'})
        elif "VAE" in generator.name:
            lr_module = LogisticRegression(n_jobs=-1, **{'C': 100, 'penalty': 'l1', 'solver': 'liblinear'})
    return train_sklearn_modules(HORIZON, column, data, generator, lr_module, name, scaler)



if __name__ == "__main__":
    for column, path, start_date, date_col, HORIZON, freq, SCALING, test_path, name in DATASETS:
        inn = functools.partial(AnomalyINN, horizon=HORIZON, cond_features=COND_FEATURES, n_layers_cond=10)

        data = pd.read_csv(path, index_col=date_col, parse_dates=[date_col],
                           infer_datetime_format=True)
        from datetime import datetime
        custom_date_parser = lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M")
        test_data = pd.read_csv(test_path, index_col=date_col, parse_dates=[date_col],
                                infer_datetime_format=True, date_parser=custom_date_parser)

        create_run_pipelines(column, data, HORIZON, inn, test_data, "lr" + name,get_sklearn_modules=get_trained_lrs)
        print("Finished main")
