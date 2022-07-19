import functools
import inspect
import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from generative_models.inn_base_functions import AnomalyINN
from classification_utils import create_run_pipelines, train_sklearn_modules

from config import *


def get_trained_rfs(name, HORIZON, column, generator, scaler, data, gs=True, filter=lambda data: data >= 1):
    # Default parameters
    # rf_module_latent = RandomForestClassifier(n_jobs=-1)
    # rf_module_data_unscaled = RandomForestClassifier(n_jobs=-1)
    # rf_module_data_scaled = RandomForestClassifier(n_jobs=-1)

    # Parameter search
    if gs == "search":
        rf_module = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid={
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2"],
        }, n_jobs=-1)
    elif gs == "default":
        rf_module = RandomForestClassifier(n_jobs=-1)
    elif gs == "optimal_technical":
        if generator is None and scaler is None:
            rf_module = RandomForestClassifier(n_jobs=-1, **{'criterion': 'gini', 'max_features': 'sqrt'})
        elif generator is None and scaler is not None:
            rf_module = RandomForestClassifier(n_jobs=-1, **{'criterion': 'gini', 'max_features': 'sqrt'})
        elif "INN" in generator.name:
            rf_module = RandomForestClassifier(n_jobs=-1, **{'criterion': 'gini', 'max_features': 'sqrt'})
        elif "VAE" in generator.name:
            rf_module = RandomForestClassifier(n_jobs=-1, **{'criterion': 'gini', 'max_features': 'sqrt'})
    elif gs == "optimal_unusual":
        if generator is None and scaler is None:
            rf_module = RandomForestClassifier(n_jobs=-1, **{'criterion': 'entropy', 'max_features': 'log2'})
        elif generator is None and scaler is not None:
            rf_module = RandomForestClassifier(n_jobs=-1, **{'criterion': 'entropy', 'max_features': 'log2'})
        elif "INN" in generator.name:
            rf_module = RandomForestClassifier(n_jobs=-1, **{'criterion': 'gini', 'max_features': 'sqrt'})
        elif "VAE" in generator.name:
            rf_module = RandomForestClassifier(n_jobs=-1, **{'criterion': 'gini', 'max_features': 'sqrt'})

    return train_sklearn_modules(HORIZON, column, data, generator, rf_module, name, scaler)


if __name__ == "__main__":
    for column, path, start_date, date_col, HORIZON, freq, SCALING, test_path, name in DATASETS:
        inn = functools.partial(AnomalyINN, horizon=HORIZON, cond_features=COND_FEATURES, n_layers_cond=10)

        data = pd.read_csv(path, index_col=date_col, parse_dates=[date_col],
                           infer_datetime_format=True)
        from datetime import datetime
        custom_date_parser = lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M")
        test_data = pd.read_csv(test_path, index_col=date_col, parse_dates=[date_col],
                                infer_datetime_format=True, date_parser=custom_date_parser)

        create_run_pipelines(column, data, HORIZON, inn, test_data, "rf" + name,get_sklearn_modules=get_trained_rfs)
        print("Finished main")
