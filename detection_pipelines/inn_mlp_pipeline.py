import functools
import inspect
import os
import sys

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from classification_utils import train_sklearn_modules, create_run_pipelines
from generative_models.inn_base_functions import AnomalyINN

from config import *


def get_trained_mlps(name, HORIZON, column, generator, scaler, data, gs=True, filter=lambda data: data >= 1):
    # Default parameters
    # mlp_module_latent = MLPClassifier()
    # mlp_module_data_unscaled = MLPClassifier()
    # mlp_module_data_scaled = MLPClassifier()

    # Parameter search
    if gs == "search":
        mlp_module = GridSearchCV(MLPClassifier(), param_grid={
            "hidden_layer_sizes": [(25,), (50,), (75,), (100,), (125,), (150,), (25, 25), (50, 50), (75, 75), (100, 100),
                                (125, 125), (150, 150), (25, 25, 25), (50, 50, 50), (75, 75, 75), (100, 100, 100),
                                (125, 125, 125), (150, 150, 150)],
            "activation": ["logistic", "tanh", "relu"],
            "alpha": [0.00001, 0.0001, 0.001],
            "batch_size": [10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 200],
        }, n_jobs=-1)
    elif gs == "default":
        mlp_module = MLPClassifier()
    elif gs == "optimal_technical":
        if generator is None and scaler is None:
            mlp_module = MLPClassifier(
                **{'activation': 'relu', 'alpha': 0.001, 'batch_size': 32, 'hidden_layer_sizes': (125,)})
        elif generator is None and scaler is not None:
            mlp_module = MLPClassifier(**{'activation': 'relu', 'alpha': 0.0001, 'batch_size': 14, 'hidden_layer_sizes': (125, 125)})
        elif "INN" in generator.name:
            mlp_module = MLPClassifier(**{'activation': 'relu', 'alpha': 0.001, 'batch_size': 15, 'hidden_layer_sizes': (100,)})
        elif "VAE" in generator.name:
            mlp_module = MLPClassifier(**{'activation': 'relu', 'alpha': 0.0001, 'batch_size': 14, 'hidden_layer_sizes': (75, 75)})
    elif gs == "optimal_unusual":
        if generator is None and scaler is None:
            mlp_module = MLPClassifier(
                **{'activation': 'relu', 'alpha': 1e-05, 'batch_size': 64, 'hidden_layer_sizes': (100, 100, 100)})
        elif generator is None and scaler is not None:
            mlp_module = MLPClassifier(**{'activation': 'relu', 'alpha': 0.0001, 'batch_size': 12, 'hidden_layer_sizes': (50, 50)})
        elif "INN" in generator.name:
            mlp_module = MLPClassifier(**{'activation': 'logistic', 'alpha': 0.001, 'batch_size': 11, 'hidden_layer_sizes': (150,)})
        elif "VAE" in generator.name:
            mlp_module = MLPClassifier(**{'activation': 'relu', 'alpha': 1e-05, 'batch_size': 11, 'hidden_layer_sizes': (75, 75)})



    return train_sklearn_modules(HORIZON, column, data, generator, mlp_module, name, scaler)


if __name__ == "__main__":
    for column, path, start_date, date_col, HORIZON, freq, SCALING, test_path, name in DATASETS:
        inn = functools.partial(AnomalyINN, horizon=HORIZON, cond_features=COND_FEATURES, n_layers_cond=10)

        data = pd.read_csv(path, index_col=date_col, parse_dates=[date_col], infer_datetime_format=True)
        from datetime import datetime
        custom_date_parser = lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M")
        test_data = pd.read_csv(test_path, index_col=date_col, parse_dates=[date_col],
                                infer_datetime_format=True, date_parser=custom_date_parser)

        create_run_pipelines(column, data, HORIZON, inn, test_data, "mlp" + name,get_sklearn_modules=get_trained_mlps)
        print("Finished main")
