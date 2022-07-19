import functools
import inspect
import os
import sys

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from generative_models.inn_base_functions import AnomalyINN
from classification_utils import evaluate_classifiers, get_trained_inn_wrappers,  train_sklearn_modules, create_run_pipelines
from config import *

def get_trained_knns(name, HORIZON, column, generator, scaler, data, gs=True, filter=lambda data: data >= 1):
    # Default parameters
    # knn_module_latent = KNeighborsClassifier(n_jobs=-1)
    # knn_module_data_unscaled = KNeighborsClassifier(n_jobs=-1)
    # knn_module_data_scaled = KNeighborsClassifier(n_jobs=-1)

    # Parameter search
    if gs == "search":
        knn_module = GridSearchCV(KNeighborsClassifier(n_jobs=-1), param_grid={
            "n_neighbors": [1, 3, 5, 7, 10],
            "weights": ["uniform", "distance"],
            "p": [1, 2, 3],
        }, n_jobs=-1)
    elif gs == "default":
        knn_module = KNeighborsClassifier(n_jobs=-1)
    elif gs == "optimal_technical":
        if generator is None and scaler is None:
            knn_module = KNeighborsClassifier(n_jobs=-1, **{'n_neighbors': 1, 'p': 2, 'weights': 'uniform'})
        elif generator is None and scaler is not None:
            knn_module = KNeighborsClassifier(n_jobs=-1, **{'n_neighbors': 1, 'p': 2, 'weights': 'uniform'})
        elif "INN" in generator.name:
            knn_module = KNeighborsClassifier(n_jobs=-1, **{'n_neighbors': 1, 'p': 2, 'weights': 'uniform'})
        elif "VAE" in generator.name:
            knn_module = KNeighborsClassifier(n_jobs=-1, **{'n_neighbors': 1, 'p': 2, 'weights': 'uniform'})
            # TRAIN LR
    elif gs == "optimal_unusual":
        if generator is None and scaler is None:
            knn_module = KNeighborsClassifier(n_jobs=-1, **{'n_neighbors': 5, 'p': 3, 'weights': 'uniform'})
        elif generator is None and scaler is not None:
            knn_module = KNeighborsClassifier(n_jobs=-1, **{'n_neighbors': 5, 'p': 3, 'weights': 'uniform'})
        elif "INN" in generator.name:
            knn_module = KNeighborsClassifier(n_jobs=-1, **{'n_neighbors': 1, 'p': 1, 'weights': 'uniform'})
        elif "VAE" in generator.name:
            knn_module = KNeighborsClassifier(n_jobs=-1, **{'n_neighbors': 10, 'p': 2, 'weights': 'uniform'})

    return train_sklearn_modules(HORIZON, column, data, generator, knn_module, name, scaler, filter=filter)

if __name__ == "__main__":
    for column, path, start_date, date_col, HORIZON, freq, SCALING, test_path, name in DATASETS:
        inn = functools.partial(AnomalyINN, horizon=HORIZON, cond_features=COND_FEATURES, n_layers_cond=10)

        data = pd.read_csv(path, index_col=date_col, parse_dates=[date_col],
                           infer_datetime_format=True)
        from datetime import datetime
        custom_date_parser = lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M")
        test_data = pd.read_csv(test_path, index_col=date_col, parse_dates=[date_col],
                                infer_datetime_format=True, date_parser=custom_date_parser)

        create_run_pipelines(column, data, HORIZON, inn, test_data, "knn" + name,get_sklearn_modules=get_trained_knns)
        print("Finished main")
