import functools
import inspect
import os
import sys

import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from classification_utils import create_run_pipelines, train_sklearn_modules
from generative_models.inn_base_functions import AnomalyINN

from config import *


def get_trained_xgboosts(name, HORIZON, column, generator, scaler, data, gs=True, filter=lambda data: data >= 1):
    # Default parameters


    # Parameter search
    if gs == "search":
            xgboost_module = GridSearchCV(XGBClassifier(n_jobs=-1), param_grid={
                "booster": ["gbtree", "gblinear", "dart"],
                "reg_lambda": [0, 0.1, 0.5, 1, 2, 4],
                "importance_type": ["gain", "weight", "cover", "total_gain", "total_cover"],
            }, n_jobs=-1)
    elif gs == "default":
        xgboost_module= XGBClassifier(n_jobs=-1)
    elif gs == "optimal_technical":
        if generator is None and scaler is None:
            xgboost_module = XGBClassifier(n_jobs=-1, **{'booster': 'gbtree', 'importance_type': 'gain', 'reg_lambda': 0})
        elif generator is None and scaler is not None:
            xgboost_module = XGBClassifier(n_jobs=-1, **{'booster': 'gbtree', 'importance_type': 'gain', 'reg_lambda': 0})
        elif "INN" in generator.name:
            xgboost_module = XGBClassifier(n_jobs=-1, **{'booster': 'gbtree', 'importance_type': 'gain', 'reg_lambda': 0.1})
        elif "VAE" in generator.name:
            xgboost_module = XGBClassifier(n_jobs=-1, **{'booster': 'gbtree', 'importance_type': 'gain', 'reg_lambda': 1})
    elif gs == "optimal_unusual":
        if generator is None and scaler is None:
            xgboost_module = XGBClassifier(n_jobs=-1, **{'booster': 'gbtree', 'importance_type': 'gain', 'reg_lambda': 0})
        elif generator is None and scaler is not None:
            xgboost_module = XGBClassifier(n_jobs=-1, **{'booster': 'gbtree', 'importance_type': 'gain', 'reg_lambda': 0})
        elif "INN" in generator.name:
            xgboost_module = XGBClassifier(n_jobs=-1, **{'booster': 'gbtree', 'importance_type': 'gain', 'reg_lambda': 1})
        elif "VAE" in generator.name:
            xgboost_module = XGBClassifier(n_jobs=-1, **{'booster': 'gblinear', 'importance_type': 'weight', 'reg_lambda': 0})
        # TRAIN classifiers
    return train_sklearn_modules(HORIZON, column, data, generator, xgboost_module, name, scaler)


if __name__ == "__main__":
    for column, path, start_date, date_col, HORIZON, freq, SCALING, test_path, name in DATASETS:
        inn = functools.partial(AnomalyINN, horizon=HORIZON, cond_features=COND_FEATURES, n_layers_cond=10)

        data = pd.read_csv(path, index_col=date_col, parse_dates=[date_col],
                           infer_datetime_format=True)
        from datetime import datetime
        custom_date_parser = lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M")
        test_data = pd.read_csv(test_path, index_col=date_col, parse_dates=[date_col],
                                infer_datetime_format=True, date_parser=custom_date_parser)

        create_run_pipelines(column, data, HORIZON, inn, test_data, "XGBoost" + name,get_sklearn_modules=get_trained_xgboosts)
        print("Finished main")
