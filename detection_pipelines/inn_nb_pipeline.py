import functools
import inspect
import os
import sys

import pandas as pd
from sklearn.naive_bayes import GaussianNB


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from classification_utils import train_sklearn_modules, create_run_pipelines
from generative_models.inn_base_functions import AnomalyINN

from config import *


def get_trained_nbs(name, HORIZON, column, inn_wrapper, scaler, data, gs=True, filter=lambda data: data >= 1):
    # Default parameters
    nb_module = GaussianNB()

    # TRAIN classifiers
    return train_sklearn_modules(HORIZON, column, data, inn_wrapper, nb_module, name, scaler)


if __name__ == "__main__":
    for column, path, start_date, date_col, HORIZON, freq, SCALING, test_path, name in DATASETS:
        inn = functools.partial(AnomalyINN, horizon=HORIZON, cond_features=COND_FEATURES, n_layers_cond=10)

        data = pd.read_csv(path, index_col=date_col, parse_dates=[date_col],
                           infer_datetime_format=True)
        from datetime import datetime
        custom_date_parser = lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M")
        test_data = pd.read_csv(test_path, index_col=date_col, parse_dates=[date_col],
                                infer_datetime_format=True, date_parser=custom_date_parser)

        create_run_pipelines(column, data, HORIZON, inn, test_data, "nb" + name,get_sklearn_modules=get_trained_nbs)
        print("Finished main")
