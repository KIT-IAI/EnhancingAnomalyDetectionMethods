import inspect
import os
import sys

import pandas as pd
from argparse import ArgumentParser

# from cVAE import cVAEModule

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from inn_knn_pipeline import get_trained_knns
from inn_lr_pipeline import get_trained_lrs
from inn_mlp_pipeline import get_trained_mlps
from inn_svc_pipeline import get_trained_svcs
from inn_nb_pipeline import get_trained_nbs
from inn_rf_pipeline import get_trained_rfs
from inn_xgboost_pipeline import get_trained_xgboosts

from classification_utils import create_run_pipelines

parser = ArgumentParser()
parser.add_argument("--anomalies", help="Number of anomalies", type=int, default=20)
parser.add_argument("--generator-methods", nargs="*", help="The chosen generator", choices=["cvae", "cinn"],
                    default=["cinn", "cvae"])
parser.add_argument("--hyperparams", help="Hyperparameters used for classifiers",
                    choices=['search', 'default', "optimal_technical", "optimal_unusual"], default='optimal_technical')
parser.add_argument("--base", help="Used classifier", choices=['knn', 'lr', 'mlp', 'nb', 'rf', 'svc', 'xgboost'],
                    default="lr")
parser.add_argument("--anomaly_types", help="Considered types of anomalies", choices=['1', '2', '3', '4', 'all'],
                    default="all")
parser.add_argument("--anomaly_group", help="Considered group of anomalies", choices=['technical', 'unusual'],
                    default="technical")

if __name__ == "__main__":
    args = parser.parse_args()

    column = "y"
    path = "../data/in_train_ID200.csv"
    start_date = "2011-01-01 00:15:00"
    date_col = "time"
    HORIZON = 96
    freq = "15min"
    SCALING = True
    if args.anomaly_types == "all":
        test_path = f"../data/out_train_ID200_{args.anomalies}_{args.anomalies}_{args.anomalies}_{args.anomalies}_technical.csv"
    else:
        test_path = f'../data/out_train_ID200_{args.anomalies if args.anomaly_types == "1" else "0"}_{args.anomalies if args.anomaly_types == "2" else "0"}_{args.anomalies if args.anomaly_types == "3" else "0"}_{args.anomalies if args.anomaly_types == "4" else "0"}_technical.csv'

    name = f"{args.anomaly_group}/num_anomalies_{args.anomalies}/cl_{args.anomaly_types}/{args.base}/{args.hyperparams}/"

    data = pd.read_csv(path, index_col=date_col, parse_dates=[date_col], infer_datetime_format=True)

    from datetime import datetime

    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    if args.anomaly_group == "technical":
        test_data = pd.read_csv(test_path, index_col=date_col, parse_dates=[date_col],
                                infer_datetime_format=True, date_parser=custom_date_parser)

    else:
        test_data = pd.read_csv(test_path[:-13] + "unusual.csv", index_col=date_col, parse_dates=[date_col],
                                infer_datetime_format=True)

    create_run_pipelines(column, data, HORIZON, args.generator_methods, test_data, name,
                         get_sklearn_modules=globals()[f"get_trained_{args.base}s"], gs=args.hyperparams)
    print("Finished main")
