import inspect
import os
import sys

import pandas as pd
from argparse import ArgumentParser

#from cVAE import cVAEModule

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from classification_utils import create_run_pipelines

parser = ArgumentParser()
parser.add_argument("--anomalies", help="number of anomalies", type=int, default=20)
parser.add_argument("--generator-methods", nargs="*", help="The chosen generator", choices=["cvae", "cinn", "cae"], default=["cinn", "cvae"])
parser.add_argument("--hyperparams", help="Hyperparameter option", choices=['search', 'default', "optimal_technical", "optimal_unusual"], default='optimal_unusual')
parser.add_argument("--base", help="Base Classifier", choices=['knn', 'lr', 'mlp', 'nb', 'rf', 'svc', 'xgboost'], default="lr")
parser.add_argument("--classes", help="AnomalyClass", choices=['1', '2', '3', '4', 'all'], default="all")
parser.add_argument("--type", help="AnomalyType", choices=['technical', 'unusual'], default="technical")



if __name__ == "__main__":
    args = parser.parse_args()

    column = "y"
    path = "../data/in_train_ID200.csv"
    start_date = "2011-01-01 00:15:00"
    date_col = "time"
    HORIZON = 96
    freq = "15min"
    SCALING = True
    if args.classes == "all":
        test_path = f"../data/out_train_ID200_{args.anomalies}_{args.anomalies}_{args.anomalies}_{args.anomalies}_small.csv"
    else:
        test_path = f'../data/out_train_ID200_{args.anomalies if args.classes == "1" else "0"}_{args.anomalies if args.classes == "2" else "0"}_{args.anomalies if args.classes == "3" else "0"}_{args.anomalies if args.classes == "4" else "0"}_small.csv'

    name = f"{args.type}/num_anomalies_{args.anomalies}/cl_{args.classes}/{args.base}/{args.hyperparams}/"


    data = pd.read_csv(path, index_col=date_col, parse_dates=[date_col],
                        infer_datetime_format=True)
                        
    from datetime import datetime
    custom_date_parser = lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M")
    if args.type == "technical":
        test_data = pd.read_csv(test_path, index_col=date_col, parse_dates=[date_col],
                            infer_datetime_format=True, date_parser=custom_date_parser)

    else:
        test_data = pd.read_csv(test_path[:-9] + "unusual_behaviour.csv", index_col=date_col, parse_dates=[date_col],
                            infer_datetime_format=True)

    create_run_pipelines(column, data, HORIZON, args.generator_methods, test_data, name, get_sklearn_modules=globals()[f"get_trained_{args.base}s"], gs=args.hyperparams)
    print("Finished main")
