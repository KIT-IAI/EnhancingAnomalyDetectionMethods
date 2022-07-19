import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
from pywatts.core.pipeline import Pipeline
from pywatts.modules import FunctionModule
import argparse

params = {
    "technical": {
        "type1": {
            "length_params": {
                'distribution': 'uniform',
                'min': 3,
                'max': 96
            }
        },
        "type2": {
            "length_params": {
                'distribution': 'uniform',
                'min': 3,
                'max': 96
            },
        },
        "type3": {
            "anomaly_params": {
                'is_extreme': True,
                'range_r': (0.61, 1.62),
            }
        },
        "type4": {
            "anomaly_params": {
                'range_r': (1.15, 8.1),
            }
        },
    },
    "unusual": {
        "type1": {
            "length_params": {
                'distribution': 'uniform',
                'min': 48,
                'max': 96 + 48
            }
        },
        "type2": {
            "length_params": {
                'distribution': 'uniform',
                'min': 48,
                'max': 96 + 48
            },
        },
        "type3": {
            "length_params": {
                'distribution': 'uniform',
                'min': 48,
                'max': 96 + 48
            }
        },
        "type4": {
            "length_params": {
                'distribution': 'uniform',
                'min': 48,
                'max': 96 + 48
            }
        },
    }
}


def create_pipeline(hparams):
    """
    Generate anomalies of types (1) to (4) for a given power time series
    """
    if hparams.anomaly_group == "unusual":
        from pywatts.modules.generation.unusual_behaviour_generation import UnusualBehaviour as AnomalyModule
    else:
        from pywatts.modules.generation.power_anomaly_generation_module import PowerAnomalyGeneration as AnomalyModule

    pipeline = Pipeline(path=os.path.join('run'))
    seed = 42
    # Type 1: Negative power spike potentially followed by zero values and finally a positive power spike
    anomaly_type1 = AnomalyModule(
        'y_hat', anomaly='type1', count=hparams.type1, label=1, seed=seed + 1,
        **params[hparams.anomaly_group]["type1"]
    )(x=pipeline['y'], labels=None)

    # Type 2: Drop to potentially zero followed by a positive power spike
    anomaly_type2 = AnomalyModule(
        'y_hat', anomaly='type2', count=hparams.type2, label=2, seed=seed + 2,
        **params[hparams.anomaly_group]["type2"]

    )(x=anomaly_type1['y_hat'], labels=anomaly_type1['labels'])

    # Type 3: Sudden negative power spike
    anomaly_type3 = AnomalyModule(
        'y_hat', anomaly='type3', count=hparams.type3, label=3, seed=seed + 4,
        **params[hparams.anomaly_group]["type3"]

    )(x=anomaly_type2['y_hat'], labels=anomaly_type2['labels'])

    # Type 4: Sudden positive power spike
    anomaly_type4 = AnomalyModule(
        'y_hat', anomaly='type4', count=hparams.type4, label=4, seed=seed + 5,
        **params[hparams.anomaly_group]["type4"]

    )(x=anomaly_type3['y_hat'], labels=anomaly_type3['labels'])

    FunctionModule(lambda x: x, name='y')(x=pipeline['y'])
    FunctionModule(lambda x: x, name='anomalies')(x=anomaly_type4['labels'])
    FunctionModule(lambda x: x, name='y_hat')(x=anomaly_type4['y_hat'])

    return pipeline


def str2intorfloat(v):
    """ String to int or float formatter for argparse. """
    try:
        return int(v)
    except:
        return float(v)


def parse_hparams(args=None):
    """ Parse command line arguments and return. """
    # prepare argument parser
    parser = argparse.ArgumentParser(
        description='Anomaly generation pipeline for energy and power time series.'
    )
    # csv path file
    parser.add_argument('--csv_path', type=str, help='Path to the data CSV file.', default="../data/in_train_ID200.csv")
    # data_index
    parser.add_argument('--column', type=str, help='Name of the target column.', default="y")
    # time_index
    parser.add_argument('--time', type=str, default='time', help='Name of the time index.')

    # anomaly params: Type 1
    parser.add_argument('--type1', type=str2intorfloat, default=None,
                        help='Percentage or absolute number of type 1 anomalies.')
    # anomaly params: Type 2
    parser.add_argument('--type2', type=str2intorfloat, default=None,
                        help='Percentage or absolute number of type 2 anomalies.')
    # anomaly params: Type 3
    parser.add_argument('--type3', type=str2intorfloat, default=None,
                        help='Percentage or absolute number of type 3 anomalies.')
    # anomaly params: Type 4
    parser.add_argument('--type4', type=str2intorfloat, nargs='?', const=True, default=None,
                        help='Percentage or absolute number of type 4 anomalies.')

    parser.add_argument('--anomaly_group', choices=["technical", "unusual"], default="unusual",
                        help='Decide if the anomalies are technical or unusual behaviour.')
    # convert argument strings
    parsed_hparams = parser.parse_args(args=args)

    return parsed_hparams


def load_data(hparams):
    """ Load the CSV file specified by hparams dict. """
    dataset = pd.read_csv(
        hparams.csv_path, index_col=hparams.time, parse_dates=True
    )

    rename_dict = {}
    rename_dict[hparams.column] = 'y'
    dataset.rename(columns=rename_dict, inplace=True)

    return dataset


def run_pipeline(hparams):
    """ Run complete power or anomaly generation pipeline (including data loading/saving). """
    dataset = load_data(hparams)
    pipeline = create_pipeline(hparams)
    result, _ = pipeline.train(dataset)
    result = pd.DataFrame(result, index=result["y"]["time"])
    del result["y"]
    result["y"] = result["y_hat"]
    del result["y_hat"]
    return result


if __name__ == '__main__':
    hparams = parse_hparams()
    if hparams.type1 is None and hparams.type2 is None and hparams.type3 is None and hparams.type4 is None:
        num_anomalies = [5, 10, 20, 25, 30, 40, 50, 100] if hparams.anomaly_group == "technical" else [10, 20, 30, 40, 50]
        for n in num_anomalies:
            hparams.type1 = n
            hparams.type2 = n
            hparams.type3 = n
            hparams.type4 = n
            result = run_pipeline(hparams)
            result.to_csv(
                f'../data/out_train_ID200_{hparams.type1}_{hparams.type2}_{hparams.type3}_{hparams.type4}_{hparams.anomaly_group}.csv',
                index_label="time")

    else:
        result = run_pipeline(hparams)
        result.to_csv(
            f'../data/out_train_ID200_{hparams.type1}_{hparams.type2}_{hparams.type3}_{hparams.type4}_{hparams.anomaly_group}.csv',
            index_label="time")
