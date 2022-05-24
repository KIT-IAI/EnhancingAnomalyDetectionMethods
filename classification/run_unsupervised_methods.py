import inspect
import os
import sys

import pandas as pd
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline


current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from modules.ae_anomaly_detector import AutoencoderDetection
from classification.classification_utils import train_sklearn_modules, get_generator, get_preprocessing_pipeline
from classification.classification_utils import get_trained_inn_wrappers, evaluate_classifiers
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from argparse import ArgumentParser

from sklearn.covariance import EllipticEnvelope

DATASETS = [
    ("y", "../data/in_train_ID200.csv", "2011-01-01 00:15:00", "time", 24 * 4, "15min", True,
     "../data/out_train_ID200_50_50_50_50_small_full.csv"),
]

COND_FEATURES = 4
TRAINING_LENGTH = 10000

parser = ArgumentParser()
parser.add_argument("--anomalies", help="number of anomalies", type=int, default=20)
parser.add_argument("--generator-methods", nargs="*", help="The chosen generator", choices=["cvae", "cinn", "cae"], default=["cinn"])#, "cvae", "cae"])
parser.add_argument("--contaminations", help="Contaminations", nargs="*", type=float, default=[0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
parser.add_argument("--base", help="Base Classifier", choices=['iForest', 'LOF', 'Envelope', "AE", "VAE"], default="VAE")
parser.add_argument("--classes", help="AnomalyClass", choices=['1', '2', '3', '4', 'all'], default="all")
parser.add_argument("--type", help="AnomalyType", choices=['technical', 'unusual'], default="technical")



def create_run_pipelines(column, generator_methods, method, HORIZON, test_data, contamination):
    name = f"{args.type}/num_anomalies_{args.anomalies}/cl_{args.classes}/{args.base}/con_{contamination}"
    for i in range(2):
        gen_scaler_list = []
        for supervised in [False]:
            for gen in generator_methods:
                generator = get_generator(gen, supervised=supervised, contamination=contamination)
                trained_generator, scaler = get_trained_inn_wrappers(HORIZON, "y", generator, test_data, name=name)
                sk_module = method(f"{name}/sup_{supervised}_{gen}", HORIZON, "y", generator,
                                   scaler, test_data, contamination=contamination)
                gen_scaler_list.append((f"{name}/{gen}_sup:{supervised}", scaler, trained_generator, sk_module))

            # Each scaler has the same weights, thus we only use the last. Yes this is ugly...
            sk_module = method(f"{name}/sup_{supervised}_scaled", HORIZON, "y", None, scaler,
                               test_data, contamination=contamination)
            gen_scaler_list.append((f"{name}/scaled_sup:{supervised}", scaler, None, sk_module))
            sk_module = method(f"{name}/sup_{supervised}_unscaled", HORIZON, "y", None, None,
                               test_data, contamination=contamination)
            gen_scaler_list.append((f"{name}/unscaled_sup:{supervised}", None, None, sk_module))

        # Do not wonder about side-effects when using IsolationTree -> After each instatiation it is some bit different...

        # EVALUATE
        evaluate_classifiers(name, HORIZON, column, gen_scaler_list, test_data[:],  supervised=False)



def get_trained_iForests(name, HORIZON, column, inn_wrapper, scaler, data, contamination):
    sk_module = IsolationForest(random_state=42, contamination=1-contamination)
    return train_sklearn_modules(HORIZON, column, data, inn_wrapper, sk_module, name, scaler, supervised=False)

def get_trained_LOFs(name, HORIZON, column, inn_wrapper, scaler, data, contamination):
    sk_module = LocalOutlierFactor(novelty=True, contamination=1-contamination)
    return train_sklearn_modules(HORIZON, column, data, inn_wrapper, sk_module, name, scaler, supervised=False)

def get_trained_Envelopes(name, HORIZON, column, inn_wrapper, scaler, data, contamination):
    sk_module = EllipticEnvelope(contamination=1-contamination)
    return train_sklearn_modules(HORIZON, column, data, inn_wrapper, sk_module, name, scaler, supervised=False)


def get_trained_VAEs(name, HORIZON, column, generator, scaler, data, contamination):
    ae_detector = AutoencoderDetection(threshold=contamination, method="vae")
    return train_ae_based_detection(HORIZON, ae_detector, column, data, generator, name, scaler)

def get_trained_AEs(name, HORIZON, column, generator, scaler, data, contamination):
    ae_detector = AutoencoderDetection(threshold=contamination)
    return train_ae_based_detection(HORIZON, ae_detector, column, data, generator, name, scaler)


def train_ae_based_detection(HORIZON, ae_detector, column, data, generator, name, scaler):
    pipeline2 = Pipeline("../results/" + name + "train")
    preprocessing_pipeline2 = get_preprocessing_pipeline(HORIZON, scaler, name="results/preprocessing_" + name,
                                                         scale_computation_mode=ComputationMode.Transform)
    preprocessing_pipeline2 = preprocessing_pipeline2(input=pipeline2[column])
    if generator is not None:
        latent = generator(input_data=preprocessing_pipeline2["target_scaled"],
                           cal_input=preprocessing_pipeline2["calendar"],
                           stats_input=preprocessing_pipeline2["stats"],
                           computation_mode=ComputationMode.Transform)
        ae_detector(x=latent, computation_mode=ComputationMode.Train)
    elif scaler is not None:
        ae_detector(x=preprocessing_pipeline2["target_scaled"], computation_mode=ComputationMode.Train)
    else:
        ae_detector(x=preprocessing_pipeline2["target_unscaled"], computation_mode=ComputationMode.Train)
    pipeline2.train(data)
    return ae_detector


if __name__ == "__main__":
    args = parser.parse_args()

    column = "y"
    path = "../data/in_train_ID200.csv"
    start_date = "2011-01-01 00:15:00"
    date_col = "time"
    HORIZON = 96
    freq = "15min"
    SCALING = True
    name = f"Small_{args.anomalies}_{args.base}_{args.classes}"

    if args.classes == "all":
        test_path = f"../data/out_train_ID200_{args.anomalies}_{args.anomalies}_{args.anomalies}_{args.anomalies}_small.csv"
    else:
        test_path = f'../data/out_train_ID200_{args.anomalies if args.classes == "1" else "0"}_{args.anomalies if args.classes == "2" else "0"}_{args.anomalies if args.classes == "3" else "0"}_{args.anomalies if args.classes == "4" else "0"}_small.csv'

    from datetime import datetime
    custom_date_parser = lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M")
    if args.type == "technical":
        test_data = pd.read_csv(test_path, index_col=date_col, parse_dates=[date_col],
                            infer_datetime_format=True, date_parser=custom_date_parser)
    else:
        test_data = pd.read_csv(test_path[:-9] + "unusual_behaviour.csv", index_col=date_col, parse_dates=[date_col],
                            infer_datetime_format=True)
    for c in args.contaminations:
        create_run_pipelines(column, args.generator_methods, globals()[f"get_trained_{args.base}s"],
                             HORIZON, test_data, contamination=c)
        print("finished")

