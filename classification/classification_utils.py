import functools
import os

import pandas as pd

from generative_models.cVAE import cVAEModule
from generative_models.anomalyINN import INNWrapper

from classification.config import GS, TRAINING_LENGTH
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.modules import Sampler, CalendarExtraction, CalendarFeature, SKLearnWrapper, FunctionModule, Slicer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from pywatts.summaries import ConfusionMatrix,  F1Score
from pywatts.modules import StatisticFeature
from utils import get_flat_output, get_reshaping


def evaluate_classifiers(name, HORIZON, column, gen_scaler_list, data, filter=lambda data: data >= 1, supervised=True):
    pipeline3 = Pipeline(f"../results/{name}evaluate")
    classifiers = {}
    for gen, scaler, generator, clf in gen_scaler_list:
        preprocessing_pipeline3 = get_preprocessing_pipeline(HORIZON, scaler, name="results/preprocessing_evaluation",
                                                             scale_computation_mode=ComputationMode.Transform)
        preprocessing_pipeline3 = preprocessing_pipeline3(input=pipeline3[column])
        if generator is not None:
            latent = generator(input_data=preprocessing_pipeline3["target_scaled"],
                         cal_input=preprocessing_pipeline3["calendar"],
                         stats_input=preprocessing_pipeline3["stats"],
                         computation_mode=ComputationMode.Transform)
            clf.name = f"Latent {gen}"
            classifier = clf(x=latent, computation_mode=ComputationMode.Transform)
        elif scaler is not None:
            clf.name = "Scaled"
            classifier = clf(
                x=preprocessing_pipeline3["target_scaled"],
                computation_mode=ComputationMode.Transform)
        else:
            clf.name = "Unscaled"
            classifier = clf(x=preprocessing_pipeline3["target_unscaled"],
                             computation_mode=ComputationMode.Transform)
        classifier.step.module.is_fitted = True
        classifier.step.is_fitted = True
        if not supervised:
            classifier = FunctionModule(functools.partial(get_flat_output,
                                                          filter=lambda x: (x.values.reshape(-1, 1) * -1) >= 1))(data=classifier)
        classifiers[gen] = classifier

    anomalies_sampled = Sampler(HORIZON)(x=pipeline3["anomalies"])

    gt = FunctionModule(functools.partial(get_flat_output, filter=filter))(data=anomalies_sampled)
    gt = Slicer(start=HORIZON * 1)(x=gt)

    F1Score(average="macro")(y_true=gt, **classifiers)
    ConfusionMatrix()(y_true=gt, **classifiers)
    _, summary = pipeline3.test(data, summary=True)


def get_trained_inn_wrappers(HORIZON, column, generator, train_data, name="name"):
    scaler = SKLearnWrapper(StandardScaler())
    # Create Pipeline
    pipeline = Pipeline(f"../results/{name}/generator_train")

    preprocessing_pipeline = get_preprocessing_pipeline(HORIZON, scaler, name="results/preprocessing_inn")
    preprocessing_pipeline = preprocessing_pipeline(input=pipeline[column])
    generator(cal_input=preprocessing_pipeline["calendar"],
                stats_input=preprocessing_pipeline["stats"],
                input_data=preprocessing_pipeline["target_scaled"])
    pipeline.train(train_data)
    return generator, scaler



def get_preprocessing_pipeline(HORIZON, scaler, name="results/preprocessing",
                               scale_computation_mode=ComputationMode.Default,
                               calendar_extractor=CalendarExtraction(country="BadenWurttemberg",
                                                                     features=[CalendarFeature.hour,
                                                                               CalendarFeature.weekend,
                                                                               CalendarFeature.month])):
    pipeline = Pipeline(name + "preprocessing")

    calendar = calendar_extractor(x=pipeline["input"])
    target = Sampler(sample_size=HORIZON)(x=pipeline["input"])
    sliced = Slicer(start=HORIZON * 1 , name="target_unscaled")(x=target)

    if scaler is not None:
        scaled = scaler(x=sliced, computation_mode=scale_computation_mode)
        FunctionModule(get_reshaping("target_scaled", horizon=96), name="target_scaled")(x=scaled)
    t =StatisticExtractor(dim="horizon", features=[StatisticFeature.mean])(x=target)
    Slicer(start=HORIZON * 1 , name="stats")(x=t)
    t = Sampler(sample_size=HORIZON)(x=calendar)
    Slicer(start=HORIZON * 1 , name="calendar")(x=t)
    t = Sampler(HORIZON)(x=pipeline["input"])
    Slicer(start=HORIZON * 1 ,name="target_unscaled")(x=t)
    return pipeline


def train_sklearn_modules(HORIZON, column, data, generator, sk_module, name, scaler, filter=lambda data: data >= 1, supervised=True):
    pipeline2 = Pipeline("../results/" + name + "train")
    preprocessing_pipeline2 = get_preprocessing_pipeline(HORIZON, scaler, name="results/preprocessing_" + name,
                                                         scale_computation_mode=ComputationMode.Transform)
    preprocessing_pipeline2 = preprocessing_pipeline2(input=pipeline2[column])
    method_name = ""
    if supervised:
        anomalies_sampled = Sampler(HORIZON)(x=pipeline2["anomalies"])
        gt = FunctionModule(functools.partial(get_flat_output, filter=filter))(data=anomalies_sampled)
        gt = Slicer(start=HORIZON * 1 )(x=gt)
        if generator is not None:
            method_name = generator.name
            latent = generator(input_data=preprocessing_pipeline2["target_scaled"],
                             cal_input=preprocessing_pipeline2["calendar"],
                             stats_input=preprocessing_pipeline2["stats"],
                             computation_mode=ComputationMode.Transform)
            SKLearnWrapper(sk_module, name=generator.name)(x=latent,
                                                           target=gt,
                                                           computation_mode=ComputationMode.Train)
        elif scaler is not None:
            method_name = "scaled"
            SKLearnWrapper(sk_module, name="scaled")(x=preprocessing_pipeline2["target_scaled"],
                                                     target=gt,
                                                     computation_mode=ComputationMode.Train)
        else:
            method_name = "unscaled"
            SKLearnWrapper(sk_module, name="unscaled")(x=preprocessing_pipeline2["target_unscaled"],
                                                       target=gt,
                                                       computation_mode=ComputationMode.Train)

    else:
        if generator is not None:
            latent = generator(input_data=preprocessing_pipeline2["target_scaled"],
                             cal_input=preprocessing_pipeline2["calendar"],
                             stats_input=preprocessing_pipeline2["stats"],
                             computation_mode=ComputationMode.Transform)
            SKLearnWrapper(sk_module, name=generator.name)(x=latent,
                                                           computation_mode=ComputationMode.Train)
        elif scaler is not None:
            SKLearnWrapper(sk_module, name="scaled")(x=preprocessing_pipeline2["target_scaled"],
                                                     computation_mode=ComputationMode.Train)
        else:
            SKLearnWrapper(sk_module, name="unscaled")(x=preprocessing_pipeline2["target_unscaled"],
                                                       computation_mode=ComputationMode.Train)

    pipeline2.train(data)
    if isinstance(sk_module, GridSearchCV):
        cv_results = pd.DataFrame(sk_module.cv_results_)
        if not os.path.exists("../results/" + name +  "/gs_results"):
            os.makedirs("../results/" + name +  "/gs_results")
        cv_results.to_csv("../results/" + name +  f"/gs_results/{method_name}.csv")
        print(name + "scaled\n", cv_results.to_string())
    return SKLearnWrapper(sk_module)


def get_generator(generator_method, supervised=True, contamination=0.8):

    if generator_method == "cinn":
        generator = INNWrapper("cINN", epochs=50, supervised=supervised, contamination=contamination)
    elif generator_method == "cvae":
        generator = cVAEModule("cVAE", epochs=100, supervised=supervised, contamination=contamination)
    else:
        raise Exception("No valid generator defined")
    return generator

def create_run_pipelines(column, train_data, HORIZON, generator_methods, test_data, name, get_sklearn_modules, gs=GS):
    for i in range(2):
        gen_scaler_list = []
        for gen in generator_methods:
            generator = get_generator(gen)
            trained_generator, scaler = get_trained_inn_wrappers(HORIZON, "y", generator, train_data[:TRAINING_LENGTH])
            sk_module = get_sklearn_modules(
                name, HORIZON, column, trained_generator, scaler, test_data[:5000], gs=gs)
            gen_scaler_list.append((gen, scaler, generator, sk_module))

        # Each scaler has the same weights, thus we only use the last. Yes this is ugly...
        sk_module = get_sklearn_modules(
                name, HORIZON, column, None, scaler, test_data[:5000], gs=gs)
        gen_scaler_list.append(("scaled", scaler, None, sk_module))
        sk_module = get_sklearn_modules(
                name, HORIZON, column, None, None, test_data[:5000], gs=gs)
        gen_scaler_list.append(("unscaled", None, None, sk_module))


        # Evaluate INN_kNN
        evaluate_classifiers(name, HORIZON, column, gen_scaler_list, test_data[TRAINING_LENGTH:])

        print("Finished creation " + name)

