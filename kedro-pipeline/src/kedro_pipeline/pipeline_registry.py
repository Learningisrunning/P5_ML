"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline, pipeline


from kedro_pipeline.pipelines import (
    training_pipeline as tp,
    predict_pipeline as pp,
    data_drift as dd,
 )

def register_pipelines() -> Dict[str, Pipeline]:

    training_pipeline = tp.create_pipeline()
    predict_pipeline = pp.create_pipeline()
    data_drift = dd.create_pipeline()

    return {
        "tp": training_pipeline,
        "predict_pipeline": predict_pipeline,
        "dd": data_drift,
        "__default__": training_pipeline + predict_pipeline + data_drift
    }