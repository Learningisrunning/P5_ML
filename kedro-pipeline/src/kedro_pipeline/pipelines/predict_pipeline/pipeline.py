from .nodes import (
    run_pipeline_data_cleaning,
    pipeline_transformation_to_BoW,
    MultinomialNB_BoW_predict,
)
from kedro.pipeline import Pipeline, node, pipeline

def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=run_pipeline_data_cleaning,
                inputs=["df_test","input_data"],
                outputs="processed_data_test",
                name="data_cleaning_test_node",
            ),
            node(
                func=pipeline_transformation_to_BoW,
                inputs=["processed_data_test","BoW_traitement"],
                outputs="processed_data_boW_test",
                name="BoW_transformation_test_node",
            ),
            node(
                func=MultinomialNB_BoW_predict,
                inputs=["trained_model", "processed_data_boW_test", "mlb"],
                outputs="prediction",
                name="predict_model_node",
            ),
        ]
    )