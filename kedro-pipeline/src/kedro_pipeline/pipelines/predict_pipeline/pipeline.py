from .nodes import (
    load_data,
    split_data,
    run_pipeline_data_cleaning,
    pipeline_transformation_to_BoW,
    accuracy,
    MultinomialNB_BoW_predict,
    MultinomialNB_BoW_train,
)
from kedro.pipeline import Pipeline, node, pipeline

def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=run_pipeline_data_cleaning,
                inputs="data_set_from_stack",
                outputs="processed_data_test",
                name="data_cleaning_test_node",
            ),
            node(
                func=pipeline_transformation_to_BoW,
                inputs="processed_data_test",
                outputs="processed_data_boW_test",
                name="BoW_transformation_test_node",
            ),
            node(
                func=MultinomialNB_BoW_predict,
                inputs=["trained_model", "processed_data_boW_test"],
                outputs=["prediction", "test_set"],
                name="predict_model_node",
            ),
            node(
                func=accuracy,
                inputs=["prediction", "test_set"],
                outputs="accuracy_score",
                name="accuracy_node",
            ),
        ]
    )