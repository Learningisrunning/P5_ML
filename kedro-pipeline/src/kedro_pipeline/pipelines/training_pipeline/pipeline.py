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
                func=load_data,
                inputs="data_set_from_stack",
                outputs="raw_data",
                name="load_data_node",
            ),
            node(
                func=split_data,
                inputs="raw_data",
                outputs=["df_train", "df_test"],
                name="split_data_node",
            ),
            node(
                func=run_pipeline_data_cleaning,
                inputs="df_train",
                outputs="processed_data_train",
                name="data_cleaning_train_node",
            ),
            node(
                func=pipeline_transformation_to_BoW,
                inputs="processed_data_train",
                outputs=["processed_data_boW_train", "BoW_traitement"],
                name="BoW_transformation_train_node",
            ),
            node(
                func=MultinomialNB_BoW_train,
                inputs="processed_data_boW_train",
                outputs=["trained_model", "mlb"],
                name="train_model_node",
            ),
        ]
    )