from kedro.pipeline import node, Pipeline, pipeline
from .nodes import (
    monitor_data_drift,split_the_data,run_pipeline_data_cleaning,add_the_month,split_the_month)

def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=split_the_data,
                inputs="data_set_from_stack",
                outputs=["df_rest", "df_recents", "df_for_the_months"],
                name="split_data",
            ),
            node(
                func=run_pipeline_data_cleaning,
                inputs="df_rest",
                outputs="processed_data_rest",
                name="data_cleaning_train_node_rest",
            ),
            node(
                func=run_pipeline_data_cleaning,
                inputs="df_recents",
                outputs="processed_data_recents",
                name="data_cleaning_train_node_recents",
            ),
            node(
                func=add_the_month,
                inputs=["processed_data_recents","df_for_the_months"],
                outputs="df_recent_before_split",
                name="add_the_months",
            ),
            node(
                func=split_the_month,
                inputs="df_recent_before_split",
                outputs="processed_data_boW_recents_liste",
                name="split_the_month",
            ),
            node(
                func=monitor_data_drift,
                inputs=["processed_data_rest", "processed_data_boW_recents_liste"],
                outputs=None,
                name="monitor_data_drift_node",
            ),
        ]
    )
