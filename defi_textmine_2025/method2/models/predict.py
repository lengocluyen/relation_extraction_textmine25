"""
python -m defi_textmine_2025.method2.models.predict
"""

import logging
from defi_textmine_2025.method2.data.loading import load_data
from defi_textmine_2025.settings import (
    INTERIM_DIR,
    OUTPUT_DIR,
    LOGGING_DIR,
    RANDOM_SEED,
    get_now_time_as_str,
)
from defi_textmine_2025.set_logging import config_logging
from defi_textmine_2025.data.utils import save_data

start_date_as_str = get_now_time_as_str()
config_logging(f"{LOGGING_DIR}/method2/predict.log")

import pandas as pd
import torch
from defi_textmine_2025.method2.models.model_custom_classes import (
    BertMlp,
    BertBasedModel,
)
from defi_textmine_2025.method2.models.task_definition import Task
from defi_textmine_2025.method2.models.predict_toolbox import (
    apply_task_step_model,
    load_trained_model,
)
from defi_textmine_2025.method2.models.shared_toolbox import (
    BASE_CHECKPOINT_NAME,
    get_target_columns,
    load_fold_data,
    task2step2ismultilabel,
)

BATCH_SIZE = 128


def predict(task_name, step_name, model_class, num_fold, train_df, val_df, input_df):
    logging.warning(
        f"Applying step model: {BATCH_SIZE=}, [{model_class.__name__}"
        f" for {task_name}-{step_name} on {num_fold=}]"
    )
    target_columns = get_target_columns(task_name, step_name)
    logging.info(f"{target_columns=}")
    ismultilabel = task2step2ismultilabel[task_name][step_name]
    logging.info(f"{ismultilabel=}")
    logging.warning(f"Preparing data for the sub-task {task_name} and {step_name=}...")
    task = Task.init_predefined_sub_task(
        task_name, pd.concat([train_df, val_df], axis=0)
    )
    task_input_data = task.filter_prediction_data(input_df)
    model = load_trained_model(
        task_name,
        step_name,
        num_fold,
        BASE_CHECKPOINT_NAME,
        model_class,
        n_classes=len(target_columns),
        device=device,
    )
    pred_onehot, pred_probas = apply_task_step_model(
        trained_model=model,
        input_data=task_input_data,
        task_name=task_name,
        step_name=step_name,
        batch_size=BATCH_SIZE,
        device=device,
        is_step_multilabel=ismultilabel,
    )
    pred_onehot_df = pd.DataFrame(
        data=pred_onehot, columns=target_columns, index=task_input_data.index
    ).add_suffix(f"{task_name}-{step_name}")
    pred_probas_df = (
        pd.DataFrame(
            data=pred_probas, columns=target_columns, index=task_input_data.index
        )
        .add_suffix("_proba")
        .add_suffix(f"-{task_name}-{step_name}")
    )
    return pd.concat([pred_onehot_df, pred_probas_df], axis=1)


if __name__ == "__main__":
    num_fold = 1  # int(sys.argv[1])
    input_path = "data/defi-text-mine-2025/interim/reduced_text_w_entity_bracket/test"  # int(sys.argv[2])
    input_df = load_data(input_path, index_col=0).head(1000)
    logging.info(f"{input_df.shape=}")
    pipelines = [
        [("roottask", "RI"), ("subtask1", "RC")],
        [("roottask", "RI"), ("subtask2", "RC")],
        [("roottask", "RI"), ("subtask3", "RC")],
    ]
    logging.warning(f"Prediction with {pipelines=} - [{num_fold=}]")
    model_class: BertBasedModel = BertMlp
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"{device=}")
    logging.warning("loading data...")
    train_df, val_df = load_fold_data(num_fold)
    logging.info(f"Loaded data {train_df.shape=}, {val_df.shape=}")
    for pipeline in pipelines:
        logging.warning(f"Applying {pipeline=}")
        for task_name, step_name in pipeline:
            # if "RC" in step_name:
            #     # we assume NO_RELATION_CLASS and WITH_RELATION_CLASS are columns
            #     # of input_data
            #     with_rel_data = task_input_data[]
            y_pred2d = predict(
                task_name, step_name, model_class, num_fold, train_df, val_df, input_df
            )
            input_df.loc[y_pred2d.index, y_pred2d.columns] = y_pred2d
            save_data(input_df, f"{OUTPUT_DIR}/test_pred.csv", with_index=False)
            break
        break
