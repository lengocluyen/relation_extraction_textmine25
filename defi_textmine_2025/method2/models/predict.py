"""
python -m defi_textmine_2025.method2.models.perdict
"""

import os
import logging
from defi_textmine_2025.settings import (
    INTERIM_DIR,
    OUTPUT_DIR,
    LOGGING_DIR,
    get_now_time_as_str,
)
from defi_textmine_2025.set_logging import config_logging

start_date_as_str = get_now_time_as_str()
config_logging(f"{LOGGING_DIR}/method2/predict-{start_date_as_str}.log")

import pandas as pd
import torch
from defi_textmine_2025.method2.models.model_custom_classes import (
    BertMlp,
    BertBasedModel,
)
from defi_textmine_2025.method2.models.task_definition import Task
from defi_textmine_2025.data.utils import load_csv
from defi_textmine_2025.method2.models.predict_toolbox import (
    apply_task_step_model,
    get_predictions,
    load_trained_model,
)
from defi_textmine_2025.method2.models.shared_toolbox import (
    BASE_CHECKPOINT_NAME,
    get_target_columns,
    load_fold_data,
    task2step2ismultilabel,
)

BATCH_SIZE = 128


if __name__ == "__main__":
    num_fold = 1  # int(sys.argv[1])
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
    test_df = load_csv(f"{INTERIM_DIR}/reduced_text_w_entity_bracket/test")
    logging.info(f"Loaded data {train_df.shape=}, {val_df.shape=}, {test_df.shape=}")
    for pipeline in pipelines:
        logging.warning(f"Applying {pipeline=}")
        for task_name, step_name in pipeline:
            logging.warning(
                f"Applying step model: {BATCH_SIZE=}, [{model_class.__name__}"
                f" for {task_name}-{step_name} on {num_fold=}]"
            )
            target_columns = get_target_columns(task_name, step_name)
            logging.info(f"{target_columns=}")
            ismultilabel = task2step2ismultilabel[task_name][step_name]
            logging.info(f"{ismultilabel=}")
            logging.warning(
                f"Preparing data for the sub-task {task_name} and {step_name=}..."
            )
            task = Task.init_predefined_sub_task(
                task_name, pd.concat([train_df, val_df], axis=0)
            )
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
                task=task,
                step_name=step_name,
                input_data=test_df.head(),
                batch_size=BATCH_SIZE,
                device=device,
                is_step_multilabel=ismultilabel,
            )
        break
