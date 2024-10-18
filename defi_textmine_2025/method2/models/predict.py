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
from defi_textmine_2025.method2.models.predict_toolbox import get_predictions
from defi_textmine_2025.method2.models.shared_toolbox import (
    BASE_CHECKPOINT_NAME,
    get_model_checkpoint_basename,
    get_model_checkpoint_path,
    get_target_columns,
    init_model,
    load_fold_data,
    load_model,
    task2step2ismultilabel,
    get_data_loaders,
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

    target_columns = get_target_columns(task_name, step_name)
    logging.info(f"{target_columns=}")
    ismultilabel = task2step2ismultilabel[task_name][step_name]
    logging.info(f"{ismultilabel=}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"{device=}")
    logging.warning(f"Preparing data for the sub-task {task_name} and {step_name=}...")
    train_df, val_df = load_fold_data(num_fold)
    test_df = load_csv(f"{INTERIM_DIR}/reduced_text_w_entity_bracket/test")
    task = Task.init_predefined_sub_task(
        task_name, pd.concat([train_df, val_df], axis=0)
    )

    logging.info(
        f"{BATCH_SIZE=}, [{model_class.__name__} for {task_name}-{step_name} on {num_fold=}]"
    )

    for data_loader in (train_data_loader, val_data_loader, test_data_loader):
        predictions, prediction_probs = get_predictions(
            model,
            data_loader,
            device,
            multilabel=ismultilabel,
        )
        break
