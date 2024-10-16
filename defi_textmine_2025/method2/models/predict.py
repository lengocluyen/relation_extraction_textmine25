"""
python -m defi_textmine_2025.method2.models.predict
"""

from collections import defaultdict
import os
import logging
from defi_textmine_2025.settings import (
    LOGGING_DIR,
    get_now_time_as_str,
)
from defi_textmine_2025.set_logging import config_logging

start_date_as_str = get_now_time_as_str()
config_logging(f"{LOGGING_DIR}/method2/predict-{start_date_as_str}.log")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AdamW
from defi_textmine_2025.method2.models.shared_toolbox import (
    BASE_CHECKPOINT_NAME,
    MAX_N_TOKENS,
    get_model_checkpoint_basename,
    get_model_checkpoint_path,
    init_model,
    task_name2targetcolumns,
    task_name2ismultilabel,
    get_data_loaders,
    get_task_data,
)

BATCH_SIZE = 96

if __name__ == "__main__":
    num_fold = 1  # int(sys.argv[1])
    task_name = "RC1"
    model_name: str = "BERT+MLP"

    logging.info(f"Training {model_name} for {task_name} on {num_fold=}")
    target_columns = task_name2targetcolumns[task_name]
    logging.info(f"{target_columns=}")
    ismultilabel = task_name2ismultilabel[task_name]
    logging.info(f"{ismultilabel=}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"{device=}")
    logging.warning(f"Preparing data for the sub-task {task_name}...")
    # task_name=None: to apply the model over every texts
    train_df, val_df = get_task_data(task_name=None, num_fold=num_fold)
    train_data_loader, val_data_loader = get_data_loaders(
        task_name, train_df, val_df, BATCH_SIZE, BATCH_SIZE
    )
    logging.warning("Initializing the model")
    n_examples = train_df.shape[0]
    n_classes = (
        train_df[target_columns].nunique()
        if isinstance(target_columns, str)
        else len(target_columns)
    )
    logging.info(f"{n_examples=}, {n_classes=}")
    model_state_basename = get_model_checkpoint_basename(
        task_name, num_fold, BASE_CHECKPOINT_NAME
    )
    model_path = get_model_checkpoint_path(task_name, num_fold, BASE_CHECKPOINT_NAME)
    model = init_model(task_name, model_name, n_classes, BASE_CHECKPOINT_NAME)
    model.to(device)
    model = 
    # model.to(device)
    logging.info(f"Initialized model:\n{model}")
