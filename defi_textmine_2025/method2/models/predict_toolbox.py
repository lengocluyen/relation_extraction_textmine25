import logging
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from defi_textmine_2025.method2.models.model_custom_classes import BertBasedModel
from defi_textmine_2025.method2.models.shared_toolbox import (
    get_data_loaders,
    get_model_checkpoint_path,
    init_model,
    load_model,
)
from defi_textmine_2025.method2.models.task_definition import Task


def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    multilabel: bool = True,
):
    model = model.eval()

    predictions = []
    prediction_probs = []

    with torch.no_grad():
        for data in tqdm(data_loader, "Prediction"):
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids)
            # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
            if multilabel:
                # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
                outputs = torch.sigmoid(outputs).cpu().detach()  # probabilities
            else:  # single-label
                outputs = torch.softmax(outputs, dim=1).cpu().detach()  # probabilities
            # thresholding at 0.5
            preds = outputs.round()

            predictions.extend(preds)
            prediction_probs.extend(outputs)

    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)

    return predictions, prediction_probs


def load_trained_model(
    task_name: str,
    step_name: str,
    num_fold: int,
    base_name_checkpoint: str,
    model_class: BertBasedModel.__class__,
    n_classes: int,
    device: torch.device,
) -> BertBasedModel:
    logging.warning("Initialize the model")
    model = init_model(task_name, model_class, n_classes, base_name_checkpoint)
    model.to(device)
    train_ckpt_path = get_model_checkpoint_path(
        task_name, step_name, num_fold, base_name_checkpoint, model_class
    )
    logging.info(f"{train_ckpt_path=}")
    model = load_model(train_ckpt_path, model, None, device)[0]
    logging.info(f"model:\n{model}")
    return model


def apply_task_step_model(
    trained_model: BertBasedModel,
    task: Task,
    step_name: str,
    input_data: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    is_step_multilabel: bool = True,
) -> Tuple[np.ndarray]:
    input_data = task.filter_prediction_data(input_data)
    assert not input_data.empty, f"No row matches entity types of {task=}!"
    logging.info(
        f"Task filtered data for prediction: {input_data.shape=}, {input_data.columns=}"
    )
    return get_predictions(
        model=trained_model,
        data_loader=get_data_loaders(
            task.name,
            step_name,
            dfs=[input_data],
            batch_sizes=[batch_size],
            shuffles=[False],
        )[0],
        device=device,
        multilabel=is_step_multilabel,
    )
