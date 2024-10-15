"""
python -m defi_textmine_2025.method2.models.train
"""

from collections import defaultdict
import os
import sys
import random
from typing import Tuple
import logging

from defi_textmine_2025.settings import (
    INTERIM_DIR,
    LOGGING_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    get_now_time_as_str,
)
from defi_textmine_2025.set_logging import config_logging

config_logging(f"{LOGGING_DIR}/method2/train-{get_now_time_as_str()}.log")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW

from defi_textmine_2025.method2.data.relation_and_entity_classes import (
    NO_RELATION_CLASS,
    REDUCED_TAGGED_TEXT_COL,
    RELATION_CLASSES,
    NO_RELATION_CLASS,
    WITH_RELATION_CLASS,
)
from defi_textmine_2025.method2.data.target_encoding import ORDERED_CLASSES
from defi_textmine_2025.data.utils import (
    compute_class_weights,
    get_cat_var_distribution,
)
from defi_textmine_2025.method2.models.dataset import CustomDataset
from defi_textmine_2025.method2.models.model_custom_classes import BertMlp
from defi_textmine_2025.bert_dataset_and_models import eval_model, train_model

# reproducibility
# following https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# torch.use_deterministic_algorithms(True)

N_FOLDS = 5
cv_fold_data_dir = f"{INTERIM_DIR}/method2-5_fold_cv"

BASE_CHECKPOINT = "camembert/camembert-base"
tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT)

TRAIN_BATCH_SIZE = 90
VAL_BATCH_SIZE = 72
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_N_TOKENS = 150
MAX_EPOCHS = 100
PATIENCE = 10

RI_TARGET_COL = NO_RELATION_CLASS
RC_1_TARGET_COLS = ["GENDER_MALE", "GENDER_FEMALE"]
RC_2_TARGET_COLS = [
    "HAS_CATEGORY",
    "HAS_CONSEQUENCE",
    "HAS_QUANTITY",
    "IS_OF_NATIONALITY",
    "HAS_COLOR",
    "IS_DEAD_ON",
    "WEIGHS",
    "IS_REGISTERED_AS",
    "IS_BORN_ON",
    "HAS_FOR_LENGTH",
    "WAS_CREATED_IN",
    "WAS_DISSOLVED_IN",
    "HAS_FOR_WIDTH",
    "HAS_FOR_HEIGHT",
    "HAS_LONGITUDE",
    "HAS_LATITUDE",
    "INITIATED",
    "DIED_IN",
    "IS_OF_SIZE",
    "DEATHS_NUMBER",
    "INJURED_NUMBER",
]
RC_3_TARGET_COLS = [
    "IS_LOCATED_IN",
    "HAS_CONTROL_OVER",
    "OPERATES_IN",
    "IS_IN_CONTACT_WITH",
    "STARTED_IN",
    "IS_AT_ODDS_WITH",
    "IS_PART_OF",
    "START_DATE",
    "END_DATE",
    "IS_COOPERATING_WITH",
    "RESIDES_IN",
    "HAS_FAMILY_RELATIONSHIP",
    "CREATED",
    "IS_BORN_IN",
]


task_name2targetcolumns = {
    # binary
    "RI": [NO_RELATION_CLASS, WITH_RELATION_CLASS],
    # binary
    "RC1": [label for label in ORDERED_CLASSES if label in RC_1_TARGET_COLS],
    # multiclass single label
    "RC2": [label for label in ORDERED_CLASSES if label in RC_2_TARGET_COLS],
    # multilabel
    "RC3": [label for label in ORDERED_CLASSES if label in RC_3_TARGET_COLS],
}

task_name2ismultilabel = {
    "RI": False,  # binary
    "RC1": False,  # binary
    "RC2": False,  # multiclass single label
    "RC3": True,  # multilabel
}
assert len(RC_1_TARGET_COLS) + len(RC_2_TARGET_COLS) + len(RC_3_TARGET_COLS) == 37


def load_fold_data(k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """return train and validation labeled dataframes

    Args:
        k (int): number of the fold to load

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train_df, val_df
    """

    return (
        pd.read_parquet(f"{cv_fold_data_dir}/{split_name}-fold{k}-mth2.parquet")
        for split_name in ("train", "validation")
    )


def tokenize_function(example: dict):
    return tokenizer(
        # max n_token without loosing entity, see setp0_data_preparation
        example[REDUCED_TAGGED_TEXT_COL],
        truncation=True,
        max_length=MAX_N_TOKENS,
    )


def get_task_data(task_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = load_fold_data(num_fold)
    train_df = train_df.assign(**{WITH_RELATION_CLASS: 1 - train_df[NO_RELATION_CLASS]})
    val_df = val_df.assign(**{WITH_RELATION_CLASS: 1 - val_df[NO_RELATION_CLASS]})
    logging.info(f"Loaded data {train_df.shape=}, {val_df.shape=}")

    out_of_scope_columns = [
        label
        for a_task, labels in task_name2targetcolumns.items()
        for label in labels
        if a_task != task_name
    ]

    train_df = train_df[
        train_df[task_name2targetcolumns[task_name]].sum(axis=1) > 0
    ].drop(out_of_scope_columns, axis=1)
    val_df = val_df[val_df[task_name2targetcolumns[task_name]].sum(axis=1) > 0].drop(
        out_of_scope_columns, axis=1
    )
    logging.info(f"Filtered data {train_df.shape=}, {val_df.shape=}")
    logging.info(f"Filtered data {train_df.columns=}, {val_df.columns=}")
    return train_df, val_df


if __name__ == "__main__":

    num_fold = 1  # int(sys.argv[1])
    task_name = "RI"

    logging.info(f"{task_name=}")
    target_columns = task_name2targetcolumns[task_name]
    logging.info(f"{target_columns=}")
    ismultilabel = task_name2ismultilabel[task_name]
    logging.info(f"{ismultilabel=}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"{device=}")
    logging.warning(f"Preparing data for the sub-task {task_name}...")
    train_df, val_df = get_task_data(task_name)

    n_examples = train_df.shape[0]
    n_classes = (
        train_df[target_columns].nunique()
        if isinstance(target_columns, str)
        else len(target_columns)
    )
    logging.info(f"{n_examples=}, {n_classes=}")
    logging.warning("Init and tokenize the dataset")
    train_ds = CustomDataset(
        train_df,
        tokenizer,
        max_n_tokens=MAX_N_TOKENS,
        text_column=REDUCED_TAGGED_TEXT_COL,
        label_columns=target_columns,
    )
    logging.info(f"{train_ds=}")
    val_ds = CustomDataset(
        val_df,
        tokenizer,
        max_n_tokens=MAX_N_TOKENS,
        text_column=REDUCED_TAGGED_TEXT_COL,
        label_columns=target_columns,
    )
    logging.info(f"{val_ds=}")
    logging.warning("Init dataloaders")
    # Data loaders
    train_data_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_data_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    logging.warning(f"Computute class weights with {n_examples=}, {n_classes=}")
    class_weights_df = compute_class_weights(
        train_df,
        label_columns=(
            [target_columns] if isinstance(target_columns, str) else target_columns
        ),
    )
    class_distr_weight_df = pd.concat(
        [get_cat_var_distribution(train_df[target_columns]), class_weights_df],
        axis=1,
    )
    logging.info(f"Class weights: \n{class_distr_weight_df}")
    class_weights = torch.from_numpy(class_weights_df.values).to(device)
    logging.info(f"{class_weights=}")
    logging.warning("Initialize the model")
    model = BertMlp(
        name="relation_identification",
        embedding_model=AutoModel.from_pretrained(BASE_CHECKPOINT, return_dict=True),
        embedding_size=768 if "base" in BASE_CHECKPOINT else 1024,
        hidden_dim=128,
        n_classes=n_classes,
    )
    model.to(device)
    logging.info(f"Initialized model:\n{model}")
    logging.warning("Start the training loop...")
    # Training loop
    best_f1_macro = 0
    n_not_better_steps = 0
    history = defaultdict(list)
    model_dir_path = os.path.join(MODELS_DIR, "method2")
    model_state_basename = (
        f"{task_name}-fold{num_fold}-BERT+MLP-{BASE_CHECKPOINT.split('/')[-1]}"
    )
    model_dict_state_path = os.path.join(model_dir_path, f"{model_state_basename}.bin")
    if not os.path.exists(os.path.dirname(model_dict_state_path)):
        os.makedirs(os.path.dirname(model_dict_state_path))
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    logging.info(
        f"{TRAIN_BATCH_SIZE=}, {VAL_BATCH_SIZE=}, {LEARNING_RATE=}, {WEIGHT_DECAY=}, "
        f"{MAX_N_TOKENS=}, {MAX_EPOCHS=}, {PATIENCE=}"
    )
    for epoch in range(1, MAX_EPOCHS + 1):
        print(
            f"Epoch {epoch}/{MAX_EPOCHS} [patience count: {n_not_better_steps} "
            f"/ {PATIENCE} - {best_f1_macro=:.3f}]"
        )
        train_loss, train_acc, train_f1 = train_model(
            model,
            train_data_loader,
            optimizer,
            class_weights,
            device,
            multilabel=ismultilabel,
        )
        val_loss, val_acc, val_f1 = eval_model(
            model, val_data_loader, class_weights, device, multilabel=ismultilabel
        )

        logging.info(f"{train_loss=:.4f}, {train_acc=:.3f}, {train_f1=:.3f}")
        logging.info(f"{val_loss=:.4f}, {val_acc=:.3f}, {val_f1=:.3f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1_macro"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1)

        plt.rcParams["figure.figsize"] = (10, 7)
        plt.plot(history["train_f1_macro"], label="train F1 macro")
        plt.plot(history["val_f1_macro"], label="validation F1 macro")
        plt.plot(history["train_loss"], label="train loss")
        plt.plot(history["val_loss"], label="validation loss")
        plt.title("Training history")
        plt.ylabel("F1 macro / loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.ylim([0, 1])
        plt.grid()
        plt.savefig(
            f"{LOGGING_DIR}/method2/training_history_{model_state_basename}.png"
        )

        # Save best model
        if val_f1 > best_f1_macro:
            logging.warning(
                f"#NEW_BEST_VAL_F1 {best_f1_macro:.3f} -> Saving new best model at {model_dict_state_path}"
            )
            torch.save(model.state_dict(), model_dict_state_path)
            best_f1_macro = val_f1
            n_not_better_steps = 0
        else:
            n_not_better_steps += 1
            if n_not_better_steps >= PATIENCE:
                logging.info("Early stopping")
                break
