import os
import random
from typing import List, Tuple
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from defi_textmine_2025.bert_dataset_and_models import BertBasedModel
from defi_textmine_2025.method2.data.target_encoding import ORDERED_CLASSES
from defi_textmine_2025.method2.models.model_custom_classes import BertMlp
from defi_textmine_2025.settings import (
    INTERIM_DIR,
    MODELS_DIR,
    RANDOM_SEED,
)
from defi_textmine_2025.method2.data.relation_and_entity_classes import (
    NO_RELATION_CLASS,
    REDUCED_TAGGED_TEXT_COL,
    RELATION_CLASSES,
    WITH_RELATION_CLASS,
)
from defi_textmine_2025.method2.models.dataset import CustomDataset

# reproducibility
# following https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# torch.use_deterministic_algorithms(True)

N_FOLDS = 5
cv_fold_data_dir = f"{INTERIM_DIR}/method2-5_fold_cv"

MAX_N_TOKENS = 200
BASE_CHECKPOINT_NAME = "camembert/camembert-base"
logging.info(f"Initializing the tokenizer {BASE_CHECKPOINT_NAME}")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_CHECKPOINT_NAME, clean_up_tokenization_spaces=True
)

RI_TARGET_COLS = [WITH_RELATION_CLASS, NO_RELATION_CLASS]
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
assert len(RC_1_TARGET_COLS) + len(RC_2_TARGET_COLS) + len(RC_3_TARGET_COLS) == 37

SUBTASK_NAME2ORDEREDLABELS: dict = {
    "roottask": [label for label in ORDERED_CLASSES if label in RELATION_CLASSES],
    # binary: gender
    "subtask1": [label for label in ORDERED_CLASSES if label in RC_1_TARGET_COLS],
    # multiclass single label: non-or-once-coocurrent relation types
    "subtask2": [label for label in ORDERED_CLASSES if label in RC_2_TARGET_COLS],
    # multilabel: often co-occurrent relation types
    "subtask3": [label for label in ORDERED_CLASSES if label in RC_3_TARGET_COLS],
}

SUBTASK_NAME_TO_RELATIONS_TO_DROP_IN_TRAIN_DATA: dict = {
    "roottask": {},
    # binary: gender
    "subtask1": {
        "['GENDER_MALE', 'GENDER_FEMALE']",
        "['GENDER_MALE', 'IS_IN_CONTACT_WITH']",
    },
    # multiclass single label: non-or-once-coocurrent relation types
    "subtask2": {
        "['DEATHS_NUMBER', 'IS_OF_SIZE']",
        "['DEATHS_NUMBER', 'INJURED_NUMBER']",
        "['DEATHS_NUMBER', 'INJURED_NUMBER']",
        "['INITIATED', 'DIED_IN']",
        "['INITIATED', 'DIED_IN']",
    },
    # multilabel: often co-occurrent relation types
    "subtask3": {},
}

INPUT_COLUMNS = [
    "text_index",
    "e1_id",
    "e2_id",
    "e1_type",
    "e2_type",
    "text",
    "relations",
    "reduced_text",
]

# task_name2targetcolumns = {
#     # binary
#     "RI": [NO_RELATION_CLASS, WITH_RELATION_CLASS],
#     # binary
#     "RC1": [label for label in ORDERED_CLASSES if label in RC_1_TARGET_COLS],
#     # multiclass single label
#     "RC2": [label for label in ORDERED_CLASSES if label in RC_2_TARGET_COLS],
#     # multilabel
#     "RC3": [label for label in ORDERED_CLASSES if label in RC_3_TARGET_COLS],
# }

task_name2ismultilabel = {
    "roottask": False,  # multilabel: the initial task w/ the whole relation types without seperation
    "subtask1": False,  # binary
    "subtask2": False,  # multiclass single label
    "subtask3": True,  # multilabel
}

task2step2ismultilabel = {
    "roottask": {  # the initial task w/
        "RI": False,
        "RC": True,
        "RI&RC": True,
        "RC_multilabel": True,
    },
    "subtask1": {  # gender relation types
        "RI": False,
        "RC": False,
        "RI&RC": False,
        "RC_multilabel": True,
    },
    "subtask2": {  # multi-class single label: no-coocurrent relatioon types
        "RI": False,
        "RC": False,
        "RI&RC": False,
        "RC_multilabel": True,
    },
    "subtask3": {  # multilabel: co-oocurrent relation types
        "RI": False,
        "RC": True,
        "RI&RC": True,
        "RC_multilabel": True,
    },
}
# assert len(RC_1_TARGET_COLS) + len(RC_2_TARGET_COLS) + len(RC_3_TARGET_COLS) == 37


model_name2class = {"BERT+MLP": BertMlp}


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


# def get_task_data(task_name: str, num_fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     train_df, val_df = load_fold_data(num_fold)
#     train_df = train_df.assign(**{WITH_RELATION_CLASS: 1 - train_df[NO_RELATION_CLASS]})
#     val_df = val_df.assign(**{WITH_RELATION_CLASS: 1 - val_df[NO_RELATION_CLASS]})
#     logging.info(f"Loaded data {train_df.shape=}, {val_df.shape=}")
#     if task_name:
#         logging.info("Filtering the data...")
#         out_of_scope_columns = [
#             label
#             for a_task, labels in task_name2targetcolumns.items()
#             for label in labels
#             if a_task != task_name and label in train_df
#         ]

#         train_df = train_df[
#             train_df[task_name2targetcolumns[task_name]].sum(axis=1) > 0
#         ].drop(out_of_scope_columns, axis=1)
#         val_df = val_df[
#             val_df[task_name2targetcolumns[task_name]].sum(axis=1) > 0
#         ].drop(out_of_scope_columns, axis=1)
#         logging.info(f"Filtered data {train_df.shape=}, {val_df.shape=}")
#         logging.info(f"Filtered data {train_df.columns=}, {val_df.columns=}")
#     return train_df, val_df


def get_target_columns(task_name: str, step_name: str) -> List[str]:
    if task_name in SUBTASK_NAME2ORDEREDLABELS:
        if step_name == "RC":
            return SUBTASK_NAME2ORDEREDLABELS[task_name]
        elif step_name == "RI":
            return RI_TARGET_COLS
        elif step_name == "RI&RC":
            return RI_TARGET_COLS + SUBTASK_NAME2ORDEREDLABELS[task_name]
        elif step_name == "RC_multilabel":
            return SUBTASK_NAME2ORDEREDLABELS[task_name]
        else:
            raise ValueError(f"Unsupported {step_name=}")
    else:
        raise ValueError(f"Unsupported {task_name=}")


def get_data_loaders(
    task_name: str,
    step_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_batch_size: int,
    val_batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    logging.warning("Init and tokenize the dataset")
    target_columns = get_target_columns(task_name, step_name)
    train_ds = CustomDataset(
        train_df,
        tokenizer,
        max_n_tokens=MAX_N_TOKENS,
        text_column=REDUCED_TAGGED_TEXT_COL,
        label_columns=target_columns,
    )
    # logging.info(f"{train_ds=}")
    val_ds = CustomDataset(
        val_df,
        tokenizer,
        max_n_tokens=MAX_N_TOKENS,
        text_column=REDUCED_TAGGED_TEXT_COL,
        label_columns=target_columns,
    )
    # logging.info(f"{val_ds=}")
    logging.warning("Init dataloaders")
    # Data loaders
    train_data_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_data_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_data_loader, val_data_loader


def get_model_checkpoint_basename(
    task_name: str,
    step_name: str,
    num_fold: str,
    base_checkpoint: str = "camembert/camembert-base",
    model_class: BertBasedModel.__class__ = BertMlp,
) -> str:
    return f"{task_name}-{step_name}-fold{num_fold}-{model_class.__name__}-{base_checkpoint.split('/')[-1]}"


def get_model_checkpoint_path(
    task_name: str,
    step_name: str,
    num_fold: str,
    base_ckpt_name: str = BASE_CHECKPOINT_NAME,
    model_class: BertBasedModel.__class__ = BertMlp,
) -> str:
    model_checkpoint_basename = get_model_checkpoint_basename(
        task_name, step_name, num_fold, base_ckpt_name, model_class
    )
    model_dir_path = os.path.join(MODELS_DIR, "method2")
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    return os.path.join(model_dir_path, f"{model_checkpoint_basename}.ckpt")


def init_bert_mlp_model(
    task_name: str, n_classes: int, base_ckpt_name: str = BASE_CHECKPOINT_NAME
) -> BertMlp:
    return BertMlp(
        name=task_name,
        embedding_model=AutoModel.from_pretrained(base_ckpt_name, return_dict=True),
        embedding_size=768 if "base" in base_ckpt_name else 1024,
        hidden_dim=128,
        n_classes=n_classes,
    )


def init_model(
    task_name: str,
    model_class: BertBasedModel.__class__,
    n_classes: int = 2,
    base_ckpt_name: str = BASE_CHECKPOINT_NAME,
) -> BertBasedModel:
    match model_class.__name__:
        case BertMlp.__name__:
            return init_bert_mlp_model(task_name, n_classes, base_ckpt_name)
        case _:
            raise ValueError(f"Unsupported {model_class.__name__=}")


def load_model(
    # task_name: str,
    # num_fold: int,
    # base_ckpt_name: str,
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training"""
    # checkpoint_path = get_model_checkpoint_path(task_name, num_fold, base_ckpt_name)
    checkpoint: dict = torch.load(
        checkpoint_path, weights_only=False, map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    last_epoch = checkpoint["last_epoch"]
    train_loop_history = checkpoint["train_loop_history"]
    best_val_f1_macro = checkpoint["best_val_f1_macro"]
    return (
        model,
        optimizer,
        last_epoch,
        train_loop_history,
        best_val_f1_macro,
    )


def save_model(
    # task_name: str,
    # step_name: str,
    # num_fold: int,
    # base_ckpt_name: str,
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    last_epoch: int,
    train_loop_history: dict,
    best_val_f1_macro: float,
):
    """https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training"""
    # checkpoint_path = get_model_checkpoint_path(task_name, num_fold, base_ckpt_name)
    torch.save(
        {
            "last_epoch": last_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loop_history": train_loop_history,
            "best_val_f1_macro": best_val_f1_macro,
        },
        checkpoint_path,
    )
