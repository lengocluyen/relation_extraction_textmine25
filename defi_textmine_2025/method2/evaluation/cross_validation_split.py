"""
python -m defi_textmine_2025.method2.evaluation.cross_validation_split
"""

import logging

from defi_textmine_2025.settings import RANDOM_SEED
from defi_textmine_2025.settings import EDA_DIR, INTERIM_DIR, LOGGING_DIR
from defi_textmine_2025.set_logging import config_logging

config_logging(f"{LOGGING_DIR}/method2/cross_validation_split.log")

import pandas as pd

from transformers import AutoTokenizer
from datasets import Dataset

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold, StratifiedKFold
from defi_textmine_2025.data.utils import (
    INTERIM_DIR,
    TARGET_COL,
    load_csv,
    get_cat_var_distribution,
    save_data,
)
from defi_textmine_2025.method2.data.target_encoding import (
    encode_target_to_onehot,
    onehot_encoder,
    all_target_columns,
)

N_FOLDS = 5


# after defi_textmine_2025/method2/data/reduce_texts.py

labeled_df = load_csv(f"{INTERIM_DIR}/reduced_text_w_entity_bracket/train", index_col=0)
logging.info(f"Loaded {labeled_df.shape}")

labeled_df = encode_target_to_onehot(
    labeled_df, TARGET_COL, onehot_encoder
).reset_index(drop=True)

logging.info(f"After onehot encoding:\n{labeled_df.head(2)}")

logging.info(f"{N_FOLDS}-fold cross validation split...")
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
k = 1
for train_index, val_index in kf.split(
    labeled_df
):  # , y=labeled_df[onehot_encoder.classes_]
    train_df, val_df = labeled_df.loc[train_index], labeled_df.loc[val_index]
    save_data(
        train_df, f"{INTERIM_DIR}/method2-{N_FOLDS}_fold_cv/train-fold{k}-mth2.parquet"
    )
    save_data(
        val_df,
        f"{INTERIM_DIR}/method2-{N_FOLDS}_fold_cv/validation-fold{k}-mth2.parquet",
    )
    logging.info(
        f"fold {k} {train_index=} {val_index=} {train_df.shape=} {val_df.shape=}"
    )
    save_data(
        get_cat_var_distribution(train_df[all_target_columns]).sort_values(by="count"),
        f"{EDA_DIR}/method2-{N_FOLDS}_fold_cv/train-fold{k}_relation_distrib-{RANDOM_SEED=}-mth2.csv",
    )
    save_data(
        get_cat_var_distribution(val_df[all_target_columns]).sort_values(by="count"),
        f"{EDA_DIR}/method2-{N_FOLDS}_fold_cv/validation-fold{k}_relation_distrib-{RANDOM_SEED=}-mth2.csv",
    )
    k += 1
