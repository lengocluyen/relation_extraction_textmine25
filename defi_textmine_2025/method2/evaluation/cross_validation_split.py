"""
python -m defi_textmine_2025.method2.evaluation.cross_validation_split
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s|%(levelname)s|%(module)s.py:%(lineno)s] %(message)s",
    datefmt="%H:%M:%S",
)
import pandas as pd

from transformers import AutoTokenizer
from datasets import Dataset

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold, StratifiedKFold
from defi_textmine_2025.data.utils import (
    EDA_DIR,
    INTERIM_DIR,
    TARGET_COL,
    load_csv,
    get_cat_var_distribution,
    save_data,
)
from defi_textmine_2025.method2.data.utils import (
    RELATIONS_TO_DROP,
    NO_RELATION_CLASS,
    RELATION_CLASSES,
    encode_target_to_onehot,
)

N_FOLDS = 5

# after defi_textmine_2025/method2/data/reduce_texts.py
logging.info(f"{RELATIONS_TO_DROP=}")

onehot_encoder = MultiLabelBinarizer().fit([list(RELATION_CLASSES)])
logging.info(f"{onehot_encoder.classes_=}")

labeled_df = load_csv(f"{INTERIM_DIR}/reduced_text_w_entity_bracket/train", index_col=0)
logging.info(f"Loaded {labeled_df.shape}")

labeled_df = labeled_df.query(f"relations not in {RELATIONS_TO_DROP}")
logging.info(f"after dropping RELATIONS_TO_DROP  {labeled_df.shape}")

labeled_df = encode_target_to_onehot(
    labeled_df, TARGET_COL, onehot_encoder
).reset_index(drop=True)

logging.info(f"After onehot encoding:\n{labeled_df.head(2)}")

logging.info(f"{N_FOLDS}-fold cross validation split...")
RANDOM_SEED = 0
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
        get_cat_var_distribution(train_df[RELATION_CLASSES]).sort_values(by="count"),
        f"{EDA_DIR}/method2-{N_FOLDS}_fold_cv/train-fold{k}_relation_distrib-{RANDOM_SEED=}-mth2.csv",
    )
    save_data(
        get_cat_var_distribution(val_df[RELATION_CLASSES]).sort_values(by="count"),
        f"{EDA_DIR}/method2-{N_FOLDS}_fold_cv/validation-fold{k}_relation_distrib-{RANDOM_SEED=}-mth2.csv",
    )
    k += 1
