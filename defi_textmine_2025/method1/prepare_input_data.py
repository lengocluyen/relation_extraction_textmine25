"""
python -m defi_textmine_2025.method1.prepare_input_data
"""

import os
from datetime import datetime
from typing import List
import json
import glob
import pandas as pd
import numpy as np
from defi_textmine_2025.data.utils import (
    EDA_DIR,
    LOGGING_DIR,
    INTERIM_DIR,
    TARGET_COL,
)
import logging

log_file_path = os.path.join(
    LOGGING_DIR,
    f'{datetime.now().strftime("method1-prepare_input_data-%Y%m%dT%H%M%S")}.log',
)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s|%(levelname)s|%(module)s.py:%(lineno)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),
    ],
)
logging.info(f"{log_file_path=}")
logging.info("## Imports")

from tqdm import tqdm

# Create new `pandas` methods which use `tqdm` progress
# (can use tqdm_gui, optional kwargs, etc.)
tqdm.pandas()

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from defi_textmine_2025.data.problem_formulation import TextToMultiLabelDataGenerator

METHOD_INTERIM_DIR = os.path.join(INTERIM_DIR, "method1")
if not os.path.exists(METHOD_INTERIM_DIR):
    os.makedirs(METHOD_INTERIM_DIR)

categories_to_check = [
    "END_DATE",
    "GENDER_MALE",
    "WEIGHS",
    "DIED_IN",
    "HAS_FAMILY_RELATIONSHIP",
    "IS_DEAD_ON",
    "IS_IN_CONTACT_WITH",
    "HAS_CATEGORY",
    "HAS_CONTROL_OVER",
    "IS_BORN_IN",
    "IS_OF_SIZE",
    "HAS_LATITUDE",
    "IS_PART_OF",
    "IS_OF_NATIONALITY",
    "IS_COOPERATING_WITH",
    "DEATHS_NUMBER",
    "HAS_FOR_HEIGHT",
    "INITIATED",
    "WAS_DISSOLVED_IN",
    "HAS_COLOR",
    "CREATED",
    "IS_LOCATED_IN",
    "WAS_CREATED_IN",
    "IS_AT_ODDS_WITH",
    "HAS_CONSEQUENCE",
    "HAS_FOR_LENGTH",
    "INJURED_NUMBER",
    "START_DATE",
    "STARTED_IN",
    "GENDER_FEMALE",
    "HAS_LONGITUDE",
    "RESIDES_IN",
    "HAS_FOR_WIDTH",
    "IS_BORN_ON",
    "HAS_QUANTITY",
    "OPERATES_IN",
    "IS_REGISTERED_AS",
]

mlb = MultiLabelBinarizer()
mlb.fit([categories_to_check])
logging.info(f"{mlb.classes_=}")


def load_csv(dir_or_file_path: str, index_col=None, sep=",") -> pd.DataFrame:
    if os.path.isdir(dir_or_file_path):
        all_files = glob.glob(os.path.join(dir_or_file_path, "*.csv"))
    else:
        assert dir_or_file_path.endswith(".csv")
        all_files = [dir_or_file_path]
    assert len(all_files) > 0
    return pd.concat(
        [
            pd.read_csv(filename, index_col=index_col, header=0, sep=sep)
            for filename in all_files
        ],
        axis=0,
        ignore_index=True,
    )


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            data,
            pd.DataFrame(
                mlb.transform(data[TARGET_COL]), columns=mlb.classes_, index=data.index
            ),
        ],
        axis=1,
    )  # .drop([TARGET_COL], axis=1)


def format_relations_str_to_list(labels_as_str: str) -> List[str]:
    return (
        json.loads(labels_as_str.replace("{", "[").replace("}", "]").replace("'", '"'))
        if not pd.isnull(labels_as_str)
        else []
    )


def process_csv_to_csv(in_dir_or_file_path: str, out_dir_path: str) -> None:
    """Convert labels, i.e. list of relations category, into one-hot vectors

    Args:
        in_dir_or_file_path (str): str
        out_dir_path (str): str
    """
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
    if os.path.isdir(in_dir_or_file_path):
        all_files = glob.glob(os.path.join(in_dir_or_file_path, "*.csv"))
    else:
        assert in_dir_or_file_path.endswith(".csv")
        all_files = [in_dir_or_file_path]
    for filename in (pb := tqdm(all_files)):
        pb.set_description(filename)
        preprocessed_data_filename = os.path.join(
            out_dir_path, os.path.basename(filename)
        )
        df = load_csv(filename).assign(
            **{
                TARGET_COL: lambda df: df[TARGET_COL].apply(
                    format_relations_str_to_list
                )
            }
        )
        process_data(df).to_csv(preprocessed_data_filename, sep="\t")


logging.info("## One-hot encoding the target...")
preprocessed_data_dir = os.path.join(
    METHOD_INTERIM_DIR, "one_hot_multilabel_tagged_text_dataset"
)
labeled_preprocessed_data_dir_path = os.path.join(preprocessed_data_dir, "train")
if not os.path.exists(labeled_preprocessed_data_dir_path):
    os.makedirs(labeled_preprocessed_data_dir_path)

    generated_data_dir_path = os.path.join(
        INTERIM_DIR, "multilabel_tagged_text_dataset"
    )
    assert os.path.exists(generated_data_dir_path)
    process_csv_to_csv(
        os.path.join(generated_data_dir_path, "train"),
        labeled_preprocessed_data_dir_path,
    )
else:
    logging.warning(
        f"One-hot encoding already done @ {labeled_preprocessed_data_dir_path} \n Delete the folder to redo!"
    )

logging.info("## Load preprocessed data...")
labeled_df = load_csv(labeled_preprocessed_data_dir_path, index_col=0, sep="\t")
logging.info(f"{labeled_df.shape=}")


train_csv_path = os.path.join(METHOD_INTERIM_DIR, "train_onehot.csv")
val_csv_path = os.path.join(METHOD_INTERIM_DIR, "validation_onehot.csv")
if os.path.exists(train_csv_path) and os.path.exists(val_csv_path):
    logging.info(
        f"## Loading existing Train-Val split @ {train_csv_path=} and {val_csv_path=}"
    )
    train_df = pd.read_csv(train_csv_path)
    valid_df = pd.read_csv(val_csv_path)
else:
    logging.info("## Train-Validation split...")
    # np.random.randint(10000)
    # chosen such that each class has at least 3 validation examples
    random_seed = 3508
    logging.info(f"{random_seed=}")
    train_df, valid_df = train_test_split(
        labeled_df, test_size=0.3, shuffle=True, random_state=random_seed
    )
    train_df.to_csv(train_csv_path)
    logging.info(f"Saving {train_df.shape=} @ {train_csv_path}")
    valid_df.to_csv(val_csv_path)
    logging.info(f"Saving {valid_df.shape=} @ {val_csv_path}")

df_train_with_relation = train_df[train_df[mlb.classes_].sum(axis=1) >= 1]
logging.info(f"{df_train_with_relation.shape=}")

df_train_without_relation = train_df[train_df[mlb.classes_].sum(axis=1) == 0]
logging.info(f"{df_train_without_relation.shape=}")

augmented_train_path = os.path.join(METHOD_INTERIM_DIR, "augmented_train.csv")
if os.path.exists(augmented_train_path):
    logging.info(f"## Loading existing augmented train data @ {augmented_train_path=}")
    augmented_train_df = pd.read_csv(augmented_train_path)
else:
    logging.info("## Augment train data...")
    logging.info("### Generate new texts by reducing original ones...")
    classes_to_augment = mlb.classes_
    augmented_train_df = train_df.loc[:]
    for text, group_df in tqdm(train_df.groupby("text")):
        dfs = [augmented_train_df]
        if group_df.iloc[0][classes_to_augment].sum() > 0:
            new_text1 = (
                TextToMultiLabelDataGenerator.remove_sentence_without_both_entities(
                    text, "<e1>", "<e2>"
                )
            )
            new_text2 = (
                TextToMultiLabelDataGenerator.remove_sentence_without_any_entity(
                    text, "<e1>", "<e2>"
                )
            )
            for new_text in {new_text1, new_text2}:
                if new_text:
                    dfs.append(group_df.assign(text=new_text))
        augmented_train_df = pd.concat(dfs, axis=0)
    augmented_train_df = augmented_train_df.reset_index(drop=True)
    augmented_train_df.to_csv(augmented_train_path)
    logging.info(f"Saving {augmented_train_df.shape=} @ {augmented_train_path}")

augmented_train_with_relation_df = augmented_train_df[
    augmented_train_df[mlb.classes_].sum(axis=1) >= 1
]
logging.info(f"{augmented_train_with_relation_df.shape=}")

augmented_train_without_relation_df = augmented_train_df[
    augmented_train_df[mlb.classes_].sum(axis=1) == 0
]
logging.info(f"{augmented_train_without_relation_df.shape=}")

oversampled_train_path = os.path.join(METHOD_INTERIM_DIR, "oversampled_train.csv")
if os.path.exists(oversampled_train_path):
    logging.info(
        f"## Loading existing oversampled train data @ {oversampled_train_path=}"
    )
    oversampled_train_df = pd.read_csv(oversampled_train_path)
else:
    logging.info("### Oversampling minority categories by random duplicating ...")
    OVERSAMPLING_TARGETED_SIZE = 1000
    classes_to_augment = mlb.classes_
    oversampled_train_df = train_df[train_df[mlb.classes_].sum(axis=1) == 0]
    for category in (pb := tqdm(classes_to_augment)):
        categ_train_df = augmented_train_df.query(f"{category}>0")
        pb.set_description(f"{category} previous_size={categ_train_df.shape[0]}")
        if categ_train_df.shape[0] < OVERSAMPLING_TARGETED_SIZE:
            categ_train_df = pd.concat(
                [categ_train_df]
                * int(
                    np.round(OVERSAMPLING_TARGETED_SIZE / categ_train_df.shape[0]) + 1
                ),
                axis=0,
            ).head(OVERSAMPLING_TARGETED_SIZE)
        oversampled_train_df = pd.concat([oversampled_train_df, categ_train_df], axis=0)
    oversampled_train_df.to_csv(oversampled_train_path)
    logging.info(f"Saving {oversampled_train_df.shape=} @ {oversampled_train_path}")

oversampled_train_with_relation_df = oversampled_train_df[
    oversampled_train_df[mlb.classes_].sum(axis=1) >= 1
]
logging.info(f"{oversampled_train_with_relation_df.shape=}")

oversampled_train_without_relation_df = oversampled_train_df[
    oversampled_train_df[mlb.classes_].sum(axis=1) == 0
]
logging.info(f"{oversampled_train_without_relation_df.shape=}")

logging.info("## Size and probability to appear in a batch ...")

train_category_sizes = train_df[mlb.classes_].sum(axis=0)
val_category_sizes = valid_df[mlb.classes_].sum(axis=0)
train_category_sizes_and_proba_in_batch_df = pd.DataFrame(
    {"train": train_category_sizes, "valid": val_category_sizes}
).sort_values("train", ascending=True)

BATCH_SIZE = 16
train_category_sizes_and_proba_in_batch_df = (
    train_category_sizes_and_proba_in_batch_df.assign(
        train_in_batch_proba=train_category_sizes.map(
            lambda categ_size: 1
            - np.prod(
                [
                    (train_df.shape[0] - categ_size - i) / (train_df.shape[0] - i)
                    for i in range(BATCH_SIZE)
                ]
            )
        )
    )
)

augmented_train_category_sizes = augmented_train_df[mlb.classes_].sum()
train_category_sizes_and_proba_in_batch_df = (
    train_category_sizes_and_proba_in_batch_df.assign(
        augmented_train=augmented_train_category_sizes,
        augmented_train_in_batch_proba=augmented_train_category_sizes.map(
            lambda categ_size: 1
            - np.prod(
                [
                    (augmented_train_df.shape[0] - categ_size - i)
                    / (augmented_train_df.shape[0] - i)
                    for i in range(BATCH_SIZE)
                ]
            )
        ),
    )
)

oversampled_train_category_sizes = oversampled_train_df[mlb.classes_].sum()
train_category_sizes_and_proba_in_batch_df = (
    train_category_sizes_and_proba_in_batch_df.assign(
        oversampled_train=oversampled_train_category_sizes,
        oversampled_train_in_batch_proba=oversampled_train_category_sizes.map(
            lambda categ_size: 1
            - np.prod(
                [
                    (augmented_train_df.shape[0] - categ_size - i)
                    / (augmented_train_df.shape[0] - i)
                    for i in range(BATCH_SIZE)
                ]
            )
        ),
    )
)
train_category_sizes_and_proba_in_batch_path = os.path.join(
    EDA_DIR, f"method1_train_category_sizes_and_proba_in_batch_size{BATCH_SIZE}.csv"
)
logging.info(
    f"Saving {train_category_sizes_and_proba_in_batch_df.shape=} @ {train_category_sizes_and_proba_in_batch_path}"
)
train_category_sizes_and_proba_in_batch_df.to_csv(
    train_category_sizes_and_proba_in_batch_path
)
