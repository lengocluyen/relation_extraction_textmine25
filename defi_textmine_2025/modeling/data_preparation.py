# Entity classes and categories (replace with your actual lists)
import glob
import json
import logging
import os
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from tqdm import tqdm

from defi_textmine_2025.data.utils import INTERIM_DIR, TARGET_COL


entity_classes = {
    "TERRORIST_OR_CRIMINAL",
    "LASTNAME",
    "LENGTH",
    "NATURAL_CAUSES_DEATH",
    "COLOR",
    "STRIKE",
    "DRUG_OPERATION",
    "HEIGHT",
    "INTERGOVERNMENTAL_ORGANISATION",
    "TRAFFICKING",
    "NON_MILITARY_GOVERNMENT_ORGANISATION",
    "TIME_MIN",
    "DEMONSTRATION",
    "TIME_EXACT",
    "FIRE",
    "QUANTITY_MIN",
    "MATERIEL",
    "GATHERING",
    "PLACE",
    "CRIMINAL_ARREST",
    "CBRN_EVENT",
    "ECONOMICAL_CRISIS",
    "ACCIDENT",
    "LONGITUDE",
    "BOMBING",
    "MATERIAL_REFERENCE",
    "WIDTH",
    "FIRSTNAME",
    "MILITARY_ORGANISATION",
    "CIVILIAN",
    "QUANTITY_MAX",
    "CATEGORY",
    "POLITICAL_VIOLENCE",
    "EPIDEMIC",
    "TIME_MAX",
    "TIME_FUZZY",
    "NATURAL_EVENT",
    "SUICIDE",
    "CIVIL_WAR_OUTBREAK",
    "POLLUTION",
    "ILLEGAL_CIVIL_DEMONSTRATION",
    "NATIONALITY",
    "GROUP_OF_INDIVIDUALS",
    "QUANTITY_FUZZY",
    "RIOT",
    "WEIGHT",
    "THEFT",
    "MILITARY",
    "NON_GOVERNMENTAL_ORGANISATION",
    "LATITUDE",
    "COUP_D_ETAT",
    "ELECTION",
    "HOOLIGANISM_TROUBLEMAKING",
    "QUANTITY_EXACT",
    "AGITATING_TROUBLE_MAKING",
}
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

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit([categories_to_check])
logging.info(f"{mlb.classes_=}")

# Directories
generated_data_dir_path = os.path.join(INTERIM_DIR, "multilabel_tagged_text_dataset")
assert os.path.exists(generated_data_dir_path)

preprocessed_data_dir = os.path.join(
    INTERIM_DIR, "one_hot_multilabel_tagged_text_dataset"
)
labeled_preprocessed_data_dir_path = os.path.join(preprocessed_data_dir, "train")


# Functions to load and process data
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
    )


def format_relations_str_to_list(labels_as_str: str) -> List[str]:
    return (
        json.loads(labels_as_str.replace("{", "[").replace("}", "]").replace("'", '"'))
        if not pd.isnull(labels_as_str)
        else []
    )


def process_csv_to_csv(in_dir_or_file_path: str, out_dir_path: str) -> None:
    if os.path.isdir(in_dir_or_file_path):
        all_files = glob.glob(os.path.join(in_dir_or_file_path, "*.csv"))
    else:
        assert in_dir_or_file_path.endswith(".csv")
        all_files = [in_dir_or_file_path]
    os.makedirs(out_dir_path, exist_ok=True)
    for filename in tqdm(all_files, desc="Processing CSV files"):
        preprocessed_data_filename = os.path.join(
            out_dir_path, os.path.basename(filename)
        )
        data = load_csv(filename).assign(
            **{
                TARGET_COL: lambda df: df[TARGET_COL].apply(
                    format_relations_str_to_list
                )
            }
        )
        processed_data = process_data(data)
        processed_data.to_csv(preprocessed_data_filename, sep="\t")


# Load and preprocess data
if not os.path.exists(labeled_preprocessed_data_dir_path):
    process_csv_to_csv(
        os.path.join(generated_data_dir_path, "train"),
        labeled_preprocessed_data_dir_path,
    )
else:
    logging.warning(
        "One-Hot-label dataset dir already exists at:"
        f" {labeled_preprocessed_data_dir_path}."
        "\n Delete this directory to recompute one-hot dataset."
    )
labeled_df = load_csv(labeled_preprocessed_data_dir_path, index_col=0, sep="\t")
df_train, df_valid = train_test_split(
    labeled_df, test_size=0.2, shuffle=True, random_state=42
)
