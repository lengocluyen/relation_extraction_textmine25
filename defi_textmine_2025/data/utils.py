from typing import Any, Dict, List, Tuple
import os
import json
import pandas as pd

TARGET_COL = "relations"
INPUT_COLS = ["text", "entities"]
ID_COL = "id"

CHALLENGE_ID = "defi-text-mine-2025"
CHALLENGE_DIR = f"data/{CHALLENGE_ID}"
assert os.path.exists(CHALLENGE_DIR), f"path not found: {CHALLENGE_DIR=}"
train_raw_data_path = os.path.join(CHALLENGE_DIR, "raw", "train.csv")
test_raw_data_path = os.path.join(CHALLENGE_DIR, "raw", "test_01-07-2024.csv")
sample_submission_path = os.path.join(CHALLENGE_DIR, "raw", "sample_submission.csv")

assert os.path.exists(train_raw_data_path)
assert os.path.exists(test_raw_data_path)
assert os.path.exists(sample_submission_path)

EDA_DIR = os.path.join(CHALLENGE_DIR, "eda")
INTERIM_DIR = os.path.join(CHALLENGE_DIR, "interim")
MODELS_DIR = os.path.join(CHALLENGE_DIR, "models")
OUTPUT_DIR = os.path.join(CHALLENGE_DIR, "output")
for dir_path in [EDA_DIR, INTERIM_DIR, MODELS_DIR, OUTPUT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

submission_path = os.path.join(OUTPUT_DIR, "submission.csv")


# def clean_text(text: str) -> str:
#     RAW_STR_TO_CLEAN_STR = {
#         "\n": "",
#         "‘’": '"',
#         "’’": '"',
#         "”": '"',
#         "“": '"',
#         "’": '"',
#         " ": " ",
#     }
#     for raw_str, clean_str in RAW_STR_TO_CLEAN_STR.items():
#         text = re.sub(raw_str, clean_str, text)
#     return text.strip()


def load_labeled_raw_data() -> pd.DataFrame:
    return pd.read_csv(train_raw_data_path, index_col=ID_COL)


def load_test_raw_data() -> pd.DataFrame:
    return pd.read_csv(test_raw_data_path, index_col=ID_COL)


def clean_raw_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.assign(
        **{
            # don't clean text since it is recommended to give the raw text to
            #  BERT-base models
            # "text": lambda df: df.text.apply(clean_text),
            "entities": lambda df: df.entities.apply(json.loads),
            TARGET_COL: lambda df: (
                df[TARGET_COL].apply(json.loads)
                if TARGET_COL in df.columns
                else None  # pd.NA
            ),
        }
    )


def print_value_types(data: pd.DataFrame) -> None:
    for col in data.columns:
        value = data.iloc[0][col]
        col_type = type(value)
        if col_type is list:
            print(
                col,
                "[ ",
                (
                    type(value[0])
                    if type(value[0]) is list
                    else [type(e) for e in value[0]]
                ),
                " ]",
            )
        else:
            print(col, col_type)


def save_data(data: pd.DataFrame, csv_path: str, with_index: bool = True) -> None:
    """save data into a file at file_path

    Args:
        data (pd.DataFrame): data to save
        file_path (str): destination file
        with_index(bool): whether to save the index too
    """
    dest_dir_path = os.path.dirname(csv_path)
    if not os.path.exists(dest_dir_path):
        os.makedirs(dest_dir_path)
    data.to_csv(csv_path, index=with_index)


def convert_text_to_entity_spans(
    text: str, text_entities: List[Dict[str, Any]]
) -> List[Tuple[str, int]]:
    """convert a text into a list of spans, more practical for problem reformulation
      and data augmentation

    Args:
        text (str): original text
        text_entities (List[Dict[str, Any]]): entities mentioned into the text, as given
          in the dataset i.e.
            [
                {
                    id:0,
                    mentions:[{"value": "feu", "start": 483, "end": 486},...],
                    "type": "CIVILIAN"
                },
                ...
            ]

    Returns:
        List[Tuple[str, int]]: list of spans in the format:
            [(text_span, entity_id)] e.g. [(..., None), ..., (feu, 0), ...]
            entity_id is None for NOTA spans (none of the above) i.e. spans
            that are not entity mentions.
    """
    start2mentions = {
        m["start"]: m | {"id": e["id"], "type": e["type"]}
        for e in text_entities
        for m in e["mentions"]
    }
    # order text spans with entity id (None for not entity)
    span_and_entity_id_pairs = []
    next_start = 0
    for start in sorted(list(start2mentions.keys())):
        e_id = start2mentions[start]["id"]
        if next_start != start:
            span_and_entity_id_pairs.append((text[next_start:start], None))
        next_start = start2mentions[start]["end"]
        span_and_entity_id_pairs.append((text[start:next_start], e_id))
    if next_start < len(text):
        span_and_entity_id_pairs.append((text[next_start:], None))
    return span_and_entity_id_pairs
