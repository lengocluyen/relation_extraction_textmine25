from collections.abc import Generator
from dataclasses import dataclass, field
import json
import logging
import os
from typing import Any, Dict, List, Tuple
import re

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
            # don't clean text since it is recommended to give the raw text to BERT-base models
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


@dataclass
class TextToMultiLabelDataGenerator:
    excluded_entity_pairs: List[Tuple[str, str]] = field(default_factory=list)
    first_entity_tag: str = field(default="e1")
    second_entity_tag: str = field(default="e2")
    text_col: str = field(default="text")
    text_index_col: str = field(default="text_index")

    def __post_init__(self):
        logging.info(f"{self.excluded_entity_pairs=}")
        assert self.first_entity_tag != self.second_entity_tag

    def tag_entities(
        self, text: str, x: Dict[str, Any], y: Dict[str, Any]
    ) -> pd.DataFrame:
        """Mark the 2 given entities as the are the argument of a possible ordered
          relation to generate the 2 possible tagged texts where:

        1. x is the first entity of the relations, and y the second
        2. y is the first entity of the relations, and x the second

        Args:
            text (str): the text as stated in the original dataset
            x (Dict[str, Any]): an entity mentioned in the text as annotated
              in the original dataset e.g.
              {
                "id": 0,
                "mentions": [
                    {"value": "accident", "start": 70, "end": 78},
                    {"value": "accident de circulation", "start": 100, "end": 123}
                ]
              }
            y (Dict[str, Any]): an entity mentioned in the text as annotated
              in the original dataset; y is different from x.

        Returns:
            pd.DataFrame: with two columns with respectively the ids of the first and
              second entities in the marked text, and a last column with the marked text
        """
        logging.debug("starting")
        start2mentions = {
            m["start"]: m | {"id": e["id"], "type": e["type"]}
            for e in [x, y]
            for m in e["mentions"]
        }
        entities_ids = {x["id"], y["id"]}
        first_entity_id_to_tagged_text = {_id: "" for _id in entities_ids}
        # order text spans with entity id (None for not entity)
        id_start_end_pairs = []
        next_start = 0
        for start in sorted(list(start2mentions.keys())):
            e_id = start2mentions[start]["id"]
            if next_start != start:
                id_start_end_pairs.append((None, (next_start, start)))
            next_start = start2mentions[start]["end"]
            id_start_end_pairs.append((e_id, (start, next_start)))
        if next_start < len(text):
            id_start_end_pairs.append((None, (next_start, len(text))))
        # build the tagged texts
        for first_e_id in entities_ids:
            tagged_text = ""
            for e_id, (start, end) in id_start_end_pairs:
                entity_span = text[start:end]
                if e_id is not None:
                    tag = (
                        self.first_entity_tag
                        if e_id == first_e_id
                        else self.second_entity_tag
                    )
                    entity_type = start2mentions[start]["type"]
                    tagged_text += "<{}><{}>{}</{}>".format(
                        tag, entity_type, entity_span, tag
                    )
                else:
                    tagged_text += entity_span
            first_entity_id_to_tagged_text[first_e_id] = tagged_text
        # filter only possible
        rows = []
        if (x["type"], y["type"]) not in self.excluded_entity_pairs:
            rows.append([x["id"], y["id"], first_entity_id_to_tagged_text[x["id"]]])
        if (y["type"], x["type"]) not in self.excluded_entity_pairs and x["id"] != y[
            "id"
        ]:
            rows.append([y["id"], x["id"], first_entity_id_to_tagged_text[y["id"]]])
        logging.debug("ending")
        return (
            pd.DataFrame(
                rows,
                columns=[
                    self.first_entity_tag,
                    self.second_entity_tag,
                    self.text_col,
                ],
            )
            if len(rows) > 0
            else pd.DataFrame()
        )

    def convert_relations_to_dataframe(
        self, relations: List[Tuple[int, str, int]]
    ) -> pd.DataFrame:
        """convert all the relations labeled in a text into a dataframe

        Args:
            relations (List[Tuple[int, str, int]]): relations of labeled in a text

        Returns:
            pd.DataFrame: resulting dataframe with 3 columns e1, e2, relations
              (i.e. the set of relations between e1 and e2) and a line per pair of
              entities e1 and e2 that are into relation.
        """
        # logging.info("starting")
        columns = [
            self.first_entity_tag,
            self.second_entity_tag,
            TARGET_COL,
        ]
        if not relations:
            return pd.DataFrame(columns=columns)
        entity_pair_to_relations = {}
        for e1, r, e2 in relations:
            entity_pair = (e1, e2)
            if entity_pair not in entity_pair_to_relations:
                entity_pair_to_relations[entity_pair] = set()
            entity_pair_to_relations[entity_pair].add(r)
        logging.debug("ending")
        return pd.DataFrame(
            [
                [e1, e2, list(e1_e2_relations)]
                for (e1, e2), e1_e2_relations in entity_pair_to_relations.items()
            ],
            columns=columns,
        )

    def tag_all_possible_entity_pairs(
        self,
        text_index: int,
        text: str,
        text_entities: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """apply self.tag_entities() for each entity pair in text_entities

        Args:
            text_index (int): the text index in the original dataset
            text (str): the text as stated in the original dataset
            text_entities (List[Dict[str, Any]]): the entity mentioned in the text
              as given in the original dataset

        Returns:
            pd.DataFrame: the resulting dataset
        """
        logging.debug("starting")
        entity_pair_to_text_df = pd.DataFrame()
        for i in range(len(text_entities)):
            # for j in range(len(text_entities)):
            for j in range(i + 1):
                ij_entity_pair_to_text_df = self.tag_entities(
                    text, text_entities[i], text_entities[j]
                )
                entity_pair_to_text_df = pd.concat(
                    [entity_pair_to_text_df, ij_entity_pair_to_text_df], axis=0
                )
        new_columns = [self.text_index_col] + entity_pair_to_text_df.columns.to_list()
        # logging.info(f"{new_columns=}")
        entity_pair_to_text_df = entity_pair_to_text_df.assign(
            **{self.text_index_col: text_index}
        ).reset_index(drop=True)[new_columns]

        logging.debug("ending")
        return entity_pair_to_text_df

    def convert(
        self,
        text_index: int,
        text: str,
        text_entities: List[Dict[str, Any]],
        text_relations: List[Tuple[int, str, int]],
    ) -> pd.DataFrame:
        """Convert an entry (a row) of the original dataset into a dataframe with:

        - a row correspond to a pair of the given entities; each possible ordered
            pair of entities has a row i.e. permutating the entities in an already
            processed pair, will enable the generation of another row.
        - a column for the text in which are tagged the mentions of the entities of
          the corresponding pair.
        - a column for the text index
        - a column for the first entity
        - a column for the second entity
        - a column for the list of relations of the first entity to the second

        Args:
            text_index (int): the text index in the original dataset
            text (str): the text as stated in the original dataset
            text_entities (List[Dict[str, Any]]): the entity mentioned in the text
              as given in the original dataset
            text_relations (List[Tuple[int, str, int]]): the labels given
              in the original dataset for the text

        Returns:
            pd.DataFrame: the resulting dataset
        """
        logging.debug("starting")
        entity_pair_to_text_df = self.tag_all_possible_entity_pairs(
            text_index, text, text_entities
        )
        entity_pair_to_relations_df = self.convert_relations_to_dataframe(
            text_relations
        )
        logging.debug("ending")
        return entity_pair_to_text_df.join(
            entity_pair_to_relations_df.set_index(
                [
                    self.first_entity_tag,
                    self.second_entity_tag,
                ]
            ),
            on=[
                self.first_entity_tag,
                self.second_entity_tag,
            ],
        )

    def generate_row_multilabel_data(
        self, clean_df: pd.DataFrame, only_w_relation: bool = True
    ) -> Generator[pd.DataFrame]:
        """yields a dataframe per text with all the generated data

        Args:
            clean_df (pd.DataFrame): original dataset with entities
                and relations as list, and not str as original
            only_w_relation (bool): if True, just the tagged texts
                with non null relations (labels) will be generated

        Yields:
            Generator[pd.DataFrame]: generated dataset from a sentence annotation.
        """
        for text_index in clean_df.index:
            generated_df = self.convert(
                text_index,
                clean_df.loc[text_index].text,
                clean_df.loc[text_index].entities,
                clean_df.loc[text_index].relations,
            )
            if only_w_relation:
                generated_df = generated_df[~pd.isnull(generated_df[TARGET_COL])]
            yield generated_df
