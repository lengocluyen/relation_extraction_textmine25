from typing import List, Set, Tuple
from dataclasses import dataclass, field
import logging
from defi_textmine_2025.method2.data.relation_and_entity_classes import (
    NO_RELATION_CLASS,
    WITH_RELATION_CLASS,
)

# from defi_textmine_2025.set_logging import config_logging
# from defi_textmine_2025.settings import LOGGING_DIR
# config_logging(f"{LOGGING_DIR}/method2/task_definition.log")

from defi_textmine_2025.data.utils import load_csv
import pandas as pd
from defi_textmine_2025.method2.models.shared_toolbox import (
    INPUT_COLUMNS,
    SUBTASK_NAME2ORDEREDLABELS,
    SUBTASK_NAME_TO_RELATIONS_TO_DROP_IN_TRAIN_DATA,
    load_fold_data,
)


@dataclass
class Task:
    name: str
    relation_types: List[str]
    entity_type_pairs: List[Tuple[str, str]]
    relations_to_drop_in_train_data: Set[str]
    identical_entities: bool = field(default=False)

    def __post_init__(self) -> None:
        assert len(self.relation_types) == len(
            set(self.relation_types)
        ), f"Types should not be duplicated in {self.relation_types=}!"

    @classmethod
    def init_predefined_sub_task(cls, sub_task_name, data: pd.DataFrame):
        assert sub_task_name in SUBTASK_NAME2ORDEREDLABELS
        relation_types = SUBTASK_NAME2ORDEREDLABELS[sub_task_name]
        entity_type_pairs = set(
            data[data[relation_types].sum(axis=1) > 0]
            .set_index(["e1_type", "e2_type"])
            .index.unique()
            .tolist()
        )
        identical_entities = sub_task_name == "subtask1"
        relations_to_drop_in_train_data = list(
            SUBTASK_NAME_TO_RELATIONS_TO_DROP_IN_TRAIN_DATA[sub_task_name]
        )
        return Task(
            sub_task_name,
            relation_types,
            entity_type_pairs,
            relations_to_drop_in_train_data,
            identical_entities,
        )

    def filter_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        result_df = pd.DataFrame()
        for e1_type, e2_type in self.entity_type_pairs:
            if self.identical_entities:
                sub_df = data.query(
                    f"e1_type=='{e1_type}' & e2_type=='{e2_type}' & e1_id==e2_id"
                )
            else:
                sub_df = data.query(f"e1_type=='{e1_type}' & e2_type=='{e2_type}'")
            result_df = pd.concat([result_df, sub_df], axis=0)
        return result_df

    def filter_prediction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.filter_rows(data)
        cols_to_drop = [col for col in data.columns if col not in INPUT_COLUMNS]
        return df.drop(cols_to_drop, axis=1)

    def filter_train_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.filter_rows(data).query(
            f"relations not in {self.relations_to_drop_in_train_data}"
        )
        cols_to_drop = [
            col
            for col in data.columns
            if col not in self.relation_types + INPUT_COLUMNS
        ]
        return (
            df.drop(cols_to_drop, axis=1)
            .assign(
                **{
                    WITH_RELATION_CLASS: (
                        df[self.relation_types].sum(axis=1) > 0
                    ).astype(float)
                }
            )
            .assign(**{NO_RELATION_CLASS: lambda _df: 1.0 - _df[WITH_RELATION_CLASS]})
        )


if __name__ == "__main__":
    train_df, val_df = load_fold_data(1)
    labeled_df = pd.concat([train_df, val_df], axis=0)
    test_df = load_csv(
        "data/defi-text-mine-2025/interim/reduced_text_w_entity_bracket/test"
    )

    logging.info(
        f"{train_df.shape=}, {val_df.shape=}, {labeled_df.shape=} {test_df.shape=}, {train_df.columns=}"
    )
    name2task = {}
    for name, rel_types in SUBTASK_NAME2ORDEREDLABELS.items():
        subtask: Task = Task.init_predefined_sub_task(name, labeled_df)
        name2task[name] = subtask
        for split_name, split_df in zip(
            ["train", "val", "test"], [train_df, val_df, test_df]
        ):
            logging.warning(f"{name=} -> {split_name}...")
            if split_name in ["train", "val"]:
                filtered_df = subtask.filter_train_data(split_df)
                logging.info(
                    f"\tfor train: {filtered_df.shape=}, {filtered_df.columns=}"
                )
                logging.info(
                    f"\t\t{filtered_df[filtered_df[WITH_RELATION_CLASS].astype(bool)].shape=}"
                )
                logging.info(
                    f"\t\t{filtered_df[filtered_df[NO_RELATION_CLASS].astype(bool)].shape=}"
                )
            filtered_df = subtask.filter_prediction_data(split_df)
            logging.info(f"\tfor pred: {filtered_df.shape=}, {filtered_df.columns=}")
        # break
    for name1 in name2task:
        for name2 in name2task:
            if name1 == name2:
                continue
            logging.warning(
                f"entity types intesection({name1}, {name2})="
                f"{name2task[name1].entity_type_pairs.intersection(name2task[name2].entity_type_pairs)}"
            )
# python -m defi_textmine_2025.method2.models.task_definition
