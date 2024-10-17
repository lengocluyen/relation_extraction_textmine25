import json
import logging

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from defi_textmine_2025.method2.data.relation_and_entity_classes import (
    NO_RELATION_CLASS,
    RELATION_CLASSES,
)


all_target_columns = [NO_RELATION_CLASS] + RELATION_CLASSES
onehot_encoder = MultiLabelBinarizer().fit([all_target_columns])
logging.debug(f"{onehot_encoder.classes_=}")

ORDERED_CLASSES = onehot_encoder.classes_.tolist()
logging.info(f"{ORDERED_CLASSES=}")


def format_relations_str_to_list(labels_as_str: str) -> list[str]:
    return (
        json.loads(labels_as_str.replace("{", "[").replace("}", "]").replace("'", '"'))
        if not pd.isnull(labels_as_str)
        else [NO_RELATION_CLASS]
    )


def encode_target_to_onehot(
    data: pd.DataFrame,
    labels_as_list_column: str,
    onehot_label_encoder: MultiLabelBinarizer,
) -> pd.DataFrame:
    assert hasattr(
        onehot_label_encoder, "classes_"
    ), "Fit the onehot_label_encoder` first!"
    onehot_columns = onehot_label_encoder.classes_
    logging.info(f"{onehot_columns=}")
    data.loc[:, onehot_columns] = onehot_label_encoder.transform(
        data[labels_as_list_column].apply(format_relations_str_to_list)
    ).astype(float)
    return data
