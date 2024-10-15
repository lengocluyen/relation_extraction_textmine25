from concurrent.futures import ProcessPoolExecutor
import os
import logging
from typing import List, Tuple

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s|%(levelname)s|%(module)s.py:%(lineno)s] %(message)s",
    datefmt="%H:%M:%S",
)

from defi_textmine_2025.method2.data.relation_and_entity_classes import (
    FULL_TAGGED_TEXT_COL,
    REDUCED_TAGGED_TEXT_COL,
)

import sys
import pandas as pd
import stanza
from stanza import DownloadMethod
import re

lang = "fr"

E1_TAG_PATTERN = re.compile(r".*[\{\}<>].*")
E2_TAG_PATTERN = re.compile(r".*[\[\]].*")


def reduce_to_text_of_interest_csv_to_csv(in_csv_path: str, out_csv_path: str) -> None:
    """Reduce all the text in in_csv_path

    Args:
        in_csv_path (str): input csv
        out_csv_path (str): resulting csv i.e. originla data in in_csv
            with the additional column `reduced_text`
    """
    nlp = stanza.Pipeline(
        lang=lang,
        processors="tokenize",
        download_method=DownloadMethod.REUSE_RESOURCES,
    )

    def reduce_to_text_of_interest(tagged_text: str) -> str:
        """Reduce the entity-tagged text to sentences that has, resp. in priority,
        2 entities or a at least an entity

        Args:
            tagged_text (str): text containing entities that are tagged following the
                patterns E1_TAG_PATTERN and E2_TAG_PATTERN

        Returns:
            str: the reduced text with only the sentences of interest.
        """
        at_least_one_entity_sentences = []
        two_entities_sentences = []
        doc = nlp(tagged_text)
        for i, sentence in enumerate(doc.sentences):
            if E1_TAG_PATTERN.match(sentence.text) and E2_TAG_PATTERN.match(
                sentence.text
            ):
                two_entities_sentences.append(sentence.text)
            if E1_TAG_PATTERN.match(sentence.text) or E2_TAG_PATTERN.match(
                sentence.text
            ):
                at_least_one_entity_sentences.append(sentence.text)
        return " ".join(
            two_entities_sentences
            if len(two_entities_sentences) > 0
            else at_least_one_entity_sentences
        )

    logging.warning(f"Start {in_csv_path} -> {out_csv_path} ...")
    out_dir = os.path.dirname(out_csv_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # load data
    df = pd.read_csv(in_csv_path)
    # apply the processing
    df[REDUCED_TAGGED_TEXT_COL] = df[FULL_TAGGED_TEXT_COL].apply(
        reduce_to_text_of_interest
    )
    # save data
    df.to_csv(out_csv_path)
    logging.warning(f"Finish {in_csv_path} -> {out_csv_path}")


def get_matching_in_out_csv_path_pairs(
    in_dir: str, out_dir: str
) -> List[Tuple[str, str]]:
    """For each csv file in in_dir, compute the mathing csv file path in out_dir
    forllowing the same path tree in in_dir

    Args:
        in_dir (str): input csv root dir
        out_dir (str): output csv root dir

    Returns:
        List[Tuple[str, str]]: pairs of matching input-output csv paths
    """
    matching_pairs = []
    for _, sub_dirs, filenames in os.walk(in_dir):
        # print(root_dir, sub_dirs, filenames)
        for filename in filenames:
            in_csv_path = os.path.join(in_dir, filename)
            if os.path.exists(in_csv_path) and filename.endswith(".csv"):
                matching_pairs.append((in_csv_path, os.path.join(out_dir, filename)))
        for sub_dir in sub_dirs:
            matching_pairs.extend(
                get_matching_in_out_csv_path_pairs(
                    os.path.join(in_dir, sub_dir), os.path.join(out_dir, sub_dir)
                )
            )
    return matching_pairs


def helper(io_csv_path_pair: Tuple[str, str]):
    """To run reduce_to_text_of_interest_csv_to_csv in parallel using
    multiprocessing

    Args:
        io_csv_path_pair (Tuple[str, str]): pair of matching input-output csv paths
    """
    reduce_to_text_of_interest_csv_to_csv(io_csv_path_pair[0], io_csv_path_pair[1])


if __name__ == "__main__":
    n_workers, in_path, out_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    if (
        os.path.isfile(in_path)
        and in_path.endswith(".csv")
        and out_path.endswith(".csv")
    ):
        reduce_to_text_of_interest_csv_to_csv(in_path, out_path)
    elif os.path.isdir(in_path):
        in_out_path_pairs = get_matching_in_out_csv_path_pairs(in_path, out_path)
        logging.info(f"{len(in_out_path_pairs)=}")
        executor = ProcessPoolExecutor(max_workers=n_workers)
        for _ in executor.map(helper, in_out_path_pairs):
            pass

# python -m defi_textmine_2025.method2.data.reduce_texts 1 data/defi-text-mine-2025/interim/entity_bracket_tagging_dataset/train/2431.csv data/defi-text-mine-2025/logs/2431-reduced_text.csv
# python -m defi_textmine_2025.method2.data.reduce_texts 7 data/defi-text-mine-2025/interim/entity_bracket_tagging_dataset/test/ data/defi-text-mine-2025/interim/reduced_text_w_entity_bracket/test/
# python -m defi_textmine_2025.method2.data.reduce_texts 14 data/defi-text-mine-2025/interim/entity_bracket_tagging_dataset/train/ data/defi-text-mine-2025/interim/reduced_text_w_entity_bracket/train/
