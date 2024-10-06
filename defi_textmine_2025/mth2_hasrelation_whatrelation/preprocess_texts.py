from concurrent.futures import ProcessPoolExecutor
import os
import logging
from typing import List, Tuple

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s|%(levelname)s|%(module)s.py:%(lineno)s] %(message)s",
    datefmt="%H:%M:%S",
)
import sys
import pandas as pd
import stanza
from stanza import DownloadMethod
import re

lang = "fr"

e1_pattern = re.compile(r".*[\{\}<>].*")
e2_pattern = re.compile(r".*[\[\]].*")


def reduce_to_text_of_interest_csv_to_csv(in_csv_path: str, out_csv_path: str):
    nlp = stanza.Pipeline(
        lang=lang,
        processors="tokenize",
        download_method=DownloadMethod.REUSE_RESOURCES,
    )

    def reduce_to_text_of_interest(tagged_text: str) -> str:
        final_text_sentences = []
        doc = nlp(tagged_text)
        for i, sentence in enumerate(doc.sentences):
            if e1_pattern.match(sentence.text) and e2_pattern.match(sentence.text):
                final_text_sentences = [sentence.text]
                break
            if e1_pattern.match(sentence.text) or e2_pattern.match(sentence.text):
                final_text_sentences.append(sentence.text)
        return " ".join(final_text_sentences)

    logging.warning(f"Start {in_csv_path} -> {out_csv_path} ...")
    out_dir = os.path.dirname(out_csv_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # load data
    df = pd.read_csv(in_csv_path)
    # apply the processing
    df["reduced_text"] = df["text"].apply(reduce_to_text_of_interest)
    # save data
    df.to_csv(out_csv_path)
    logging.warning(f"Finish {in_csv_path} -> {out_csv_path}")


def get_matching_in_out_csv_path_pairs(
    in_dir: str, out_dir: str
) -> List[Tuple[str, str]]:
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


def helper(path_pair: Tuple[str, str]):
    reduce_to_text_of_interest_csv_to_csv(path_pair[0], path_pair[1])


if __name__ == "__main__":
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    # reduce_to_text_of_interest_csv_to_csv()
    in_out_path_pairs = get_matching_in_out_csv_path_pairs(in_dir, out_dir)
    logging.info(f"{len(in_out_path_pairs)=}")
    executor = ProcessPoolExecutor(max_workers=20)
    for _ in executor.map(helper, in_out_path_pairs):
        pass


# python -m defi_textmine_2025.mth2_hasrelation_whatrelation.preprocess_texts data/defi-text-mine-2025/interim/entity_bracket_tagging_dataset/test/13.csv data/defi-text-mine-2025/interim/reduced_text_w_entity_bracket/test/13.csv
# python -m defi_textmine_2025.mth2_hasrelation_whatrelation.preprocess_texts data/defi-text-mine-2025/interim/entity_bracket_tagging_dataset/test/ data/defi-text-mine-2025/interim/reduced_text_w_entity_bracket/test/
# python -m defi_textmine_2025.mth2_hasrelation_whatrelation.preprocess_texts data/defi-text-mine-2025/interim/entity_bracket_tagging_dataset/ data/defi-text-mine-2025/interim/reduced_text_w_entity_bracket/
