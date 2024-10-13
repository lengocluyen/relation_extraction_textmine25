import os
import logging

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s|%(levelname)s|%(module)s.py:%(lineno)s] %(message)s",
    datefmt="%H:%M:%S",
)

import pandas as pd
from defi_textmine_2025.data.problem_formulation import (
    EntityBracketTaggingDataGenerator,
)
from defi_textmine_2025.data.utils import (
    INTERIM_DIR,
    clean_raw_dataset,
    load_labeled_raw_data,
    load_test_raw_data,
    save_data,
)

# output_dir_path: input of defi_textmine_2025/method2/data/reduce_texts.py
output_dir_path = os.path.join(INTERIM_DIR, "entity_bracket_tagging_dataset2")
assert not os.path.exists(
    output_dir_path
), f"You must delete the output folder first {output_dir_path}!"

logging.info("Load train data")
labeled_raw_df = load_labeled_raw_data()
logging.info("Load test data")
test_raw_df = load_test_raw_data()

logging.info("Clean train data")
labeled_clean_df = clean_raw_dataset(labeled_raw_df)
logging.info("Clean test data")
test_clean_df = clean_raw_dataset(test_raw_df)

logging.info("Compute entity_relation_cat_df...")
entity_relation_cat_df = pd.concat(
    [
        pd.DataFrame(
            [
                [
                    text_idx,
                    e1_id,
                    e2_id,
                    text_entities[e1_id]["type"],
                    r_cat,
                    text_entities[e2_id]["type"],
                ]
                for e1_id, r_cat, e2_id in text_relations
            ],
            columns=["text_id", "e1_id", "e2_id", "e1_cat", "r_cat", "e2_cat"],
        )
        for text_idx, text, text_entities, text_relations in labeled_clean_df.reset_index().values
    ],
    axis=0,
)
logging.info(f"{entity_relation_cat_df.shape=}\n{entity_relation_cat_df.head()}")

entity_cat_pair_in_relation_df = entity_relation_cat_df[
    ["e1_cat", "e2_cat", "e1_id", "e2_id"]
]
entity_cat_pair_in_relation_df

logging.info("Compute entity_type_pairs_in_binary_relation...")
entity_type_pairs_in_binary_relation = set(
    entity_cat_pair_in_relation_df.query("e1_id != e2_id")[["e1_cat", "e2_cat"]]
    .set_index(["e1_cat", "e2_cat"])
    .index.to_list()
)
logging.info(
    f"{len(entity_type_pairs_in_binary_relation)} "
    f"{entity_type_pairs_in_binary_relation=}"
)

logging.info("Compute entity_types_in_unary_relation...")
entity_types_in_unary_relation = set(
    entity_cat_pair_in_relation_df.query("e1_id == e2_id")[["e1_cat"]]
    .set_index(["e1_cat"])
    .index.to_list()
)
logging.info(f"{len(entity_types_in_unary_relation)} {entity_types_in_unary_relation=}")

data_generator = EntityBracketTaggingDataGenerator(
    allowed_binary_relation_entity_type_pairs=entity_type_pairs_in_binary_relation,
    allowed_unary_relation_entity_types=entity_types_in_unary_relation,
)

for split_name, clean_df in zip(
    ["test", "train"],
    [test_clean_df, labeled_clean_df],
):
    dest_dir_path = os.path.join(output_dir_path, split_name)
    for multilabel_data in (
        pb := tqdm(
            data_generator.generate_row_multilabel_data(
                clean_df, only_w_relation=False
            ),
            total=clean_df.shape[0],
            desc=f"{dest_dir_path} <- ",
        )
    ):
        text_index = multilabel_data.iloc[0][data_generator.text_index_col]
        dest_csv_file = os.path.join(dest_dir_path, f"{text_index}.csv")
        pb.set_description(f"{dest_csv_file} <-")
        save_data(multilabel_data, dest_csv_file, False)


# python -m defi_textmine_2025.method2.data.tag_entity_in_texts
