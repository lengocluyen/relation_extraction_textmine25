"""
python -m defi_textmine_2025.method1.predict
"""

import glob
import os
from datetime import datetime
import pandas as pd
from defi_textmine_2025.data.utils import (
    LOGGING_DIR,
    INTERIM_DIR,
    MODELS_DIR,
    TARGET_COL,
    submission_path,
    load_test_raw_data,
)
import logging

log_file_path = os.path.join(
    LOGGING_DIR,
    f'{datetime.now().strftime("method1-evaluate-%Y%m%dT%H%M%S")}.log',
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
import torch
from transformers import CamembertTokenizer, CamembertModel

from defi_textmine_2025.bert_dataset_and_models import (
    CustomDataset,
    LinearHeadBertBasedModel,
    get_predictions,
)


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


PRETRAINED_EMBEDDING_CHECKPOINT = "camembert/camembert-base"
EMBEDDING_SIZE = 768  # 768 # 1024
TASK_NAME = "multilabel_tagged_text"
BATCH_SIZE = 32

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

mlb = MultiLabelBinarizer()
mlb.fit([categories_to_check])
logging.info(f"{mlb.classes_=}")
target_list = mlb.classes_
logging.info(f"{len(target_list)} categories = {target_list}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"{device=}")

model_dir_path = os.path.join(
    MODELS_DIR, "method1", f"finetuned-{PRETRAINED_EMBEDDING_CHECKPOINT}"
)
model_dict_state_path = os.path.join(model_dir_path, "mutilabel_model_state.bin")
assert os.path.exists(
    model_dict_state_path
), f"Model state file unfound at {model_dict_state_path}"

logging.info("## Loading test data ...")
generated_data_dir_path = os.path.join(INTERIM_DIR, "multilabel_tagged_text_dataset")
test_csv_path = os.path.join(generated_data_dir_path, "test")
assert os.path.exists(test_csv_path), f"Unfound {test_csv_path=}"
logging.info(f"### Loading {test_csv_path=} ...")
test_df = load_csv(test_csv_path)

logging.info("## init the tokenizer with special tokens ...")
# Hyperparameters
MAX_LEN = 300
# tokenizer = BertTokenizer.from_pretrained(BASE_CHECKPOINT)
tokenizer = CamembertTokenizer.from_pretrained(PRETRAINED_EMBEDDING_CHECKPOINT)
task_special_tokens = ["<e1>", "</e1>", "<e2>", "</e2>"] + [
    f"<{entity_class}>" for entity_class in entity_classes
]
# add special tokens to the tokenizer
logging.info(f"OLD {len(tokenizer)=}")
num_added_tokens = tokenizer.add_tokens(task_special_tokens, special_tokens=True)
logging.info(f"{num_added_tokens=}")
logging.info(f"NEW {len(tokenizer)=}")

logging.info("### Init train and validation datasets...")
test_dataset = CustomDataset(
    test_df.drop(TARGET_COL, axis=1).assign(
        **{cat: [0] * test_df.shape[0] for cat in target_list}
    ),
    tokenizer,
    MAX_LEN,
    target_list,
)
logging.info("#### testing the dataset")
logging.info(next(iter(test_dataset)))

logging.info("## Create test data loaders...")
test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=21
)

logging.info("## Init the model and Load its state ...")
model = LinearHeadBertBasedModel(
    tokenizer=tokenizer,
    embedding_model=CamembertModel.from_pretrained(
        PRETRAINED_EMBEDDING_CHECKPOINT, return_dict=True
    ),
    embedding_size=EMBEDDING_SIZE,
    hidden_dim=128,
    n_classes=len(target_list),
)
model.load_state_dict(torch.load(model_dict_state_path))
# if model is not in GPU, then load it to GPU
if not next(model.parameters()).is_cuda:
    model = model.to(device)
logging.info(f"Loaded {model=}")

logging.info("## Getting predictions ...")
titles, predictions, prediction_probs, target_values = get_predictions(
    model, test_data_loader, device
)

logging.info("## Prepare submission...")
ml_labeled_test_df = pd.concat(
    [
        test_df.drop(TARGET_COL, axis=1),
        pd.Series(
            mlb.inverse_transform(
                pd.DataFrame(
                    predictions.numpy(), columns=target_list, index=test_df.index
                )[mlb.classes_].values
            ),
            name=TARGET_COL,
            index=test_df.index,
        ),
    ],
    axis=1,
)
text_idx_to_relations = {
    text_index: [
        l[0]
        for l in group_df.drop(["text_index", "text"], axis=1)[
            group_df.relations.str.len() > 0
        ]
        .apply(
            lambda row: (
                [[row.iloc[0], r, row.iloc[1]] for r in row.iloc[-1]]
                if len(row.iloc[-1]) > 0
                else []
            ),
            axis=1,
        )
        .values.tolist()
    ]
    for text_index, group_df in tqdm(ml_labeled_test_df.groupby("text_index"))
}
submission_df = (
    pd.DataFrame(
        {
            "id": list(text_idx_to_relations.keys()),
            TARGET_COL: list(text_idx_to_relations.values()),
        }
    )
    .set_index("id")
    .loc[load_test_raw_data().index]
)
submission_df = submission_df.assign(
    relations=submission_df.relations.map(lambda x: str(x).replace("'", '"'))
)
logging.info(f"### Saving {submission_df.shape=} @ {submission_path}")
submission_df.to_csv(submission_path)
