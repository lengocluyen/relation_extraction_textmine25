"""
python -m defi_textmine_2025.method1.evaluate
"""

import os
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report
from defi_textmine_2025.data.utils import (
    LOGGING_DIR,
    INTERIM_DIR,
    MODELS_DIR,
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

logging.info("## Loading data ...")
METHOD_INTERIM_DIR = os.path.join(INTERIM_DIR, "method1")
train_csv_path = os.path.join(METHOD_INTERIM_DIR, "oversampled_train.csv")
valid_csv_path = os.path.join(METHOD_INTERIM_DIR, "validation_onehot.csv")
assert os.path.exists(train_csv_path), f"Unfound {train_csv_path=}"
assert os.path.exists(valid_csv_path), f"Unfound {valid_csv_path=}"
logging.info(f"### Loading {train_csv_path=} ...")
train_df = pd.read_csv(train_csv_path)
logging.info(f"### Loading {valid_csv_path=} ...")
valid_df = pd.read_csv(valid_csv_path)

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
train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN, target_list)
valid_dataset = CustomDataset(valid_df, tokenizer, MAX_LEN, target_list)
logging.info("#### testing the dataset")
logging.info(next(iter(train_dataset)))

logging.info("## Create train and validation data loaders...")
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

val_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
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

for splitloader, splitname in zip(
    [train_data_loader, val_data_loader], ["train", "validation"]
):
    logging.info(f"## Getting predictions on {splitname} data...")
    titles, predictions, prediction_probs, target_values = get_predictions(
        model, splitloader, device
    )
    scores_path = os.path.join(
        METHOD_INTERIM_DIR, f"{splitname}-classification_report.txt"
    )
    logging.info(f"## Saving {splitname} scores @ {scores_path}")
    with open(scores_path, "w") as fw:
        fw.write(
            classification_report(
                target_values, predictions, target_names=target_list, zero_division=0
            )
        )
