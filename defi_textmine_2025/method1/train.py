"""
python -m defi_textmine_2025.method1.train
"""

import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from defi_textmine_2025.data.utils import (
    LOGGING_DIR,
    INTERIM_DIR,
    MODELS_DIR,
    compute_class_weights,
)
import logging

log_file_path = os.path.join(
    LOGGING_DIR,
    f'{datetime.now().strftime("method1-train-%Y%m%dT%H%M%S")}.log',
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
from collections import defaultdict

from sklearn.preprocessing import MultiLabelBinarizer
import torch
from transformers import AdamW, CamembertTokenizer, CamembertModel

from defi_textmine_2025.bert_dataset_and_models import (
    CustomDataset,
    LinearHeadBertBasedModel,
    train_model,
    eval_model,
)

PRETRAINED_EMBEDDING_CHECKPOINT = "camembert/camembert-base"
EMBEDDING_SIZE = 768  # 768 # 1024
TASK_NAME = "multilabel_tagged_text"
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 32

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

model_dir_path = os.path.join(
    MODELS_DIR, "method1", f"finetuned-{PRETRAINED_EMBEDDING_CHECKPOINT}"
)
if not os.path.exists(model_dir_path):
    os.makedirs(model_dir_path)
model_dict_state_path = os.path.join(model_dir_path, "mutilabel_model_state.bin")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"{device=}")

logging.info("## Loading data ...")
METHOD_INTERIM_DIR = os.path.join(INTERIM_DIR, "method1")
train_csv_path = os.path.join(METHOD_INTERIM_DIR, "augmented_train.csv")
valid_csv_path = os.path.join(METHOD_INTERIM_DIR, "validation_onehot.csv")
assert os.path.exists(train_csv_path), f"Unfound {train_csv_path=}"
assert os.path.exists(valid_csv_path), f"Unfound {valid_csv_path=}"
logging.info(f"### Loading {train_csv_path=} ...")
train_df = pd.read_csv(train_csv_path)
logging.info(f"### Loading {valid_csv_path=} ...")
valid_df = pd.read_csv(valid_csv_path)

logging.info(f"{train_df.shape=}, {valid_df.shape=}")

logging.info("## Create the tokenized datasets for model input with special tokens...")
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

logging.info("### Test the tokenizer...")
test_text = "La <e2><NON_MILITARY_GOVERNMENT_ORGANISATION>police</e2> tchèque a <e2><NON_MILITARY_GOVERNMENT_ORGANISATION>mis la main</e2> sur le couple responsable d'un trafic d'œuvres d'art. Il s'agit de <e1><TERRORIST_OR_CRIMINAL>Patel</e1> et Mirna Maroski. Une <e2><NON_MILITARY_GOVERNMENT_ORGANISATION>perquisition</e2> à leur domicile a permis de retrouver une centaine de tableaux d'artistes européens. Il y avait également des pots en céramique et en porcelaine d'origine chinoise, ainsi que plusieurs faux documents de voyage. Les époux Maroski ont été conduits au poste de <e2><NON_MILITARY_GOVERNMENT_ORGANISATION>police</e2> dans un véhicule blindé. Mirna Maroski s'est évanouie une fois arrivée au poste. Elle a été amenée en ambulance au CHU de Motol où elle a été soignée. Monsieur Sergueï Alekseï, le directeur de l'hôpital, a demandé à ses collaborateurs d'être vigilants et de ne pas se laisser corrompre par la criminelle."
# generate encodings
encodings = tokenizer.encode_plus(
    test_text,
    add_special_tokens=True,
    max_length=MAX_LEN,
    truncation=True,
    padding="max_length",
    return_attention_mask=True,
    return_tensors="pt",
)
# we get a dictionary with three keys (see: https://huggingface.co/transformers/glossary.html)
logging.info(f"{encodings=}")
logging.info(f"{tokenizer.batch_decode(encodings['input_ids'])=}")
logging.info("### Init train and validation datasets...")
train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN, target_list)
valid_dataset = CustomDataset(valid_df, tokenizer, MAX_LEN, target_list)
logging.info("#### testing the dataset")
logging.info(next(iter(train_dataset)))

logging.info("## Create train and validation data loaders...")
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=22
)

val_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=22
)

logging.info("## Compute class weights to handle imbalance into the loss function...")
# Source: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights
# Scaling by total/2 helps keep the loss to a similar magnitude.
n_examples = train_df.shape[0]
n_classes = len(target_list)

class_weights = compute_class_weights(train_df, target_list)
class_weights_tensor = torch.Tensor(class_weights.values).to(
    device, dtype=torch.float16
)
class_weights_path = os.path.join(METHOD_INTERIM_DIR, "class_weights.csv")
logging.info(f"Saving {class_weights.shape=} @ {class_weights_path}")
pd.concat(
    [train_df[target_list].sum(axis=0).rename("size"), class_weights], axis=1
).to_csv(class_weights_path)

logging.info("## Prepare the model to trained...")
model = LinearHeadBertBasedModel(
    tokenizer=tokenizer,
    embedding_model=CamembertModel.from_pretrained(
        PRETRAINED_EMBEDDING_CHECKPOINT, return_dict=True
    ),
    embedding_size=EMBEDDING_SIZE,
    hidden_dim=128,
    n_classes=len(target_list),
)
# Freezing BERT layers: (tested, weaker convergence)
# for param in model.embedding_model.parameters():
#     param.requires_grad = False
model.to(device)
logging.info(f"{model=}")

# define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)

logging.info("## Model Training...")
if os.path.exists(
    model_dict_state_path
):  # to continue the training from a previous checkpoint
    logging.warning("The training will continue from a previous checkpoint...")
    logging.info("Loading the previous checkpoint...")
    model.load_state_dict(torch.load(model_dict_state_path))
    # if model is not in GPU, then load it to GPU
    if not next(model.parameters()).is_cuda:
        model = model.to(device)
elif not os.path.exists(os.path.dirname(model_dict_state_path)):
    os.makedirs(os.path.dirname(model_dict_state_path))

logging.info(f"Recall {log_file_path=}")

EPOCHS = 30
PATIENCE = 5
n_not_better_steps = 0
history = defaultdict(list)
best_val_loss = 10000

for epoch in range(1, EPOCHS + 1):
    logging.info(f"Epoch {epoch}/{EPOCHS}")
    model, train_acc, train_f1_macro, train_loss = train_model(
        model, train_data_loader, optimizer, class_weights_tensor, device
    )
    val_acc, val_f1_macro, val_loss = eval_model(
        model, val_data_loader, class_weights_tensor, device
    )

    logging.info(
        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f} train_f1_macro={train_f1_macro:.4f}, val_f1_macro={val_f1_macro:.4f}"
    )

    history["train_acc"].append(train_acc)
    history["train_f1_macro"].append(train_f1_macro)
    history["train_loss"].append(train_loss)
    history["val_acc"].append(val_acc)
    history["val_f1_macro"].append(val_f1_macro)
    history["val_loss"].append(val_loss)

    logging.info(f"Recall {log_file_path=}")

    learning_curve_fig_path = os.path.join(METHOD_INTERIM_DIR, "learning_curve.png")
    logging.info(f"## Plotting the learning curve @ {learning_curve_fig_path} ...")
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.plot(history["train_f1_macro"], label="train F1 macro")
    plt.plot(history["val_f1_macro"], label="validation F1 macro")
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.title("Training history")
    plt.ylabel("F1 macro / loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.ylim([0, 1])
    plt.grid()
    plt.savefig(learning_curve_fig_path)
    # save the best model
    if val_loss < best_val_loss:
        logging.info(
            f"Saving improved model state with {val_loss=} < {best_val_loss=} @ {model_dict_state_path}"
        )
        torch.save(model.state_dict(), model_dict_state_path)
        best_val_loss = val_loss
        n_not_better_steps = 0
    else:  # check for early stopping
        n_not_better_steps += 1
        if n_not_better_steps >= PATIENCE:
            break
