"""
Usage: python -m defi_textmine_2025.mth2_hasrelation_whatrelation.mth2_baseline_grumodels
"""

from datetime import datetime
import os
import json
import glob
import logging
from typing import List
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    CamembertTokenizer,
    CamembertModel,
    AdamW,
    PreTrainedTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from defi_textmine_2025.data.utils import LOGGING_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s|%(levelname)s|%(module)s.py:%(lineno)s] %(message)s",
    datefmt="%H:%M:%S",
    filename=os.path.join(
        LOGGING_DIR,
        f'{datetime.now().strftime("mth2_baseline_grumodels-%Y%m%dT%H%M%S")}.log',
    ),
)

# Custom imports (ensure these are defined in your environment)
from defi_textmine_2025.data.utils import (
    load_test_raw_data,
    TARGET_COL,
    INTERIM_DIR,
    MODELS_DIR,
    submission_path,
)

# Define constants
BASE_CHECKPOINT = "camembert/camembert-base"
EMBEDDING_SIZE = 768
TASK_NAME = "multilabel_tagged_text"

# Entity classes and categories (replace with your actual lists)
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

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit([categories_to_check])
logging.info(f"{mlb.classes_=}")

# Directories
generated_data_dir_path = os.path.join(INTERIM_DIR, "reduced_text_w_entity_bracket")
assert os.path.exists(generated_data_dir_path)

preprocessed_data_dir = os.path.join(
    INTERIM_DIR, "one_hot_multilabel_tagged_text_dataset"
)
labeled_preprocessed_data_dir_path = os.path.join(preprocessed_data_dir, "train")

model_dir_path = os.path.join(MODELS_DIR, f"finetuned-{BASE_CHECKPOINT}")
model_dict_state_path = os.path.join(
    model_dir_path, "MLTC_model_state_camembert_large.bin"
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"{device=}")


# Functions to load and process data
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


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            data,
            pd.DataFrame(
                mlb.transform(data[TARGET_COL]), columns=mlb.classes_, index=data.index
            ),
        ],
        axis=1,
    )


def format_relations_str_to_list(labels_as_str: str) -> List[str]:
    return (
        json.loads(labels_as_str.replace("{", "[").replace("}", "]").replace("'", '"'))
        if not pd.isnull(labels_as_str)
        else []
    )


def process_csv_to_csv(in_dir_or_file_path: str, out_dir_path: str) -> None:
    if os.path.isdir(in_dir_or_file_path):
        all_files = glob.glob(os.path.join(in_dir_or_file_path, "*.csv"))
    else:
        assert in_dir_or_file_path.endswith(".csv")
        all_files = [in_dir_or_file_path]
    os.makedirs(out_dir_path, exist_ok=True)
    for filename in tqdm(all_files, desc="Processing CSV files"):
        preprocessed_data_filename = os.path.join(
            out_dir_path, os.path.basename(filename)
        )
        data = load_csv(filename).assign(
            **{
                TARGET_COL: lambda df: df[TARGET_COL].apply(
                    format_relations_str_to_list
                )
            }
        )
        processed_data = process_data(data)
        processed_data.to_csv(preprocessed_data_filename, sep="\t")


# Load and preprocess data
logging.info("process_csv_to_csv labels to one-hot...")
process_csv_to_csv(
    os.path.join(generated_data_dir_path, "train"), labeled_preprocessed_data_dir_path
)
logging.info("loading one-hot labeled data...")
labeled_df = load_csv(labeled_preprocessed_data_dir_path, index_col=0, sep="\t")
logging.info("Train-val split...")
df_train, df_valid = train_test_split(
    labeled_df, test_size=0.2, shuffle=True, random_state=42
)

target_list = mlb.classes_
logging.info(f"{len(target_list)} categories = {target_list}")

# Hyperparameters
MAX_LEN = 200
logging.info(f"Init tokenizer w/ {MAX_LEN=}...")
tokenizer = CamembertTokenizer.from_pretrained(BASE_CHECKPOINT)
task_special_tokens = ["<e1>", "</e1>", "<e2>", "</e2>"] + [
    f"<{entity_class}>" for entity_class in entity_classes
]
# Add special tokens to the tokenizer
num_added_tokens = tokenizer.add_tokens(task_special_tokens, special_tokens=True)


# Custom Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, target_list):
        self.tokenizer = tokenizer
        self.df = df
        self.texts = df["reduced_text"].tolist()
        self.targets = df[target_list].values
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "targets": torch.FloatTensor(self.targets[index]),
            "text": text,
        }


# Create datasets and dataloaders
train_dataset = CustomDataset(df_train, tokenizer, MAX_LEN, target_list)
valid_dataset = CustomDataset(df_valid, tokenizer, MAX_LEN, target_list)

TRAIN_BATCH_SIZE = 80
VALID_BATCH_SIZE = 16

train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True
)
val_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False
)


# Model Definition
class TransformerAttentionBertModel(nn.Module):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        embedding_model: PreTrainedModel,
        n_classes: int,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 8,
        dropout_rate: float = 0.1,
    ):
        super(TransformerAttentionBertModel, self).__init__()
        self.embedding_model = embedding_model
        self.embedding_model.resize_token_embeddings(len(tokenizer))
        self.embedding_size = embedding_model.config.hidden_size

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=num_attention_heads,
            dropout=dropout_rate,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(self.embedding_size, n_classes),
        )

    def forward(
        self,
        input_ids: torch.tensor,
        attn_mask: torch.tensor,
        token_type_ids: torch.tensor = None,
    ):
        # Obtain embeddings from the pre-trained model
        outputs = self.embedding_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = (
            outputs.last_hidden_state
        )  # Shape: [batch_size, seq_len, hidden_size]

        # Transpose for Transformer encoder (required shape: [seq_len, batch_size, hidden_size])
        hidden_states = hidden_states.permute(1, 0, 2)

        # Create a source mask based on attention mask
        src_key_padding_mask = attn_mask == 0  # Shape: [batch_size, seq_len]

        # Pass through Transformer encoder
        transformer_output = self.transformer_encoder(
            hidden_states,
            src_key_padding_mask=src_key_padding_mask,
        )  # Shape: [seq_len, batch_size, hidden_size]

        # Take the mean of the encoder outputs
        transformer_output = transformer_output.permute(
            1, 0, 2
        )  # Shape: [batch_size, seq_len, hidden_size]
        pooled_output = torch.mean(
            transformer_output, dim=1
        )  # Shape: [batch_size, hidden_size]

        # Pass through classification head
        logits = self.classification_head(
            pooled_output
        )  # Shape: [batch_size, n_classes]

        return logits


# Initialize model
model = TransformerAttentionBertModel(
    tokenizer=tokenizer,
    embedding_model=CamembertModel.from_pretrained(BASE_CHECKPOINT, return_dict=True),
    n_classes=len(target_list),
    num_transformer_layers=2,
    num_attention_heads=8,
    dropout_rate=0.1,
)
model.to(device)

logging.info(f"{model=}")

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
EPOCHS = 50
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

# Source: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights
# Scaling by total/2 helps keep the loss to a similar magnitude.
n_examples = labeled_df.shape[0]
n_classes = len(target_list)


def get_cat_var_distribution(cat_var: pd.Series) -> pd.DataFrame:
    return pd.concat(
        [cat_var.value_counts(), cat_var.value_counts(normalize=True)], axis=1
    )


def compute_class_weights(lbl_df: pd.DataFrame, class_names: list) -> pd.Series:
    return (
        lbl_df[target_list]
        .sum(axis=0)
        .map(lambda x: (1 / x) * (n_examples / n_classes))
        .rename("weight")
    )


class_weights = compute_class_weights(labeled_df, target_list)
class_weights_tensor = torch.Tensor(class_weights.values).to(
    device, dtype=torch.float16
)

logging.info(
    pd.concat(
        [labeled_df[target_list].sum(axis=0).rename("class_size"), class_weights],
        axis=1,
    )
)


# Loss function
def loss_fn(
    outputs: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor
) -> float:
    """BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid followed
    by a BCELoss as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

        Args:
            outputs (torch.Tensor): predicted output
            targets (torch.Tensor): expected output (groundtruth)
            class_weights (torch.Tensor): weight of the different classes of the target,
              to handle the imbalance of the dataset

        Returns:
            float: loss value
    """
    return torch.nn.BCEWithLogitsLoss(weight=class_weights)(outputs, targets)


# Training and evaluation functions
def train_model(model, data_loader, optimizer, scheduler, _class_weights_tensor):
    model.train()
    losses = []
    correct_predictions = 0
    num_samples = 0

    for data in tqdm(data_loader, desc="Training"):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["targets"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, targets, _class_weights_tensor)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        preds = torch.sigmoid(outputs).round()
        correct_predictions += (preds == targets).sum().item()
        num_samples += targets.numel()

        # break  # TODO: REMOVE

    return np.mean(losses), correct_predictions / num_samples


def eval_model(model, data_loader, _class_weights_tensor):
    model.eval()
    losses = []
    correct_predictions = 0
    num_samples = 0
    predictions = []
    prediction_probs = []
    target_values = []

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["targets"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets, _class_weights_tensor)
            losses.append(loss.item())

            preds = torch.sigmoid(outputs).round()
            correct_predictions += (preds == targets).sum().item()
            num_samples += targets.numel()

            predictions.extend(preds.cpu())
            prediction_probs.extend(torch.sigmoid(outputs).cpu())
            target_values.extend(targets.cpu())

            # break  # TODO: REMOVE

    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    target_values = torch.stack(target_values)

    f1_macro = f1_score(target_values, predictions, average="macro", zero_division=0)

    return np.mean(losses), correct_predictions / num_samples, f1_macro


# Training loop
PATIENCE = 3
best_f1_macro = 0
n_not_better_steps = 0
history = defaultdict(list)

if not os.path.exists(os.path.dirname(model_dict_state_path)):
    os.makedirs(os.path.dirname(model_dict_state_path))

for epoch in range(1, EPOCHS + 1):
    logging.info(f"Epoch {epoch}/{EPOCHS}")
    train_loss, train_acc = train_model(
        model, train_data_loader, optimizer, scheduler, class_weights_tensor
    )
    val_loss, val_acc, val_f1_macro = eval_model(
        model, val_data_loader, class_weights_tensor
    )

    logging.info(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
    logging.info(
        f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, Val F1 Macro: {val_f1_macro:.4f}"
    )

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1_macro"].append(val_f1_macro)

    # Save best model
    if val_f1_macro > best_f1_macro:
        torch.save(model.state_dict(), model_dict_state_path)
        best_f1_macro = val_f1_macro
        n_not_better_steps = 0
    else:
        n_not_better_steps += 1
        if n_not_better_steps >= PATIENCE:
            logging.info("Early stopping")
            break

# Plot training history
plt.figure(figsize=(10, 7))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.plot(history["val_f1_macro"], label="Validation F1 Macro")
plt.xlabel("Epoch")
plt.ylabel("Loss / F1 Macro")
plt.title("Training History")
plt.legend()
plt.grid()
plt.savefig("./training_history_transformers.png")

# Load best model
model.load_state_dict(torch.load(model_dict_state_path))
model.to(device)


# Get predictions on validation set
def get_predictions(model, data_loader):
    model.eval()
    predictions = []
    prediction_probs = []
    target_values = []
    texts = []

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Getting Predictions"):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["targets"].to(device)
            text = data["text"]

            outputs = model(input_ids, attention_mask)
            preds = torch.sigmoid(outputs).round()

            predictions.extend(preds.cpu())
            prediction_probs.extend(torch.sigmoid(outputs).cpu())
            target_values.extend(targets.cpu())
            texts.extend(text)
            # break  # TODO: REMOVE

    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    target_values = torch.stack(target_values)

    return texts, predictions, prediction_probs, target_values


texts, predictions, prediction_probs, target_values = get_predictions(
    model, val_data_loader
)

# Classification report
report = classification_report(
    target_values, predictions, target_names=target_list, zero_division=0
)
logging.info(report)

# Prepare test data
df_test = load_csv(os.path.join(generated_data_dir_path, "test"))
logging.info(df_test.head())

test_dataset = CustomDataset(
    df_test.assign(**{cat: [0] * df_test.shape[0] for cat in target_list}),
    tokenizer,
    MAX_LEN,
    target_list,
)
TEST_BATCH_SIZE = 16
test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False
)

# Get predictions on test set
texts_test, predictions_test, prediction_probs_test, _ = get_predictions(
    model, test_data_loader
)

# Prepare submission
ml_labeled_test_df = pd.concat(
    [
        df_test.drop(TARGET_COL, axis=1, errors="ignore"),
        pd.Series(
            mlb.inverse_transform(
                pd.DataFrame(predictions_test.numpy(), columns=target_list)[
                    mlb.classes_
                ].values
            ),
            name=TARGET_COL,
            index=df_test.index,
        ),
    ],
    axis=1,
)
logging.info(ml_labeled_test_df.head())

# Adjust as per your submission requirements
text_idx_to_relations = {
    text_index: [
        l[0]
        for l in group_df.drop(["text_index", "text"], axis=1, errors="ignore")[
            group_df[TARGET_COL].str.len() > 0
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
submission_df.to_csv(submission_path)
logging.info(f"Generated submission at {submission_path}")
