import os
from typing import List
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import sys
import logging

import torch.utils.data 

logging.basicConfig(
     level=logging.INFO, 
     format= '[%(asctime)s|%(levelname)s|%(module)s.py:%(lineno)s] %(message)s',
     datefmt='%H:%M:%S'
 )
#import tqdm.notebook as tq
import torch.utils
import tqdm as tq
from tqdm import tqdm
# Create new `pandas` methods which use `tqdm` progress
# (can use tqdm_gui, optional kwargs, etc.)
tqdm.pandas()
from collections import defaultdict

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW, CamembertTokenizer, CamembertModel
from torch.nn import MultiheadAttention

from defi_textmine_2025.data import load_test_raw_data
from defi_textmine_2025.data import TARGET_COL, INTERIM_DIR, MODELS_DIR, submission_path

# BASE_CHECKPOINT = "bert-base-uncased"
# BASE_CHECKPOINT = "bert-base-multilingual-cased"
BASE_CHECKPOINT = "camembert/camembert-base"
TASK_NAME = "multilabel_tagged_text"

entity_classes = {'TERRORIST_OR_CRIMINAL', 'LASTNAME', 'LENGTH', 'NATURAL_CAUSES_DEATH', 'COLOR', 'STRIKE', 'DRUG_OPERATION', 'HEIGHT', 'INTERGOVERNMENTAL_ORGANISATION', 'TRAFFICKING', 'NON_MILITARY_GOVERNMENT_ORGANISATION', 'TIME_MIN', 'DEMONSTRATION', 'TIME_EXACT', 'FIRE', 'QUANTITY_MIN', 'MATERIEL', 'GATHERING', 'PLACE', 'CRIMINAL_ARREST', 'CBRN_EVENT', 'ECONOMICAL_CRISIS', 'ACCIDENT', 'LONGITUDE', 'BOMBING', 'MATERIAL_REFERENCE', 'WIDTH', 'FIRSTNAME', 'MILITARY_ORGANISATION', 'CIVILIAN', 'QUANTITY_MAX', 'CATEGORY', 'POLITICAL_VIOLENCE', 'EPIDEMIC', 'TIME_MAX', 'TIME_FUZZY', 'NATURAL_EVENT', 'SUICIDE', 'CIVIL_WAR_OUTBREAK', 'POLLUTION', 'ILLEGAL_CIVIL_DEMONSTRATION', 'NATIONALITY', 'GROUP_OF_INDIVIDUALS', 'QUANTITY_FUZZY', 'RIOT', 'WEIGHT', 'THEFT', 'MILITARY', 'NON_GOVERNMENTAL_ORGANISATION', 'LATITUDE', 'COUP_D_ETAT', 'ELECTION', 'HOOLIGANISM_TROUBLEMAKING', 'QUANTITY_EXACT', 'AGITATING_TROUBLE_MAKING'}
categories_to_check = ['END_DATE', 'GENDER_MALE', 'WEIGHS', 'DIED_IN', 'HAS_FAMILY_RELATIONSHIP', 'IS_DEAD_ON', 'IS_IN_CONTACT_WITH', 'HAS_CATEGORY', 'HAS_CONTROL_OVER', 'IS_BORN_IN', 'IS_OF_SIZE', 'HAS_LATITUDE', 'IS_PART_OF', 'IS_OF_NATIONALITY', 'IS_COOPERATING_WITH', 'DEATHS_NUMBER', 'HAS_FOR_HEIGHT', 'INITIATED', 'WAS_DISSOLVED_IN', 'HAS_COLOR', 'CREATED', 'IS_LOCATED_IN', 'WAS_CREATED_IN', 'IS_AT_ODDS_WITH', 'HAS_CONSEQUENCE', 'HAS_FOR_LENGTH', 'INJURED_NUMBER', 'START_DATE', 'STARTED_IN', 'GENDER_FEMALE', 'HAS_LONGITUDE', 'RESIDES_IN', 'HAS_FOR_WIDTH', 'IS_BORN_ON', 'HAS_QUANTITY', 'OPERATES_IN', 'IS_REGISTERED_AS']

mlb = MultiLabelBinarizer()
mlb.fit([categories_to_check])
logging.info(f"{mlb.classes_=}")

generated_data_dir_path = os.path.join(INTERIM_DIR, "multilabel_tagged_text_dataset")
assert os.path.exists(generated_data_dir_path)

preprocessed_data_dir = os.path.join(INTERIM_DIR, "one_hot_multilabel_tagged_text_dataset")
labeled_preprocessed_data_dir_path = os.path.join(preprocessed_data_dir,"train")
#! mkdir -p {labeled_preprocessed_data_dir_path}

model_dir_path = os.path.join(MODELS_DIR, f"finetuned-{BASE_CHECKPOINT}")
#! mkdir -p {model_dir_path}
model_dict_state_path = os.path.join(model_dir_path,"MLTC_model_state.bin")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_csv(dir_or_file_path: str, index_col=None, sep=',') -> pd.DataFrame:
    if os.path.isdir(dir_or_file_path):
        all_files = glob.glob(os.path.join(dir_or_file_path , "*.csv"))  
    else:
        assert dir_or_file_path.endswith(".csv")
        all_files = [dir_or_file_path]
    assert len(all_files) > 0
    return pd.concat([pd.read_csv(filename, index_col=index_col, header=0, sep=sep) for filename in all_files], axis=0, ignore_index=True)

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([data, pd.DataFrame(mlb.transform(data[TARGET_COL]), columns=mlb.classes_, index=data.index)], axis=1) # .drop([TARGET_COL], axis=1)


def format_relations_str_to_list(labels_as_str: str) -> List[str]:
    return json.loads(
        labels_as_str.replace("{", "[").replace("}", "]").replace("'", '"')
    )  if not pd.isnull(labels_as_str) else []


def process_csv_to_csv(in_dir_or_file_path: str, out_dir_path: str) -> None:
    """Convert labels, i.e. list of relations category, into one-hot vectors

    Args:
        in_dir_or_file_path (str): str
        out_dir_path (str): str
    """
    if os.path.isdir(in_dir_or_file_path):
        all_files = glob.glob(os.path.join(in_dir_or_file_path , "*.csv"))  
    else:
        assert in_dir_or_file_path.endswith(".csv")
        all_files = [in_dir_or_file_path]
    for filename in (pb:=tqdm(all_files)):
        pb.set_description(filename)
        preprocessed_data_filename = os.path.join(out_dir_path, os.path.basename(filename))
        process_data(load_csv(filename).assign(**{TARGET_COL: lambda df: df[TARGET_COL].apply(format_relations_str_to_list)})).to_csv(preprocessed_data_filename, sep="\t")


process_csv_to_csv(os.path.join(generated_data_dir_path, "train"), labeled_preprocessed_data_dir_path)
labeled_df = load_csv(labeled_preprocessed_data_dir_path, index_col=0, sep='\t')
df_train, df_valid = train_test_split(labeled_df, test_size = 0.2, shuffle=True, random_state=42)
pd.DataFrame({"train": df_train[mlb.classes_].sum(axis=0), "valid": df_valid[mlb.classes_].sum(axis=0)}).sort_values("train", ascending=False)
df_train_with_relation = df_train[df_train[mlb.classes_].sum(axis=1) >= 1]
df_train_without_relation = df_train[df_train[mlb.classes_].sum(axis=1) == 0]

#df_train = pd.concat([df_train_with_relation, df_train_without_relation.sample(n=int(df_train_with_relation.shape[0] / 3))], axis=0)
                     
target_list = mlb.classes_
logging.info(f"{len(target_list)} categories = {target_list}")

# Hyperparameters
MAX_LEN = 300 # TODO: increase
# tokenizer = BertTokenizer.from_pretrained(BASE_CHECKPOINT)
tokenizer = CamembertTokenizer.from_pretrained(BASE_CHECKPOINT)
task_special_tokens = ["<e1>", "</e1>", "<e2>", "</e2>"] + [
    f"<{entity_class}>" for entity_class in entity_classes
]
# add special tokens to the tokenizer
num_added_tokens = tokenizer.add_tokens(task_special_tokens, special_tokens=True)

# Test the tokenizer
test_text = "La <e2><NON_MILITARY_GOVERNMENT_ORGANISATION>police</e2> tchèque a <e2><NON_MILITARY_GOVERNMENT_ORGANISATION>mis la main</e2> sur le couple responsable d'un trafic d'œuvres d'art. Il s'agit de <e1><TERRORIST_OR_CRIMINAL>Patel</e1> et Mirna Maroski. Une <e2><NON_MILITARY_GOVERNMENT_ORGANISATION>perquisition</e2> à leur domicile a permis de retrouver une centaine de tableaux d'artistes européens. Il y avait également des pots en céramique et en porcelaine d'origine chinoise, ainsi que plusieurs faux documents de voyage. Les époux Maroski ont été conduits au poste de <e2><NON_MILITARY_GOVERNMENT_ORGANISATION>police</e2> dans un véhicule blindé. Mirna Maroski s'est évanouie une fois arrivée au poste. Elle a été amenée en ambulance au CHU de Motol où elle a été soignée. Monsieur Sergueï Alekseï, le directeur de l'hôpital, a demandé à ses collaborateurs d'être vigilants et de ne pas se laisser corrompre par la criminelle."
# generate encodings
encodings = tokenizer.encode_plus(test_text, 
                                  add_special_tokens = True,
                                  max_length = MAX_LEN,
                                  truncation = True,
                                  padding = "max_length", 
                                  return_attention_mask = True, 
                                  return_tensors = "pt")

tokenizer.batch_decode(encodings['input_ids'])

"""
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, target_list):
        self.tokenizer = tokenizer
        self.df = df
        # self.e1 = list(df['e1'])
        # self.e1 = list(df['e1'])
        # self.text_indexes = list(df['text_index'])
        self.title = list(df['text'])
        self.targets = self.df[target_list].values
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        text = str(self.title[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index]),
            'title': text,
            # 'text_index': self.text_index[index],
            # 'e1': self.e1[index],
            # 'e2': self.e2[index],
        }
"""
import torch
from torch.utils.data import Dataset
from transformers import pipeline, set_seed, CamembertTokenizer, CamembertModel, AdamW, get_linear_schedule_with_warmup
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32


class AugmentationDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

    
class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, target_list, augment_data=False, num_augmentations=1, device=None, save_path="augmented_texts.json"):
        self.tokenizer = tokenizer
        self.df = df
        self.title = list(df['text'])
        self.targets = self.df[target_list].values
        self.max_len = max_len
        self.augment_data = augment_data
        self.num_augmentations = num_augmentations
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = save_path

        # Initialize text generation and paraphrase pipelines
        if self.augment_data:
            if os.path.exists(self.save_path):
                with open(self.save_path, "r") as file:
                    self.augmented_texts = json.load(file)
            else:
                self.generator = pipeline('text-generation', model='gpt2', tokenizer='gpt2', device=0 if self.device.type == 'cuda' else -1)
                self.paraphraser = pipeline('text2text-generation', model='t5-base', tokenizer='t5-base', device=0 if self.device.type == 'cuda' else -1)
                set_seed(42)
                self.augmented_texts = self.augment_all_texts()
                with open(self.save_path, "w") as file:
                    json.dump(self.augmented_texts, file)

    def augment_all_texts(self):
        augmentation_dataset = AugmentationDataset(self.title)
        augmentation_dataloader = torch.utils.data.DataLoader(augmentation_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

        augmented_texts = []
        for batch in augmentation_dataloader:
            batch = list(batch)

            # Process synthetic texts
            synthetic_results = self.generator(batch, max_new_tokens=self.max_len // 2, num_return_sequences=1, pad_token_id=self.generator.tokenizer.eos_token_id)
            synthetic_texts = []
            for result in synthetic_results:
                if isinstance(result, list):
                    synthetic_texts.extend([r['generated_text'] for r in result if 'generated_text' in r])
                elif isinstance(result, dict) and 'generated_text' in result:
                    synthetic_texts.append(result['generated_text'])

            # Process paraphrased texts
            paraphrase_results = self.paraphraser([f"paraphrase: {text}" for text in batch], max_length=self.max_len // 2, num_return_sequences=1)
            paraphrase_texts = []
            for result in paraphrase_results:
                if isinstance(result, list):
                    paraphrase_texts.extend([r['generated_text'] if 'generated_text' in r else r['text'] for r in result])
                elif isinstance(result, dict):
                    paraphrase_texts.append(result['generated_text'] if 'generated_text' in result else result['text'])

            # Combine original, synthetic, and paraphrased texts
            for original, synthetic, paraphrase in zip(batch, synthetic_texts, paraphrase_texts):
                augmented_text = original + " " + synthetic + " " + paraphrase
                augmented_texts.append(augmented_text)

        return augmented_texts

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        text = str(self.title[index])
        text = " ".join(text.split())

        if self.augment_data:
            text = self.augmented_texts[index]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index]),
            'title': text,
        }



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = CustomDataset(df_train, tokenizer, MAX_LEN, target_list, augment_data=True, num_augmentations=1,device=device, save_path="augmented_texts.json")
valid_dataset = CustomDataset(df_valid, tokenizer, MAX_LEN, target_list, augment_data=False,num_augmentations=1, device=device, save_path="augmented_texts_valid.json")



# Data loaders
train_data_loader = torch.utils.data.DataLoader(train_dataset, 
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_data_loader = torch.utils.data.DataLoader(valid_dataset, 
    batch_size=VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)
"""
class BERTClass(torch.nn.Module):
    def __init__(self, tokenizer: CamembertTokenizer):
        super(BERTClass, self).__init__()
        self.bert_model = CamembertModel.from_pretrained(BASE_CHECKPOINT, return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, len(target_list))
        # if you want to add new tokens to the vocabulary, then in general you’ll need to resize the embedding layers with
        # Source https://discuss.huggingface.co/t/adding-new-tokens-while-preserving-tokenization-of-adjacent-tokens/12604
        self.bert_model.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

model = BERTClass(tokenizer)

# # Freezing BERT layers: (tested, weaker convergence)
# for param in model.bert_model.parameters():
#     param.requires_grad = False

model.to(device)
"""
"""


class BertWithMHA(torch.nn.Module):
    def __init__(self, tokenizer: CamembertTokenizer, num_labels: int):
        super(BertWithMHA, self).__init__()
        self.bert_model = CamembertModel.from_pretrained(BASE_CHECKPOINT, return_dict=True)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.linear1 = torch.nn.Linear(768, 512)  # Hidden layer for feature extraction
        self.linear2 = torch.nn.Linear(512, num_labels)  # Output layer
        self.mha = MultiheadAttention(embed_dim=768, num_heads=4)  # Multi-head attention layer
        self.bert_model.resize_token_embeddings(len(tokenizer))
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = self.dropout1(output.pooler_output)

        # Optional: Apply Multi-Head Attention
        attention_output, _ = self.mha(pooled_output.unsqueeze(1), pooled_output.unsqueeze(1), pooled_output.unsqueeze(1))
        attention_output = attention_output.squeeze(1)

        hidden_output = self.linear1(self.dropout2(attention_output))
        hidden_output = torch.nn.functional.relu(hidden_output)  # ReLU activation

        output = self.linear2(hidden_output)
        return output
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"    
num_labels = len(target_list)
model = BertWithMHA(tokenizer, num_labels)


"""


class ImprovedBERTClass(nn.Module):
    def __init__(self, tokenizer: CamembertTokenizer, num_labels: int, use_hidden_states=False, dropout_prob: float = 0.3, base_checkpoint: str = 'camembert-large'):
        super(ImprovedBERTClass, self).__init__()

        self.bert_model = CamembertModel.from_pretrained(BASE_CHECKPOINT, return_dict=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.use_hidden_states = use_hidden_states

        if len(tokenizer) != self.bert_model.config.vocab_size:
            self.bert_model.resize_token_embeddings(len(tokenizer))

        self.linear = nn.Linear(self.bert_model.config.hidden_size, num_labels)

        self.self_attention = nn.MultiheadAttention(self.bert_model.config.hidden_size, num_heads=4)
        self.self_attention_dropout = nn.Dropout(dropout_prob)
        self.layer_norm1 = nn.LayerNorm(self.bert_model.config.hidden_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(self.bert_model.config.hidden_size, self.bert_model.config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.bert_model.config.hidden_size * 4, self.bert_model.config.hidden_size)
        )
        self.ffn_dropout = nn.Dropout(dropout_prob)
        self.layer_norm2 = nn.LayerNorm(self.bert_model.config.hidden_size)

    def forward(self, input_ids, attn_mask, token_type_ids=None):
        output = self.bert_model(input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)

        if self.use_hidden_states:
            hidden_state = output.last_hidden_state[:, 0, :]  # Using the [CLS] token hidden state
            hidden_state = self.dropout(hidden_state)
        else:
            hidden_state = self.dropout(output.pooler_output)

        attention_output, _ = self.self_attention(hidden_state.unsqueeze(0), hidden_state.unsqueeze(0), hidden_state.unsqueeze(0))
        attention_output = attention_output.squeeze(0)
        attention_output = self.self_attention_dropout(attention_output)
        attention_output = self.layer_norm1(attention_output + hidden_state)

        ffn_output = self.ffn(attention_output)
        ffn_output = self.ffn_dropout(ffn_output)
        ffn_output = self.layer_norm2(ffn_output + attention_output)

        output = self.linear(ffn_output)
        return output
    
# # Freezing BERT layers: (tested, weaker convergence)
# for param in model.bert_model.parameters():
#     param.requires_grad = False
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"    
num_labels = len(target_list)
use_hidden_states = False  # Set to True if you want to use hidden states instead of pooler output
model = ImprovedBERTClass(tokenizer, num_labels, use_hidden_states)
model.to(device)

# BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class. 
# This version is more numerically stable than using a plain Sigmoid followed 
# by a BCELoss as, by combining the operations into one layer, 
# we take advantage of the log-sum-exp trick for numerical stability.
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


# Training of the model for one epoch
def train_model(training_loader, model, optimizer):
    predictions = []
    prediction_probs = []
    target_values = []
    losses = []
    correct_predictions = 0
    num_samples = 0
    # set model to training mode (activate dropout, batch norm)
    model.train()
    # initialize the progress bar
    loop = tq.tqdm(enumerate(training_loader), total=len(training_loader), 
                      leave=True, colour='steelblue')
    for batch_idx, data in loop:
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        # forward
        outputs = model(ids, mask, token_type_ids) # (batch,predict)=(32,37)
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())
        # training accuracy, apply sigmoid, round (apply thresh 0.5)
        # outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
        # targets = targets.cpu().detach().numpy()
        # correct_predictions += np.sum(outputs==targets)
        # num_samples += targets.size   # total number of elements in the 2D array
        outputs = torch.sigmoid(outputs).cpu().detach()
        # thresholding at 0.5
        preds = outputs.round()
        targets = targets.cpu().detach()
        correct_predictions += np.sum(preds.numpy()==targets.numpy())
        num_samples += targets.numpy().size   # total number of elements in the 2D array
        
        # thresholding at 0.5
        preds = outputs.round()        
        predictions.extend(preds)
        prediction_probs.extend(outputs)
        target_values.extend(targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # grad descent step
        optimizer.step()
        # Update progress bar
        #loop.set_description(f"")
        #loop.set_postfix(batch_loss=loss)

    # returning: trained model, model accuracy, mean loss
    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    target_values = torch.stack(target_values)    

    return model, float(correct_predictions)/num_samples, f1_score(target_values, predictions, average="macro", zero_division=0), np.mean(losses)
    # return model, float(correct_predictions)/num_samples, np.mean(losses)

# torch.cuda.empty_cache()
# train_model(train_data_loader, model, optimizer)

def get_predictions(model, data_loader):
    """
    Outputs:
      predictions - 
    """
    model = model.eval()
    
    titles = []
    predictions = []
    prediction_probs = []
    target_values = []

    with torch.no_grad():
      for data in tqdm(data_loader, "training"):
        title = data["title"]
        ids = data["input_ids"].to(device, dtype = torch.long)
        mask = data["attention_mask"].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data["targets"].to(device, dtype = torch.float)
        
        outputs = model(ids, mask, token_type_ids)
        # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
        outputs = torch.sigmoid(outputs).detach().cpu()
        # thresholding at 0.5
        preds = outputs.round()
        targets = targets.detach().cpu()

        titles.extend(title)
        predictions.extend(preds)
        prediction_probs.extend(outputs)
        target_values.extend(targets)
    
    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    target_values = torch.stack(target_values)
    
    return titles, predictions, prediction_probs, target_values


def eval_model(validation_loader, model):    
    predictions = []
    prediction_probs = []
    target_values = []
    losses = []
    correct_predictions = 0
    num_samples = 0
    # set model to eval mode (turn off dropout, fix batch norm)
    model.eval()

    with torch.no_grad():
        # for batch_idx, data in tqdm(enumerate(validation_loader, 0), "evaluating"):
        for data in tqdm(validation_loader, "evaluating"):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

            # validation accuracy
            # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
            outputs = torch.sigmoid(outputs).cpu().detach()
            # thresholding at 0.5
            preds = outputs.round()
            targets = targets.cpu().detach()
            correct_predictions += np.sum(preds.numpy()==targets.numpy())
            num_samples += targets.numpy().size   # total number of elements in the 2D array

            predictions.extend(preds)
            prediction_probs.extend(outputs)
            target_values.extend(targets)
    
    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    target_values = torch.stack(target_values)    

    return float(correct_predictions)/num_samples, f1_score(target_values, predictions, average="macro", zero_division=0), np.mean(losses)


# eval_model(train_data_loader, model)

EPOCHS = 15
# THRESHOLD = 0.5 # threshold for the sigmoid
PATIENCE = 4
n_not_better_steps = 0
history = defaultdict(list)
best_f1_macro = 0
# assert not os.path.exists(model_dict_state_path), "The trained model is already serialized at {model_dict_state_path}"
# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-5)  
#optimizer = AdamW(model.parameters(), lr = 1e-2)
#optimizer = AdamW(model.parameters(), lr = 1e-3)  
if not os.path.exists(os.path.dirname(model_dict_state_path)):
    os.makedirs(os.path.dirname(model_dict_state_path))

for epoch in range(1, EPOCHS+1):
    tx = f'Epoch {epoch}/{EPOCHS}'
    print(tx)
    logging.info(tx)
    model, train_acc, train_f1_macro, train_loss = train_model(train_data_loader, model, optimizer)
    val_acc, val_f1_macro, val_loss = eval_model(val_data_loader, model)

    tx2 = f'train_loss={train_loss:.4f}, val_loss={val_loss:.4f} train_f1_macro={train_f1_macro:.4f}, val_f1_macro={val_f1_macro:.4f}'
    print(tx2)
    logging.info(tx2)

    history['train_acc'].append(train_acc)
    history['train_f1_macro'].append(train_f1_macro)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_f1_macro'].append(val_f1_macro)
    history['val_loss'].append(val_loss)
    # save the best model
    if val_f1_macro > best_f1_macro:
        torch.save(model.state_dict(), model_dict_state_path)
        best_f1_macro = val_f1_macro
        n_not_better_steps = 0
    else: # check for early stopping
        n_not_better_steps += 1
        if n_not_better_steps >= PATIENCE:
            break

plt.rcParams["figure.figsize"] = (10,7)
plt.plot(history['train_f1_macro'], label='train F1 macro')
plt.plot(history['val_f1_macro'], label='validation F1 macro')
plt.plot(history['train_loss'], label='train loss')
plt.plot(history['val_loss'], label='validation loss')
plt.title('Training history')
plt.ylabel('F1 macro / loss')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])
plt.grid()
plt.savefig("./luyen_impro_luyen.png")

# Loading pretrained model (best model)
#model = BERTClass(tokenizer)
model = ImprovedBERTClass(tokenizer=tokenizer,num_labels=num_labels)
model.load_state_dict(torch.load(model_dict_state_path))
model = model.to(device)

titles, predictions, prediction_probs, target_values = get_predictions(model, val_data_loader)

tx3 = classification_report(target_values, predictions, target_names=target_list, zero_division=0)
print(tx3)
logging.info(tx3)

