from collections import OrderedDict
import tqdm.notebook as tq
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, target_list):
        self.tokenizer = tokenizer
        self.df = df
        self.title = list(df["text"])
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
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "token_type_ids": inputs["token_type_ids"].flatten(),
            "targets": torch.FloatTensor(self.targets[index]),
            "title": text,
        }


class BertBasedModel(nn.Module):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        embedding_model: PreTrainedModel,
        head_model: torch.nn.Sequential,
    ):
        super(BertBasedModel, self).__init__()
        self.embedding_model = embedding_model
        # if you want to add new tokens to the vocabulary, then in general you’ll need
        # to resize the embedding layers with
        # Source https://discuss.huggingface.co/t/adding-new-tokens-while-preserving-tokenization-of-adjacent-tokens/12604
        self.embedding_model.resize_token_embeddings(len(tokenizer))
        self.head_model = head_model

    def forward(
        self,
        input_ids: torch.tensor,
        attn_mask: torch.tensor,
        token_type_ids: torch.tensor,
    ) -> torch.tensor:
        output = self.embedding_model(
            input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids
        )
        output = self.head_model(output.pooler_output)
        return output


class LinearHeadBertBasedModel(BertBasedModel):
    def __init__(
        self,
        tokenizer: PreTrainedModel,
        embedding_model: PreTrainedModel,
        embedding_size: int,
        n_classes: int,
    ):
        head_model = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(embedding_size, n_classes)
        )
        super(LinearHeadBertBasedModel, self).__init__(
            tokenizer, embedding_model, head_model
        )


class Transpose(nn.Module):

    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, tensor):
        return tensor.transpose(1, 2)


class Squeeze(nn.Module):

    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, tensor):
        return tensor.squeeze()


class Conv1dHeadBertBasedModel(BertBasedModel):
    def __init__(
        self,
        tokenizer: PreTrainedModel,
        embedding_model: PreTrainedModel,
        n_classes: int,
        embedding_size: int,
        frac: int = 2,
        kernel: int = 5,
    ):
        """Sources:
        - https://gist.github.com/cjmcmurtrie/bcf2bce22715545559f52c28716813f2
        - https://stackoverflow.com/questions/71309113/cnn-model-and-bert-with-text

        Args:
            tokenizer (PreTrainedModel): _description_
            embedding_model (PreTrainedModel): _description_
            n_classes (int): _description_
            embedding_size (int): _description_
            frac (int, optional): _description_. Defaults to 2.
            kernel (int, optional): _description_. Defaults to 5.
        """
        head_model = nn.Sequential(
            Transpose(),
            nn.Conv1d(embedding_size, embedding_size // frac, kernel, padding="valid"),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(
                embedding_size // frac, embedding_size // frac, kernel, padding="valid"
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(embedding_size // frac, embedding_size // frac, 27),
            nn.ReLU(),
            Squeeze(),
            nn.Linear(embedding_size // frac, n_classes),
        )
        super(Conv1dHeadBertBasedModel, self).__init__(
            tokenizer, embedding_model, head_model
        )


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
    loop = tq.tqdm(
        enumerate(training_loader),
        total=len(training_loader),
        leave=True,
        colour="steelblue",
    )
    for batch_idx, data in loop:
        ids = data["input_ids"].to(device, dtype=torch.long)
        mask = data["attention_mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.float)

        # forward
        outputs = model(ids, mask, token_type_ids)  # (batch,predict)=(32,37)
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
        correct_predictions += np.sum(preds.numpy() == targets.numpy())
        num_samples += targets.numpy().size  # total number of elements in the 2D array

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
        # loop.set_description(f"")
        # loop.set_postfix(batch_loss=loss)

    # returning: trained model, model accuracy, mean loss
    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    target_values = torch.stack(target_values)

    return (
        model,
        float(correct_predictions) / num_samples,
        f1_score(target_values, predictions, average="macro", zero_division=0),
        np.mean(losses),
    )
    # return model, float(correct_predictions)/num_samples, np.mean(losses)


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
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)

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


def eval_model(model: LinearHeadBertBasedModel, validation_loader, device):
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
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

            # validation accuracy
            # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
            outputs = torch.sigmoid(outputs).cpu().detach()
            # thresholding at 0.5
            preds = outputs.round()
            targets = targets.cpu().detach()
            correct_predictions += np.sum(preds.numpy() == targets.numpy())
            num_samples += (
                targets.numpy().size
            )  # total number of elements in the 2D array

            predictions.extend(preds)
            prediction_probs.extend(outputs)
            target_values.extend(targets)

    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    target_values = torch.stack(target_values)

    return (
        float(correct_predictions) / num_samples,
        f1_score(target_values, predictions, average="macro", zero_division=0),
        np.mean(losses),
    )
