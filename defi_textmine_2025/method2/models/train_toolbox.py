import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score


def loss_fn(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    multilabel: bool = True,
) -> float:
    """
    BCEWithLogitsLoss for multilabel and CrossEntropyLoss for single-label.

    BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
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
    if multilabel:
        return torch.nn.BCEWithLogitsLoss(weight=class_weights)(outputs, targets)
    else:
        return torch.nn.CrossEntropyLoss(weight=class_weights)(outputs, targets)


def train_model(
    model: nn.Module,
    training_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    _class_weights_tensor: torch.Tensor,
    device: torch.device,
    multilabel: bool = True,
):
    predictions = []
    prediction_probs = []
    target_values = []
    losses = []
    correct_predictions = 0
    num_samples = 0
    # set model to training mode (activate dropout, batch norm)
    model.train()
    # initialize the progress bar
    loop = tqdm(
        enumerate(training_loader),
        desc="Training",
        total=len(training_loader),
        leave=True,
        # colour="steelblue",
    )
    for batch_idx, data in loop:
        ids = data["input_ids"].to(device, dtype=torch.long)
        mask = data["attention_mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.float)

        # forward
        outputs = model(ids, mask, token_type_ids)  # (batch,predict)=(32,37)
        loss = loss_fn(outputs, targets, _class_weights_tensor, multilabel)
        losses.append(loss.item())
        # training accuracy, apply sigmoid, round (apply thresh 0.5)
        # outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
        # targets = targets.cpu().detach().numpy()
        # correct_predictions += np.sum(outputs==targets)
        # num_samples += targets.size   # total number of elements in the 2D array
        if multilabel:
            outputs = torch.sigmoid(outputs).cpu().detach()
        else:  # single-label
            outputs = torch.softmax(outputs, dim=1).cpu().detach()
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
        loop.set_postfix({"batch_loss": f"{loss.cpu().detach().numpy():.5f}"})
        # break

    # returning: trained model, model accuracy, mean loss
    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    target_values = torch.stack(target_values)

    return (
        np.mean(losses),
        float(correct_predictions) / num_samples,
        f1_score(target_values, predictions, average="macro", zero_division=0),
    )


def eval_model(
    model: nn.Module,
    validation_loader: DataLoader,
    _class_weights_tensor: torch.Tensor,
    device: torch.device,
    multilabel: bool = True,
):
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
        for data in tqdm(validation_loader, "Evaluation"):
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)  # logits

            loss = loss_fn(outputs, targets, _class_weights_tensor, multilabel)
            losses.append(loss.item())

            # validation accuracy
            if multilabel:
                # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
                outputs = torch.sigmoid(outputs).cpu().detach()  # probabilities
            else:  # single-label
                outputs = torch.softmax(outputs, dim=1).cpu().detach()  # probabilities
            # thresholding at 0.5
            preds: torch.Tensor = outputs.round()  # 0 1 labels
            targets: torch.Tensor = targets.cpu().detach()  # 0 1 labels
            correct_predictions += np.sum(preds.numpy() == targets.numpy())
            num_samples += (
                targets.numpy().size
            )  # total number of elements in the 2D array

            predictions.extend(preds)
            prediction_probs.extend(outputs)
            target_values.extend(targets)
            # break

    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    target_values = torch.stack(target_values)

    return (
        np.mean(losses),
        float(correct_predictions) / num_samples,
        f1_score(target_values, predictions, average="macro", zero_division=0),
    )
