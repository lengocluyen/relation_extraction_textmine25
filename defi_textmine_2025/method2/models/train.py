"""
python -m defi_textmine_2025.method2.models.train
"""

from collections import defaultdict
import os
import logging
from defi_textmine_2025.settings import (
    LOGGING_DIR,
    get_now_time_as_str,
)
from defi_textmine_2025.set_logging import config_logging

start_date_as_str = get_now_time_as_str()
task_name = "RC3"
config_logging(f"{LOGGING_DIR}/method2/train-{task_name}-{start_date_as_str}.log")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AdamW

from defi_textmine_2025.data.utils import (
    compute_class_weights,
    get_cat_var_distribution,
)
from defi_textmine_2025.bert_dataset_and_models import eval_model, train_model
from defi_textmine_2025.method2.models.shared_toolbox import (
    BASE_CHECKPOINT_NAME,
    MAX_N_TOKENS,
    get_model_checkpoint_basename,
    get_model_checkpoint_path,
    init_model,
    load_model,
    save_model,
    task_name2targetcolumns,
    task_name2ismultilabel,
    get_data_loaders,
    get_task_data,
)

TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 96
LEARNING_RATE = 2e-7
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 100
PATIENCE = 10


if __name__ == "__main__":

    num_fold = 1  # int(sys.argv[1])
    model_name: str = "BERT+MLP"

    logging.info(f"Training {model_name} for {task_name} on {num_fold=}")
    target_columns = task_name2targetcolumns[task_name]
    logging.info(f"{target_columns=}")
    ismultilabel = task_name2ismultilabel[task_name]
    logging.info(f"{ismultilabel=}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"{device=}")
    logging.warning(f"Preparing data for the sub-task {task_name}...")
    train_df, val_df = get_task_data(task_name, num_fold)
    train_data_loader, val_data_loader = get_data_loaders(
        task_name, train_df, val_df, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE
    )
    n_examples = train_df.shape[0]
    n_classes = (
        train_df[target_columns].nunique()
        if isinstance(target_columns, str)
        else len(target_columns)
    )
    logging.info(f"{n_examples=}, {n_classes=}")

    logging.warning(f"Computute class weights with {n_examples=}, {n_classes=}")
    class_weights_df = compute_class_weights(
        train_df,
        label_columns=(
            [target_columns] if isinstance(target_columns, str) else target_columns
        ),
    )
    class_distr_weight_df = pd.concat(
        [get_cat_var_distribution(train_df[target_columns]), class_weights_df],
        axis=1,
    )
    logging.info(f"Class weights: \n{class_distr_weight_df}")
    class_weights = torch.from_numpy(class_weights_df.values).to(device)
    logging.info(f"{class_weights=}")
    model_state_basename = get_model_checkpoint_basename(
        task_name, num_fold, BASE_CHECKPOINT_NAME, model_name
    )
    logging.warning("Initialize the model")
    model = init_model(task_name, model_name, n_classes, BASE_CHECKPOINT_NAME)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    last_epoch = 0
    best_val_f1_macro = 0
    train_loop_history = defaultdict(list)
    model_ckpt_path = get_model_checkpoint_path(
        task_name, num_fold, BASE_CHECKPOINT_NAME
    )
    if os.path.exists(model_ckpt_path):
        logging.warning(
            f"Resume training from last best checkpoint at {model_ckpt_path}"
        )
        (
            model,
            optimizer,
            last_epoch,
            train_loop_history,
            best_val_f1_macro,
        ) = load_model(
            task_name, num_fold, BASE_CHECKPOINT_NAME, model, optimizer, device
        )

    logging.info(f"model:\n{model}")

    logging.warning("Start the training loop...")
    # Training loop

    logging.info(
        f"{TRAIN_BATCH_SIZE=}, {VAL_BATCH_SIZE=}, {LEARNING_RATE=}, {WEIGHT_DECAY=}, "
        f"{MAX_N_TOKENS=}, {MAX_EPOCHS=}, {PATIENCE=}"
    )
    patience_count = 0
    for epoch in range(last_epoch + 1, MAX_EPOCHS + 1):
        logging.info(
            f"Starting epoch {epoch}/{MAX_EPOCHS} [patience count: {patience_count} "
            f"/ {PATIENCE} - {best_val_f1_macro=:.5f}]"
        )
        train_loss, train_acc, train_f1 = train_model(
            model,
            train_data_loader,
            optimizer,
            class_weights,
            device,
            multilabel=ismultilabel,
        )
        val_loss, val_acc, val_f1 = eval_model(
            model, val_data_loader, class_weights, device, multilabel=ismultilabel
        )

        logging.info(f"{train_loss=:.4f}, {train_acc=:.3f}, {train_f1=:.5f}")
        logging.info(f"{val_loss=:.4f}, {val_acc=:.3f}, {val_f1=:.5f}")

        train_loop_history["train_loss"].append(train_loss)
        train_loop_history["train_acc"].append(train_acc)
        train_loop_history["train_f1_macro"].append(train_f1)
        train_loop_history["val_loss"].append(val_loss)
        train_loop_history["val_acc"].append(val_acc)
        train_loop_history["val_f1_macro"].append(val_f1)

        plt.figure()
        plt.rcParams["figure.figsize"] = (10, 7)
        plt.plot(
            train_loop_history["train_f1_macro"],
            label=f"train F1 macro [max={np.max(train_loop_history['train_f1_macro']):.5f}]",
        )
        plt.plot(
            train_loop_history["val_f1_macro"],
            label=f"validation F1 macro [max={np.max(train_loop_history['val_f1_macro']):.5f}]",
        )
        plt.plot(
            train_loop_history["train_loss"],
            label=f"train loss [min={np.min(train_loop_history['train_loss']):.5f}]",
        )
        plt.plot(
            train_loop_history["val_loss"],
            label=f"validation loss [min={np.min(train_loop_history['val_loss']):.5f}]",
        )
        plt.title(f"Training history - {task_name} sub-task")
        plt.ylabel("F1 macro / loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.ylim([0, 1])
        plt.grid()
        plt.savefig(
            f"{LOGGING_DIR}/method2/training_history_{model_state_basename}"
            f"-{start_date_as_str}.png"
        )

        # Save best model
        if val_f1 > best_val_f1_macro:
            best_val_f1_macro = val_f1
            patience_count = 0
            logging.warning(
                f"#NEW_BEST_VAL_F1 {val_f1:.5f} -> Saving new best model"
                f" at {model_ckpt_path}"
            )
            save_model(
                task_name,
                num_fold,
                BASE_CHECKPOINT_NAME,
                model,
                optimizer,
                epoch,
                train_loop_history,
                best_val_f1_macro,
            )
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logging.info("Early stopping")
                break
