import json
import logging
import os
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

# Local imports
import training.pipeline as pipeline
import training.sample_fit as sample_fit
from data.dataset import AudioDataset
from utils.arguments import (
    DataArguments,
    AudioArguments,
    ModelArguments,
    TrainingArguments,
)
from utils.constants import (
    SEED,
    UNKNOWN_DIAGNOSES_FILE,
    SAMPLES_DF_FILE,
    AUDIO_DATA_FILE,
)
from utils.helpers import get_model_file, get_aggregate_file


# Ignore excessive warnings
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# Set random seeds and deterministic pytorch for reproducibility
random.seed(SEED)  # python random seed
torch.manual_seed(SEED)  # pytorch random seed
np.random.seed(SEED)  # numpy random seed
torch.backends.cudnn.deterministic = True

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wandb.login()


def load_data(data_args, train_args):
    samples_df_path = os.path.join(data_args.processed_path, SAMPLES_DF_FILE)
    audio_data_path = os.path.join(data_args.processed_path, AUDIO_DATA_FILE)

    samples_df = pd.read_csv(samples_df_path)
    data = np.load(audio_data_path)

    if train_args.target[0] != 0:
        print("Removing patients with multiple diagnoses")
        diagnosis_filter = ~(
            samples_df.multilabel | samples_df.diagnosis.isin(train_args.exclude)
        ).values
        samples_df = samples_df[diagnosis_filter].reset_index(drop=True)
        data = data[diagnosis_filter]

    if 3 in train_args.target or 5 in train_args.target:
        unknown_diagnoses_file = os.path.join(
            data_args.data_path, UNKNOWN_DIAGNOSES_FILE
        )
        with open(unknown_diagnoses_file) as f:
            print("Removing patients with unknown diagnosis")
            unknown_diagnoses = json.load(f)
            diagnosis_filter = ~(samples_df.patient.isin(unknown_diagnoses)).values
            samples_df = samples_df[diagnosis_filter].reset_index(drop=True)
            data = data[diagnosis_filter]

    print("Filtering YAO data")
    position_filter = samples_df.position.isin([f"P{i + 1}" for i in range(8)])
    samples_df = samples_df[position_filter].reset_index(drop=True)
    data = data[position_filter]

    return samples_df, data


def evaluate_and_save(
    model,
    loader_1,
    loader_2,
    fold_1,
    fold_2,
    criterion,
    best_score,
    model_args,
    train_args,
    val_first=True,
):
    if val_first:
        val_loader = loader_1
        val_fold = fold_1
        test_fold = fold_2
    else:
        val_loader = loader_2
        val_fold = fold_2
        test_fold = fold_1

    metrics, pos_score = sample_fit.evaluate(model, val_loader, criterion, device)

    avg_score = [score for score in pos_score.values()]
    avg_score = np.array(avg_score).mean()
    if avg_score > best_score:
        best_score = avg_score
        print(f"Best overall model ({val_fold}-{test_fold})\n")
        target_str = "+".join([str(t) for t in train_args.target])
        model_dir = os.path.join(train_args.out_path, "models")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model_file = get_model_file(
            model_args.model_name, target_str, val_fold, test_fold
        )
        model_file = os.path.join(model_dir, model_file)
        torch.save(model.state_dict(), model_file)

    return metrics, best_score


def fit_samples(
    model,
    train_loader,
    loader_1,
    loader_2,
    fold_1,
    fold_2,
    optimizer,
    criterion,
    cv_index,
    model_args,
    train_args,
):
    best_score_1 = 0
    best_score_2 = 0

    print("Steps per epoch:", len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=len(train_loader),
        epochs=train_args.epochs,
    )  # added

    for epoch in range(train_args.epochs):
        step = cv_index * train_args.epochs + epoch
        train_loss = sample_fit.train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch,
            device,
            scheduler=scheduler,
            log_interval=10,
        )
        wandb.log(
            {
                "Train Loss": train_loss,
            },
            step=step,
        )

        if epoch + 1 >= train_args.validation_start:
            # 1. Evaluate on first fold
            metrics, best_score_1 = evaluate_and_save(
                model,
                loader_1,
                loader_2,
                fold_1,
                fold_2,
                criterion,
                best_score_1,
                model_args,
                train_args,
                val_first=True,
            )
            wandb.log(metrics, step=step)

            # 2. Evaluate on second fold
            metrics, best_score_2 = evaluate_and_save(
                model,
                loader_1,
                loader_2,
                fold_1,
                fold_2,
                criterion,
                best_score_2,
                model_args,
                train_args,
                val_first=False,
            )


def position_aggregation(
    samples_df,
    data_loader,
    val_fold,
    test_fold,
    audio_args,
    model_args,
    train_args,
):
    # Get sample features
    target = train_args.target
    target_str = "+".join([str(t) for t in target])

    sample_outputs = pipeline.make_features(
        samples_df,
        data_loader,
        val_fold,
        test_fold,
        audio_args,
        model_args,
        train_args,
        device,
    )

    output_df = samples_df[["patient", "position"]]
    output_df = output_df.assign(output=sample_outputs).rename(
        columns={"output": f"output_{target_str}"}
    )

    agg_dir = os.path.join(train_args.out_path, "aggregate")
    Path(agg_dir).mkdir(parents=True, exist_ok=True)
    agg_file = get_aggregate_file(
        model_args.model_name, target_str, val_fold, test_fold
    )
    agg_file = os.path.join(agg_dir, agg_file)
    output_df.to_csv(agg_file, index=False)


def model_pipeline(
    samples_df,
    data,
    fold_1,
    fold_2,
    cv_index,
    audio_args,
    model_args,
    train_args,
):
    (
        model,
        train_loader,
        loader_1,
        loader_2,
        optimizer,
        criterion,
    ) = pipeline.make_sample_model(
        samples_df, data, fold_1, fold_2, audio_args, model_args, train_args, device
    )

    if train_args.do_train:
        fit_samples(
            model,
            train_loader,
            loader_1,
            loader_2,
            fold_1,
            fold_2,
            optimizer,
            criterion,
            cv_index,
            model_args,
            train_args,
        )

    ds = AudioDataset(
        samples_df,
        target=train_args.target,
        data=data,
        audio_args=audio_args,
        train=False,
    )

    data_loader = DataLoader(ds, batch_size=1, shuffle=False)

    position_aggregation(
        samples_df,
        data_loader,
        val_fold=fold_1,
        test_fold=fold_2,
        audio_args=audio_args,
        model_args=model_args,
        train_args=train_args,
    )
    position_aggregation(
        samples_df,
        data_loader,
        val_fold=fold_2,
        test_fold=fold_1,
        audio_args=audio_args,
        model_args=model_args,
        train_args=train_args,
    )


def main():
    parser = HfArgumentParser(
        (DataArguments, AudioArguments, ModelArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args, audio_args, model_args, train_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            data_args,
            audio_args,
            model_args,
            train_args,
        ) = parser.parse_args_into_dataclasses()

    samples_df, data = load_data(data_args, train_args)

    mode = "online" if train_args.online_logging else "offline"
    with wandb.init(project="deep-breath", mode=mode):
        for cv_index, (fold_1, fold_2) in enumerate(combinations(range(5), 2)):
            print()
            print("#" * 25, fold_1, fold_2, "#" * 25)
            model_pipeline(
                samples_df,
                data,
                fold_1,
                fold_2,
                cv_index,
                audio_args,
                model_args,
                train_args,
            )
            print()


if __name__ == "__main__":
    main()
