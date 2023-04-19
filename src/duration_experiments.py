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
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

# Local imports
import training.pipeline as pipeline
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
from utils.helpers import get_aggregate_file


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

    agg_dir = os.path.join(
        train_args.out_path, "aggregate", str(audio_args.max_duration)
    )
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

    max_durations = np.linspace(start=2.5, stop=30, num=12)

    # Don't overwrite saved models
    train_args.save_ouputs = False

    for cv_index, (fold_1, fold_2) in enumerate(combinations(range(5), 2)):
        for max_duration in max_durations:
            audio_args.max_duration = max_duration
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
