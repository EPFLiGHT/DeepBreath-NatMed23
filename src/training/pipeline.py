import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import AudioDataset
from data.sampler import BalancedSampler
from models.SampleModel import SampleModel
from utils.helpers import get_model_file, get_output_file, get_attention_file
import torch.utils.data.dataloader
from nn.modules.loss import BCELoss
from numpy import ndarray
from optim.adamw import AdamW
from pandas.core.frame import DataFrame
from torch.nn.modules.loss import BCELoss
from torch.optim.adamw import AdamW
from typing import Any, Iterator, Tuple
from utils.arguments import AudioArguments, ModelArguments, TrainingArguments
from utils.data.dataloader import DataLoader


def make_sample_loader(
    ds: AudioDataset, train_args: TrainingArguments
) -> torch.utils.data.dataloader.DataLoader:
    batch_size = train_args.batch_size

    if train_args.balanced_sampling:
        balanced_sampler = BalancedSampler(
            ds.samples_df,
            pos_label=train_args.target,
            label_col="diagnosis",
            additional_cols=["location"],
        )
        data_loader = DataLoader(
            ds,
            batch_sampler=balanced_sampler.get_sampler(
                batch_size=batch_size, alpha=train_args.sampling_alpha
            ),
        )
    else:
        data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return data_loader


def make_optimizer(params: Iterator[Any], train_args: TrainingArguments) -> AdamW:
    if train_args.optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params=params,
            momentum=train_args.momentum,
            lr=train_args.learning_rate,
            weight_decay=train_args.weight_decay,
        )
    elif train_args.optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            params=params,
            lr=train_args.learning_rate,
            weight_decay=train_args.weight_decay,
        )
    elif train_args.optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            params=params,
            lr=train_args.learning_rate,
            weight_decay=train_args.weight_decay,
        )

    return optimizer


def make_loss() -> BCELoss:
    criterion = nn.BCELoss()
    return criterion


def make_sample_model(
    samples_df: DataFrame,
    data: ndarray,
    fold_1: int,
    fold_2: int,
    audio_args: AudioArguments,
    model_args: ModelArguments,
    train_args: TrainingArguments,
    device: torch.device,
) -> Tuple[SampleModel, DataLoader, DataLoader, DataLoader, AdamW, BCELoss]:
    loc_selection = (samples_df.location.isin(train_args.train_loc)).values
    samples_df = samples_df[loc_selection].reset_index(drop=True)
    data = data[loc_selection]

    indices_1 = (samples_df.fold == fold_1).values
    indices_2 = (samples_df.fold == fold_2).values
    train_indices = ~(indices_1 | indices_2)

    train_ds = AudioDataset(
        samples_df[train_indices],
        target=train_args.target,
        data=data[train_indices],
        audio_args=audio_args,
    )
    ds_1 = AudioDataset(
        samples_df[indices_1],
        target=train_args.target,
        data=data[indices_1],
        audio_args=audio_args,
        train=False,
    )
    ds_2 = AudioDataset(
        samples_df[indices_2],
        target=train_args.target,
        data=data[indices_2],
        audio_args=audio_args,
        train=False,
    )

    train_loader = make_sample_loader(train_ds, train_args)
    loader_1 = DataLoader(ds_1, batch_size=1, shuffle=False)
    loader_2 = DataLoader(ds_2, batch_size=1, shuffle=False)

    # Make the model
    model = SampleModel(audio_config=audio_args, model_config=model_args).to(device)

    # Make the loss and optimizer
    criterion = make_loss()
    optimizer = make_optimizer(model.parameters(), train_args)

    return model, train_loader, loader_1, loader_2, optimizer, criterion


def make_features(
    samples_df: DataFrame,
    data_loader: torch.utils.data.dataloader.DataLoader,
    val_fold: int,
    test_fold: int,
    audio_args: AudioArguments,
    model_args: ModelArguments,
    train_args: TrainingArguments,
    device: torch.device,
) -> ndarray:
    outputs = np.zeros(len(samples_df))
    model_name = model_args.model_name
    target_str = "+".join([str(t) for t in train_args.target])

    max_samples = int(audio_args.sr * audio_args.max_duration)
    max_stft_samples = (max_samples // audio_args.hop_length) + 1
    frame_values = np.zeros((len(samples_df), max_stft_samples + 1))
    attention_values = np.zeros((len(samples_df), max_stft_samples + 1))

    model = SampleModel(audio_config=audio_args, model_config=model_args).to(device)
    model_file = get_model_file(model_name, target_str, val_fold, test_fold)
    model_path = os.path.join(train_args.out_path, "models", model_file)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        for batch_dict in data_loader:  # added
            # Load the input features and labels from the dataset
            sample_idx = batch_dict["sample_idx"].item()
            data = batch_dict["data"].to(device)

            # Make prediction
            output_dict = model(data)  # added
            output = output_dict["diagnosis_output"]

            # Add output and attention values
            outputs[sample_idx] = output.item()

            if "framewise_att" in output_dict.keys():
                framewise_output = (
                    output_dict["framewise_output"].cpu().numpy().flatten()
                )
                framewise_att = output_dict["framewise_att"].cpu().numpy().flatten()
                n_frames = len(framewise_output)
                assert n_frames <= max_stft_samples
                frame_values[sample_idx, :n_frames] = framewise_output
                frame_values[sample_idx, -1] = n_frames
                attention_values[sample_idx, :n_frames] = framewise_att
                attention_values[sample_idx, -1] = n_frames

    if train_args.save_outputs:
        # Create folder if it does not exist
        feature_dir = os.path.join(train_args.out_path, "features")
        Path(feature_dir).mkdir(parents=True, exist_ok=True)

        # Save model outputs
        output_file = get_output_file(model_name, target_str, val_fold, test_fold)
        output_file = os.path.join(feature_dir, output_file)
        np.save(output_file, outputs)

        # Save model attention values
        attn_file = get_attention_file(model_name, target_str, val_fold, test_fold)
        attn_file = os.path.join(feature_dir, attn_file)
        np.save(attn_file, attention_values)

    return outputs
