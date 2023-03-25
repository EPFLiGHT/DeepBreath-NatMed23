from os.path import join
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.dataset import AudioDataset
from ..data.sampler import BalancedSampler
from ..models.SampleModel import SampleModel


def make_sample_loader(ds, config):
    batch_size = config.batch_size

    if config.balanced_sampling:
        balanced_sampler = BalancedSampler(
            ds.samples_df,
            pos_label=config.target,
            label_col="diagnosis",
            additional_cols=["location"],
        )
        data_loader = DataLoader(
            ds,
            batch_sampler=balanced_sampler.get_sampler(
                batch_size=batch_size, alpha=config.sampling_alpha
            ),
        )
    else:
        data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return data_loader


def make_optimizer(model, optimizer, parameters):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, **parameters)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **parameters)
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), **parameters)

    return optimizer


def make_loss():
    criterion = nn.BCELoss()
    return criterion


def make_sample_model(samples_df, data, fold_1, fold_2, config, device):
    loc_selection = (samples_df.location.isin(config.train_loc)).values
    samples_df = samples_df[loc_selection].reset_index(drop=True)
    data = data[loc_selection]

    indices_1 = (samples_df.fold == fold_1).values
    indices_2 = (samples_df.fold == fold_2).values
    train_indices = ~(indices_1 | indices_2)

    train_ds = AudioDataset(
        samples_df[train_indices],
        data=data[train_indices],
        target=config.target,
        preprocessing=config.preprocessing,
        pre_config=config.pre_config,
        split_config=config.split_config,
    )
    ds_1 = AudioDataset(
        samples_df[indices_1],
        data=data[indices_1],
        target=config.target,
        preprocessing=config.preprocessing,
        pre_config=config.pre_config,
        split_config=config.split_config,
        train=False,
    )
    ds_2 = AudioDataset(
        samples_df[indices_2],
        data=data[indices_2],
        target=config.target,
        preprocessing=config.preprocessing,
        pre_config=config.pre_config,
        split_config=config.split_config,
        train=False,
    )

    train_loader = make_sample_loader(train_ds, config)
    loader_1 = DataLoader(ds_1, batch_size=1, shuffle=False)
    loader_2 = DataLoader(ds_2, batch_size=1, shuffle=False)

    # Make the model
    model = SampleModel(**config.network).to(device)

    # Make the loss and optimizer
    criterion = make_loss()
    optimizer = make_optimizer(
        model, config.optimizer["name"], config.optimizer["parameters"]
    )

    return model, train_loader, loader_1, loader_2, optimizer, criterion


def make_features(samples_df, data_loader, val_fold, test_fold, config, device):
    outputs = np.zeros(len(samples_df))
    target_str = "+".join([str(t) for t in config.target])

    max_samples = int(config.split_config["sr"] * config.split_config["max_duration"])
    max_stft_samples = (max_samples // config.feat_config["hop_length"]) + 1
    frame_values = np.zeros((len(samples_df), max_stft_samples + 1))
    attention_values = np.zeros((len(samples_df), max_stft_samples + 1))

    model = SampleModel(**config.network).to(device)
    model_dir = join(config.out_folder, "models")
    model_file = join(
        model_dir,
        config.model_file.format(
            config.network["feature_model"], target_str, val_fold, test_fold
        ),
    )
    model.load_state_dict(torch.load(model_file, map_location=device))
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

    if config.save_outputs:
        # Create folder if it does not exist
        out_dir = join(config.out_folder, "features")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Save model outputs
        output_file = join(
            out_dir,
            config.output_file.format(
                config.network["feature_model"], target_str, val_fold, test_fold
            ),
        )
        np.save(output_file, outputs)

        # Save model attention values
        attn_file = join(
            out_dir,
            config.attn_file.format(
                config.network["feature_model"], target_str, val_fold, test_fold
            ),
        )
        np.save(attn_file, attention_values)

    return outputs
