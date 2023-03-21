import argparse
import logging
import random
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader

# Local imports
import training.pipeline as pipeline
import training.sample_fit as sample_fit
from config import SEED, pipeline_config
from data.dataset import AudioDataset

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


unknown_diagnosis = ['Pn_POA_Ca6', 'Pn_POA_Ca12', 'Pn_POA_Ca13', 'Pn_POA_Ca18', 'Pn_POA_Ca20',
                     'Pn_POA_Ca23', 'Pn_POA_Ca24', 'Pn_POA_Ca26', 'Pn_POA_Ca32', 'Pn_POA_Ca37',
                     'Pn_POA_Ca40', 'Pn_POA_Ca41', 'Pn_POA_Ca44', 'Pn_POA_Ca45', 'Pn_POA_Ca47',
                     'Pn_POA_Ca48', 'Pn_POA_Ca49', 'Pn_POA_Ca56', 'Pn_POA_Ca60', 'Pn_POA_Ca67',
                     'Pn_POA_Ca69', 'Pn_POA_Ca76', 'Pn_POA_Ca79', 'Pn_POA_Ca89', 'Pn_POA_Ca90',
                     'Pn_POA_Ca92', 'Pn_POA_Ca93', 'Pn_POA_Ca99', 'Pn_POA_Ca103', 'Pn_POA_Ca104',
                     'Pn_POA_Ca105', 'Pn_POA_Ca113', 'Pn_POA_Ca123', 'Pn_POA_Ca127', 'Pn_POA_Ca136',
                     'Pn_POA_Ca140', 'Pn_POA_Ca142', 'Pn_POA_Ca144', 'Pn_POA_Ca148', 'Pn_POA_Ca152',
                     'Pn_POA_Ca166', 'Pn_POA_Ca170', 'Pn_POA_Ca173', 'Pn_POA_Ca179', 'Pn_POA_Ca182',
                     'Pn_POA_Ca186', 'Pn_POA_Ca188', 'Pn_POA_Ca189', 'Pn_POA_Ca190', 'Pn_POA_Ca191',
                     'Pn_POA_Ca193', 'Pn_POA_Ca213', 'Pn_POA_Ca218', 'Pn_POA_Ca223', 'Pn_POA_Ca224',
                     'Pn_POA_Ca225', 'Pn_POA_Ca227', 'Pn_POA_Ca228', 'Pn_POA_Ca232', 'Pn_POA_Ca234',
                     'Pn_POA_Ca236', 'Pn_POA_Ca242', 'Pn_POA_Ca245']


def position_aggregation(samples_df, data_loader, val_fold, test_fold, config, device):
    # Get sample features
    target = config.target
    target_str = '+'.join([str(t) for t in target])

    sample_outputs = pipeline.make_features(samples_df, data_loader, val_fold, test_fold, config, device)

    output_df = samples_df[["patient", "position"]]
    output_df = output_df.assign(output=sample_outputs).rename(columns={"output": f"output_{target_str}"})

    out_dir = join(config.out_folder, "aggregate")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = join(out_dir,
                    config.aggregate_file.format(config.network["feature_model"], target_str, val_fold, test_fold))
    output_df.to_csv(out_path, index=False)


def evaluate_and_save(model, loader_1, loader_2, fold_1, fold_2, criterion, best_score, config, device, val_first=True):
    if val_first:
        val_loader = loader_1
        val_fold = fold_1
        test_fold = fold_2
    else:
        val_loader = loader_2
        val_fold = fold_2
        test_fold = fold_1

    metrics, pos_score, example_spect = sample_fit.evaluate(model, val_loader, criterion, device)

    avg_score = [score for score in pos_score.values()]
    avg_score = np.array(avg_score).mean()
    if avg_score > best_score:
        best_score = avg_score
        print(f"Best overall model ({val_fold}-{test_fold})\n")
        target_str = '+'.join([str(t) for t in config.target])
        out_dir = join(config.out_folder, "models")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model_path = join(out_dir,
                          config.model_file.format(config.network["feature_model"], target_str, val_fold, test_fold))
        torch.save(model.state_dict(), model_path)

    return metrics, best_score


def fit_samples(model, train_loader, loader_1, loader_2, fold_1, fold_2, optimizer, criterion, cv_index, config):
    # wandb.watch(model, log="all")

    best_score_1 = 0
    best_score_2 = 0

    print("Steps per epoch:", len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader),
                                                    epochs=config.epochs)  # added

    for epoch in range(config.epochs):
        step = cv_index * config.epochs + epoch
        # train_loss = sample_fit.train_epoch_mod(model, train_loader, optimizer, criterion, epoch, device,
        train_loss = sample_fit.train_epoch(model, train_loader, optimizer, criterion, epoch, device,
                                            scheduler=scheduler, log_interval=10)
        wandb.log({
            "Train Loss": train_loss,
        }, step=step)

        if epoch + 1 >= config.validation_start:
            # 1. Evaluate on first fold
            metrics, best_score_1 = evaluate_and_save(model, loader_1, loader_2, fold_1, fold_2, criterion,
                                                      best_score_1, config, device, val_first=True)
            wandb.log(metrics, step=step)

            # 2. Evaluate on second fold
            metrics, best_score_2 = evaluate_and_save(model, loader_1, loader_2, fold_1, fold_2, criterion,
                                                      best_score_2, config, device, val_first=False)


def model_pipeline(samples_df, data, fold_1, fold_2, cv_index, config):
    model, train_loader, loader_1, loader_2, optimizer, criterion = pipeline.make_sample_model(samples_df, data, fold_1,
                                                                                               fold_2, config, device)
    if not no_fit:
        fit_samples(model, train_loader, loader_1, loader_2, fold_1, fold_2, optimizer, criterion, cv_index, config)

    ds = AudioDataset(samples_df, data=data, target=config.target, preprocessing=config.preprocessing,
                      pre_config=config.pre_config, split_config=config.split_config, train=False)

    data_loader = DataLoader(ds, batch_size=1, shuffle=False)

    position_aggregation(samples_df, data_loader, val_fold=fold_1, test_fold=fold_2, config=config, device=device)
    position_aggregation(samples_df, data_loader, val_fold=fold_2, test_fold=fold_1, config=config, device=device)


def load_data(config):
    samples_df = pd.read_csv(config.samples_df_path)
    data = np.load(config.samples_path)

    if config.target[0] != 0:
        print("Removing patients with multiple diagnoses")
        diagnosis_filter = ~(samples_df.multilabel | samples_df.diagnosis.isin(config.exclude)).values
        samples_df = samples_df[diagnosis_filter].reset_index(drop=True)
        data = data[diagnosis_filter]

    if 3 in config.target or 5 in config.target:
        print("Removing patients with unknown diagnosis")
        diagnosis_filter = ~(samples_df.patient.isin(unknown_diagnosis)).values
        samples_df = samples_df[diagnosis_filter].reset_index(drop=True)
        data = data[diagnosis_filter]

    print("Filtering YAO data")
    position_filter = (samples_df.position.isin([f"P{i + 1}" for i in range(8)]))
    samples_df = samples_df[position_filter].reset_index(drop=True)
    data = data[position_filter]

    return samples_df, data


def experiment(config, online=False):
    # Initialize a new wandb run
    mode = "online" if online else "offline"
    with wandb.init(project="deep-breath", config=config, mode=mode):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        samples_df, data = load_data(config)

        for cv_index, (fold_1, fold_2) in enumerate(config.cv_folds):
            print()
            print('#' * 25, fold_1, fold_2, '#' * 25)
            model_pipeline(samples_df, data, fold_1, fold_2, cv_index, config)
            print()


if __name__ == "__main__":
    # Defines all parser arguments when launching the script directly in terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('target_list', type=int, nargs='+', help="Diagnosis code(s) of target class")
    parser.add_argument("-nf", "--no_fit", help="Skip model training, use saved models to produce features",
                        action="store_true")
    args = parser.parse_args()

    no_fit = args.no_fit
    pipeline_config["target"] = args.target_list

    experiment(pipeline_config)
