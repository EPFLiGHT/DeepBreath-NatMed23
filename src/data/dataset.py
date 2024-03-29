import dataclasses
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing.features import AudioFeatures
from utils.arguments import AudioArguments


class AudioDataset(Dataset):
    def __init__(
        self,
        samples_df: pd.DataFrame,
        target: List[int],
        data: np.ndarray,
        audio_args: AudioArguments,
        train: bool = True,
    ) -> None:
        self.audio_args = audio_args

        self.samples_df = samples_df.reset_index().rename(
            columns={"index": "old_index"}
        )
        self.data = self._preprocess_data(data)

        labels = (self.samples_df.diagnosis.isin(target)).values.astype(int)
        self.samples_df["label"] = labels

        self.targets = np.zeros((labels.size, 2))
        self.targets[np.arange(labels.size), labels] = 1

        locations = sorted(self.samples_df.location.unique())
        self.location_code = {loc: i for i, loc in enumerate(locations)}

        self.train = train

    @property
    def samples_df(self):
        return self._samples_df

    @samples_df.setter
    def samples_df(self, df):
        self._samples_df = df

    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        filters = AudioFeatures(features=[], config=dataclasses.asdict(self.audio_args))
        preprocessed_data = np.zeros(data.shape)
        for i, n_samples in enumerate(self.samples_df.end.values):
            audio = data[i][:n_samples]
            filtered_audio = filters.transform(audio)
            preprocessed_data[i][:n_samples] = filtered_audio
        return preprocessed_data

    def get_class_counts(self):
        class_sample_count = np.unique(self.samples_df["label"], return_counts=True)[1]
        return torch.from_numpy(class_sample_count.astype(np.float32))

    def __len__(self) -> int:
        return len(self.samples_df)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        y = np.array([self.samples_df["label"].values[i]]).astype(np.float32)

        sample_length = int(self.audio_args.sr * self.audio_args.split_duration)
        n_samples = self.samples_df.iloc[i].end
        audio = self.data[i][:n_samples]
        if self.train:
            if n_samples < sample_length:
                new_audio = np.zeros(sample_length, dtype=audio.dtype)
                random_start = np.random.choice(sample_length - n_samples)
                new_audio[random_start : (random_start + n_samples)] = audio
                audio = new_audio
            else:
                end_idx = n_samples - sample_length
                random_start = np.random.choice(end_idx)
                audio = audio[random_start : (random_start + sample_length)]
        else:
            # Added to evaluate the impact of recording duration on classification performance
            max_samples = int(self.audio_args.sr * self.audio_args.max_duration)
            if n_samples > max_samples:
                audio = audio[:max_samples]

            if i == 0:
                print("#" * 10, "INFO", "#" * 10)
                print("Number of samples:", n_samples)
                print(
                    "Max Duration:",
                    self.audio_args.max_duration,
                    "- Max Samples:",
                    max_samples,
                )
                print("Final number of samples:", len(audio))
                print("Head:", audio[:5])
                print("Tail:", audio[-5:])
                print()

        x = audio.astype(np.float32)

        batch_dict = {
            "sample_idx": torch.from_numpy(np.array(i)),
            "data": torch.from_numpy(x),
            "target": torch.from_numpy(y),
        }

        return batch_dict
