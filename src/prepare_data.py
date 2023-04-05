import os
import random
import sys

import librosa
import numpy as np
import pandas as pd
from transformers import HfArgumentParser

from preprocessing.helpers import prepare_data
from utils.arguments import DataArguments, AudioArguments
from utils.constants import (
    SEED,
    PATIENT_DF_FILE,
    SAMPLES_DF_FILE,
    AUDIO_DATA_FILE,
)

# Ignore excessive warnings
import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# Set random seeds and deterministic pytorch for reproducibility
random.seed(SEED)  # python random seed
np.random.seed(SEED)  # numpy random seedimport numpy as np


def get_samples(
    patient_df,
    sr,
    max_duration,
    interim_path,
    audio_data_path,
    samples_df_path,
):
    sample_length = int(sr * max_duration)

    audio_data = []
    samples_df = []

    for _, r in patient_df.iterrows():
        patient_id = r.patient
        print(patient_id)

        audio_folder = os.path.join(interim_path, patient_id, "audio")
        audio_files = sorted([f for f in os.listdir(audio_folder)])

        for f in audio_files:
            filename = os.path.join(audio_folder, f)
            samples, sr = librosa.load(filename, sr=None, duration=max_duration)
            n_samples = samples.size
            assert sr == sr

            # Add sample to dataset
            samples = librosa.util.fix_length(samples, size=sample_length)
            audio_data.append(samples)
            code = f.split(".")[0]
            position = code.split("_")[3]
            new_sample = dict(r)
            new_sample["file"] = filename
            new_sample["position"] = position
            new_sample["end"] = n_samples
            samples_df.append(new_sample)

    audio_data = np.array(audio_data)
    np.save(audio_data_path, audio_data)

    samples_df = pd.DataFrame(samples_df)
    samples_df.to_csv(samples_df_path, index=False)


def main():
    parser = HfArgumentParser((AudioArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        audio_args, data_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        audio_args, data_args = parser.parse_args_into_dataclasses()

    sr, max_duration = audio_args.sr, audio_args.max_duration
    # import pdb; pdb.set_trace()
    project_locations = {
        "As": data_args.asthmoscope_locations,
        "Pn": data_args.pneumoscope_locations,
    }

    patient_df_path = os.path.join(data_args.processed_path, PATIENT_DF_FILE)
    audio_data_path = os.path.join(data_args.processed_path, AUDIO_DATA_FILE)
    samples_df_path = os.path.join(data_args.processed_path, SAMPLES_DF_FILE)

    patient_df = prepare_data(
        data_args.data_path, project_locations, out_path=data_args.interim_path
    )
    patient_df.to_csv(patient_df_path, index=False)

    get_samples(
        patient_df=patient_df,
        sr=sr,
        max_duration=max_duration,
        interim_path=data_args.interim_path,
        audio_data_path=audio_data_path,
        samples_df_path=samples_df_path,
    )


if __name__ == "__main__":
    main()
