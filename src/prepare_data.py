import argparse
import librosa
import numpy as np
import pandas as pd
import random
from os import listdir
from os.path import join

import utils.config as config
from preprocessing.helpers import prepare_data

# Ignore excessive warnings
import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# Set random seeds and deterministic pytorch for reproducibility
random.seed(config.SEED)  # python random seed
np.random.seed(config.SEED)  # numpy random seedimport numpy as np


def get_samples(patient_df, out_path, audio_array_path, audio_meta_path):
    sample_length = config.MAX_DURATION * config.RATE

    audio_dataset = []
    samples_df = []

    for _, r in patient_df.iterrows():
        patient_id = r.patient
        print(patient_id)

        audio_folder = join(out_path, patient_id, "audio")
        audio_files = sorted([f for f in listdir(audio_folder)])

        for f in audio_files:
            filename = join(audio_folder, f)
            samples, sr = librosa.load(filename, sr=None, duration=config.MAX_DURATION)
            n_samples = samples.size
            assert sr == config.RATE

            # Add sample to dataset
            samples = librosa.util.fix_length(samples, size=sample_length)
            audio_dataset.append(samples)
            code = f.split(".")[0]
            position = code.split("_")[3]
            new_sample = dict(r)
            new_sample["file"] = filename
            new_sample["position"] = position
            new_sample["end"] = n_samples
            samples_df.append(new_sample)

    audio_dataset = np.array(audio_dataset)
    np.save(audio_array_path, audio_dataset)

    samples_df = pd.DataFrame(samples_df)
    samples_df.to_csv(audio_meta_path, index=False)


if __name__ == "__main__":
    # Defines all parser arguments when launching the script directly in terminal
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-al",
        "--asthmoscope_locations",
        nargs="+",
        help="Asthmoscope study centres used to create a dataset",
    )
    parser.add_argument(
        "-pl",
        "--pneumoscope_locations",
        nargs="+",
        help="Pneumoscope study centres used to create a dataset",
    )
    parser.add_argument(
        "-dp",
        "--download_path",
        type=str,
        default="../data/raw/canonical_data/pediatric_studies",
        help="Path to downloaded data",
    )
    parser.add_argument(
        "-op",
        "--out_path",
        type=str,
        default="../data",
        help="Destination folder for selected recordings",
    )
    args = parser.parse_args()

    project_locations = {
        "As": args.asthmoscope_locations
        if args.asthmoscope_locations is not None
        else ["GVA"],
        "Pn": args.pneumoscope_locations
        if args.pneumoscope_locations is not None
        else ["GVA", "POA", "DKR", "MAR", "RBA", "YAO"],
    }
    _download_path = args.download_path
    _out_path = args.out_path
    _interim_path = join(_out_path, "interim")
    _patient_path = join(_out_path, "processed", config.PATIENT_DF_FILE)
    _audio_array_path = join(_out_path, "processed", config.AUDIO_DATA_FILE)
    _audio_meta_path = join(_out_path, "processed", config.SAMPLES_DF_FILE)

    _patient_df = prepare_data(
        _download_path, project_locations, out_path=_interim_path
    )
    _patient_df.to_csv(_patient_path, index=False)
    get_samples(_patient_df, _interim_path, _audio_array_path, _audio_meta_path)
