from math import ceil
from os import listdir
from os.path import isdir, isfile, join
from pathlib import Path
from shutil import copy, copytree

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydub import AudioSegment

import preprocessing.features as feats
from utils.config import feat_config


def code_with_extension(base_folder, code, extension):
    filename = "{}.{}".format(code, extension)
    return join(base_folder, filename)


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def normalize_and_save(file, output_file, target_dBFS=-20.0):
    sound = AudioSegment.from_file(file, "wav")
    normalized_sound = match_target_amplitude(sound, target_dBFS)
    normalized_sound.export(output_file, format="wav")


def diagnoses_to_int(diagnoses, delimiters=["+", "?"]):
    if isinstance(diagnoses, str):
        for char in delimiters:
            diagnoses = diagnoses.replace(char, " ")
        diagnoses = list(set([int(float(d)) for d in diagnoses.split()]))
    else:
        diagnoses = [int(diagnoses)]

    return diagnoses


def prepare_data(
    root_path,
    project_locations,
    db_columns=["patient", "diagnosis", "new_diagnosis"],
    out_path="../data/interim",
    n_splits=5,
):
    patient_df = []
    for project, locations in project_locations.items():
        for location in locations:
            print(f"{project}: {location}")
            case_folder = join(root_path, f"{project}_{location}_Cases")
            control_folder = join(root_path, f"{project}_{location}_Controls")
            clinical_path = join(
                root_path, "features", f"{project}_{location}_clinical_database.csv"
            )

            if isfile(clinical_path):
                clinical_db = pd.read_csv(clinical_path)
                if "new_diagnosis" not in clinical_db.columns:
                    clinical_db["new_diagnosis"] = clinical_db.diagnosis
                    clinical_db["diagnosis"] = -1
                else:
                    clinical_db["new_diagnosis"] = clinical_db.new_diagnosis.fillna(
                        clinical_db.diagnosis
                    )
                clinical_db = clinical_db[db_columns].dropna(
                    subset=["patient", "new_diagnosis"]
                )
                clinical_db = clinical_db.rename(
                    columns={"diagnosis": "old_diagnosis", "new_diagnosis": "diagnosis"}
                )
                clinical_db["location"] = location

                for _, r in clinical_db.iterrows():
                    patient_id, diagnoses, old_diagnoses = (
                        r["patient"],
                        r["diagnosis"],
                        r["old_diagnosis"],
                    )
                    diagnoses = diagnoses_to_int(diagnoses)
                    old_diagnosis = diagnoses_to_int(old_diagnoses)[0]
                    multilabel = len(diagnoses) > 1
                    if multilabel:
                        print(
                            "\t", patient_id, "has multiple diagnoses:", r["diagnosis"]
                        )

                    r["old_diagnosis"] = old_diagnosis
                    r["all_diagnoses"] = diagnoses
                    r["diagnosis"] = diagnoses[0]
                    r["multilabel"] = multilabel

                    patient_folder = join(
                        control_folder if diagnoses[0] == 0 else case_folder, patient_id
                    )
                    audio_folder = join(patient_folder, "audio")
                    label_folder = join(patient_folder, "labels")

                    if isdir(audio_folder) and len(listdir(audio_folder)):
                        out_audio_folder = join(out_path, patient_id, "audio")
                        out_label_folder = join(out_path, patient_id, "labels")
                        recording_codes = sorted(
                            [
                                f.split(".")[0]
                                for f in listdir(audio_folder)
                                if isfile(join(audio_folder, f))
                            ]
                        )

                        for code in recording_codes:
                            Path(out_audio_folder).mkdir(parents=True, exist_ok=True)
                            audio_file = code_with_extension(audio_folder, code, "wav")
                            output_file = code_with_extension(
                                out_audio_folder, code, "wav"
                            )
                            normalize_and_save(audio_file, output_file)

                            label_file = code_with_extension(label_folder, code, "txt")
                            if isfile(label_file):
                                Path(out_label_folder).mkdir(
                                    parents=True, exist_ok=True
                                )
                                copy(label_file, out_label_folder)

                        patient_df.append(r.to_dict())
            elif project == "As":
                if isdir(case_folder):
                    patients = sorted(
                        [d for d in listdir(case_folder) if isdir(join(case_folder, d))]
                    )
                    for patient_id in patients:
                        patient_folder = join(case_folder, patient_id)
                        audio_folder = join(patient_folder, "audio")

                        if isdir(audio_folder):
                            out_audio_folder = join(out_path, patient_id, "audio")
                            recording_codes = sorted(
                                [
                                    f.split(".")[0]
                                    for f in listdir(audio_folder)
                                    if isfile(join(audio_folder, f))
                                ]
                            )
                            add_patient = False

                            for code in recording_codes:
                                code_split = code.split("_")
                                geo_ts, timing = code_split[-2], code_split[-1]
                                site, check = geo_ts[0], geo_ts[1]
                                if (
                                    (timing == "a") and (site == "S") and (check == "1")
                                ):  # or site == 'H'
                                    Path(out_audio_folder).mkdir(
                                        parents=True, exist_ok=True
                                    )
                                    audio_file = code_with_extension(
                                        audio_folder, code, "wav"
                                    )
                                    output_file = code_with_extension(
                                        out_audio_folder, code, "wav"
                                    )
                                    normalize_and_save(audio_file, output_file)
                                    add_patient = True

                            if add_patient:
                                r = {
                                    "patient": patient_id,
                                    "location": location,
                                    "old_diagnosis": 4,
                                    "all_diagnoses": [4],
                                    "diagnosis": 4,
                                    "multilabel": False,
                                }  # Asthma diagnosis
                                patient_df.append(r)

    patient_df = pd.DataFrame(patient_df)

    grouped_patients = (
        patient_df.groupby(["location", "diagnosis"])["patient"]
        .apply(list)
        .reset_index()
    )
    patient_diagnosis = patient_df[
        ["patient", "old_diagnosis", "all_diagnoses", "multilabel"]
    ]

    df_with_folds = []
    for _, r in grouped_patients.iterrows():
        base_fold = np.random.randint(n_splits)
        for i, patient in enumerate(r["patient"]):
            patient_fold = (base_fold + i) % 5
            row = {
                "patient": patient,
                "location": r["location"],
                "fold": patient_fold,
                "diagnosis": r["diagnosis"],
            }
            df_with_folds.append(row)

    patient_df = pd.DataFrame(df_with_folds).merge(patient_diagnosis, on="patient")
    patient_df["multilabel"] = patient_df.multilabel.astype("bool")

    return patient_df


def label_matrix(labels_df, duration=10, frames_per_second=50):
    n_frames = int(duration * frames_per_second)

    labels_df["Label"] = labels_df.Label.apply(lambda s: s.split())
    labels_df = labels_df.explode("Label")

    unique_labels = labels_df.Label.unique().tolist()
    n_labels = len(unique_labels)
    label_idx = {l: i for i, l in enumerate(unique_labels)}

    label_mat = np.zeros((n_labels, n_frames))

    for _, r in labels_df.iterrows():
        label = r.Label
        start, end = r.Start, r.End
        start_idx = max(0, int(start * frames_per_second))
        end_idx = ceil(end * frames_per_second)
        for i in range(start_idx, end_idx):
            if i < n_frames:
                label_mat[label_idx[label], i] = 1.0

    return label_mat, unique_labels


def display_spectrogram(
    spectrogram,
    title="Spectrogram",
    config=feat_config,
    duration=None,
    spect_path=None,
    labels_df=None,
    frame_attention=None,
    frame_probs=None,
    fig_width=15,
    plot_height=5,
):
    n_plots = 1

    if labels_df is not None:
        n_plots += 1
        label_mat, unique_labels = label_matrix(labels_df, duration=duration)

    if frame_attention is not None:
        n_plots += 2

    logspect = feats.log_spectrogram(spectrogram, ref=np.max)

    fig, ax = plt.subplots(
        nrows=n_plots, ncols=1, sharex=False, figsize=(fig_width, n_plots * plot_height)
    )

    if labels_df is not None:
        ax[0].matshow(label_mat, aspect="auto", origin="upper")
        ax[0].xaxis.set_ticks([])
        ax[0].xaxis.set_ticklabels([])
        ax[0].yaxis.set_ticklabels([""] + unique_labels)
        ax[0].set_ylabel("Label")
        ax[0].set_title("Annotations")

    if frame_attention is not None:
        p_recording = round(
            (frame_attention * frame_probs).sum() / 16, 2
        )  # up-sampling, divide by 16
        ax[n_plots - 3].plot(range(len(frame_probs)), frame_probs)
        ax[n_plots - 3].set_xlim([0, len(frame_probs)])
        ax[n_plots - 3].xaxis.set_ticks([])
        ax[n_plots - 3].xaxis.set_ticklabels([])
        ax[n_plots - 3].set_ylim([0.0, 1.05])
        ax[n_plots - 3].set_ylabel("p")
        ax[n_plots - 3].set_title(
            "Frame Probabilities (0: healthy, 1: pathological) - Pr_recording={}".format(
                p_recording
            )
        )

        ax[n_plots - 2].plot(range(len(frame_attention)), frame_attention)
        ax[n_plots - 2].set_xlim([0, len(frame_attention)])
        ax[n_plots - 2].xaxis.set_ticks([])
        ax[n_plots - 2].xaxis.set_ticklabels([])
        ax[n_plots - 2].set_ylabel("Attention")
        ax[n_plots - 2].set_title("Attention Values")

    spect_ax = ax if n_plots == 1 else ax[n_plots - 1]
    _ = librosa.display.specshow(
        logspect,
        x_axis="time",
        y_axis="mel",
        sr=config["sr"],
        hop_length=config["hop_length"],
        fmax=config["fmax"],
        ax=spect_ax,
    )
    spect_ax.set_ylabel("Frequency bins")
    spect_ax.set_title(title)

    if spect_path:
        fig.savefig(spect_path)
        plt.cla()
        plt.clf()
        plt.close("all")


def display_mfcc(mfccs, config=feat_config, mfcc_path=None, figsize=(16, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        mfccs, x_axis="time", sr=config["sr"], hop_length=config["hop_length"], ax=ax
    )
    fig.colorbar(img, ax=ax)
    ax.set(title="MFCC")

    if mfcc_path:
        fig.savefig(mfcc_path)
        plt.cla()
        plt.clf()
        plt.close("all")


def audio_processing(
    features,
    preprocessing,
    audio_path,
    label_path,
    spect_path,
    duration=None,
    frame_attention=None,
    frame_probs=None,
):
    if duration is not None:
        wav = feats.read_audio(audio_path, duration=duration)
        if wav.size < duration * feat_config["sr"]:
            pad_size = duration * feat_config["sr"] - wav.size
            wav = np.pad(wav, (0, pad_size), mode="constant")
    else:
        wav = feats.read_audio(audio_path, duration=30.0)
        duration = feats.get_duration(wav)

    audio_features = feats.AudioFeatures(features, preprocessing=preprocessing)
    out = audio_features.transform(wav)
    spect = out[0]

    if isfile(label_path):
        labels_df = pd.read_csv(
            label_path,
            sep="\t",
            header=None,
            names=["Start", "End", "Label"],
            dtype=str,
        )
        labels_df["Start"] = labels_df.Start.apply(lambda s: float(s.replace(",", ".")))
        labels_df["End"] = labels_df.End.apply(lambda s: float(s.replace(",", ".")))
        labels_df["Label"] = labels_df.Label.fillna("NaN")
        display_spectrogram(
            spect,
            title="Spectrogram",
            duration=duration,
            spect_path=spect_path,
            labels_df=labels_df,
            frame_attention=frame_attention,
            frame_probs=frame_probs,
        )
    else:
        display_spectrogram(
            spect,
            title="Spectrogram",
            duration=duration,
            spect_path=spect_path,
            frame_attention=frame_attention,
            frame_probs=frame_probs,
        )


def generate_spectrograms(
    base_folder, preprocessing=[], features=["stftw"], duration=10
):
    audio_folder = join(base_folder, "audio")
    label_folder = join(base_folder, "labels")
    spect_folder = join(base_folder, "spectrograms")

    if isdir(audio_folder):
        Path(spect_folder).mkdir(parents=True, exist_ok=True)
        recording_codes = sorted(
            [
                f.split(".")[0]
                for f in listdir(audio_folder)
                if isfile(join(audio_folder, f))
            ]
        )
        for code in recording_codes:
            audio_path = code_with_extension(audio_folder, code, "wav")
            label_path = code_with_extension(label_folder, code, "txt")
            spect_path = code_with_extension(spect_folder, code, "png")
            audio_processing(
                features, preprocessing, audio_path, label_path, spect_path, duration
            )


def all_spectrograms(
    data_path, patient_df, preprocessing=[], features=["stftw"], duration=10
):
    for _, r in patient_df.iterrows():
        patient_id = r.patient
        print(patient_id)
        patient_folder = join(data_path, patient_id)
        generate_spectrograms(patient_folder, preprocessing, features, duration)


def copy_spectrograms(folder, diagnosis_folder):
    spect_folder = join(folder, "spectrograms")
    if isdir(spect_folder):
        spectrogram_files = sorted(
            [f for f in listdir(spect_folder) if isfile(join(spect_folder, f))]
        )
        for f in spectrogram_files:
            position = f.split(".")[-2].split("_")[3]
            position_folder = join(diagnosis_folder, position)
            Path(position_folder).mkdir(parents=True, exist_ok=True)
            spect_path = join(spect_folder, f)
            copy(spect_path, position_folder)


def aggregate_spectrograms(data_path, patient_df, out_folder="../data/spectrograms"):
    patient_df_exploded = patient_df.explode("all_diagnoses")
    for diagnosis_nbr in sorted(patient_df_exploded.diagnosis.unique()):
        print(diagnosis_nbr)
        diagnosis_folder = join(out_folder, str(int(diagnosis_nbr)))
        patient_selection = patient_df_exploded[
            patient_df_exploded.diagnosis == diagnosis_nbr
        ].patient.values
        for patient_id in patient_selection:
            patient_folder = join(data_path, patient_id)
            print("\t", patient_folder)
            copy_spectrograms(patient_folder, diagnosis_folder)


def aggregate_audios(data_path, patient_df, out_folder="../audios"):
    diagnoses = patient_df.all_diagnoses.apply(
        lambda ls: "+".join([str(d) for d in ls])
    )
    for diagnosis in sorted(diagnoses.unique()):
        for location in patient_df.location.unique():
            print(diagnosis, location)
            diag_loc_folder = join(out_folder, str(diagnosis), location)
            Path(diag_loc_folder).mkdir(parents=True, exist_ok=True)
            diag_loc_filter = (patient_df.location == location) & (
                diagnoses == diagnosis
            )
            patient_selection = patient_df[diag_loc_filter].patient.values
            for patient_id in patient_selection:
                patient_folder = join(data_path, patient_id)
                print("\t", patient_folder)
                audio_path = join(patient_folder, "audio")
                patient_out = join(diag_loc_folder, patient_id)
                copytree(audio_path, patient_out)
