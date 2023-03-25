from os import listdir
from os.path import isdir, isfile, join
from pathlib import Path
from shutil import copy

import numpy as np
import pandas as pd
from pydub import AudioSegment


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
