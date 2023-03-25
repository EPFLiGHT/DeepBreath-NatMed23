from itertools import combinations
from os.path import join

from .constants import (
    RATE,
    MAX_DURATION,
    PATIENT_DF_FILE,
    SAMPLES_DF_FILE,
    AUDIO_DATA_FILE,
)


pre_config = {
    "sr": RATE,
    "freq_highcut": 150,  # 80 100 150  -> Latest: 100
    "order_highcut": 10,
    "freq_lowcut": 800,  # 900 1000 1250  -> Latest: 900
    "order_lowcut": 4,
}

feat_config = {
    "transform": "logmel",
    "sr": RATE,
    "n_fft": 256,
    "hop_length": 64,  # 128
    "center": True,
    "n_mels": 32,  # 32 64  -> Latest: 32
    "fmin": 250,  # 200 250
    "fmax": 750,  # 750 900 1000  -> Latest: 900
    "n_mfcc": 40,
    "roll_percent": 0.85,
}

split_config = {
    "sr": RATE,
    "max_duration": MAX_DURATION,
    "pad_audio": False,
    "split_duration": 5,
    "overlap": 0.5,
    "center": False,
}

network = {
    "feature_model": "dense",  # cnn6 cnn10 dense
    "n_feats": feat_config["n_mels"],
    "conv_channels": 32,  # 32
    "fc_features": 128,
    "classes_num": 1,  # 2
    "conv_dropout": 0.2,
    "fc_dropout": 0.5,
    "feat_config": feat_config,
    "time_drop_width": 20,  # changed
    "freq_drop_width": 4,
}

optimizer = {
    "name": "adamw",  # adamw
    "parameters": {"lr": 1e-4, "weight_decay": 0.005},  # 1e-4  # 0.005
}

pipeline_config = dict(
    preprocessing=["highpass", "lowpass"],
    augment=False,
    features=["logmel"],
    pre_config=pre_config,
    feat_config=feat_config,
    split_config=split_config,
    patient_df_path=join("../data/processed", PATIENT_DF_FILE),
    samples_df_path=join("../data/processed", SAMPLES_DF_FILE),
    samples_path=join("../data/processed", AUDIO_DATA_FILE),
    train_loc=["GVA", "POA"],
    stetho=["L"],
    cv_folds=list(combinations(range(5), 2)),
    epochs=100,
    validation_start=60,
    batch_size=64,
    balanced_sampling=True,  # changeme
    sampling_alpha=0.6,  # changeme
    network=network,
    exclude=[],
    optimizer=optimizer,
    out_folder="../out",
    model_file="{}_D{}_V{}_T{}.pt",
    feature_file="{}_features_D{}_V{}_T{}.npy",
    output_file="{}_outputs_D{}_V{}_T{}.npy",
    attn_file="{}_attn_D{}_V{}_T{}.npy",
    aggregate_file="{}_agg_D{}_V{}_T{}.csv",
    save_outputs=False,
)
