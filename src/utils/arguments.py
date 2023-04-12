from dataclasses import dataclass, field
from typing import List


@dataclass
class DataArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    data_path: str = field(
        default="../data/raw/canonical_data/pediatric_studies",
        metadata={"help": ("TODO")},
    )

    interim_path: str = field(
        default="../data/interim",
        metadata={"help": ("TODO")},
    )

    processed_path: str = field(
        default="../data/processed",
        metadata={"help": ("TODO")},
    )

    asthmoscope_locations: List[str] = field(
        default_factory=lambda: ["GVA"],
        metadata={"help": ("TODO")},
    )

    pneumoscope_locations: List[str] = field(
        default_factory=lambda: ["GVA", "POA", "DKR", "MAR", "RBA", "YAO"],
        metadata={"help": ("TODO")},
    )


@dataclass
class AudioArguments:
    """
    TODO
    """

    sr: int = field(
        default=4000,
        metadata={"help": ("TODO")},
    )

    max_duration: float = field(
        default=30.0,
        metadata={"help": ("TODO")},
    )

    split_duration: float = field(
        default=5.0,
        metadata={"help": ("TODO")},
    )

    freq_highcut: int = field(
        default=150,
        metadata={"help": ("TODO")},
    )

    order_highcut: int = field(
        default=10,
        metadata={"help": ("TODO")},
    )

    freq_lowcut: int = field(
        default=800,
        metadata={"help": ("TODO")},
    )

    order_lowcut: int = field(
        default=4,
        metadata={"help": ("TODO")},
    )

    preprocessing: List[str] = field(
        default_factory=lambda: ["highpass", "lowpass"],
        metadata={"help": ("TODO")},
    )

    transform: str = field(
        default="logmel",
        metadata={"help": ("TODO")},
    )

    n_fft: int = field(
        default=256,
        metadata={"help": ("TODO")},
    )

    hop_length: int = field(
        default=64,
        metadata={"help": ("TODO")},
    )

    center: bool = field(
        default=True,
        metadata={"help": ("TODO")},
    )

    n_mels: int = field(
        default=32,
        metadata={"help": ("TODO")},
    )

    fmin: int = field(
        default=250,
        metadata={"help": ("TODO")},
    )

    fmax: int = field(
        default=750,
        metadata={"help": ("TODO")},
    )

    n_mfcc: int = field(
        default=40,
        metadata={"help": ("TODO")},
    )

    roll_percent: float = field(
        default=0.85,
        metadata={"help": ("TODO")},
    )

    time_drop_width: int = field(
        default=20,
        metadata={"help": ("TODO")},
    )

    freq_drop_width: int = field(
        default=4,
        metadata={"help": ("TODO")},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
        default="dense",
        metadata={"help": ("TODO")},
    )

    fc_features: int = field(
        default=128,
        metadata={"help": ("TODO")},
    )

    conv_dropout: float = field(
        default=0.2,
        metadata={"help": ("TODO")},
    )

    fc_dropout: float = field(
        default=0.2,
        metadata={"help": ("TODO")},
    )

    classes_num: int = field(
        default=1,
        metadata={"help": ("TODO")},
    )


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    out_path: str = field(
        default="../out",
        metadata={"help": ("TODO")},
    )

    do_train: bool = field(
        default=True,
        metadata={"help": ("TODO")},
    )

    target: List[int] = field(
        default_factory=lambda: [0],
        metadata={"help": ("TODO")},
    )

    train_loc: List[str] = field(
        default_factory=lambda: ["GVA", "POA"],
        metadata={"help": ("TODO")},
    )

    epochs: int = field(
        default=100,
        metadata={"help": ("TODO")},
    )

    validation_start: int = field(
        default=60,
        metadata={"help": ("TODO")},
    )

    batch_size: int = field(
        default=64,
        metadata={"help": ("TODO")},
    )

    balanced_sampling: bool = field(
        default=True,
        metadata={"help": ("TODO")},
    )

    sampling_alpha: float = field(
        default=0.6,
        metadata={"help": ("TODO")},
    )

    optimizer_name: str = field(
        default="adamw",
        metadata={"help": ("TODO")},
    )

    learning_rate: float = field(
        default=1e-4,
        metadata={"help": ("TODO")},
    )

    weight_decay: float = field(
        default=5e-3,
        metadata={"help": ("TODO")},
    )

    momentum: float = field(
        default=0.9,
        metadata={"help": ("TODO")},
    )

    save_outputs: bool = field(
        default=True,
        metadata={"help": ("TODO")},
    )
