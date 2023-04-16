from dataclasses import dataclass, field
from typing import List


@dataclass
class DataArguments:
    """
    Arguments related to data loading and preprocessing.
    """

    data_path: str = field(
        default="../data/raw/canonical_data/pediatric_studies",
        metadata={"help": ("Path to the directory containing the raw data files.")},
    )

    interim_path: str = field(
        default="../data/interim",
        metadata={
            "help": ("Path to the directory to store the interim preprocessed data.")
        },
    )

    processed_path: str = field(
        default="../data/processed",
        metadata={"help": ("Path to the directory to store the processed data.")},
    )

    asthmoscope_locations: List[str] = field(
        default_factory=lambda: ["GVA"],
        metadata={"help": ("List of Asthmoscope locations to include in the dataset.")},
    )

    pneumoscope_locations: List[str] = field(
        default_factory=lambda: ["GVA", "POA", "DKR", "MAR", "RBA", "YAO"],
        metadata={"help": ("List of Pneumoscope locations to include in the dataset.")},
    )


@dataclass
class AudioArguments:
    """
    Arguments for audio (pre-)processing.
    """

    sr: int = field(
        default=4000,
        metadata={"help": ("Sampling rate of audio files.")},
    )

    max_duration: float = field(
        default=30.0,
        metadata={"help": ("Maximum duration of audio files in seconds.")},
    )

    split_duration: float = field(
        default=5.0,
        metadata={
            "help": (
                "Duration of audio files to split into smaller segments in seconds."
            )
        },
    )

    freq_highcut: int = field(
        default=150,
        metadata={"help": ("High-pass filter cutoff frequency in Hz.")},
    )

    order_highcut: int = field(
        default=10,
        metadata={"help": ("High-pass filter order.")},
    )

    freq_lowcut: int = field(
        default=800,
        metadata={"help": ("Low-pass filter cutoff frequency in Hz.")},
    )

    order_lowcut: int = field(
        default=4,
        metadata={"help": ("Low-pass filter order.")},
    )

    preprocessing: List[str] = field(
        default_factory=lambda: ["highpass", "lowpass"],
        metadata={"help": ("Preprocessing filters to apply to the audio file.")},
    )

    transform: str = field(
        default="logmel",
        metadata={"help": ("Type of transform to apply to the audio file.")},
    )

    n_fft: int = field(
        default=256,
        metadata={"help": ("FFT window size.")},
    )

    hop_length: int = field(
        default=64,
        metadata={"help": ("Hop length for the STFT.")},
    )

    center: bool = field(
        default=True,
        metadata={"help": ("Whether to center the STFT.")},
    )

    n_mels: int = field(
        default=32,
        metadata={"help": ("Number of Mel bands to generate.")},
    )

    fmin: int = field(
        default=250,
        metadata={"help": ("Minimum frequency of Mel bands.")},
    )

    fmax: int = field(
        default=750,
        metadata={"help": ("Maximum frequency of Mel bands.")},
    )

    n_mfcc: int = field(
        default=40,
        metadata={"help": ("Number of MFCCs to generate.")},
    )

    roll_percent: float = field(
        default=0.85,
        metadata={
            "help": (
                "Percentage of the roll-off frequency to use for the high-frequency range."
            )
        },
    )

    time_drop_width: int = field(
        default=20,
        metadata={"help": ("The width of the time dropout mask in frames.")},
    )

    freq_drop_width: int = field(
        default=4,
        metadata={"help": ("The width of the frequency dropout mask in Mel bins.")},
    )


@dataclass
class ModelArguments:
    """
    Arguments related to the model itself.
    """

    model_name: str = field(
        default="dense",
        metadata={"help": ("The name of the model to use.")},
    )

    conv_dropout: float = field(
        default=0.2,
        metadata={"help": ("The dropout rate to use after convolutional layers.")},
    )

    fc_dropout: float = field(
        default=0.5,
        metadata={"help": ("The dropout rate to use after fully connected layers.")},
    )

    classes_num: int = field(
        default=1,
        metadata={"help": ("The number of classes for the classification task.")},
    )


@dataclass
class TrainingArguments:
    """
    Arguments for model training.
    """

    target: List[int] = field(
        metadata={"help": ("Target class(es) for binary classification.")},
    )

    exclude: List[int] = field(
        default_factory=list,
        metadata={"help": ("List of classes that should be excluded from training.")},
    )

    out_path: str = field(
        default="../out",
        metadata={
            "help": ("Output directory for the trained model and evaluation results.")
        },
    )

    do_train: bool = field(
        default=True,
        metadata={"help": ("Whether to run training.")},
    )

    online_logging: bool = field(
        default=False,
        metadata={"help": ("Whether to upload training logs to Weights & Biases.")},
    )

    train_loc: List[str] = field(
        default_factory=lambda: ["GVA", "POA"],
        metadata={"help": ("Training locations for dataset selection.")},
    )

    epochs: int = field(
        default=100,
        metadata={"help": ("Number of training epochs.")},
    )

    validation_start: int = field(
        default=60,
        metadata={"help": ("Epoch when validation starts.")},
    )

    batch_size: int = field(
        default=64,
        metadata={"help": ("Batch size for training.")},
    )

    balanced_sampling: bool = field(
        default=True,
        metadata={"help": ("Whether to use balanced sampling for training.")},
    )

    sampling_alpha: float = field(
        default=0.6,
        metadata={"help": ("Alpha parameter for class weights in balanced sampling.")},
    )

    optimizer_name: str = field(
        default="adamw",
        metadata={"help": ("Optimizer to use for training.")},
    )

    learning_rate: float = field(
        default=1e-4,
        metadata={"help": ("Learning rate for the optimizer.")},
    )

    weight_decay: float = field(
        default=5e-3,
        metadata={"help": ("Weight decay for the optimizer.")},
    )

    momentum: float = field(
        default=0.9,
        metadata={"help": ("Momentum for the optimizer.")},
    )

    save_outputs: bool = field(
        default=True,
        metadata={"help": ("Whether to save training outputs.")},
    )
