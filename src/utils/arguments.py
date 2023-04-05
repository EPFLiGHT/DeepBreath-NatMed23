from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments


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

    asthmoscope_locations: Optional[List[str]] = field(
        default_factory=lambda: ["GVA"],
        metadata={"help": ("TODO")},
    )

    pneumoscope_locations: Optional[List[str]] = field(
        default_factory=lambda: ["GVA", "POA", "DKR", "MAR", "RBA", "YAO"],
        metadata={"help": ("TODO")},
    )

    train_loc: Optional[List[str]] = field(
        default_factory=lambda: ["GVA", "POA"],
        metadata={"help": ("TODO")},
    )

    out_path: str = field(
        default="../out",
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

    preprocessing: Optional[List[str]] = field(
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

    conv_channels: int = field(
        default=32,
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


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(
        default=None, metadata={"help": "Language id for summarization."}
    )

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines or csv file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training, validation, or test file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
