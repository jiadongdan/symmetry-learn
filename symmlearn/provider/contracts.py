"""Provider protocol constants shared by the Python API and worker."""

from copy import deepcopy

from symmlearn.features.eight_channel.contract import CHANNEL_NAMES
from symmlearn.finetuning.artifacts import ADAPTER_SCHEMA_VERSION
from symmlearn.models.cnn_8ch_pg17.specification import (
    DEFAULT_OPTIONS as MODEL_DEFAULT_OPTIONS,
    MODEL_IDENTIFIER,
)


PROVIDER_CONTRACT_VERSION = "symmetry-learn-provider-v1"
WORKER_SCHEMA_VERSION = "scientific-symmetry-worker-v1"
DEFAULT_OPTIONS = deepcopy(MODEL_DEFAULT_OPTIONS)
