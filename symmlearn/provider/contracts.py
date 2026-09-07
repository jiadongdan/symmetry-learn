"""Provider protocol constants shared by the Python API and worker."""

PROVIDER_CONTRACT_VERSION = "symmetry-learn-provider-v1"
WORKER_SCHEMA_VERSION = "scientific-symmetry-worker-v1"
ADAPTER_SCHEMA_VERSION = "symmetry-adapter-head-v1"
MODEL_IDENTIFIER = "cnn_8ch_pg17"

CHANNEL_NAMES = (
    "image",
    "reflection_strength",
    "reflection_sin_2theta",
    "reflection_cos_2theta",
    "rotation_2_fold",
    "rotation_3_fold",
    "rotation_4_fold",
    "rotation_6_fold",
)

DEFAULT_OPTIONS = {
    "n_max": 20,
    "symmetry_patch_size": 51,
    "rotation_folds": [2, 3, 4, 6],
    "reflection_p": 2.0,
    "normalize_rotation_maps": False,
    "classifier_patch_size": 64,
    "minimum_shots_per_class": 3,
    "maximum_shots_per_class": 50,
    "adapter_bottleneck": 16,
    "epochs": 150,
    "learning_rate": 0.0005,
    "weight_decay": 0.0,
    "seed": 42,
    "stride": 4,
    "batch_size": 512,
    "device": "auto",
}
