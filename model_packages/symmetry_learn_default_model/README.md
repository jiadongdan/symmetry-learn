# symmetry-learn-default-model

This data-only Python distribution provides the default pretrained checkpoint
for `symmetry-learn`. It contains no model architecture or training code.

The main `symmetry-learn` distribution locates this package through
`importlib.resources`, verifies the declared file size and SHA-256 digest, and
strictly loads the checkpoint into the matching registered architecture.

The bundled checkpoint contains the complete pretrained `model_state_dict`
and training metrics. Pretraining optimizer and scheduler states are omitted
because they are not used by inference or task-specific adapter fine-tuning.

This package is installed automatically as a dependency of `symmetry-learn`.
Additional model weights are distributed separately and are never downloaded
silently.
