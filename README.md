# symmetry-learn

[![CI](https://github.com/jiadongdan/symmetry-learn/actions/workflows/tests.yml/badge.svg)](https://github.com/jiadongdan/symmetry-learn/actions/workflows/tests.yml)
[![Torch CI](https://github.com/jiadongdan/symmetry-learn/actions/workflows/torch-tests.yml/badge.svg)](https://github.com/jiadongdan/symmetry-learn/actions/workflows/torch-tests.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/jiadongdan/symmetry-learn/actions/workflows/tests.yml)

A Python library for symmetry analysis of STEM (Scanning Transmission Electron Microscopy) images using deep learning. It detects rotational and reflectional symmetry in atomic-resolution images by combining crystallographic knowledge with convolutional neural networks.

## Installation

Install the released library with its bundled default model:

```bash
pip install symmetry-learn
```

The default `cnn_8ch_pg17` architecture and its `pg17-symmetry-v1` checkpoint
are available immediately after installation. The checkpoint is delivered by
the required `symmetry-learn-default-model` companion distribution.

For source development, install the model asset package before the main
editable package:

```bash
git clone https://github.com/jiadongdan/symmetry-learn.git
cd symmetry-learn
pip install -e ./model_packages/symmetry_learn_default_model
pip install -e .
```

To include test dependencies:

```bash
pip install -e ".[test]"
```

## Package Structure

```
symmlearn/
├── models/        Model architectures, specifications, registry, and weight resolution
├── features/      Registered model-input pipelines, including the eight-channel contract
├── patches/       Support validation, patch extraction, and dense coordinate grids
├── finetuning/    Reusable adapters, optimization engine, and adapter artifacts
├── inference/     Batched probability prediction and dense output maps
├── workflows/     Authoritative few-shot workflows composed from the public modules
├── provider/      Thin versioned Python and subprocess interfaces for orchestrators
├── lattice/       Plane groups, Wyckoff positions, and image simulation
├── maps/          Rotational and reflectional symmetry maps
├── measure/       Patch-based symmetry measurement
├── sampling/      Stratified, Sobol, and Poisson-disk sampling
├── training/      Legacy general training utilities
├── workflow/      Legacy experimental patch extraction workflow
└── utils/         File, memory, random-state, and visualization helpers

model_packages/
└── symmetry_learn_default_model/
    └── .../pg17-symmetry-v1.pth
```

Model architectures and model-specific fine-tuning recipes live under
`symmlearn.models`. Actual bundled parameters live in the data-only companion
distribution so code and large immutable model assets can be versioned
independently.

## Requirements

- Python >= 3.9
- numpy
- scipy
- matplotlib
- scikit-image
- scikit-learn
- [ase](https://wiki.fysik.dtu.dk/ase/) (Atomic Simulation Environment)
- spglib
- torch
- psutil

## Provider Interface

`symmlearn.provider` is the stable interface boundary for external tools such
as `symmetry-harness`. Numerical work is implemented in the public `features`,
`patches`, `finetuning`, `inference`, and `workflows` packages. Agent
interaction, point-selection UI, run orchestration, model installation consent,
and reports belong to the separate Harness.

Inspect capabilities without loading a model or checkpoint:

```bash
symmetry-learn-provider --capabilities
```

Use the direct array API:

```python
from symmlearn.provider import compute_features, few_shot_analyze

features, feature_record = compute_features(image, device="cuda")
result = few_shot_analyze(
    image,
    coordinates_xy=coordinates_xy,
    labels=labels,
    class_names=["Phase A", "Phase B"],
    options={"device": "cuda"},
)
```

The default workflow requires at least three support points per local class;
five points per class are recommended. It freezes the pretrained network,
inserts residual adapters after its convolutional and linear blocks, replaces
the 17-class head with a task-specific head, and optimizes only the adapters
and new head.

An explicit trusted checkpoint remains supported and overrides the bundled
default:

```python
result = few_shot_analyze(
    image,
    coordinates_xy=coordinates_xy,
    labels=labels,
    class_names=["Phase A", "Phase B"],
    checkpoint_path="custom.pth",
    checkpoint_sha256="<sha256>",
)
```

External orchestrators can use the versioned worker contract through
`python -m symmlearn.provider.worker --capabilities`, `--probe JOB.json`, or
`--job JOB.json`. Existing explicit-checkpoint jobs remain compatible. A job
that omits `checkpoint_path` uses the installed default weight. Optional model
weights are never downloaded silently.
