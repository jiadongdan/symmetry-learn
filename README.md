# symmetry-learn

[![CI](https://github.com/jiadongdan/symmetry-learn/actions/workflows/tests.yml/badge.svg)](https://github.com/jiadongdan/symmetry-learn/actions/workflows/tests.yml)
[![Torch CI](https://github.com/jiadongdan/symmetry-learn/actions/workflows/torch-tests.yml/badge.svg)](https://github.com/jiadongdan/symmetry-learn/actions/workflows/torch-tests.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/jiadongdan/symmetry-learn/actions/workflows/tests.yml)

A Python library for symmetry analysis of STEM (Scanning Transmission Electron Microscopy) images using deep learning. It detects rotational and reflectional symmetry in atomic-resolution images by combining crystallographic knowledge with convolutional neural networks.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/jiadongdan/symmetry-learn.git
cd symmetry-learn
pip install -e .
```

To include test dependencies:

```bash
pip install -e ".[test]"
```

## Package Structure

```
symmlearn/
├── lattice/       Crystallographic structures — plane groups, Wyckoff positions,
│                  unit cell generation, and image simulation from atomic coordinates
├── maps/          Symmetry maps — rotational and reflectional symmetry maps
│                  computed via Zernike polynomial kernels
├── measure/       Patch-based measurement — SymmMeasurement for extracting and
│                  analysing image patches
├── provider/      Versioned Python and worker APIs for feature extraction,
│                  checkpoint validation, few-shot adaptation, and prediction
├── sampling/      Sampling strategies — stratified, Sobol, and Poisson-disk
│                  sampling for generating patch locations
├── training/      Deep learning — CNN model and dataset utilities for
│                  symmetry classification
├── workflow/      High-level workflows — extract_exp_data() for patch extraction
│                  and label assignment from experimental images
└── utils/         Utilities — file I/O, random state helpers, atom visualisation
```

## Requirements

- Python >= 3.9
- numpy
- scipy
- matplotlib
- scikit-image
- scikit-learn
- [ase](https://wiki.fysik.dtu.dk/ase/) (Atomic Simulation Environment)
- spglib

Optional (for deep learning features):

- torch
- torchvision

## Provider Interface

`symmlearn.provider` is the stable numerical boundary for external tools such
as `symmetry-harness`. The library remains usable directly from Python, while
agent interaction, point-selection UI, run orchestration, and reports belong to
the separate Harness.

Install the optional Provider runtime:

```bash
python -m pip install -e ".[provider,test]"
```

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
    checkpoint_path="best_model.pth",
    checkpoint_sha256="<sha256>",
    options={"device": "cuda"},
)
```

External orchestrators can use the versioned worker contract through
`python -m symmlearn.provider.worker --capabilities`, `--probe JOB.json`, or
`--job JOB.json`. The Provider never downloads or guesses checkpoints.
