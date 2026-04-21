# symmetry-learn

![Tests](https://github.com/jiadongdan/symmetry-learn/actions/workflows/tests.yml/badge.svg)
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
