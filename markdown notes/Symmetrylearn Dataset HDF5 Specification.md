
## File Format

HDF5 (`.h5` or `.hdf5`)

## Structure (channel first)
```
dataset.h5
├── train/
│   ├── data      # (N_train, C, W, H) array of input samples
│   └── labels    # (N_train,) array of class labels
├── val/
│   ├── data      # (N_val, C, W, H) array of input samples
│   └── labels    # (N_val,) array of class labels
└── test/
    ├── data      # (N_test, C, W, H) array of input samples
    └── labels    # (N_test,) array of class labels
```

## Datasets

| Path           | Shape                | Dtype     | Description                |
| -------------- | -------------------- | --------- | -------------------------- |
| `train/data`   | `(N_train, C, W, H)` | `float32` | Training images/features   |
| `train/labels` | `(N_train,)`         | `int64`   | Training class indices     |
| `val/data`     | `(N_val, C, W, H)`   | `float32` | Validation images/features |
| `val/labels`   | `(N_val,)`           | `int64`   | Validation class indices   |
| `test/data`    | `(N_test, C, W, H)`  | `float32` | Test images/features       |
| `test/labels`  | `(N_test,)`          | `int64`   | Test class indices         |

## Usage
```python
import h5py

with h5py.File('dataset.h5', 'r') as f:
    X_train = f['train/data'][:]
    y_train = f['train/labels'][:]
```