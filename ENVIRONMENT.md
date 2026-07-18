# MORSA Environment

This file describes the software environment for the public MORSA code release
and the model-fitting efficiency analyses.

## Python stack

A reference environment is:

- Python 3.12.8
- PyTorch 2.10.0+cu128
- torchvision 0.25.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- one visible NVIDIA GPU

Install PyTorch and torchvision separately from `requirements-linux.txt`,
because their wheels must match the local CUDA runtime, driver and operating
system. Use the official PyTorch installation selector for the target machine:

```text
https://pytorch.org/get-started/locally/
```

After PyTorch is installed, install the remaining scientific dependencies with:

```bash
python -m pip install -r requirements-linux.txt
```

## UNI checkpoint boundary

The UNI checkpoint is not redistributed with this repository. Request access to
`MahmoodLab/UNI`, store `pytorch_model.bin` outside version control, and provide
its path through `--uni_model_path` or `UNI_MODEL_PATH`.

## Timing Boundary

The efficiency comparison reports model fitting and model-side profiling after
precomputed WSI-derived `cluster_features` are available. It excludes raw WSI
tiling, UNI feature extraction and k-means feature summarization, which were
completed upstream rather than on the current local workstation.
