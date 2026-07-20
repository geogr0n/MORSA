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

The efficiency comparison sums five optimizer-based training loops for neural
models and the complete 13-candidate ridge sweeps for analytical MORSA-Mean.
The matched trained-versus-analytical MORSA-Mean comparison excludes prototype
averaging and fold-specific PCA from both measurements. All runtime values
exclude raw WSI tiling, UNI feature extraction and k-means feature
summarization, which are upstream preprocessing steps.
