# UNI Checkpoint

MORSA uses a frozen UNI pathology foundation-model checkpoint for WSI feature extraction.

The UNI checkpoint is not distributed with this repository. Readers should request access from the official gated Hugging Face repository:

- Model repository: `MahmoodLab/UNI`
- URL: `https://huggingface.co/MahmoodLab/UNI`
- Model file used in this study: `pytorch_model.bin`
- Upstream licence: `cc-by-nc-nd-4.0`

The checkpoint used for the reported experiments was named `pytorch_model.bin` and had SHA256 checksum:

```text
56EF09B44A25DC5C7EEDC55551B3D47BCD17659A7A33837CF9ABC9EC4E2FFB40
```

Place the downloaded checkpoint outside version control and pass its location with `--uni_model_path` or `UNI_MODEL_PATH`.

Trained MORSA checkpoints are not included. Models can be retrained with the
released code after obtaining the required source data; the Zenodo results
dataset supports independent inspection of the reported analyses.
