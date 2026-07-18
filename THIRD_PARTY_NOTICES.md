# Third-Party Notices

This repository contains code adapted from public upstream projects. These notes
preserve provenance for the MORSA (Morphology-Omics Representation via Structural Alignment)
code release.

## Repository Licence

The MORSA (Morphology-Omics Representation via Structural Alignment) code release is
distributed under the GNU General Public License, version 3 or later. See
[LICENSE](LICENSE).

## Upstream Projects

### gevaertlab/sequoia-pub

- Upstream repository: <https://github.com/gevaertlab/sequoia-pub>
- Repository-level upstream licence: MIT
- Relevant adapted components include SEQUOIA/ViS-style training, data-loading
  and model-routing code.

The upstream `sequoia-pub` repository ships an MIT `LICENSE` at its repository
root. Preserve this attribution when redistributing modified code.

### owkin/HE2RNA_code

- Upstream repository: <https://github.com/owkin/HE2RNA_code>
- Upstream licence: GNU General Public License v3.0 or later
- Relevant adapted component: `src/he2rna_backbone.py`

The public `sequoia-pub/src/he2rna.py` file retains the Owkin copyright notice
and GNU GPL v3-or-later notice. Accordingly, the HE2RNA-style baseline path in
this repository is treated as GPL-governed code.

## Notes

- This notice is not legal advice.
- Raw third-party WSI/RNA data, controlled-access data, the gated UNI checkpoint
  and trained MORSA checkpoints are not redistributed in this code release.
