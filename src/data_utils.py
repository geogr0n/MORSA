import os
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split


WSI_EXTENSIONS = {".svs", ".ndpi", ".mrxs", ".tif", ".tiff", ".scn", ".vms", ".vmu", ".bif"}


def _wsi_stem(value):
    name = Path(str(value)).name
    suffix = Path(name).suffix.lower()
    return name[: -len(suffix)] if suffix in WSI_EXTENSIONS else name


def custom_collate_fn(batch):
    """Remove bad entries from the dataloader
    Args:
        batch (torch.Tensor): batch of dataset entries
    Returns:
        collate: Default collation for the dataloader
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def filter_no_features(df, feature_path, feature_name):
    print(f'Filtering WSIs that do not have {feature_name} features')
    projects = np.unique(df.tcga_project)
    all_wsis_with_features = []
    remove = []
    for proj in projects:
        wsis_with_features = os.listdir(os.path.join(feature_path, proj))
        for wsi in wsis_with_features:
            try:
                with h5py.File(os.path.join(feature_path, proj, wsi, wsi+'.h5'), "r") as f:
                    cols = list(f.keys())
                    if feature_name not in cols:
                        remove.append(wsi)
            except Exception:
                remove.append(wsi)
        all_wsis_with_features += wsis_with_features

    # Match reference names and feature directories in the same stem namespace.
    available_stems = {_wsi_stem(value) for value in all_wsis_with_features}
    remove_stems = {_wsi_stem(value) for value in remove}
    reference_stems = df['wsi_file_name'].astype(str).map(_wsi_stem)
    remove_stems.update(reference_stems[~reference_stems.isin(available_stems)].tolist())

    print(f'Original shape: {df.shape}')
    df = df[~reference_stems.isin(remove_stems)].reset_index(drop=True)
    print(f'New shape: {df.shape}')
    return df


def patient_kfold(dataset, n_splits=5, random_state=0, valid_size=0.1):
    """Perform cross-validation with patient split.
    """
    indices = np.arange(len(dataset))

    patients_unique = np.unique(dataset.patient_id)

    skf = KFold(n_splits, shuffle=True, random_state=random_state)
    ind = skf.split(patients_unique)

    train_idx = []
    valid_idx = []
    test_idx = []

    for ind_train, ind_test in ind:

        patients_train = patients_unique[ind_train]
        patients_test = patients_unique[ind_test]

        test_idx.append(indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] ==
                                       np.array(patients_test)[np.newaxis], axis=1)])

        if valid_size > 0:
            patients_train, patients_valid = train_test_split(
                patients_train, test_size=valid_size, random_state=random_state)
            valid_idx.append(indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] ==
                                            np.array(patients_valid)[np.newaxis], axis=1)])

        train_idx.append(indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] ==
                                        np.array(patients_train)[np.newaxis], axis=1)])

    return train_idx, valid_idx, test_idx
