import argparse
import gc
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def _string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _decode_key(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _tile_keys_from_patch_h5(f_read):
    if "tile_keys" in f_read:
        return [_decode_key(x) for x in f_read["tile_keys"][:]]
    keys = []
    for key in f_read.keys():
        if "_" not in key:
            continue
        try:
            x, y = key.split("_", 1)
            int(x)
            int(y)
        except ValueError:
            continue
        keys.append(key)
    return keys


def _coords_from_keys(keys):
    return np.asarray([[int(key.split("_", 1)[0]), int(key.split("_", 1)[1])] for key in keys], dtype=np.int64)


def find_uni_model_path(requested_path=None):
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        requested_path,
        os.environ.get("UNI_MODEL_PATH"),
        str(project_root / "pytorch_model.bin"),
        str(project_root / "checkpoints" / "pytorch_model.bin"),
        str(project_root / "models" / "UNI" / "pytorch_model.bin"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)

    searched = "\n".join(f"  - {p}" for p in candidates if p)
    raise FileNotFoundError(
        "Cannot find UNI pytorch_model.bin. Set --uni_model_path or UNI_MODEL_PATH.\n"
        f"Searched:\n{searched}"
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract UNI features from HDF5 patches")
    parser.add_argument("--ref_file", required=True, type=str, help="Path to reference csv file")
    parser.add_argument("--patch_data_path", required=True, type=str, help="Directory where patch HDF5 files are saved")
    parser.add_argument("--feature_path", required=True, type=str, help="Output directory for features")
    parser.add_argument("--max_patch_number", type=int, default=4000, help="Max number of patches per slide")
    parser.add_argument("--seed", type=int, default=99, help="Random seed")
    parser.add_argument("--tcga_projects", default=None, type=str, nargs="*", help="Project labels to process")
    parser.add_argument("--start", type=int, default=0, help="Start slide index for parallelization")
    parser.add_argument("--end", type=int, default=None, help="End slide index for parallelization")
    parser.add_argument(
        "--uni_model_path",
        default=os.environ.get("UNI_MODEL_PATH"),
        type=str,
        help="Path to UNI pytorch_model.bin. Defaults to UNI_MODEL_PATH or project-root/pytorch_model.bin",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.max_patch_number <= 0:
        parser.error("--max_patch_number must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("-" * 10)
    print("Args for this experiment")
    print(args)
    print("-" * 10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = int(os.environ.get("UNI_BATCH_SIZE", "64"))
    if batch_size <= 0:
        raise ValueError("UNI_BATCH_SIZE must be positive")

    transforms_val = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    uni_model_path = find_uni_model_path(args.uni_model_path)
    print(f"Using UNI model: {uni_model_path}")

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.load_state_dict(torch.load(str(uni_model_path), map_location="cpu"), strict=True)
    model.to(device)
    model.eval()

    print("Loading dataset...")
    df = pd.read_csv(args.ref_file)
    required_columns = {"wsi_file_name", "tcga_project"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Reference CSV is missing columns: {sorted(missing_columns)}")
    df = df.drop_duplicates(["wsi_file_name"])

    if args.tcga_projects:
        df = df[df["tcga_project"].isin(args.tcga_projects)]

    df = df.iloc[args.start : args.end]
    if df.empty:
        raise ValueError("No slides remain after filtering")

    print("=" * 60)
    print("Start UNI feature extraction")
    print(f"Slides: {df.shape[0]}")
    print(f"Max patches per slide: {args.max_patch_number}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)

    total_slides = df.shape[0]
    completed_count = 0
    failures = []

    for _, row in tqdm(df.iterrows(), total=total_slides, desc="Overall Progress"):
        wsi_name = row["wsi_file_name"]
        wsi_stem = Path(wsi_name).stem
        project = row["tcga_project"]

        patch_folder = os.path.join(args.patch_data_path, wsi_stem)
        if not os.path.exists(patch_folder):
            failures.append(f"Missing patch directory: {patch_folder}")
            continue

        patch_h5 = os.path.join(patch_folder, wsi_stem + ".hdf5")
        output_dir = os.path.join(args.feature_path, project, wsi_stem)
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(os.path.join(output_dir, "complete_uni.txt")):
            print(f"{wsi_stem}: UNI features already obtained")
            continue

        try:
            with h5py.File(patch_h5, "r") as f_read:
                keys = _tile_keys_from_patch_h5(f_read)
                if not keys:
                    raise ValueError(f"No patch datasets found in {patch_h5}")
                if len(keys) > args.max_patch_number:
                    keys = random.sample(keys, args.max_patch_number)
                coords = _coords_from_keys(keys)

                features_tiles = []
                for idx in tqdm(range(0, len(keys), batch_size), desc=f"Processing {wsi_stem}"):
                    batch_keys = keys[idx : idx + batch_size]
                    batch_images = []
                    for key in batch_keys:
                        image = Image.fromarray(f_read[key][:]).convert("RGB")
                        batch_images.append(transforms_val(image))

                    batch_tensor = torch.stack(batch_images).to(device)
                    with torch.no_grad():
                        features = model(batch_tensor)
                        features_tiles.extend(features.detach().cpu().numpy())

                    del batch_tensor, features, batch_images
                    torch.cuda.empty_cache()

            features_tiles = np.asarray(features_tiles)
            n_tiles = len(features_tiles)
            output_h5 = os.path.join(output_dir, wsi_stem + ".h5")
            with h5py.File(output_h5, "w") as f_write:
                f_write.create_dataset("uni_features", data=features_tiles)
                f_write.create_dataset("coords", data=coords)
                f_write.create_dataset("tile_keys", data=np.asarray(keys, dtype=object), dtype=_string_dtype())
                f_write.attrs["tile_row_order"] = "uni_features[i] corresponds to coords[i] and tile_keys[i]"
                f_write.attrs["source_patch_h5"] = f"{wsi_stem}/{wsi_stem}.hdf5"
                f_write.attrs["max_patch_number"] = int(args.max_patch_number)
                f_write.attrs["seed"] = int(args.seed)

            with open(os.path.join(output_dir, "complete_uni.txt"), "w") as f_sum:
                f_sum.write(f"Total n patch = {n_tiles}")

            del features_tiles
            gc.collect()
            torch.cuda.empty_cache()

            completed_count += 1
            print("=" * 60)
            print(f"{wsi_stem} complete; extracted {n_tiles} patch features")
            print(f"Progress: {completed_count}/{total_slides} ({completed_count/total_slides*100:.1f}%)")
            print("=" * 60)
        except Exception as exc:
            failures.append(f"{wsi_stem}: {exc}")
            gc.collect()
            torch.cuda.empty_cache()

    if failures:
        preview = "\n".join(failures[:20])
        suffix = "" if len(failures) <= 20 else f"\n... {len(failures) - 20} more"
        raise RuntimeError(f"UNI feature extraction failed for {len(failures)} slide(s):\n{preview}{suffix}")


if __name__ == "__main__":
    main()
