import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build k-means cluster features from UNI features")
    parser.add_argument("--ref_file", type=str, required=True, help="Reference CSV containing WSI and project names")
    parser.add_argument("--feature_path", type=str, required=True, help="Directory containing UNI feature files")
    parser.add_argument("--num_clusters", type=int, default=100, help="Number of clusters for k-means")
    parser.add_argument("--tcga_projects", default=None, type=str, nargs="*", help="tcga_project values to process")
    parser.add_argument("--start", type=int, default=0, help="Start slide index for parallelization")
    parser.add_argument("--end", type=int, default=None, help="End slide index for parallelization")
    parser.add_argument("--gtex", help="Treat the input as GTEx data", action="store_true")
    parser.add_argument("--gtex_tissue", type=str, default=None, help="GTEx tissue label")
    parser.add_argument("--seed", type=int, default=0, help="K-means random seed")

    args = parser.parse_args()
    if args.gtex and not args.gtex_tissue:
        parser.error("--gtex_tissue is required when --gtex is set")
    if args.num_clusters <= 0:
        parser.error("--num_clusters must be positive")

    np.random.seed(args.seed)

    print("-" * 10)
    print("Args for this experiment")
    print(args)
    print("-" * 10)

    df = pd.read_csv(args.ref_file)
    required_columns = {"wsi_file_name"}
    if not args.gtex:
        required_columns.add("tcga_project")
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Reference CSV is missing columns: {sorted(missing_columns)}")
    df = df.drop_duplicates(["wsi_file_name"])

    if args.tcga_projects:
        df = df[df["tcga_project"].isin(args.tcga_projects)]

    print(f"Total number of slides = {df.shape[0]}")

    df = df.iloc[args.start : args.end]
    if df.empty:
        raise ValueError("No slides remain after filtering")

    print(f"New number of slides = {df.shape[0]}")

    failures = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        wsi = row["wsi_file_name"]
        if args.gtex:
            project = args.gtex_tissue
        else:
            project = row["tcga_project"]
        wsi = Path(wsi).stem

        path = f"{args.feature_path}/{project}/{wsi}"
        try:
            f = h5py.File(f"{path}/{wsi}.h5", "r+")
        except Exception as exc:
            failures.append(f"Cannot open {path}/{wsi}.h5: {exc}")
            continue

        try:
            features = f["uni_features"]
        except Exception:
            failures.append(f"No uni_features dataset in {path}/{wsi}.h5")
            f.close()
            continue

        if features.shape[0] < args.num_clusters:
            failures.append(
                f"{wsi} has {features.shape[0]} patches but requires {args.num_clusters} clusters"
            )
            f.close()
            continue

        if "cluster_features" in f.keys():
            print(f"{wsi}: cluster_features already available")
            f.close()
            continue

        kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed).fit(features)
        clusters = kmeans.labels_
        coords = f["coords"][:] if "coords" in f else None

        mean_features = []
        cluster_coords = []
        cluster_counts = []
        for pos in tqdm(range(args.num_clusters), desc=f"Clustering {wsi}"):
            indexes = np.where(clusters == pos)[0]
            mean_features.append(np.mean(features[indexes], axis=0))
            cluster_counts.append(int(indexes.shape[0]))
            if coords is not None:
                cluster_coords.append(np.mean(coords[indexes], axis=0))

        try:
            f.create_dataset("cluster_features", data=np.asarray(mean_features))
            f.create_dataset("cluster_assignments", data=np.asarray(clusters, dtype=np.int64))
            f.create_dataset("cluster_tile_counts", data=np.asarray(cluster_counts, dtype=np.int64))
            if coords is not None:
                f.create_dataset("cluster_coords", data=np.asarray(cluster_coords, dtype=np.float32))
            f.attrs["cluster_row_order"] = "cluster_features[j] is cluster j; tile i maps by cluster_assignments[i]"
            f.attrs["kmeans_random_state"] = int(args.seed)
            f.attrs["num_clusters"] = int(args.num_clusters)
            f.close()
        except Exception as exc:
            failures.append(f"{wsi}: error creating cluster features: {exc}")
            f.close()

    if failures:
        preview = "\n".join(failures[:20])
        suffix = "" if len(failures) <= 20 else f"\n... {len(failures) - 20} more"
        raise RuntimeError(f"K-means feature aggregation failed for {len(failures)} slide(s):\n{preview}{suffix}")

    print("Done!")
