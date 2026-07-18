import argparse
import gc
import json
import os

if os.environ.get("OPENSLIDE_DLL_DIR") and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.environ["OPENSLIDE_DLL_DIR"])

from multiprocessing import Manager, Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.color import rgb2hsv
from skimage.exposure.exposure import is_low_contrast
from skimage.filters import threshold_otsu
from skimage.io import imsave


def _string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _write_string_dataset(hdf, name, values):
    if name in hdf:
        del hdf[name]
    hdf.create_dataset(name, data=np.asarray(values, dtype=object), dtype=_string_dtype())


def _write_numeric_dataset(hdf, name, values, dtype):
    if name in hdf:
        del hdf[name]
    hdf.create_dataset(name, data=np.asarray(values, dtype=dtype))


def _open_slide(slide_path):
    try:
        import openslide_bin  # noqa: F401
    except ImportError:
        pass
    from openslide import OpenSlide

    return OpenSlide(slide_path)

def get_mask_image(img_RGB, RGB_min=50):
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return mask

def get_mask(slide, level='max', RGB_min=50):
    #read svs image at a certain level  and compute the otsu mask
    if level == 'max':
        level = len(slide.level_dimensions) - 1
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),level,slide.level_dimensions[level]).convert('RGB')),
                           axes=[1, 0, 2])

    tissue_mask = get_mask_image(img_RGB, RGB_min)
    return tissue_mask, level

def extract_patches(
    slide_path,
    mask_path,
    patch_size,
    patches_output_dir,
    slide_id,
    max_patches_per_slide=2000,
    seed=5,
):
    patch_folder = os.path.join(patches_output_dir, slide_id)
    if not os.path.isdir(patch_folder):
        os.makedirs(patch_folder)

    patch_folder_mask = os.path.join(mask_path, slide_id)
    if not os.path.isdir(patch_folder_mask):
        os.makedirs(patch_folder_mask)

    if os.path.exists(os.path.join(patch_folder, "complete.txt")):
        print(f'{slide_id}: patches have already been extracted')
        return True

    path_hdf5 = os.path.join(patch_folder, f"{slide_id}.hdf5")
    hdf = h5py.File(path_hdf5, 'w')
    slide = None

    try:
        slide = _open_slide(slide_path)
        mask, mask_level = get_mask(slide)
        mask = binary_dilation(mask, iterations=3)
        mask = binary_erosion(mask, iterations=3)
        np.save(os.path.join(patch_folder_mask, "mask.npy"), mask)

        mask_level = len(slide.level_dimensions) - 1

        PATCH_LEVEL = 0
        BACKGROUND_THRESHOLD = .2
        ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0]
        ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1]

        xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

        # handle slides with 40 magnification at base level
        resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
        if not slide.properties.get('aperio.AppMag', 20): print(f"magnifications for {slide_id} is not found, using default magnification 20X")

        patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))
        print(f"patch size for {slide_id}: {patch_size_resized}")

        i = 0
        tile_keys = []
        coords = []
        indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) for y in
                    range(0, ymax, patch_size_resized[0])]

        # Generate candidate tissue patches in a deterministic shuffled order.
        if max_patches_per_slide is None:
            max_patches_per_slide = len(indices)

        np.random.RandomState(seed).shuffle(indices)

        for x, y in indices:
            # check if in background mask
            x_mask = int(x / ratio_x)
            y_mask = int(y / ratio_y)
            if mask[x_mask, y_mask] == 1:
                patch = slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB')
                try:
                    mask_patch = get_mask_image(np.array(patch))
                    mask_patch = binary_dilation(mask_patch, iterations=3)
                except Exception as e:
                    print("error with slide id {} patch {}".format(slide_id, i))
                    print(e)
                    continue
                if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                    if resize_factor != 1.0:
                        patch = patch.resize(patch_size)
                    patch = np.array(patch)
                    tile_name = f"{x}_{y}"
                    hdf.create_dataset(tile_name, data=patch)
                    tile_keys.append(tile_name)
                    coords.append((x, y))
                    i = i + 1
            if i >= max_patches_per_slide:
                break

        _write_string_dataset(hdf, "tile_keys", tile_keys)
        _write_numeric_dataset(hdf, "coords", coords, np.int64)
        hdf.attrs["patch_level"] = PATCH_LEVEL
        hdf.attrs["mask_level"] = mask_level
        hdf.attrs["patch_size_output"] = int(patch_size[0])
        hdf.attrs["patch_size_level0_x"] = int(patch_size_resized[0])
        hdf.attrs["patch_size_level0_y"] = int(patch_size_resized[1])
        hdf.attrs["max_patches_per_slide"] = int(max_patches_per_slide)
        hdf.attrs["patch_shuffle_seed"] = int(seed)
        hdf.attrs["n_patches"] = int(i)
        hdf.attrs["slide_width_level0"] = int(xmax)
        hdf.attrs["slide_height_level0"] = int(ymax)

        hdf.close()

        if i == 0:
            print("no patch extracted for slide {}".format(slide_id))
            return False
        else:
            thumbnail = np.array(
                slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level]).convert("RGB")
            )
            imsave(os.path.join(patch_folder_mask, "thumbnail_mask_level.png"), thumbnail)
            imsave(os.path.join(patch_folder_mask, "mask_mask_level.png"), (mask.T.astype(np.uint8) * 255))
            metadata = {
                "slide_id": slide_id,
                "slide_file": Path(slide_path).name,
                "level_dimensions": [list(map(int, dim)) for dim in slide.level_dimensions],
                "patch_level": int(PATCH_LEVEL),
                "mask_level": int(mask_level),
                "patch_size_output": [int(patch_size[0]), int(patch_size[1])],
                "patch_size_level0": [int(patch_size_resized[0]), int(patch_size_resized[1])],
                "resize_factor": float(resize_factor),
                "max_patches_per_slide": int(max_patches_per_slide),
                "patch_shuffle_seed": int(seed),
                "n_patches": int(i),
                "tile_key_order": "matches coords dataset and downstream uni_features row order",
            }
            with open(os.path.join(patch_folder_mask, "patch_metadata.json"), "w", encoding="utf-8") as f_meta:
                json.dump(metadata, f_meta, indent=2)
            with open(os.path.join(patch_folder, "complete.txt"), 'w') as f:
                f.write('Process complete!\n')
                f.write(f"Total n patch = {i}")
                print(f"{slide_id} complete, total n patch = {i}")
            return True

    except Exception as e:
        print("error with slide id {}".format(slide_id))
        print(e)
        return False
    finally:
        if 'hdf' in locals() and hdf is not None:
            try:
                hdf.close()
            except Exception:
                pass
        if slide is not None:
            try:
                slide.close()
            except Exception:
                pass
        gc.collect()

def get_slide_id(slide_name):
    return Path(slide_name).stem

counter = None
lock = None
total_slides = 0

def init_worker(c, l, t):
    global counter, lock, total_slides
    counter = c
    lock = l
    total_slides = t

def process(opts):
    global counter, lock, total_slides
    slide_path, patch_size, patches_output_dir, mask_path, slide_id, max_patches_per_slide, seed = opts

    success = extract_patches(slide_path, mask_path, patch_size,
                              patches_output_dir, slide_id, max_patches_per_slide, seed)

    if counter is not None and lock is not None:
        with lock:
            counter.value += 1
            print(f"\n{'='*60}")
            print(f"Progress: {counter.value}/{total_slides} ({counter.value/total_slides*100:.1f}%)")
            print(f"Completed: {slide_id}")
            print(f"{'='*60}\n")
    return bool(success)


parser = argparse.ArgumentParser(description='Extract tissue patches from whole-slide images into HDF5 files')
parser.add_argument('--ref_file', required=True, type=str, help='Reference CSV containing wsi_file_name')
parser.add_argument('--wsi_path', required=True, type=str, help='Directory containing whole-slide images')
parser.add_argument('--patch_path', required=True, type=str, help='Output directory for patch HDF5 files')
parser.add_argument('--mask_path', required=True, type=str, help='Output directory for masks and patch metadata')
parser.add_argument('--patch_size', default=256, type=int, help='patch size, '
                                                                'default 256')
parser.add_argument('--start', type=int, default=0,
                    help='First slide index to process')
parser.add_argument('--end', type=int, default=None,
                    help='Exclusive final slide index to process')
parser.add_argument('--max_patches_per_slide', default=None, type=int,
                    help='Maximum accepted tissue patches per slide')
parser.add_argument('--seed', default=5, type=int,
                    help='Deterministic candidate-patch shuffle seed')
parser.add_argument('--debug', action='store_true',
                    help='Process at most five slides and 20 patches per slide')
parser.add_argument('--parallel', default=1, type=int,
                    help='Number of worker processes; use 0 for serial execution')


if __name__ == '__main__':

    args = parser.parse_args()
    if args.patch_size <= 0:
        parser.error("--patch_size must be positive")
    if args.max_patches_per_slide is not None and args.max_patches_per_slide <= 0:
        parser.error("--max_patches_per_slide must be positive")
    if args.parallel < 0:
        parser.error("--parallel must be non-negative")
    wsi_root = Path(args.wsi_path)
    slide_paths = [
        p for p in wsi_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".svs", ".tiff", ".tif"}
    ]
    slide_by_name = {p.name: p for p in slide_paths}
    slide_list = sorted(slide_by_name)

    if args.ref_file:
        ref_file = pd.read_csv(args.ref_file)
        selected_slides = list(ref_file['wsi_file_name'])
        # wsi_file_name already includes the .svs extension
        slide_list = list(set(slide_list) & set(selected_slides))
        slide_list = sorted(slide_list)

    slide_list = slide_list[args.start:args.end]

    if args.debug:
        slide_list = slide_list[0:5]
        args.max_patches_per_slide = 20

    print(f"Found {len(slide_list)} slides")
    if not slide_list:
        raise SystemExit("No matching whole-slide images were found")
    print(f"{'='*60}")
    print("Start patch extraction")
    print(f"WSI count: {len(slide_list)}")
    print(f"Parallel workers: {args.parallel if args.parallel else 1}")
    print(f"Max patches per WSI: {args.max_patches_per_slide}")
    print(f"{'='*60}\n")

    opts = [
        (str(slide_by_name[s]), (args.patch_size, args.patch_size), args.patch_path, args.mask_path,
        get_slide_id(s), args.max_patches_per_slide, args.seed) for
        (i, s) in enumerate(slide_list)]

    if args.parallel:
        manager = Manager()
        counter = manager.Value('i', 0)
        lock = manager.Lock()

        pool = Pool(processes=args.parallel, initializer=init_worker,
                   initargs=(counter, lock, len(slide_list)))
        results = pool.map(process, opts)
        pool.close()
        pool.join()
        if not all(results):
            failed = len([x for x in results if not x])
            raise SystemExit(f"Patch extraction failed for {failed} slide(s)")

        print(f"\n{'='*60}")
        print(f"All done. Processed {len(slide_list)} WSI")
        print(f"{'='*60}\n")
    else:
        results = []
        for i, opt in enumerate(opts):
            results.append(process(opt))
            print(f"Progress: {i+1}/{len(slide_list)} ({(i+1)/len(slide_list)*100:.1f}%)")
        if not all(results):
            failed = len([x for x in results if not x])
            raise SystemExit(f"Patch extraction failed for {failed} slide(s)")
