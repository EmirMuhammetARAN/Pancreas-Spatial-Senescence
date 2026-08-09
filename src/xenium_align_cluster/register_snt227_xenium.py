"""Register the 69-year SNT227-bottom PhenoCycler section to Xenium.

This is an exploratory, reproducible registration run.  It does two distinct
things:

1. Registers the *DAPI images* using many SIFT image landmarks and RANSAC.
2. After image registration, creates conservative cell pseudo-labels from
   reciprocal nearest Xenium/PhenoCycler centroids.

A centroid match alone is not proof that two cell masks are identical.  The
``strict`` output (reciprocal nearest neighbours <= 3 um) is therefore the
appropriate starting set for training.  The ``high_confidence`` set (<= 5 um)
and ``candidate`` set (<= 8 um) are retained for sensitivity analyses, not
silently mixed into the strict training data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import cv2
import h5py
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree


RATIO_TEST = 0.72
RANSAC_THRESHOLD_PX = 2.0
STRICT_DISTANCE_UM = 3.0
HIGH_CONFIDENCE_DISTANCE_UM = 5.0
CANDIDATE_DISTANCE_UM = 8.0
RNG = np.random.default_rng(69)


def percentile_uint8(image: np.ndarray) -> np.ndarray:
    """Robustly map a DAPI image to uint8 without allowing bright debris to dominate."""
    low, high = np.percentile(image, (1.0, 99.8))
    scaled = np.clip((image.astype(np.float32) - low) / (high - low + 1e-6), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def ome_physical_pixel_sizes(tiff: tifffile.TiffFile) -> tuple[float, float]:
    """Read native Xenium pixel size in micrometres from OME XML."""
    root = ElementTree.fromstring(tiff.ome_metadata)
    pixels = next(element for element in root.iter() if element.tag.endswith("Pixels"))
    return float(pixels.attrib["PhysicalSizeX"]), float(pixels.attrib["PhysicalSizeY"])


def read_xenium_dapi(path: Path, level: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Return MIP DAPI and (um per x/y pixel) at the selected pyramid level."""
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        dapi = series.levels[level].asarray().max(axis=0)
        physical_x, physical_y = ome_physical_pixel_sizes(tiff)
        full_y, full_x = series.shape[-2:]
        level_y, level_x = dapi.shape
    return dapi, (physical_x * full_x / level_x, physical_y * full_y / level_y)


def read_phenocycler_dapi(path: Path, level: int, start_y_full: int) -> tuple[np.ndarray, int]:
    """Return DAPI channel 0 from the bottom SNT227 component and its level-space y start."""
    with tifffile.TiffFile(path) as tiff:
        level_image = tiff.series[0].levels[level].pages[0].asarray()
        full_y = tiff.series[0].shape[-2]
        downsample = full_y / level_image.shape[0]
    start_y_level = int(round(start_y_full / downsample))
    return level_image[start_y_level:, :], start_y_level


def orientation_matrices(width: int, height: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return image variants and matrices from original crop coordinates to variant coordinates."""
    identity = np.eye(3, dtype=np.float64)
    return {
        "identity": (identity, identity),
        "flip_lr": (
            np.array([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            None,
        ),
        "flip_ud": (
            np.array([[1.0, 0.0, 0.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]]),
            None,
        ),
        "rotate_180": (
            np.array([[-1.0, 0.0, width - 1.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]]),
            None,
        ),
    }


def apply_orientation(image: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return image
    if name == "flip_lr":
        return cv2.flip(image, 1)
    if name == "flip_ud":
        return cv2.flip(image, 0)
    if name == "rotate_180":
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError(f"Unknown orientation: {name}")


def estimate_affine(pc_crop: np.ndarray, xenium_dapi: np.ndarray) -> tuple[np.ndarray, str, dict[str, float]]:
    """Find the best DAPI affine registration, testing the four image orientations."""
    source = percentile_uint8(pc_crop)
    target = percentile_uint8(xenium_dapi)
    sift = cv2.SIFT_create(nfeatures=60000)
    target_keypoints, target_descriptors = sift.detectAndCompute(target, None)
    if target_descriptors is None:
        raise RuntimeError("No SIFT features were found in the Xenium DAPI image.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    best: tuple[np.ndarray, str, dict[str, float]] | None = None
    width, height = source.shape[1], source.shape[0]

    for name, (original_to_variant, _) in orientation_matrices(width, height).items():
        variant = apply_orientation(source, name)
        source_keypoints, source_descriptors = sift.detectAndCompute(variant, None)
        if source_descriptors is None:
            continue
        raw_matches = matcher.knnMatch(source_descriptors, target_descriptors, k=2)
        matches = [first for first, second in raw_matches if first.distance < RATIO_TEST * second.distance]
        if len(matches) < 25:
            continue
        source_xy = np.float32([source_keypoints[match.queryIdx].pt for match in matches])
        target_xy = np.float32([target_keypoints[match.trainIdx].pt for match in matches])
        affine, inliers = cv2.estimateAffine2D(
            source_xy,
            target_xy,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD_PX,
            maxIters=20000,
            confidence=0.999,
            refineIters=50,
        )
        if affine is None or inliers is None:
            continue
        predicted = cv2.transform(source_xy[None, :, :], affine)[0]
        residuals = np.linalg.norm(predicted - target_xy, axis=1)
        use = inliers.ravel().astype(bool)
        stats = {
            "matches": float(len(matches)),
            "inliers": float(use.sum()),
            "median_residual_level_px": float(np.median(residuals[use])),
            "p95_residual_level_px": float(np.percentile(residuals[use], 95)),
        }
        affine_h = np.vstack((affine, [0.0, 0.0, 1.0]))
        # This maps the original (unflipped) PC crop directly to Xenium level pixels.
        original_to_xenium = affine_h @ original_to_variant
        if best is None or stats["inliers"] > best[2]["inliers"]:
            best = (original_to_xenium, name, stats)

    if best is None:
        raise RuntimeError("No robust DAPI registration was found. Inspect the selected SNT227 crop.")
    return best


def load_xenium_observations(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        xy = handle["obsm"]["spatial"][()].astype(np.float64, copy=False)
        obs = handle["obs"]
        codes = obs["leiden"]["codes"][()]
        categories = decode_strings(obs["leiden"]["categories"][()])
        leiden = np.full(len(codes), "Unknown", dtype=object)
        valid = codes >= 0
        leiden[valid] = categories[codes[valid]]
        cell_id_key = "cell_id" if "cell_id" in obs else "_index"
        cell_ids = decode_strings(obs[cell_id_key][()])
    return xy, cell_ids, leiden


def quality_labels(reciprocal: np.ndarray, distances_um: np.ndarray) -> np.ndarray:
    labels = np.full(len(distances_um), "unmatched", dtype=object)
    labels[reciprocal & (distances_um <= CANDIDATE_DISTANCE_UM)] = "candidate"
    labels[reciprocal & (distances_um <= HIGH_CONFIDENCE_DISTANCE_UM)] = "high_confidence"
    labels[reciprocal & (distances_um <= STRICT_DISTANCE_UM)] = "strict"
    return labels


def write_qc_images(output_dir: Path, xenium: np.ndarray, pc_crop: np.ndarray, pc_to_xenium_level: np.ndarray) -> None:
    """Write a DAPI overlay; yellow structures indicate agreement, not cell labels."""
    warped = cv2.warpAffine(
        percentile_uint8(pc_crop),
        pc_to_xenium_level[:2],
        (xenium.shape[1], xenium.shape[0]),
        flags=cv2.INTER_LINEAR,
    )
    target = percentile_uint8(xenium)
    overlay = np.dstack((warped, target, np.zeros_like(target)))
    cv2.imwrite(str(output_dir / "dapi_overlay_red_pc_green_xenium.png"), overlay)
    cv2.imwrite(str(output_dir / "xenium_dapi.png"), target)
    cv2.imwrite(str(output_dir / "phenocycler_bottom_dapi.png"), percentile_uint8(pc_crop))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    spatial_dir = project_root / "dataset" / "spatial-transkriptomics"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenocycler-image", type=Path, default=project_root / "dataset" / "raw_images" / "SNT227_PC24038_Scan1.qptiff")
    parser.add_argument("--phenocycler-cells", type=Path, default=project_root / "dataset" / "parquets" / "features_dual_SNT227_age69.parquet")
    parser.add_argument("--xenium-image", type=Path, default=next(spatial_dir.glob("morphology.ome*227*")))
    parser.add_argument("--xenium-h5ad", type=Path, default=next(spatial_dir.glob("secondary_analysis*227alt*.h5ad")))
    parser.add_argument("--output-dir", type=Path, default=project_root / "deneme" / "snt227_xenium_registration")
    parser.add_argument("--split-y-full-px", type=int, default=26000, help="Top/bottom boundary in the raw 51,840-pixel QPTIFF.")
    args = parser.parse_args()

    for input_path in (args.phenocycler_image, args.phenocycler_cells, args.xenium_image, args.xenium_h5ad):
        if not input_path.exists():
            raise FileNotFoundError(input_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Level 4 PC has ~16x downsampling; Xenium level 5 is ~32x.  These are
    # comparably sized, sufficiently detailed images for robust global affine fitting.
    xenium_dapi, xenium_um_per_pixel = read_xenium_dapi(args.xenium_image, level=5)
    pc_dapi, crop_start_level = read_phenocycler_dapi(args.phenocycler_image, level=4, start_y_full=args.split_y_full_px)
    pc_to_xenium_level, chosen_orientation, fit = estimate_affine(pc_dapi, xenium_dapi)
    write_qc_images(args.output_dir, xenium_dapi, pc_dapi, pc_to_xenium_level)

    # Raw PC global pixels -> PC crop pixels at level 4.
    raw_to_crop = np.array(
        [[1.0 / 16.0, 0.0, 0.0], [0.0, 1.0 / 16.0, -float(crop_start_level)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    level_to_um = np.diag((xenium_um_per_pixel[0], xenium_um_per_pixel[1], 1.0))
    raw_pc_to_xenium_um = level_to_um @ pc_to_xenium_level @ raw_to_crop

    print("Reading PhenoCycler features and retaining only SNT227 bottom…")
    pc = pd.read_parquet(args.phenocycler_cells)
    pc = pc.loc[pc["global_y"] >= args.split_y_full_px].copy()
    if pc.empty:
        raise RuntimeError("The requested bottom component contains no PhenoCycler cells.")
    print(f"PhenoCycler bottom cells: {len(pc):,}")

    print("Reading Xenium cell centroids and Leiden labels…")
    xenium_xy, xenium_ids, xenium_leiden = load_xenium_observations(args.xenium_h5ad)
    print(f"Xenium cells: {len(xenium_xy):,}")

    raw_xy = pc[["global_x", "global_y"]].to_numpy(dtype=np.float64, copy=False)
    pc_homogeneous = np.column_stack((raw_xy, np.ones(len(raw_xy), dtype=np.float64)))
    registered_xy = (pc_homogeneous @ raw_pc_to_xenium_um.T)[:, :2]

    xenium_tree = cKDTree(xenium_xy)
    pc_tree = cKDTree(registered_xy)
    distances_um, xenium_index = xenium_tree.query(registered_xy, k=1, workers=-1)
    _, pc_index_from_xenium = pc_tree.query(xenium_xy, k=1, workers=-1)
    reciprocal = pc_index_from_xenium[xenium_index] == np.arange(len(pc))
    labels = quality_labels(reciprocal, distances_um)

    result = pc.copy()
    result.insert(0, "pc_cell_id", result["tile_y"].astype(str) + "_" + result["tile_x"].astype(str) + "_" + result["label"].astype(str))
    result["registered_x_um"] = registered_xy[:, 0]
    result["registered_y_um"] = registered_xy[:, 1]
    result["xenium_cell_id"] = xenium_ids[xenium_index]
    result["xenium_x_um"] = xenium_xy[xenium_index, 0]
    result["xenium_y_um"] = xenium_xy[xenium_index, 1]
    result["xenium_leiden"] = xenium_leiden[xenium_index]
    result["centroid_distance_um"] = distances_um
    result["mutual_nearest_neighbor"] = reciprocal
    result["match_quality"] = labels

    all_path = args.output_dir / "SNT227_matches_all.parquet"
    strict_path = args.output_dir / "SNT227_matches_strict_3um.parquet"
    high_path = args.output_dir / "SNT227_matches_high_confidence_5um.parquet"
    result.to_parquet(all_path, index=False)
    result.loc[result["match_quality"] == "strict"].to_parquet(strict_path, index=False)
    result.loc[result["match_quality"].isin(["strict", "high_confidence"])].to_parquet(high_path, index=False)

    counts = result["match_quality"].value_counts().to_dict()
    metadata = {
        "source": "SNT227 bottom PhenoCycler component registered to 69-year Xenium",
        "pc_split_y_full_px": args.split_y_full_px,
        "selected_orientation": chosen_orientation,
        "dapi_affine_fit": fit,
        "raw_pc_pixel_to_xenium_um_affine": raw_pc_to_xenium_um.tolist(),
        "xenium_um_per_level5_pixel_xy": xenium_um_per_pixel,
        "matching_rules": {
            "strict": "reciprocal nearest neighbour AND <= 3 um",
            "high_confidence": "reciprocal nearest neighbour AND <= 5 um",
            "candidate": "reciprocal nearest neighbour AND <= 8 um",
        },
        "counts": {key: int(value) for key, value in counts.items()},
        "distance_um_quantiles": {str(q): float(result["centroid_distance_um"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
    }
    (args.output_dir / "registration_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savetxt(args.output_dir / "raw_pc_to_xenium_um_affine.txt", raw_pc_to_xenium_um, fmt="%.12f")

    print(f"DAPI orientation: {chosen_orientation}; inliers: {int(fit['inliers']):,}; median residual: {fit['median_residual_level_px']:.3f} level-5 px")
    print(f"Strict matches (<=3 um): {counts.get('strict', 0):,}")
    print(f"High-confidence matches (<=5 um, including strict): {counts.get('strict', 0) + counts.get('high_confidence', 0):,}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
