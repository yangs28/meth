"""Extract georeferenced centers of predicted methane-plume mask components.

Read-only post-processing; never loads, retrains, or edits a Model6 checkpoint.
Modes:
  --run-id <ID> or --run-dir <PATH>  Read an existing completed GTM Model6 plume run.
  --input <GEOTIFF|NPY|NPZ>         Read an individual prediction with georeferencing.

Use --help for exact arguments. Requires numpy, scipy and rasterio.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform as transform_coordinates
from scipy import ndimage
from scipy.special import expit

REPO_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "scene_id", "scene_index", "dataset", "method", "component_id", "pixel_count",
    "centroid_row", "centroid_col", "centroid_x", "centroid_y",
    "centroid_lon", "centroid_lat", "centroid_in_plume",
    "representative_row", "representative_col", "representative_x", "representative_y",
    "representative_lon", "representative_lat", "mean_probability", "max_probability",
    "threshold", "weighting", "min_pixels", "connectivity", "crs", "pixel_area_native_sq",
    "prediction_file", "source_georeference", "tessera_status", "tessera_row", "tessera_col",
]


def geographic_point(affine: Affine, crs: CRS, row: float, col: float):
    """Transform a (possibly fractional) raster pixel-center row/column to WGS84."""
    x, y = affine * (float(col) + 0.5, float(row) + 0.5)
    lons, lats = transform_coordinates(crs, "EPSG:4326", [x], [y])
    return float(x), float(y), float(lons[0]), float(lats[0])


def tessera_cell(tile, lon: float, lat: float):
    """Look up a point in a *supplied* georeferenced embedding grid, if any."""
    if tile is None:
        return {"tessera_status": "not_requested", "tessera_row": None, "tessera_col": None}
    xs, ys = transform_coordinates("EPSG:4326", tile.crs, [lon], [lat])
    row, col = tile.index(xs[0], ys[0])
    if not (0 <= row < tile.height and 0 <= col < tile.width):
        return {"tessera_status": "outside_tile", "tessera_row": None, "tessera_col": None}
    return {"tessera_status": "within_tile_bounds_unverified_data", "tessera_row": int(row), "tessera_col": int(col)}


def calculate_plume_centroids(
    probability: np.ndarray,
    observable: np.ndarray,
    affine: Affine,
    crs: CRS | str,
    *,
    threshold: float = 0.5,
    min_pixels: int = 100,
    connectivity: int = 4,
    weighting: str = "geometric",
    value_type: str = "probability",
    tessera_tile=None,
) -> list[dict[str, Any]]:
    """Return one georeferenced centroid and one on-mask pixel per component.

    Note: area centroid may be outside a C-shaped component. This is not an
    emission-source estimate. The representative point is an actual foreground
    pixel nearest the centroid using native affine-mapped distance.
    """
    p = np.asarray(probability, dtype=np.float64)
    valid = np.asarray(observable, dtype=bool).copy()
    if p.ndim != 2 or valid.shape != p.shape:
        raise ValueError("Prediction and observable masks must be matching 2D arrays")
    if min_pixels < 1 or connectivity not in {4, 8} or weighting not in {"geometric", "probability"}:
        raise ValueError("Invalid minimum size, connectivity or weighting")
    if value_type not in {"probability", "logit", "binary"}:
        raise ValueError("value_type must be probability, logit or binary")
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    crs = CRS.from_user_input(crs)
    if crs is None or affine.is_identity or abs(affine.a * affine.e - affine.b * affine.d) < 1e-18:
        raise ValueError("Prediction needs valid, invertible georeferencing")
    valid &= np.isfinite(p)
    if value_type == "probability":
        if np.any(valid & ((p < -1e-6) | (p > 1.0 + 1e-6))):
            raise ValueError("Probability values outside [0,1]: check the input file/band")
        score = p
    elif value_type == "logit":
        score = expit(p)
    else:
        score = p
    foreground = valid & ((p > 0) if value_type == "binary" else (score >= threshold))
    if weighting == "probability" and value_type == "binary":
        raise ValueError("Probability weighting needs probability/logit input")
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labeled, count = ndimage.label(foreground, structure=structure)
    components = ndimage.find_objects(labeled)
    output: list[dict[str, Any]] = []
    for component_id, slices in enumerate(components, start=1):
        if slices is None:
            continue
        rr, cc = np.nonzero(labeled[slices] == component_id)
        if rr.size < min_pixels:
            continue
        rr = rr + slices[0].start
        cc = cc + slices[1].start
        values = score[rr, cc]
        if weighting == "probability":
            if values.sum() <= 0:
                raise ValueError("Component's probability weights must sum to positive")
            center_row = float(np.average(rr, weights=values))
            center_col = float(np.average(cc, weights=values))
        else:
            center_row = float(rr.mean())
            center_col = float(cc.mean())
        dc = cc - center_col
        dr = rr - center_row
        dx, dy = affine.a * dc + affine.b * dr, affine.d * dc + affine.e * dr
        closest = int(np.argmin(dx * dx + dy * dy))
        representative_row, representative_col = int(rr[closest]), int(cc[closest])
        cpr = int(np.floor(center_row + 0.5))
        cpc = int(np.floor(center_col + 0.5))
        center_inside = (0 <= cpr < p.shape[0] and 0 <= cpc < p.shape[1]
                         and labeled[cpr, cpc] == component_id)
        cx, cy, clon, clat = geographic_point(affine, crs, center_row, center_col)
        rx, ry, rlon, rlat = geographic_point(affine, crs, representative_row, representative_col)
        result = dict(
            component_id=component_id, pixel_count=int(rr.size),
            centroid_row=center_row, centroid_col=center_col,
            centroid_x=cx, centroid_y=cy, centroid_lon=clon, centroid_lat=clat,
            centroid_in_plume=bool(center_inside),
            representative_row=representative_row, representative_col=representative_col,
            representative_x=rx, representative_y=ry, representative_lon=rlon, representative_lat=rlat,
            mean_probability=float(values.mean()) if value_type != "binary" else None,
            max_probability=float(values.max()) if value_type != "binary" else None,
            threshold=threshold, weighting=weighting, min_pixels=min_pixels,
            connectivity=connectivity, crs=crs.to_string(),
            pixel_area_native_sq=abs(affine.a * affine.e - affine.b * affine.d),
        )
        # A TESSERA pixel can be sampled at the representative (on-plume) point.
        result.update(tessera_cell(tessera_tile, rlon, rlat))
        output.append(result)
    return output


def _single_prediction(input_path: Path, reference_raster: Path | None, band: int, array_key: str | None):
    with ExitStack() as stack:
        if input_path.suffix.lower() in {".tif", ".tiff"}:
            src = stack.enter_context(rasterio.open(input_path))
            if band < 1 or band > src.count:
                raise ValueError(f"Prediction TIFF has {src.count} band(s); requested band {band}")
            values = src.read(band)
            valid = src.read_masks(band) > 0
            affine, crs, reference = src.transform, src.crs, str(input_path)
            if reference_raster:
                ref = stack.enter_context(rasterio.open(reference_raster))
                if (src.width, src.height) != (ref.width, ref.height):
                    raise ValueError("Prediction and reference raster have different dimensions")
                if src.crs is not None and not src.transform.is_identity and (
                    src.crs != ref.crs or not src.transform.almost_equals(ref.transform)
                ):
                    raise ValueError("Prediction TIFF and reference raster have different georeferencing")
                affine, crs, reference = ref.transform, ref.crs, str(reference_raster)
                valid &= ref.read_masks(1) > 0
        elif input_path.suffix.lower() in {".npy", ".npz"}:
            if not reference_raster:
                raise ValueError("NPY/NPZ predictions need the exact --reference-raster (CRS and affine)")
            if input_path.suffix.lower() == ".npy":
                values = np.load(input_path, allow_pickle=False)
            else:
                with np.load(input_path, allow_pickle=False) as archive:
                    if array_key is None:
                        if len(archive.files) != 1:
                            raise ValueError(f"Select --array-key; options: {archive.files}")
                        array_key = archive.files[0]
                    values = np.asarray(archive[array_key])
            values = np.squeeze(values)
            ref = stack.enter_context(rasterio.open(reference_raster))
            if values.shape != (ref.height, ref.width):
                raise ValueError("Prediction array must match exact reference raster shape; do not resize")
            valid = ref.read_masks(1) > 0
            affine, crs, reference = ref.transform, ref.crs, str(reference_raster)
        else:
            raise ValueError("Only .tif, .tiff, .npy and .npz prediction files are supported")
        if crs is None or affine.is_identity:
            raise ValueError("Missing valid CRS/affine: cannot geolocate prediction")
        return np.asarray(values), valid, affine, crs, reference


def _historical_run(run_dir: Path, methods: list[str], min_pixels: int, connectivity: int,
                    weighting: str, tile) -> tuple[list[dict], dict]:
    """Read existing research/GTM_Model6/GTM_Model6_plume_* outputs, no inference."""
    manifest = json.loads((run_dir / "GTM_Model6_plume_manifest.json").read_text(encoding="utf-8"))
    results = json.loads((run_dir / "GTM_Model6_plume_results.json").read_text(encoding="utf-8"))
    if not results.get("complete"):
        raise ValueError("The saved GTM Model6 plume run is not marked complete")
    records = {int(r["index"]): r for r in results["records"]}
    rows = []
    with np.load(run_dir / "GTM_Model6_plume_inputs.npz", allow_pickle=False) as source:
        for i, scene in enumerate(manifest["observations"]):
            valid = np.asarray(source[f"{i}_valid"], dtype=bool)
            affine, crs = Affine(*scene["affine"]), CRS.from_user_input(scene["crs"])
            rec = records[i]
            path = run_dir / f"GTM_Model6_plume_prediction_{i}.npz"
            with np.load(path, allow_pickle=False) as predictions:
                for method in methods:
                    if method not in predictions.files or method not in rec["thresholds"]:
                        raise ValueError(f"Missing saved prediction or threshold for method {method} in {path}")
                    probability = np.asarray(predictions[method])
                    if probability.shape != valid.shape:
                        raise ValueError(f"Scene {i}: prediction and valid mask shapes do not agree")
                    if "shape" in scene and list(probability.shape) != list(scene["shape"]):
                        raise ValueError(f"Scene {i}: prediction shape differs from manifest")
                    threshold = float(rec["thresholds"][method])
                    found = calculate_plume_centroids(
                        probability, valid, affine, crs, threshold=threshold,
                        min_pixels=min_pixels, connectivity=connectivity, weighting=weighting,
                        tessera_tile=tile,
                    )
                    rows.extend([{**item, "scene_id": scene["id"], "scene_index": i,
                                  "dataset": scene.get("dataset"), "method": method,
                                  "prediction_file": str(path),
                                  "source_georeference": "GTM_Model6_plume_manifest.json"}
                                 for item in found])
                    print(f"Scene {i} [{method}]: {len(found)} component(s)")
    return rows, {"source": "completed_historical_GTM_Model6_plume_run",
                  "run_dir": str(run_dir), "methods": methods, "scene_count": len(manifest["observations"]),
                  "threshold_policy": "recorded per-scene/per-method threshold from results JSON"}


def _write_outputs(outdir: Path, records: list[dict], metadata: dict, *, overwrite: bool = False):
    names = ("plume_centroids.csv", "plume_centroids.geojson", "run_metadata.json")
    targets = [outdir / name for name in names]
    if any(path.exists() for path in targets) and not overwrite:
        raise FileExistsError(f"Results already exist in {outdir}; choose a new directory")
    outdir.mkdir(parents=True, exist_ok=True)
    with targets[0].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    features = [{"type": "Feature", "geometry": {"type": "Point",
                                                  "coordinates": [r["centroid_lon"], r["centroid_lat"]]},
                 "properties": {k: v for k, v in r.items() if k not in {"centroid_lon", "centroid_lat"}}}
                for r in records]
    targets[1].write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2,
                                      allow_nan=False), encoding="utf-8")
    details = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "component_count": len(records), "output_files": [str(p) for p in targets],
               "geographic_reference": "WGS84 EPSG:4326; GeoJSON geometry is the mathematical centroid",
               "representative_point": "nearest actual selected plume pixel; properties representative_lon/lat",
               "caveat": "Predicted plume component center, not a verified methane source location",
               "model_checkpoint_modified": False, **metadata}
    targets[2].write_text(json.dumps(details, indent=2, allow_nan=False), encoding="utf-8")
    return targets


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-id", help="Existing run ID below outputs/GTM_Model6/ (historical plume run)")
    mode.add_argument("--run-dir", type=Path, help="Absolute/relative directory of completed GTM Model6 plume run")
    mode.add_argument("--input", type=Path, help="Standalone prediction TIFF, NPY, or NPZ")
    parser.add_argument("--reference-raster", type=Path, help="Exact georeferenced reference raster, mandatory for NPY/NPZ")
    parser.add_argument("--band", type=int, default=1, help="1-indexed TIFF probability band")
    parser.add_argument("--array-key", help="Array name within multi-key NPZ")
    parser.add_argument("--value-type", choices=["probability", "logit", "binary"], default="probability")
    parser.add_argument("--threshold", type=float, default=0.5, help="For --input only; existing GTM runs use saved thresholds")
    parser.add_argument("--methods", default="mean", help="Comma-separated GTM Model6 methods, e.g. mean,S16")
    parser.add_argument("--min-pixels", type=int, default=100, help="Match existing GTM Model6 exporter default")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=4, help="Match existing GTM Model6 exporter default")
    parser.add_argument("--weighting", choices=["geometric", "probability"], default="geometric")
    parser.add_argument("--tessera-raster", type=Path, help="Optional existing georeferenced TESSERA embedding tile TIFF")
    parser.add_argument("--output-dir", type=Path, help="New directory for CSV/GeoJSON/JSON; defaults under outputs/plume_centroids")
    parser.add_argument("--overwrite", action="store_true", help="Replace the three existing export files explicitly")
    args = parser.parse_args(argv)
    if args.band < 1 or args.min_pixels < 1:
        parser.error("--band and --min-pixels must each be >=1")
    if args.run_id and (Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}):
        parser.error("--run-id must be one directory name, not a path")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods or any(re.fullmatch(r"[A-Za-z0-9_-]+", m) is None for m in methods):
        parser.error("--methods must contain simple comma-separated method names")
    if len(set(methods)) != len(methods):
        parser.error("--methods may not repeat names")
    out = args.output_dir
    with ExitStack() as stack:
        tile = None
        if args.tessera_raster:
            tile = stack.enter_context(rasterio.open(args.tessera_raster))
            if tile.crs is None or tile.transform.is_identity:
                raise ValueError("TESSERA raster needs georeferencing metadata")
        if args.run_id or args.run_dir:
            run = ((REPO_ROOT / "outputs" / "GTM_Model6" / args.run_id) if args.run_id
                   else args.run_dir).resolve()
            if out is None:
                out = REPO_ROOT / "outputs" / "plume_centroids" / (run.name + "_" + "-".join(methods))
            records, metadata = _historical_run(run, methods, args.min_pixels,
                                                args.connectivity, args.weighting, tile)
        else:
            srcpath = args.input.resolve()
            if out is None:
                out = REPO_ROOT / "outputs" / "plume_centroids" / srcpath.stem
            p, valid, affine, crs, reference = _single_prediction(
                srcpath, args.reference_raster, args.band, args.array_key)
            parts = calculate_plume_centroids(
                p, valid, affine, crs, threshold=args.threshold, min_pixels=args.min_pixels,
                connectivity=args.connectivity, weighting=args.weighting,
                value_type=args.value_type, tessera_tile=tile)
            records = [{**item, "scene_id": srcpath.stem, "scene_index": 0,
                        "dataset": None, "method": "standalone", "prediction_file": str(srcpath),
                        "source_georeference": reference} for item in parts]
            metadata = {"source": "standalone_prediction", "prediction": str(srcpath),
                        "threshold": args.threshold, "value_type": args.value_type,
                        "georeference": reference}
    targets = _write_outputs(out.resolve(), records, metadata, overwrite=args.overwrite)
    print(f"Finished: {len(records)} plume center(s). CSV: {targets[0]}")
    print(f"GeoJSON: {targets[1]}")
    if not records:
        print("No connected components passed the threshold and size filter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
