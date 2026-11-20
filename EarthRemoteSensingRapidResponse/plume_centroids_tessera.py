"""Extract georeferenced centers of predicted methane-plume mask components.

Read-only post-processing; never loads, retrains, or edits a Model6 checkpoint.
Modes:
  --run-id <ID> or --run-dir <PATH>  Read an existing completed GTM Model6 plume run.
  --input <GEOTIFF|NPY|NPZ>         Process one georeferenced prediction.
  --input-dir <DIRECTORY>           Recursively process saved probability GeoTIFFs.

Optional --html-report produces a self-contained, offline visual report.
Does not perform model inference. Requires numpy, scipy, rasterio.
Optional geodesic ground area requires pyproj; if unavailable it is recorded as missing.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import html
import json
import math
import re
import os
import shutil
import tempfile
import uuid
import sys
import ctypes
import itertools
from contextlib import contextmanager
import struct
import zlib
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
from rasterio.windows import Window

try:
    from pyproj import Geod
except ImportError:  # Geodesic areas are optional; never silently substitute planar area.
    Geod = None

SCHEMA_VERSION = 3
PATCH_VERSION = "3.1"
MANAGED_FILES = ("plume_centroids.csv", "plume_centroids.geojson", "run_metadata.json", "report.html")

REPO_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "scene_id", "scene_index", "dataset", "method", "component_id", "pixel_count",
    "centroid_row", "centroid_col", "centroid_x", "centroid_y",
    "centroid_lon", "centroid_lat", "centroid_in_plume",
    "representative_row", "representative_col", "representative_x", "representative_y",
    "representative_lon", "representative_lat", "mean_probability", "max_probability",
    "threshold", "weighting", "min_pixels", "connectivity", "crs", "pixel_area_native_sq",
    "prediction_file", "source_georeference", "tessera_status", "tessera_row", "tessera_col",
    "component_area_m2", "prediction_sha256",  # v2 field: projected plane m², NOT geodesic ground area
    "component_projected_area_m2", "component_ground_area_m2", "ground_area_status",
    "tessera_lookup_point", "tessera_centroid_status", "tessera_centroid_row", "tessera_centroid_col",
    "tessera_representative_status", "tessera_representative_row", "tessera_representative_col",
    "tessera_valid_band_count", "tessera_total_bands", "tessera_tile", "tessera_year",
]


def geographic_point(affine: Affine, crs: CRS, row: float, col: float):
    """Transform a (possibly fractional) raster pixel-center row/column to WGS84."""
    x, y = affine * (float(col) + 0.5, float(row) + 0.5)
    lons, lats = transform_coordinates(crs, "EPSG:4326", [x], [y])
    return float(x), float(y), float(lons[0]), float(lats[0])


def tessera_cell(tile, lon: float, lat: float) -> dict[str, Any]:
    """Locate an EPSG:4326 point in a supplied raster AND inspect pixel validity.

    A valid raster cell is not proof that the raster is a TESSERA embedding or that
    its year matches a methane overpass. Does not extract a 128-D embedding.
    """
    if tile is None:
        return {"tessera_status": "not_requested", "tessera_row": None,
                "tessera_col": None, "tessera_valid_band_count": None,
                "tessera_total_bands": None}
    xs, ys = transform_coordinates("EPSG:4326", tile.crs, [lon], [lat])
    if not np.isfinite(xs[0]) or not np.isfinite(ys[0]):
        raise ValueError("TESSERA coordinate could not be transformed into the raster CRS")
    row, col = tile.index(xs[0], ys[0])
    if not (0 <= row < tile.height and 0 <= col < tile.width):
        return {"tessera_status": "outside_tile", "tessera_row": None,
                "tessera_col": None, "tessera_valid_band_count": 0,
                "tessera_total_bands": int(tile.count)}
    # Masked raster read observes per-band nodata, dataset mask, alpha, and NaN/Inf.
    cell = tile.read(window=Window(col, row, 1, 1), masked=True)
    unmasked_finite = ~np.ma.getmaskarray(cell).reshape(-1) & np.isfinite(np.asarray(cell.data).reshape(-1))
    good = int(np.count_nonzero(unmasked_finite))
    return {"tessera_status": "valid_raster_cell" if good == tile.count else "missing_or_nonfinite_raster_data",
            "tessera_row": int(row), "tessera_col": int(col),
            "tessera_valid_band_count": good, "tessera_total_bands": int(tile.count)}


def _geodesic_component_area(rr: np.ndarray, cc: np.ndarray, affine: Affine,
                             crs: CRS) -> tuple[float | None, str]:
    """Sum WGS84 ellipsoidal areas of the individual native pixel footprints.

    Unlike a projected-area determinant, this handles EPSG:3857, geographic CRS,
    rotated affines and varying latitude. Pixel footprint != sensor independence.
    """
    if Geod is None:
        return None, "unavailable_pyproj_not_installed"
    if len(rr) == 0:
        return 0.0, "computed_wgs84_geodesic"
    # Coordinate transformations are vectorized; polygon areas are per pixel.
    longitude = []
    latitude = []
    for dc, dr in ((0, 0), (1, 0), (1, 1), (0, 1)):
        xx = affine.a * (cc + dc) + affine.b * (rr + dr) + affine.c
        yy = affine.d * (cc + dc) + affine.e * (rr + dr) + affine.f
        lons, lats = transform_coordinates(crs, "EPSG:4326", xx.tolist(), yy.tolist())
        longitude.append(np.asarray(lons, dtype=np.float64))
        latitude.append(np.asarray(lats, dtype=np.float64))
    if not all(np.all(np.isfinite(z)) for z in longitude + latitude):
        raise ValueError("Cannot compute ground area: invalid transformed pixel corners")
    geod = Geod(ellps="WGS84")
    total = 0.0
    for i in range(len(rr)):
        signed, _ = geod.polygon_area_perimeter(
            [a[i] for a in longitude], [a[i] for a in latitude])
        total += abs(float(signed))
    return total, "computed_wgs84_geodesic"


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
    tessera_point: str = "representative",
    tessera_year: str | None = None,
    ground_area_max_pixels: int = 100_000,
    require_ground_area: bool = False,
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
    if tessera_point not in {"representative", "centroid"}:
        raise ValueError("tessera_point must be representative or centroid")
    if ground_area_max_pixels < 0:
        raise ValueError("ground_area_max_pixels must be nonnegative (0 disables the limit)")
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
        det = abs(affine.a * affine.e - affine.b * affine.d)
        planar = (float(rr.size * det * crs.linear_units_factor[1] ** 2)
                  if crs.is_projected else None)
        if ground_area_max_pixels and rr.size > ground_area_max_pixels:
            ground, ground_status = None, "skipped_pixel_limit"
        else:
            ground, ground_status = _geodesic_component_area(rr, cc, affine, crs)
        if require_ground_area and ground is None:
            raise RuntimeError(f"Geodesic area unavailable for component {component_id}: {ground_status}; "
                               "install pyproj or increase --ground-area-max-pixels")
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
            pixel_area_native_sq=det,
            component_area_m2=planar,  # Deprecated v2 field; retained for compatibility.
            component_projected_area_m2=planar,
            component_ground_area_m2=ground,
            ground_area_status=ground_status,
        )
        # Explicitly record BOTH point lookups. The primary lookup is selectable.
        tc = tessera_cell(tessera_tile, clon, clat)
        tr = tessera_cell(tessera_tile, rlon, rlat)
        result.update({
            "tessera_lookup_point": tessera_point,
            "tessera_centroid_status": tc["tessera_status"],
            "tessera_centroid_row": tc["tessera_row"],
            "tessera_centroid_col": tc["tessera_col"],
            "tessera_representative_status": tr["tessera_status"],
            "tessera_representative_row": tr["tessera_row"],
            "tessera_representative_col": tr["tessera_col"],
            "tessera_tile": str(tessera_tile.name) if tessera_tile else None,
            "tessera_year": tessera_year,
        })
        result.update(tc if tessera_point == "centroid" else tr)
        output.append(result)
    return output


def _prediction_array_2d(values: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    """Preserve legitimate singleton spatial axes; remove only extra 1-sized axes.

    Accept a native (H,W) array, or unambiguous (1,H,W), (H,W,1),
    (1,1,H,W), etc. Never resize, transpose, or pick a non-singleton channel.
    """
    values = np.asarray(values)
    if values.shape == expected_shape:
        return values
    if values.ndim < 3 or values.ndim > 4:
        raise ValueError("Prediction array must match exact reference raster shape; do not resize")
    axes = [i for i, size in enumerate(values.shape) if size == 1]
    for removed in itertools.combinations(axes, values.ndim - 2):
        candidate = np.squeeze(values, axis=removed)
        if candidate.shape == expected_shape:
            return candidate
    raise ValueError("Prediction array must match exact reference raster shape; do not resize")


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
            ref = stack.enter_context(rasterio.open(reference_raster))
            values = _prediction_array_2d(values, (ref.height, ref.width))
            valid = ref.read_masks(1) > 0
            affine, crs, reference = ref.transform, ref.crs, str(reference_raster)
        else:
            raise ValueError("Only .tif, .tiff, .npy and .npz prediction files are supported")
        if crs is None or affine.is_identity:
            raise ValueError("Missing valid CRS/affine: cannot geolocate prediction")
        return np.asarray(values), valid, affine, crs, reference


def _historical_run(run_dir: Path, methods: list[str], min_pixels: int, connectivity: int,
                    weighting: str, tile, tessera_point: str = "representative",
                    tessera_year: str | None = None, ground_area_max_pixels: int = 100_000,
                    require_ground_area: bool = False) -> tuple[list[dict], dict]:
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
                        tessera_tile=tile, tessera_point=tessera_point, tessera_year=tessera_year,
                        ground_area_max_pixels=ground_area_max_pixels,
                        require_ground_area=require_ground_area,
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


def _managed_snapshot(outdir: Path, *, overwrite: bool) -> tuple | None:
    """Fail closed for symlinks, directories and user data; capture directory state.

    Called under the cooperative export lock before staging and again at commit.
    The second snapshot must match the first, including file identities and mtimes.
    """
    if outdir.is_symlink():
        raise FileExistsError(f"Refusing a symlink output directory: {outdir}")
    if not outdir.exists():
        return None
    if not outdir.is_dir():
        raise NotADirectoryError(f"Output destination is not a directory: {outdir}")
    entries = sorted(outdir.iterdir(), key=lambda p: p.name)
    unrelated = [p.name for p in entries if p.name not in MANAGED_FILES]
    if unrelated:
        raise FileExistsError(
            f"Output contains other files; refusing to replace them: {unrelated}. Choose a new --output-dir")
    unsafe = [p.name for p in entries if p.is_symlink() or not p.is_file()]
    if unsafe:
        raise FileExistsError(
            f"Output contains managed-name directories, symlinks or special files: {unsafe}; refusing to replace")
    if entries and not overwrite:
        raise FileExistsError(f"Results already exist in {outdir}; choose a new directory or --overwrite")
    parent = outdir.stat()
    state = ((parent.st_dev, parent.st_ino, parent.st_mtime_ns),
             tuple((p.name, p.stat().st_dev, p.stat().st_ino,
                    p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                   for p in entries))
    return state


@contextmanager
def _exclusive_output_lock(outdir: Path):
    """Exclusive, process-cooperative lock for one output name.

    Directory creation is atomic across Windows/POSIX. An abandoned lock fails
    closed: users must check the owning process before manually clearing it.
    """
    outdir.parent.mkdir(parents=True, exist_ok=True)
    lock = outdir.parent / f".{outdir.name}.centroid.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"Centroid output is locked: {lock}. Another export may be running. "
            "Do not remove a lock without checking the process.") from exc
    marker = lock / "owner.json"
    owner_id = uuid.uuid4().hex
    try:
        marker.write_text(json.dumps({"pid": os.getpid(), "owner_id": owner_id,
                                      "created_utc": datetime.now(timezone.utc).isoformat()}),
                          encoding="utf-8")
        yield
    finally:
        try:
            if marker.exists():
                current = json.loads(marker.read_text(encoding="utf-8"))
                if current.get("owner_id") == owner_id:
                    marker.unlink()
            lock.rmdir()  # never recursively remove an unknown directory
        except OSError as exc:
            print(f"WARNING: could not clear export lock {lock}: {exc}")


def _rename_directory_exclusive(source: Path, dest: Path) -> None:
    """Publish only if destination is absent; no replacing a late-arriving dir.

    Windows os.rename refuses existing destinations; Linux uses renameat2
    RENAME_NOREPLACE (atomic). Other platforms use best-effort checks.
    """
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError("Linux renameat2 unavailable; refusing non-atomic publish")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                              ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(source), -100, os.fsencode(dest), 1)
        if result:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), str(dest))
    else:
        # On Windows renaming a directory to an existing path raises.
        # On other POSIX platforms the second preflight plus cooperative lock
        # mitigates races, but non-cooperating writers remain a limitation.
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(f"Output appeared during publishing: {dest}")
        source.rename(dest)


def _remove_owned_directory(folder: Path) -> None:
    """Delete only our known regular artifact files, never recurse into subfolders."""
    entries = list(folder.iterdir())
    if any(e.name not in MANAGED_FILES or e.is_symlink() or not e.is_file()
           for e in entries):
        raise FileExistsError(f"Refusing unsafe cleanup of {folder}; preserve for review")
    for path in entries:
        path.unlink()
    folder.rmdir()


def _csv_safe_value(value: Any) -> Any:
    """Keep computational numbers intact; escape spreadsheet-formula text cells.

    GeoJSON and metadata retain the literal original labels. In CSV an initial
    apostrophe is a text marker used by spreadsheet applications.
    """
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    if isinstance(value, str) and value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def _csv_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _csv_safe_value(value) for key, value in record.items()}


def _verify_source_integrity(signatures: dict[str, tuple]) -> None:
    """Check source bytes and metadata again just before publication."""
    for name, (signature, digest) in signatures.items():
        source = Path(name)
        if _file_signature(source) != signature or _sha256(source) != digest:
            raise RuntimeError(f"Input changed before export: {source}")


def _write_outputs(outdir: Path, records: list[dict], metadata: dict,
                   *, overwrite: bool = False, html_text: str | None = None,
                   source_integrity: dict[str, tuple] | None = None):
    """Stage and publish a complete export, protecting other user files.

    Cooperative locking, preflight revalidation, no-clobber publish and rollback
    close the previously reproduced overwrite bugs. Abrupt power loss between
    replacing directories may still leave an inspectable hidden backup.
    """
    outdir = outdir.absolute()
    with _exclusive_output_lock(outdir):
        original = _managed_snapshot(outdir, overwrite=overwrite)
        stage = Path(tempfile.mkdtemp(prefix=f".{outdir.name}.pending-", dir=outdir.parent))
        backup = outdir.parent / f".{outdir.name}.backup-{uuid.uuid4().hex}"
        replaced_old = False
        installed_new = False
        try:
            csv_path, geo_path, meta_path = (stage / name for name in MANAGED_FILES[:3])
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(_csv_safe_record(record) for record in records)
            features = [{"type": "Feature", "geometry": {"type": "Point",
                       "coordinates": [r["centroid_lon"], r["centroid_lat"]]},
                       "properties": {k: v for k, v in r.items()
                                      if k not in {"centroid_lon", "centroid_lat"}}}
                        for r in records]
            geo_path.write_text(json.dumps({"type": "FeatureCollection", "features": features},
                                            indent=2, allow_nan=False), encoding="utf-8")
            if html_text is not None:
                (stage / "report.html").write_text(html_text, encoding="utf-8")
            integrity = {name: _sha256(stage / name) for name in
                         ("plume_centroids.csv", "plume_centroids.geojson")}
            if html_text is not None:
                integrity["report.html"] = _sha256(stage / "report.html")
            details = {
                "schema_version": SCHEMA_VERSION,
                "patch_version": PATCH_VERSION,
                "export_id": uuid.uuid4().hex,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "component_count": len(records),
                "output_files": [str(outdir / p.name) for p in stage.iterdir()] + [str(outdir / "run_metadata.json")],
                "export_sha256": integrity,
                "geographic_reference": "WGS84 EPSG:4326; GeoJSON geometry is the mathematical centroid",
                "representative_point": "nearest actual selected plume pixel; properties representative_lon/lat",
                "area_semantics": {
                    "component_area_m2": "DEPRECATED: v2-compatible projected planar area, not ellipsoidal ground area",
                    "component_projected_area_m2": "projected planar area (native pixel determinant converted to square meters)",
                    "component_ground_area_m2": "sum of WGS84 ellipsoidal pixel-footprint areas; empty if pyproj unavailable",
                },
                "tessera_data_caveat": "A valid raster cell is NOT verified TESSERA embedding provenance or temporal match; no embedding vector is extracted",
                "preview_caveat": "Offline report is a pixel-space preview, not a satellite RGB/geo-reprojected map",
                "csv_text_policy": "Apostrophe-prefix text values that spreadsheet software might interpret as formulas; GeoJSON preserves literal labels",
                "caveat": "Predicted plume component center, not a verified methane source location",
                "model_checkpoint_modified": False,
                **metadata,
            }
            meta_path.write_text(json.dumps(details, indent=2, allow_nan=False), encoding="utf-8")
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                if len(list(csv.DictReader(stream))) != len(records):
                    raise ValueError("Staged CSV record-count mismatch")
            if len(json.loads(geo_path.read_text(encoding="utf-8"))["features"]) != len(records):
                raise ValueError("Staged GeoJSON record-count mismatch")
            if json.loads(meta_path.read_text(encoding="utf-8"))["component_count"] != len(records):
                raise ValueError("Staged metadata record-count mismatch")
            # Fail if a destination or managed file appeared/changed while staging.
            final = _managed_snapshot(outdir, overwrite=overwrite)
            if final != original:
                raise FileExistsError(f"Output directory changed while preparing export: {outdir}")
            if source_integrity is not None:
                _verify_source_integrity(source_integrity)
            if original is not None:
                outdir.rename(backup)
                replaced_old = True
            _rename_directory_exclusive(stage, outdir)
            installed_new = True
        except BaseException:
            if replaced_old and not installed_new and backup.exists():
                try:
                    if outdir.exists() or outdir.is_symlink():
                        raise FileExistsError("Destination appeared while trying to restore old results")
                    _rename_directory_exclusive(backup, outdir)
                except OSError as exc:
                    print(f"WARNING: old results preserved in backup {backup}; restore manually after review: {exc}")
            raise
        finally:
            if stage.exists():
                try:
                    _remove_owned_directory(stage)
                except (OSError, ValueError) as exc:
                    print(f"WARNING: pending stage preserved for inspection: {stage} ({exc})")
        if replaced_old and backup.exists():
            try:
                _remove_owned_directory(backup)
            except OSError as exc:
                print(f"WARNING: new results published but old backup preserved: {backup} ({exc})")
        return [outdir / name for name in MANAGED_FILES[:3]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch_scene_id(path: Path) -> str:
    """Prefer original MARS UUID from folders like 02_<UUID>."""
    folder = path.parent.name
    match = re.fullmatch(
        r"\d+_([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})", folder,
    )
    return match.group(1) if match else folder


def _batch_scene_ids(paths: list[Path], input_dir: Path) -> list[str]:
    """Keep unique MARS UUIDs; disambiguate ordinary multi-TIFF folders.

    Relative-path fallback is stable for a given input directory and cannot
    silently drop a duplicate sample. It does not make non-independent samples
    scientifically independent; users should review duplicated UUIDs.
    """
    preferred = [_batch_scene_id(p) for p in paths]
    frequencies = {name: preferred.count(name) for name in set(preferred)}
    used: set[str] = set()
    identities: list[str] = []
    for index, (path, candidate) in enumerate(zip(paths, preferred)):
        if frequencies[candidate] == 1:
            name = candidate
        else:
            rel = path.relative_to(input_dir).with_suffix("")
            name = "__".join(rel.parts)
        if name in used:
            # Relative-path joins can collide with names containing '__'.
            # A stable ordinal disambiguates without silently discarding a file.
            name = f"{name}__index_{index:05d}"
        if name in used:
            raise ValueError(f"Unable to assign a unique scene ID for {path}")
        used.add(name)
        identities.append(name)
    return identities

def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    """File identity and metadata; raises for absent or replaced source files."""
    st = path.stat()
    if not path.is_file():
        raise FileNotFoundError(f"Prediction is not a regular file: {path}")
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _read_verified_prediction(path: Path, reference_raster: Path | None,
                              band: int, array_key: str | None):
    """Refuse inputs that change between hashing and raster reading."""
    sources = [path] + ([reference_raster] if reference_raster else [])
    before = {str(p): (_file_signature(p), _sha256(p)) for p in sources}
    data = _single_prediction(path, reference_raster, band, array_key)
    after = {str(p): (_file_signature(p), _sha256(p)) for p in sources}
    if before != after:
        raise RuntimeError(f"Source prediction/reference changed while being read: {path}")
    return data, before[str(path)][1], before



def _png_data_uri(rgb: np.ndarray) -> str:
    """Encode an RGB preview using only the Python standard library."""
    height, width, channels = rgb.shape
    if channels != 3 or rgb.dtype != np.uint8:
        raise ValueError("Preview must be uint8 RGB")

    def png_chunk(kind: bytes, payload: bytes) -> bytes:
        content = kind + payload
        return (struct.pack(">I", len(payload)) + content
                + struct.pack(">I", binascii.crc32(content) & 0xffffffff))

    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(height))
    png = (b"\x89PNG\r\n\x1a\n"
           + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + png_chunk(b"IDAT", zlib.compress(raw, level=6))
           + png_chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _preview_image(values: np.ndarray, valid: np.ndarray,
                   parts: list[dict], threshold: float,
                   connectivity: int, value_type: str) -> str:
    """Illustrative native-pixel probability image, NOT geographic satellite imagery."""
    score = expit(values) if value_type == "logit" else np.asarray(values, dtype=float)
    finite = valid & np.isfinite(score)
    score = np.where(finite, np.clip(score, 0, 1), 0)
    height, width = score.shape
    step = max(1, math.ceil(max(height, width) / 600))
    def pooled_max(arr: np.ndarray) -> np.ndarray:
        """Max-pool every step×step block so thin bright features survive previews."""
        if step == 1:
            return arr
        hh = math.ceil(height / step) * step
        ww = math.ceil(width / step) * step
        pad = np.pad(arr, ((0, hh - height), (0, ww - width)), mode="constant", constant_values=0)
        return pad.reshape(hh // step, step, ww // step, step).max(axis=(1, 3))

    score_small = pooled_max(score)
    valid_small = pooled_max(finite)
    # Dark blue low scores -> warm orange high scores.
    rgb = np.empty((*score_small.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (17 + 212 * score_small).astype(np.uint8)
    rgb[..., 1] = (36 + 110 * score_small).astype(np.uint8)
    rgb[..., 2] = (59 - 18 * score_small).astype(np.uint8)
    rgb[~valid_small] = (28, 32, 41)

    if parts:
        foreground = finite & (score >= threshold if value_type != "binary" else values > 0)
        structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
        labels, _ = ndimage.label(foreground, structure=structure)
        kept = np.isin(labels, [int(r["component_id"]) for r in parts]) & (labels > 0)
        edge = kept & ~ndimage.binary_erosion(kept, structure=structure)
        rgb[pooled_max(edge)] = (255, 211, 107)

    h, w = rgb.shape[:2]

    def mark(row: float, col: float, color: tuple[int, int, int], *, cross: bool) -> None:
        y = int(math.floor(float(row) / step))
        x = int(math.floor(float(col) / step))
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if (abs(dx) <= 1 or abs(dy) <= 1) if cross else (abs(dx) <= 2 and abs(dy) <= 2):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        rgb[yy, xx] = color

    for p in parts:
        mark(p["centroid_row"], p["centroid_col"], (255, 255, 255), cross=True)
        mark(p["representative_row"], p["representative_col"], (52, 234, 215), cross=False)
    return _png_data_uri(rgb)


def _html_report(cards: list[dict[str, Any]], records: list[dict[str, Any]],
                 settings: dict[str, Any]) -> str:
    """Create standalone report with embedded images; no server or extra dependencies."""
    entries = []
    for card in cards:
        items = []
        for p in card["parts"]:
            inside = "yes" if p["centroid_in_plume"] else "no"
            mean = (f"{float(p['mean_probability']):.4f}"
                    if p["mean_probability"] is not None else "—")
            items.append(
                "<tr>"
                f"<td>{int(p['component_id'])}</td>"
                f"<td>{int(p['pixel_count']):,}</td>"
                f"<td>{float(p['centroid_lat']):.9f}</td>"
                f"<td>{float(p['centroid_lon']):.9f}</td>"
                f"<td>{inside}</td>"
                f"<td>{mean}</td>"
                "</tr>"
            )
        table = ("<table><thead><tr><th>Region</th><th>Pixels</th><th>Latitude</th>"
                 "<th>Longitude</th><th>Inside?</th><th>Mean p</th></tr></thead><tbody>"
                 + ("".join(items) or '<tr><td colspan="6">No qualifying regions</td></tr>')
                 + "</tbody></table>")
        sid = html.escape(str(card["scene_id"]))
        path = html.escape(str(card["path"]))
        entries.append(
            f'<section class="scene" data-count="{len(card["parts"])}" data-name="{sid}">'
            f'<div class="scene-head"><h2>{sid}</h2><span>{len(card["parts"])} region(s)</span></div>'
            f'<p class="path">{path}</p>'
            f'<div class="scene-layout"><img src="{card["image_uri"]}" '
            f'alt="Predicted probabilities and centroid markers for {sid}" />'
            f'<div class="table-scroll">{table}</div></div></section>'
        )
    settings_label = html.escape(", ".join(f"{k}={v}" for k, v in settings.items()
                                          if k != "total_scenes"))
    shown = len(cards)
    total = int(settings.get("total_scenes", shown))
    report_scope = (f"Preview shows first {shown} of {total} processed scenes; all numerical exports include every scene."
                    if shown < total else "All processed scenes included in this report.")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model6 centroid report</title>
<style>
:root{{font-family:system-ui,Segoe UI,sans-serif;color:#e8eef6;background:#101923}}
body{{max-width:1200px;margin:auto;padding:28px 18px}}
h1{{font-size:1.7rem;margin-bottom:8px}}h2{{font-size:1rem;margin:0;overflow-wrap:anywhere}}
p{{color:#a7b8c7}}.meta{{margin-bottom:22px}}.controls{{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:18px}}
input[type=search]{{padding:10px;background:#243344;color:#fff;border:1px solid #5c7085;border-radius:7px;min-width:245px}}
.scene{{border:1px solid #405368;background:#172432;border-radius:12px;padding:16px;margin:16px 0}}
.scene-head{{display:flex;justify-content:space-between;align-items:center;gap:10px}}
.path{{font-size:.75rem;overflow-wrap:anywhere;margin:9px 0 16px}}
.scene-layout{{display:grid;grid-template-columns:minmax(180px,300px) minmax(0,1fr);gap:20px;align-items:start}}
.scene img{{width:100%;height:auto;image-rendering:pixelated;border:1px solid #536779;background:#172432}}
table{{border-collapse:collapse;font-size:.82rem;width:100%;min-width:560px}}th,td{{padding:9px;border-bottom:1px solid #34485a;text-align:left}}th{{color:#a9c6d7}}.table-scroll{{overflow-x:auto}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:.85rem}}.muted{{color:#a7b8c7}}.swatch{{display:inline-block;width:12px;height:12px;vertical-align:middle;margin-right:4px}}
@media(max-width:720px){{.scene-layout{{grid-template-columns:1fr}}.scene img{{max-width:350px}}}}
</style></head><body>
<h1>Model6 Plume Centroid Report</h1>
<p class="meta">{len(cards)} preview scene(s) · {len(records)} total connected plume region(s) · {settings_label}</p>
<p class="muted">{html.escape(report_scope)}</p>
<p class="muted">Preview uses max-pooling when downsampled, retaining thin high-score features; it is a pixel-space visualization, not satellite RGB or a mapped coastline. White cross = {html.escape(str(settings.get("weighting", "geometric")))} centroid; teal square = nearest selected plume pixel; gold = detected region boundary. The centroid is not a verified methane emission source.</p>
<div class="controls"><input id="filter" type="search" placeholder="Search scene ID" aria-label="Search scene ID">
<label><input id="detections" type="checkbox"> Only scenes with detected regions</label></div>
<div class="legend"><span><i class="swatch" style="background:#ffffff"></i>{html.escape(str(settings.get("weighting", "geometric")))} centroid</span><span><i class="swatch" style="background:#34ead7"></i>On-plume point</span><span><i class="swatch" style="background:#ffd36b"></i>Region boundary</span></div>
{''.join(entries)}
<script>
const filter=document.getElementById('filter'), detections=document.getElementById('detections');
function refresh(){{const q=filter.value.trim().toLowerCase();for(const item of document.querySelectorAll('.scene')){{item.hidden=!(item.dataset.name.toLowerCase().includes(q)&&(!detections.checked||Number(item.dataset.count)>0));}}}}
filter.addEventListener('input',refresh);detections.addEventListener('change',refresh);
</script></body></html>"""

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-id", help="Existing historical GTM Model6 plume run below outputs/GTM_Model6/")
    mode.add_argument("--run-dir", type=Path, help="Directory containing a completed historical GTM Model6 plume run")
    mode.add_argument("--input", type=Path, help="One georeferenced prediction TIFF, NPY, or NPZ")
    mode.add_argument("--input-dir", type=Path, help="Recursively process prediction GeoTIFFs from a directory")
    parser.add_argument("--pattern", default="model6_probability.tif", help="Filename glob for --input-dir (default matches our Model6 saved predictions)")
    parser.add_argument("--scene-id", help="Override scene identifier for --input mode")
    parser.add_argument("--dataset", help="Optional dataset label for standalone or batch exports")
    parser.add_argument("--checkpoint-file", type=Path, help="Record a supplied checkpoint SHA-256 in metadata only; does NOT run inference or establish lineage")
    parser.add_argument("--html-report", action="store_true", help="Create offline browser report with embedded probability previews")
    parser.add_argument("--report-limit", type=int, default=60, help="Maximum scenes visualized in offline HTML; all scenes still exported")
    parser.add_argument("--reference-raster", type=Path, help="Exact georeferenced reference raster, mandatory for NPY/NPZ")
    parser.add_argument("--band", type=int, default=1, help="1-indexed TIFF probability band")
    parser.add_argument("--array-key", help="Array name within multi-key NPZ")
    parser.add_argument("--value-type", choices=["probability", "logit", "binary"], default="probability")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold in standalone / batch modes")
    parser.add_argument("--methods", default="mean", help="Comma-separated historical GTM Model6 methods")
    parser.add_argument("--min-pixels", type=int, default=100, help="Minimum connected-region size (pilot used 10; historical default stays 100)")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=4)
    parser.add_argument("--weighting", choices=["geometric", "probability"], default="geometric")
    parser.add_argument("--tessera-raster", type=Path, help="Optional georeferenced TESSERA raster; checks cell validity, not embeddings")
    parser.add_argument("--tessera-point", choices=["representative", "centroid"], default="representative", help="Which point drives legacy tessera_row/tessera_col (both are exported)")
    parser.add_argument("--tessera-year", help="Explicit user-declared year of supplied TESSERA raster (not verified)")
    parser.add_argument("--require-ground-area", action="store_true", help="Fail if ground area cannot be computed for any emitted component")
    parser.add_argument("--ground-area-max-pixels", type=int, default=100_000,
                        help="Skip slow per-pixel geodesic area above this component size; 0 = unlimited")
    parser.add_argument("--output-dir", type=Path, help="New output directory containing CSV, GeoJSON and metadata")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing export/report files")
    args = parser.parse_args(argv)
    if args.report_limit < 1:
        parser.error("--report-limit must be >= 1")
    if args.ground_area_max_pixels < 0:
        parser.error("--ground-area-max-pixels must be >= 0")
    if args.tessera_year and re.fullmatch(r"(?:19|20|21)\d{2}", args.tessera_year) is None:
        parser.error("--tessera-year must be a four-digit year between 1900 and 2199")
    if args.require_ground_area and Geod is None:
        parser.error("Install pyproj (pip install pyproj) for --require-ground-area")
    if Geod is None:
        print("WARNING: pyproj not available; component_ground_area_m2 will be empty. "
              "The projected component_area_m2 field is not an ellipsoidal ground area.")
    if args.tessera_year and not args.tessera_raster:
        parser.error("--tessera-year requires --tessera-raster")
    if args.band < 1 or args.min_pixels < 1:
        parser.error("--band and --min-pixels must each be >=1")
    if not np.isfinite(args.threshold) or not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0,1]")
    if args.run_id and (Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}):
        parser.error("--run-id must be one directory name, not a path")
    if args.scene_id and not args.input:
        parser.error("--scene-id is available only with --input")
    if args.html_report and (args.run_id or args.run_dir):
        parser.error("--html-report currently supports --input and --input-dir")
    if args.checkpoint_file and (args.run_id or args.run_dir):
        parser.error("--checkpoint-file is available only with --input or --input-dir")
    if args.input_dir and (args.reference_raster or args.array_key):
        parser.error("--input-dir handles georeferenced TIFFs; --reference-raster and --array-key are only for --input")
    if args.scene_id and not args.scene_id.strip():
        parser.error("--scene-id cannot be blank")
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
        cards: list[dict[str, Any]] = []
        if args.run_id or args.run_dir:
            run = ((REPO_ROOT / "outputs" / "GTM_Model6" / args.run_id) if args.run_id
                   else args.run_dir).resolve()
            if out is None:
                out = REPO_ROOT / "outputs" / "plume_centroids" / (run.name + "_" + "-".join(methods))
            records, metadata = _historical_run(run, methods, args.min_pixels,
                                                args.connectivity, args.weighting, tile, args.tessera_point, args.tessera_year,
                                                args.ground_area_max_pixels, args.require_ground_area)
        else:
            if args.input_dir:
                folder = args.input_dir.resolve()
                if not folder.is_dir():
                    raise NotADirectoryError(folder)
                if (Path(args.pattern).is_absolute() or ".." in Path(args.pattern).parts
                        or "\\" in args.pattern):
                    parser.error("--pattern must be a relative filename glob")
                paths = sorted(p for p in folder.rglob(args.pattern)
                               if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
                if not paths:
                    raise FileNotFoundError(f"No georeferenced prediction TIFFs matching {args.pattern!r} below {folder}")
                if out is None:
                    out = REPO_ROOT / "outputs" / "plume_centroids" / (folder.name + "_centroids")
            else:
                paths = [args.input.resolve()]
                if out is None:
                    out = REPO_ROOT / "outputs" / "plume_centroids" / paths[0].stem
            records = []
            metadata = {
                "source": "prediction_batch" if args.input_dir else "standalone_prediction",
                "threshold": args.threshold, "value_type": args.value_type,
                "weighting": args.weighting, "min_pixels": args.min_pixels,
                "connectivity": args.connectivity, "dataset": args.dataset,
                "scene_count": len(paths), "inputs": [],
                "tessera_lookup_point": args.tessera_point,
                "tessera_raster": str(args.tessera_raster.resolve()) if args.tessera_raster else None,
                "tessera_year_user_supplied": args.tessera_year,
                "report_limit": args.report_limit if args.html_report else None,
                "ground_area_max_pixels": args.ground_area_max_pixels,
                "require_ground_area": args.require_ground_area,
            }
            if args.checkpoint_file:
                cp = args.checkpoint_file.resolve()
                if not cp.is_file():
                    raise FileNotFoundError(f"Supplied checkpoint does not exist: {cp}")
                metadata["supplied_checkpoint"] = {
                    "path": str(cp), "sha256": _sha256(cp),
                    "inference_lineage_verified": False,
                    "note": "User-supplied provenance only; this tool cannot prove which checkpoint produced a prediction",
                }
            scene_ids = (_batch_scene_ids(paths, folder) if args.input_dir else
                         [args.scene_id or paths[0].stem])
            source_signatures: dict[str, tuple] = {}
            for index, path in enumerate(paths):
                scene_id = scene_ids[index]
                (values, valid, affine, crs, reference), sha, signatures = _read_verified_prediction(
                    path, args.reference_raster, args.band, args.array_key)
                source_signatures.update(signatures)
                parts = calculate_plume_centroids(
                    values, valid, affine, crs, threshold=args.threshold,
                    min_pixels=args.min_pixels, connectivity=args.connectivity,
                    weighting=args.weighting, value_type=args.value_type,
                    tessera_tile=tile, tessera_point=args.tessera_point,
                    tessera_year=args.tessera_year,
                    ground_area_max_pixels=args.ground_area_max_pixels,
                    require_ground_area=args.require_ground_area)
                records.extend([{
                    **p, "scene_id": scene_id, "scene_index": index,
                    "dataset": args.dataset, "method": "standalone",
                    "prediction_file": str(path), "source_georeference": reference,
                    "prediction_sha256": sha,
                } for p in parts])
                metadata["inputs"].append({"scene_id": scene_id, "prediction": str(path),
                                            "prediction_sha256": sha, "components": len(parts),
                                            "source_georeference": reference})
                if args.html_report and len(cards) < args.report_limit:
                    cards.append({"scene_id": scene_id, "path": str(path), "parts": parts,
                                  "image_uri": _preview_image(values, valid, parts,
                                  args.threshold, args.connectivity, args.value_type)})
                print(f"Scene {index + 1}/{len(paths)} [{scene_id}]: {len(parts)} plume region(s)", flush=True)
            if args.html_report:
                metadata["html_report_scenes"] = len(cards)
                metadata["html_report_omitted_scenes"] = len(paths) - len(cards)
                if len(paths) > len(cards):
                    print(f"HTML preview limited to first {len(cards)} scenes out of {len(paths)}; CSV/GeoJSON include all.", flush=True)
            if not args.input_dir:
                metadata["prediction"] = str(paths[0])
                metadata["georeference"] = metadata["inputs"][0]["source_georeference"]
    # Recheck every source before publishing; earlier inputs may change while a
    # later scene is being processed. Hashes reflect the actually read bytes.
    if not (args.run_id or args.run_dir):
        _verify_source_integrity(source_signatures)
    out = out.absolute()
    html_text = (_html_report(cards, records, {"threshold": args.threshold,
                 "min_pixels": args.min_pixels, "connectivity": args.connectivity,
                 "weighting": args.weighting, "total_scenes": metadata.get("scene_count", len(cards))}) if args.html_report else None)
    targets = _write_outputs(out, records, metadata, overwrite=args.overwrite,
                             html_text=html_text,
                             source_integrity=(source_signatures if not (args.run_id or args.run_dir) else None))
    if html_text is not None:
        print(f"HTML report: {out / 'report.html'}")
    print(f"Finished: {len(records)} plume center(s). CSV: {targets[0]}")
    print(f"GeoJSON: {targets[1]}")
    if not records:
        print("No connected components passed the threshold and size filter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
