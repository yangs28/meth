"""Synthetic/regression tests; never download imagery or run any ML model."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plume_centroids import calculate_plume_centroids, main, tessera_cell

AFFINE = from_origin(-122.0, 47.0, 0.0001, 0.0001)


def calc(p, **kw):
    return calculate_plume_centroids(np.asarray(p), np.ones(np.asarray(p).shape, bool),
                                    kw.pop("affine", AFFINE), kw.pop("crs", "EPSG:4326"),
                                    min_pixels=1, **kw)


def test_geometric_centroid_for_2x2_square():
    p = np.zeros((10, 10), float)
    p[3:5, 4:6] = 0.8
    points = calc(p)
    assert len(points) == 1
    point = points[0]
    assert point["centroid_row"] == 3.5
    assert point["centroid_col"] == 4.5
    assert point["pixel_count"] == 4
    assert -122.1 < point["centroid_lon"] < -121.9
    assert point["centroid_in_plume"]


def test_c_shape_centroid_outside_and_representative_inside():
    p = np.zeros((12, 12), float)
    p[2:10, 2:4] = 0.9
    p[2:4, 3:10] = 0.9
    p[8:10, 3:10] = 0.9
    pts = calc(p)
    assert len(pts) == 1
    point = pts[0]
    assert point["centroid_in_plume"] is False
    assert p[point["representative_row"], point["representative_col"]] == 0.9


def test_probability_weighting_moves_center():
    p = np.zeros((3, 4), float)
    p[1, 1] = .2
    p[1, 2] = .8
    uniform = calc(p, threshold=.1)[0]
    weighted = calc(p, threshold=.1, weighting="probability")[0]
    assert uniform["centroid_col"] == 1.5
    assert weighted["centroid_col"] == pytest.approx(1.8)


def test_four_and_eight_connected_components():
    p = np.array([[1, 0], [0, 1]], dtype=float)
    assert len(calc(p, connectivity=4)) == 2
    assert len(calc(p, connectivity=8)) == 1


def test_projected_crs_and_rotated_affine():
    affine = Affine(20, 3, 550000, -2, -20, 5200000)
    p = np.zeros((5, 5), float)
    p[2, 3] = .9
    got = calc(p, crs="EPSG:32610", affine=affine)[0]
    assert got["centroid_x"] == pytest.approx((affine * (3.5, 2.5))[0])
    assert got["centroid_y"] == pytest.approx((affine * (3.5, 2.5))[1])
    assert -125 < got["centroid_lon"] < -115
    assert 40 < got["centroid_lat"] < 55
    assert got["pixel_area_native_sq"] == pytest.approx(394)


def test_empty_and_invalid_masks():
    p = np.zeros((5, 5))
    assert calc(p) == []
    with pytest.raises(ValueError, match="matching 2D"):
        calculate_plume_centroids(p, np.ones((4, 5), bool), AFFINE, "EPSG:4326")
    p[0, 0] = 2.5
    with pytest.raises(ValueError, match="outside"):
        calc(p)
    assert calc(p, value_type="binary")[0]["pixel_count"] == 1


def test_logit_input():
    raw_logits = np.full((4, 4), -10.0)
    raw_logits[1:3, 1:3] = 5.0
    assert len(calc(raw_logits, value_type="logit")) == 1


def test_tessera_indices_inside_outside(tmp_path):
    path = tmp_path / "tile.tif"
    with rasterio.open(path, "w", driver="GTiff", width=10, height=10, count=1,
                       dtype="float32", crs="EPSG:4326", transform=AFFINE) as dst:
        dst.write(np.ones((1, 10, 10), dtype="float32"))
    with rasterio.open(path) as tile:
        one = tessera_cell(tile, -121.99995, 46.99995)
        assert one["tessera_status"] == "within_tile_bounds_unverified_data"
        assert one["tessera_row"] == 0 and one["tessera_col"] == 0
        assert tessera_cell(tile, -100, 50)["tessera_status"] == "outside_tile"


def test_read_standalone_geotiff_end_to_end(tmp_path):
    tif = tmp_path / "prediction.tif"
    mask = np.zeros((12, 12), dtype="float32")
    mask[2:6, 3:7] = .9
    with rasterio.open(tif, "w", driver="GTiff", width=12, height=12, count=1,
                       dtype="float32", crs="EPSG:4326", transform=AFFINE) as dst:
        dst.write(mask, 1)
    out = tmp_path / "results"
    assert main(["--input", str(tif), "--output-dir", str(out), "--min-pixels", "1"]) == 0
    with (out / "plume_centroids.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["scene_id"] == "prediction"
    geo = json.loads((out / "plume_centroids.geojson").read_text())
    assert len(geo["features"]) == 1
    assert geo["features"][0]["geometry"]["coordinates"] == [
        float(rows[0]["centroid_lon"]), float(rows[0]["centroid_lat"])]
    with pytest.raises(FileExistsError):
        main(["--input", str(tif), "--output-dir", str(out), "--min-pixels", "1"])


def test_npy_requires_matching_reference_raster(tmp_path):
    arr = tmp_path / "pred.npy"
    np.save(arr, np.ones((5, 5), np.float32))
    with pytest.raises(ValueError, match="reference-raster"):
        main(["--input", str(arr), "--output-dir", str(tmp_path / "no_ref")])
    ref = tmp_path / "reference.tif"
    with rasterio.open(ref, "w", driver="GTiff", width=6, height=6, count=1,
                       dtype="float32", crs="EPSG:4326", transform=AFFINE) as dst:
        dst.write(np.zeros((6, 6), dtype="float32"), 1)
    with pytest.raises(ValueError, match="match exact reference raster shape"):
        main(["--input", str(arr), "--reference-raster", str(ref),
              "--output-dir", str(tmp_path / "bad_shape")])


def test_run_from_saved_historical_model6_outputs(tmp_path):
    root = tmp_path / "historical_run"
    root.mkdir()
    mask = np.zeros((12, 12), dtype="float32")
    mask[2:10, 2:4] = .9
    mask[2:4, 3:10] = .9
    mask[8:10, 3:10] = .9
    (root / "GTM_Model6_plume_manifest.json").write_text(json.dumps({
        "observations": [{"id": "mars_scene_001", "dataset": "MARS", "source": "source.tif",
                          "shape": [12, 12], "crs": "EPSG:4326",
                          "affine": [0.0001, 0.0, -122, 0.0, -0.0001, 47]}]}))
    (root / "GTM_Model6_plume_results.json").write_text(json.dumps({
        "complete": True, "records": [{"index": 0, "fold": 0, "group": "test",
                                       "thresholds": {"mean": 0.5}}]}))
    np.savez_compressed(root / "GTM_Model6_plume_inputs.npz", **{
        "0_valid": np.ones((12, 12), dtype=bool)})
    np.savez_compressed(root / "GTM_Model6_plume_prediction_0.npz", mean=mask)
    out = tmp_path / "centers"
    assert main(["--run-dir", str(root), "--min-pixels", "1", "--output-dir", str(out)]) == 0
    with (out / "plume_centroids.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["scene_id"] == "mars_scene_001"
    assert rows[0]["method"] == "mean"
    assert rows[0]["centroid_in_plume"] == "False"
    assert mask[int(rows[0]["representative_row"]), int(rows[0]["representative_col"])] > .5
    assert json.loads((out / "run_metadata.json").read_text())["model_checkpoint_modified"] is False


def test_completed_run_guard(tmp_path):
    root = tmp_path / "incomplete"
    root.mkdir()
    (root / "GTM_Model6_plume_manifest.json").write_text('{"observations": []}')
    (root / "GTM_Model6_plume_results.json").write_text('{"complete": false}')
    with pytest.raises(ValueError, match="not marked complete"):
        main(["--run-dir", str(root), "--output-dir", str(tmp_path / "out")])


def test_run_id_resolves_repository_layout(tmp_path, monkeypatch):
    import plume_centroids as mod
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    folder = tmp_path / "outputs" / "GTM_Model6" / "finished_test"
    folder.mkdir(parents=True)
    (folder / "GTM_Model6_plume_manifest.json").write_text(json.dumps({
        "observations": [{"id": "scene_x", "dataset": "MARS", "crs": "EPSG:4326",
                          "affine": [0.0001, 0, -122, 0, -0.0001, 47]}]}))
    (folder / "GTM_Model6_plume_results.json").write_text(json.dumps({
        "complete": True, "records": [{"index": 0, "thresholds": {"mean": .5}}]}))
    np.savez_compressed(folder / "GTM_Model6_plume_inputs.npz", **{
        "0_valid": np.ones((12, 12), dtype=bool)})
    p = np.zeros((12, 12), np.float32)
    p[1:10, 1:10] = .9
    np.savez_compressed(folder / "GTM_Model6_plume_prediction_0.npz", mean=p)
    assert main(["--run-id", "finished_test", "--min-pixels", "10"]) == 0
    out = tmp_path / "outputs" / "plume_centroids" / "finished_test_mean"
    assert (out / "plume_centroids.csv").is_file()
    with (out / "plume_centroids.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 1
