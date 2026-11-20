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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EarthRemoteSensingRapidResponse"))
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
        assert one["tessera_status"] == "valid_raster_cell"
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


def test_batch_mode_and_html_report(tmp_path):
    import base64
    import re
    import struct

    input_dir = tmp_path / 'model6_batch_02'
    ids = [
        '00116765-cbfa-4eae-a9e8-54f137bc511b',
        '001573ce-b6d0-4c44-99eb-c7083d0286d0',
    ]
    before = {}
    for index, (id_, positive) in enumerate(zip(ids, [True, False]), start=1):
        folder = input_dir / f'{index:02d}_{id_}'
        folder.mkdir(parents=True)
        path = folder / 'model6_probability.tif'
        p = np.zeros((32, 32), dtype=np.float32)
        if positive:
            p[5:21, 6:23] = .93
            p[25:28, 25:28] = .51
        with rasterio.open(path, 'w', driver='GTiff', height=32, width=32,
                           count=1, dtype='float32', transform=from_origin(777890,4273530,10,10),
                           crs='EPSG:32639') as dst:
            dst.write(p,1)
        before[str(path)] = path.read_bytes()

    out = tmp_path / 'new_export'
    assert main(['--input-dir',str(input_dir),'--min-pixels','10',
                 '--output-dir',str(out),'--dataset','MARS-S2L','--html-report']) == 0
    with (out/'plume_centroids.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1  # 9 pixel weak secondary blob excluded
    assert rows[0]['scene_id'] == ids[0]
    assert rows[0]['dataset'] == 'MARS-S2L'
    assert int(rows[0]['pixel_count']) == 16*17
    assert float(rows[0]['component_area_m2']) == 16*17*100
    assert len(rows[0]['prediction_sha256']) == 64
    meta = json.loads((out/'run_metadata.json').read_text())
    assert meta['scene_count'] == 2
    assert [x['components'] for x in meta['inputs']] == [1, 0]
    html_result = (out/'report.html').read_text(encoding='utf-8')
    assert ids[0] in html_result and ids[1] in html_result
    assert 'Only scenes with detected regions' in html_result
    image_data = re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',html_result)
    assert len(image_data) == 2
    for data in image_data:
        png = base64.b64decode(data)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", png[16:24])
        assert (width, height) == (32, 32)
    for p, value in before.items():
        assert Path(p).read_bytes() == value
    with pytest.raises(FileExistsError):
        main(['--input-dir',str(input_dir),'--min-pixels','10',
              '--output-dir',str(out),'--html-report'])


def test_standalone_scene_id_and_html_report(tmp_path):
    tif=tmp_path/'prediction.tif'
    a=np.zeros((16,16),dtype=np.float32)
    a[5:9,5:9]=.6
    with rasterio.open(tif,'w',driver='GTiff',height=16,width=16,
                       count=1,dtype='float32',crs='EPSG:4326',transform=AFFINE) as dst:
        dst.write(a,1)
    out=tmp_path/'exports'
    assert main(['--input',str(tif),'--scene-id','example_scene',
                 '--min-pixels','10','--html-report','--output-dir',str(out)]) == 0
    with (out/'plume_centroids.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==1 and rows[0]['scene_id']=='example_scene'
    assert rows[0]['component_area_m2']==''  # geographic CRS, no fabricated m²
    assert 'example_scene' in (out/'report.html').read_text()


def test_batch_duplicate_scene_id_guard(tmp_path):
    input_dir=tmp_path/'inputs'
    id_='00116765-cbfa-4eae-a9e8-54f137bc511b'
    for prefix in ['01','02']:
        folder=input_dir/f'{prefix}_{id_}'
        folder.mkdir(parents=True)
        with rasterio.open(folder/'model6_probability.tif','w',driver='GTiff',
                           width=5,height=5,count=1,dtype='float32',
                           transform=AFFINE,crs='EPSG:4326') as dst:
            dst.write(np.ones((5,5),dtype=np.float32),1)
    out=tmp_path/'output'
    assert main(['--input-dir',str(input_dir),'--min-pixels','1',
                 '--output-dir',str(out)]) == 0
    with (out / 'plume_centroids.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert len({r['scene_id'] for r in rows}) == 2
    assert all(id_ in r['scene_id'] for r in rows)


def test_binary_report_and_empty_case(tmp_path):
    tif=tmp_path/'binary.tif'
    arr=np.zeros((12,12),dtype=np.uint8)
    arr[2:5,3:6]=1
    with rasterio.open(tif,'w',driver='GTiff',width=12,height=12,count=1,
                       dtype='uint8',transform=AFFINE,crs='EPSG:4326') as dst:
        dst.write(arr,1)
    assert main(['--input',str(tif),'--value-type','binary','--min-pixels','1',
                 '--output-dir',str(tmp_path/'out'),'--html-report']) == 0
    assert 'No qualifying regions' not in (tmp_path/'out'/'report.html').read_text()
    assert main(['--input',str(tif),'--value-type','binary','--min-pixels','20',
                 '--output-dir',str(tmp_path/'empty'),'--html-report']) == 0
    assert 'No qualifying regions' in (tmp_path/'empty'/'report.html').read_text()


def test_batch_glob_no_predictions(tmp_path):
    folder=tmp_path/'empty'
    folder.mkdir()
    with pytest.raises(FileNotFoundError,match='No georeferenced prediction TIFFs'):
        main(['--input-dir',str(folder),'--output-dir',str(tmp_path/'out')])


def test_optional_checkpoint_provenance_does_not_load_weights(tmp_path):
    import hashlib
    tif = tmp_path / 'mask.tif'
    checkpoint = tmp_path / 'synthetic.pt'
    checkpoint.write_bytes(b'not a real checkpoint; metadata only')
    with rasterio.open(tif, 'w', driver='GTiff', width=10, height=10,
                       count=1, dtype='float32', transform=AFFINE, crs='EPSG:4326') as dst:
        a = np.zeros((10,10), dtype='float32')
        a[3:7,3:7] = 0.8
        dst.write(a,1)
    out = tmp_path / 'out'
    assert main(['--input', str(tif), '--min-pixels', '1',
                 '--checkpoint-file', str(checkpoint), '--output-dir', str(out)]) == 0
    metadata = json.loads((out / 'run_metadata.json').read_text())
    recorded = metadata['supplied_checkpoint']
    assert recorded['sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert recorded['inference_lineage_verified'] is False

# ---- v3 corrections and adversarial regression tests ---------------------


def _write_tif(path: Path, arr: np.ndarray, *, crs="EPSG:4326", affine=AFFINE,
               nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1],
                       height=arr.shape[0], count=1, dtype="float32", crs=crs,
                       transform=affine, nodata=nodata) as ds:
        ds.write(arr.astype("float32"), 1)
    return path


def test_v3_overwrite_removes_stale_html(tmp_path):
    path = _write_tif(tmp_path / "input.tif", np.ones((12, 12), dtype="float32"))
    out = tmp_path / "centroids"
    assert main(["--input", str(path), "--min-pixels", "10",
                 "--output-dir", str(out), "--html-report"]) == 0
    assert "1 total connected plume region(s)" in (out / "report.html").read_text()
    assert main(["--input", str(path), "--min-pixels", "150",
                 "--output-dir", str(out), "--overwrite"]) == 0
    assert not (out / "report.html").exists()
    assert (out / "run_metadata.json").is_file()
    with (out / "plume_centroids.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == []
    assert json.loads((out / "run_metadata.json").read_text())["component_count"] == 0


def test_v3_overwrite_refuses_to_delete_user_files(tmp_path):
    path = _write_tif(tmp_path / "input.tif", np.ones((10, 10), dtype="float32"))
    out = tmp_path / "out"
    out.mkdir()
    (out / "important_notes.txt").write_text("DO NOT DELETE")
    with pytest.raises(FileExistsError, match="other files"):
        main(["--input", str(path), "--min-pixels", "1",
              "--output-dir", str(out), "--overwrite"])
    assert (out / "important_notes.txt").read_text() == "DO NOT DELETE"
    assert not (out / "plume_centroids.csv").exists()


def test_v3_overwrite_rollback_on_stage_publish_failure(tmp_path, monkeypatch):
    import plume_centroids as mod
    path = _write_tif(tmp_path / "input.tif", np.ones((8, 8), dtype="float32"))
    out = tmp_path / "out"
    main(["--input", str(path), "--min-pixels", "1", "--output-dir", str(out)])
    original_bytes = {name: (out / name).read_bytes() for name in mod.MANAGED_FILES[:3]}
    old_publish = mod._rename_directory_exclusive

    def rename_but_fail_pending(self, target):
        if ".pending-" in self.name:
            raise OSError("simulated disk/publish error")
        return old_publish(self, target)

    monkeypatch.setattr(mod, "_rename_directory_exclusive", rename_but_fail_pending)
    with pytest.raises(OSError, match="simulated"):
        main(["--input", str(path), "--min-pixels", "1000",
              "--output-dir", str(out), "--overwrite"])
    assert all((out / name).read_bytes() == content
               for name, content in original_bytes.items())


def test_v3_export_hashes_all_published_artifacts(tmp_path):
    import hashlib
    path = _write_tif(tmp_path / "input.tif", np.ones((9, 9), dtype="float32"))
    out = tmp_path / "out"
    main(["--input", str(path), "--min-pixels", "2",
          "--output-dir", str(out), "--html-report"])
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["schema_version"] == 3
    assert len(meta["export_id"]) == 32
    for name, expected in meta["export_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    assert set(meta["export_sha256"]) == {
        "plume_centroids.csv", "plume_centroids.geojson", "report.html"
    }


def test_v3_mercator_area_geodesic_corrected():
    pytest.importorskip("pyproj")
    from rasterio.warp import transform as tx
    from pyproj import Geod
    import plume_centroids as mod
    x, y = tx("EPSG:4326", "EPSG:3857", [0.0], [60.0])
    aff = from_origin(x[0], y[0], 10.0, 10.0)
    result = mod.calculate_plume_centroids(
        np.ones((1, 1)), np.ones((1, 1), bool), aff, "EPSG:3857", min_pixels=1)[0]
    assert result["component_area_m2"] == pytest.approx(100.0)
    assert result["component_projected_area_m2"] == pytest.approx(100.0)
    assert result["component_ground_area_m2"] == pytest.approx(25.0, abs=1.0)
    assert result["ground_area_status"] == "computed_wgs84_geodesic"
    corners = [aff * p for p in [(0, 0), (1, 0), (1, 1), (0, 1)]]
    lon, lat = tx("EPSG:3857", "EPSG:4326", [p[0] for p in corners],
                  [p[1] for p in corners])
    expected, _ = Geod(ellps="WGS84").polygon_area_perimeter(lon, lat)
    assert result["component_ground_area_m2"] == pytest.approx(abs(expected), abs=1e-4)


def test_v3_geographic_crs_has_ground_area_but_no_projected_area():
    pytest.importorskip("pyproj")
    p = np.ones((2, 2), dtype="float32")
    item = calc(p)[0]
    assert item["component_area_m2"] is None
    assert item["component_projected_area_m2"] is None
    assert 100 < item["component_ground_area_m2"] < 1000


def test_v3_missing_pyproj_is_explicit(tmp_path, monkeypatch):
    import plume_centroids as mod
    monkeypatch.setattr(mod, "Geod", None)
    p = np.ones((2, 2), dtype="float32")
    item = calc(p)[0]
    assert item["component_ground_area_m2"] is None
    assert item["ground_area_status"] == "unavailable_pyproj_not_installed"
    tif = _write_tif(tmp_path / "input.tif", p)
    with pytest.raises(SystemExit, match="2"):
        main(["--input", str(tif), "--require-ground-area"])


def test_v3_tessera_multiband_valid_and_nodata(tmp_path):
    import plume_centroids as mod
    arr = np.ones((3, 10, 10), dtype="float32")
    arr[1, 5, 3] = -9999
    arr[2, 4, 4] = np.nan
    tile_path = tmp_path / "embeddings.tif"
    with rasterio.open(tile_path, "w", driver="GTiff", width=10, height=10,
                       count=3, dtype="float32", nodata=-9999, crs="EPSG:4326",
                       transform=AFFINE) as ds:
        ds.write(arr)
    def point(row, col):
        return mod.geographic_point(AFFINE, rasterio.crs.CRS.from_epsg(4326), row, col)[2:]
    with rasterio.open(tile_path) as tile:
        lon, lat = point(2, 2)
        valid = mod.tessera_cell(tile, lon, lat)
        assert valid["tessera_status"] == "valid_raster_cell"
        assert valid["tessera_valid_band_count"] == 3
        lon, lat = point(5, 3)
        invalid = mod.tessera_cell(tile, lon, lat)
        assert invalid["tessera_status"] == "missing_or_nonfinite_raster_data"
        assert invalid["tessera_valid_band_count"] == 2
        lon, lat = point(4, 4)
        nan = mod.tessera_cell(tile, lon, lat)
        assert nan["tessera_status"] == "missing_or_nonfinite_raster_data"
        assert nan["tessera_valid_band_count"] == 2


def test_v3_tessera_centroid_and_onplume_may_differ(tmp_path):
    import plume_centroids as mod
    p = np.zeros((12, 12), dtype="float32")
    p[2:10, 2:4] = 0.9
    p[2:4, 3:10] = 0.9
    p[8:10, 3:10] = 0.9
    tile_path = tmp_path / "tile.tif"
    with rasterio.open(tile_path, "w", driver="GTiff", width=12, height=12,
                       count=2, dtype="float32", crs="EPSG:4326", transform=AFFINE) as dst:
        dst.write(np.ones((2, 12, 12), dtype="float32"))
    with rasterio.open(tile_path) as tile:
        z = calc(p, tessera_tile=tile, tessera_year="2024")[0]
        assert z["centroid_in_plume"] is False
        assert (z["tessera_centroid_row"], z["tessera_centroid_col"]) != (
            z["tessera_representative_row"], z["tessera_representative_col"])
        assert (z["tessera_row"], z["tessera_col"]) == (
            z["tessera_representative_row"], z["tessera_representative_col"])
        assert z["tessera_year"] == "2024"
        z2 = calc(p, tessera_tile=tile, tessera_point="centroid")[0]
        assert (z2["tessera_row"], z2["tessera_col"]) == (
            z2["tessera_centroid_row"], z2["tessera_centroid_col"])


def test_v3_thin_feature_visible_after_downsample(tmp_path):
    import io, base64, re
    Image = pytest.importorskip("PIL.Image")
    p = np.zeros((1200, 1200), dtype="float32")
    p[101:151, 101] = 0.9
    path = _write_tif(tmp_path / "large.tif", p)
    out = tmp_path / "out"
    main(["--input", str(path), "--min-pixels", "10",
          "--output-dir", str(out), "--html-report"])
    raw = (out / "report.html").read_text()
    image = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', raw)
    assert image is not None
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(image.group(1)))))
    assert rgb.shape == (600, 600, 3)
    # This score falls on an ODD source column, unseen by v2's 0::2 sampling.
    assert tuple(rgb[51, 50]) != (17, 36, 59)


def test_v3_weighted_html_legend(tmp_path):
    path = _write_tif(tmp_path / "input.tif", np.ones((10, 10), dtype="float32"))
    out = tmp_path / "out"
    main(["--input", str(path), "--min-pixels", "10",
          "--weighting", "probability", "--html-report", "--output-dir", str(out)])
    html = (out / "report.html").read_text()
    assert "probability centroid" in html
    assert "Geometric centroid" not in html


def test_v3_report_limit_explicitly_reported(tmp_path):
    import uuid
    inp = tmp_path / "batch"
    for i in range(3):
        _write_tif(inp / f"{i:02d}_{uuid.uuid4()}" / "model6_probability.tif",
                   np.ones((10, 10), dtype="float32"))
    out = tmp_path / "out"
    main(["--input-dir", str(inp), "--min-pixels", "1",
          "--report-limit", "2", "--html-report", "--output-dir", str(out)])
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["scene_count"] == 3
    assert meta["component_count"] == 3
    assert meta["html_report_scenes"] == 2
    assert meta["html_report_omitted_scenes"] == 1
    assert "Preview shows first 2 of 3 processed scenes" in (out / "report.html").read_text()


def test_v3_untrusted_scene_id_html_escaped(tmp_path):
    path = _write_tif(tmp_path / "input.tif", np.ones((10, 10), dtype="float32"))
    out = tmp_path / "out"
    nasty = '" onmouseover="alert(1)'
    main(["--input", str(path), "--scene-id", nasty, "--min-pixels", "1",
          "--html-report", "--output-dir", str(out)])
    html = (out / "report.html").read_text()
    assert 'data-name="&quot; onmouseover=&quot;alert(1)"' in html
    assert 'data-name="" onmouseover=' not in html


def test_v3_output_symlink_refused(tmp_path):
    path = _write_tif(tmp_path / "input.tif", np.ones((10, 10), dtype="float32"))
    out = tmp_path / "out"
    link = tmp_path / "link"
    out.mkdir()
    try:
        link.symlink_to(out, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks require elevated permission on this platform")
    with pytest.raises(FileExistsError, match="symlink"):
        main(["--input", str(path), "--min-pixels", "1",
              "--output-dir", str(link), "--overwrite"])


def test_v3_geodesic_work_limit_is_explicit(tmp_path):
    p = np.ones((12, 12), dtype="float32")
    item = calc(p, ground_area_max_pixels=100)[0]
    assert item["component_ground_area_m2"] is None
    assert item["ground_area_status"] == "skipped_pixel_limit"
    tif = _write_tif(tmp_path / "input.tif", p)
    with pytest.raises(RuntimeError, match="increase --ground-area-max-pixels"):
        main(["--input", str(tif), "--min-pixels", "1",
              "--ground-area-max-pixels", "100", "--require-ground-area",
              "--output-dir", str(tmp_path / "failed")])
    assert not (tmp_path / "failed").exists()


def test_v3_rejects_invalid_tessera_year(tmp_path):
    path = _write_tif(tmp_path / "input.tif", np.ones((10, 10), dtype="float32"))
    with pytest.raises(SystemExit, match="2"):
        main(["--input", str(path), "--tessera-year", "2020<script>"])

# V3.1 regression tests for independently reproduced V3 defects.

def test_v31_refuses_managed_name_directory_and_preserves_contents(tmp_path):
    import plume_centroids as mod
    out = tmp_path / "out"
    nested = out / "report.html"
    nested.mkdir(parents=True)
    (nested / "important.txt").write_text("user data")
    with pytest.raises(FileExistsError, match="managed-name directories"):
        mod._write_outputs(out, [], {}, overwrite=True)
    assert (nested / "important.txt").read_text() == "user data"
    assert (out / "plume_centroids.csv").exists() is False
    assert not (tmp_path / ".out.centroid.lock").exists()


def test_v31_refuses_managed_name_symlink(tmp_path):
    import plume_centroids as mod
    out = tmp_path / "out"
    out.mkdir()
    target = tmp_path / "my_work.txt"
    target.write_text("protected")
    link = out / "report.html"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation denied by this system")
    with pytest.raises(FileExistsError, match="symlinks"):
        mod._write_outputs(out, [], {}, overwrite=True)
    assert target.read_text() == "protected"
    assert link.is_symlink()


def test_v31_export_lock_is_exclusive_and_not_removed_if_unowned(tmp_path):
    import plume_centroids as mod
    out = tmp_path / "out"
    lock = tmp_path / ".out.centroid.lock"
    lock.mkdir()
    (lock / "owner.json").write_text('{"pid":999999, "owner_id":"external"}')
    with pytest.raises(FileExistsError, match="locked"):
        mod._write_outputs(out, [], {})
    assert lock.is_dir()
    assert "external" in (lock / "owner.json").read_text()
    assert not out.exists()


def test_v31_export_lock_cleared_after_success(tmp_path):
    import plume_centroids as mod
    out = tmp_path / "out"
    mod._write_outputs(out, [], {})
    assert (out / "run_metadata.json").is_file()
    assert not (tmp_path / ".out.centroid.lock").exists()


@pytest.mark.parametrize("allow_overwrite", [False, True])
def test_v31_refuses_late_created_output_directory(tmp_path, monkeypatch, allow_overwrite):
    import plume_centroids as mod
    out = tmp_path / "out"
    original = mod._sha256
    changed = [False]

    def intrude_during_stage(path):
        if not changed[0] and ".pending-" in str(path):
            changed[0] = True
            out.mkdir()
            (out / "report.html").write_text("other process's report")
        return original(path)

    monkeypatch.setattr(mod, "_sha256", intrude_during_stage)
    with pytest.raises(FileExistsError):
        mod._write_outputs(out, [], {}, overwrite=allow_overwrite)
    assert (out / "report.html").read_text() == "other process's report"
    assert not (out / "run_metadata.json").exists()
    assert not (tmp_path / ".out.centroid.lock").exists()


def test_v31_refuses_to_overwrite_old_results_mutated_while_staging(tmp_path, monkeypatch):
    import plume_centroids as mod
    out = tmp_path / "out"
    mod._write_outputs(out, [], {}, overwrite=False)
    original = mod._sha256
    changed = [False]

    def edit_during_stage(path):
        if not changed[0] and ".pending-" in str(path):
            changed[0] = True
            (out / "plume_centroids.csv").write_text("a user's concurrent change")
        return original(path)

    monkeypatch.setattr(mod, "_sha256", edit_during_stage)
    with pytest.raises(FileExistsError, match="changed"):
        mod._write_outputs(out, [], {}, overwrite=True)
    assert (out / "plume_centroids.csv").read_text() == "a user's concurrent change"
    assert not list(tmp_path.glob(".out.backup-*"))


def test_v31_exclusive_directory_rename_refuses_a_late_destination(tmp_path):
    import plume_centroids as mod
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "plume_centroids.csv").write_text("ours")
    out = tmp_path / "out"
    out.mkdir()
    (out / "report.html").write_text("theirs")
    with pytest.raises((FileExistsError, OSError)):
        mod._rename_directory_exclusive(stage, out)
    assert stage.is_dir()
    assert (out / "report.html").read_text() == "theirs"


@pytest.mark.parametrize("shape", [(1, 3), (3, 1)])
def test_v31_singleton_spatial_axis_npy_and_npz(tmp_path, shape):
    values = np.ones(shape, dtype=np.float32) * .9
    ref = _write_tif(tmp_path / "ref.tif", np.ones(shape, np.float32))
    for kind in ("npy", "npz"):
        path = tmp_path / f"pred.{kind}"
        if kind == "npy":
            np.save(path, values)
        else:
            np.savez(path, mask=values[None, None, ...])
        out = tmp_path / f"output_{shape}_{kind}"
        args = ["--input", str(path), "--reference-raster", str(ref),
                "--min-pixels", "1", "--output-dir", str(out)]
        assert main(args) == 0
        with (out / "plume_centroids.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1 and int(rows[0]["pixel_count"]) == 3


def test_v31_rejects_ambiguous_non_singleton_channels(tmp_path):
    ref = _write_tif(tmp_path / "ref.tif", np.ones((1, 3), np.float32))
    pred = tmp_path / "pred.npy"
    np.save(pred, np.ones((2, 1, 3), np.float32))
    with pytest.raises(ValueError, match="match exact reference raster shape"):
        main(["--input", str(pred), "--reference-raster", str(ref),
              "--output-dir", str(tmp_path / "out")])


def test_v31_flat_batch_multiple_tiffs_unique_ids(tmp_path):
    folder = tmp_path / "inputs"
    folder.mkdir()
    for stem in ["scene_A", "scene_B"]:
        _write_tif(folder / (stem + ".tif"), np.ones((5, 5), np.float32))
    out = tmp_path / "out"
    main(["--input-dir", str(folder), "--pattern", "*.tif", "--min-pixels", "1",
          "--output-dir", str(out)])
    with (out / "plume_centroids.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {x["scene_id"] for x in rows} == {"scene_A", "scene_B"}


@pytest.mark.parametrize("value", ["=1+1", "  =SUM(1,2)", "-4+2", "+2", "@SUM(A1)", "\t=3"])
def test_v31_csv_text_formula_escaped_but_geojson_literal(tmp_path, value):
    path = _write_tif(tmp_path / "prediction.tif", np.ones((3, 3), np.float32))
    out = tmp_path / "out"
    main(["--input", str(path), f"--scene-id={value}",
          "--min-pixels", "1", "--output-dir", str(out)])
    with (out / "plume_centroids.csv").open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["scene_id"] == "'" + value
    geometry = json.loads((out / "plume_centroids.geojson").read_text())
    assert geometry["features"][0]["properties"]["scene_id"] == value


def test_v31_stable_normal_scene_id_unchanged(tmp_path):
    path = _write_tif(tmp_path / "prediction.tif", np.ones((3, 3), np.float32))
    out = tmp_path / "out"
    main(["--input", str(path), "--scene-id", "MARS-2026-01",
          "--min-pixels", "1", "--output-dir", str(out)])
    with (out / "plume_centroids.csv").open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["scene_id"] == "MARS-2026-01"


def test_v31_refuses_source_changed_during_raster_read(tmp_path, monkeypatch):
    import plume_centroids as mod
    path = _write_tif(tmp_path / "prediction.tif", np.ones((3, 3), np.float32))
    original_read = mod._single_prediction
    def mutate(path_, *args):
        result = original_read(path_, *args)
        with rasterio.open(path_, "r+") as f:
            f.write(np.zeros((3, 3), dtype="float32"), 1)
        return result
    monkeypatch.setattr(mod, "_single_prediction", mutate)
    with pytest.raises(RuntimeError, match="changed while being read"):
        main(["--input", str(path), "--min-pixels", "1",
              "--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_v31_refuses_earlier_source_changed_during_batch(tmp_path, monkeypatch):
    import plume_centroids as mod
    folder = tmp_path / "input"
    first = _write_tif(folder / "01" / "model6_probability.tif", np.ones((3, 3), np.float32))
    _write_tif(folder / "02" / "model6_probability.tif", np.ones((3, 3), np.float32))
    old_preview = mod._preview_image
    counter = [0]
    def edit_after_first_scene(*args):
        result = old_preview(*args)
        counter[0] += 1
        if counter[0] == 2:
            with rasterio.open(first, "r+") as f:
                f.write(np.zeros((3, 3), dtype="float32"), 1)
        return result
    monkeypatch.setattr(mod, "_preview_image", edit_after_first_scene)
    with pytest.raises(RuntimeError, match="Input changed before export"):
        main(["--input-dir", str(folder), "--min-pixels", "1", "--html-report",
              "--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_v31_refuses_source_changed_during_export_staging(tmp_path, monkeypatch):
    import plume_centroids as mod
    source = _write_tif(tmp_path / "prediction.tif", np.ones((4, 4), np.float32))
    output = tmp_path / "out"
    original_sha = mod._sha256
    changed = [False]

    def mutate_while_writing_results(path):
        if not changed[0] and ".pending-" in str(path):
            changed[0] = True
            with rasterio.open(source, "r+") as image:
                image.write(np.zeros((4, 4), np.float32), 1)
        return original_sha(path)

    monkeypatch.setattr(mod, "_sha256", mutate_while_writing_results)
    with pytest.raises(RuntimeError, match="Input changed before export"):
        main(["--input", str(source), "--min-pixels", "1",
              "--output-dir", str(output)])
    assert not output.exists()
    assert not (tmp_path / ".out.centroid.lock").exists()


def test_v31_input_source_hash_and_export_integrity(tmp_path):
    import hashlib
    source = _write_tif(tmp_path / "prediction.tif", np.ones((5, 5), np.float32))
    output = tmp_path / "out"
    main(["--input", str(source), "--min-pixels", "1",
          "--output-dir", str(output)])
    meta = json.loads((output / "run_metadata.json").read_text())
    assert meta["schema_version"] == 3
    assert meta["patch_version"] == "3.1"
    assert meta["inputs"][0]["prediction_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    for name, checksum in meta["export_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum
