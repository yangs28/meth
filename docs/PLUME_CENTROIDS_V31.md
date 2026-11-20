# Methane Project — Plume Centroids V3.1 (corrective release)

**Release:** 2026-11-19 · **Source:** exact user-supplied V3 ZIP, `plume_centroids_v3_20261119(1).zip`.  
**Deliverable:** drop-in replacement for `EarthRemoteSensingRapidResponse/plume_centroids.py` and its regression tests.  
**Does not run Model6 inference, modify a checkpoint, install a GUI, acquire imagery, or retrieve TESSERA embeddings.**

## Why this release exists

The audit of V3 found important safety and compatibility defects. V3.1 fixes them without replacing the centroid algorithm:

| Original defect | V3.1 correction |
|---|---|
| `--overwrite` could recursively delete a directory named `report.html` or another managed filename | **Reject** managed-name directories, symlinks and special files; cleanup only known regular files with nonrecursive `unlink` / `rmdir` |
| Race: a destination created after the preflight could be overwritten without `--overwrite` | Exclusive per-output-name lock, pre- and post-staging directory state checks, and a no-clobber publish on Windows and Linux |
| A legitimate `1 × N` or `N × 1` NumPy raster was broken by `np.squeeze` | Preserve native 2D spatial axes; only discard explicitly extra singleton batch/channel dimensions |
| Multiple TIFFs under one parent caused duplicate IDs | Preserve unique MARS UUIDs; use stable relative-path IDs when names would collide |
| Spreadsheet software could interpret externally provided labels as formulas | Prefix potentially formula-like **CSV text values** with a leading apostrophe; preserve literal values in GeoJSON and HTML |
| Source files could change between raster read and SHA-256 recording | Compare source metadata and SHA-256 before/after reading, after batch processing, and just before publishing |

`schema_version: 3` and all original CSV columns are retained; metadata adds `patch_version: "3.1"`. V3.1 does not modify the original thresholding, 4/8-connectivity, geometric/probability-weighted centroids, pixel-center transform, geodesic area, or TESSERA grid-cell lookup formulas.

## Requirements

Python 3.11+ with NumPy, SciPy, Rasterio; `pyproj` for `component_ground_area_m2` (`--require-ground-area` fails rather than inventing area if unavailable). `pytest` is required only for the included test suite; Pillow is used by an optional preview test, not by the application.

## Install in your existing Windows repository — backup first

1. Download `plume_centroids_v31_20261119.zip` into Downloads.
2. Open VS Code PowerShell and run:

```powershell
Set-Location 'C:\Users\Nugge\Documents\GitHub\meth'

$Zip = Join-Path $env:USERPROFILE 'Downloads\plume_centroids_v31_20261119.zip'
$Staging = Join-Path $env:TEMP ('plume_centroids_v31_' + [guid]::NewGuid().ToString('N'))
$Script = '.\EarthRemoteSensingRapidResponse\plume_centroids.py'
$Test = '.\tests\plume_centroids\test_plume_centroids.py'

if (-not (Test-Path -LiteralPath $Zip)) { throw "ZIP not found: $Zip" }
if (-not (Test-Path -LiteralPath $Script)) { throw "Script not found: $Script" }
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
Copy-Item -LiteralPath $Script -Destination "$Script.backup_$Stamp"
if (Test-Path -LiteralPath $Test) {
    Copy-Item -LiteralPath $Test -Destination "$Test.backup_$Stamp"
}
Expand-Archive -LiteralPath $Zip -DestinationPath $Staging
Copy-Item -LiteralPath (Join-Path $Staging 'EarthRemoteSensingRapidResponse\plume_centroids.py') -Destination $Script -Force
Copy-Item -LiteralPath (Join-Path $Staging 'tests\plume_centroids\test_plume_centroids.py') -Destination $Test -Force
Write-Host 'V3.1 installed; backups preserved.' -ForegroundColor Green
```

3. Verify:

```powershell
$Python = 'E:\MethData\venvs\model6_train_cuda128\Scripts\python.exe'
& $Python -m pytest -q .\tests\plume_centroids\test_plume_centroids.py
& $Python .\EarthRemoteSensingRapidResponse\plume_centroids.py --help
```

**Development check:** 58 tests passed in Python 3.13 on Linux with optional dependencies installed. A real Windows replay remains necessary; do not assume an identical pass count if optional test dependencies are absent (some tests may be skipped).

## Verify against your existing Model6 eight-scene pilot

**Use a new output directory, not `--overwrite`, for this first replay.**

```powershell
& $Python .\EarthRemoteSensingRapidResponse\plume_centroids.py `
    --input-dir '.\outputs\plume_centroids\model6_batch_02' `
    --dataset 'MARS-S2L' `
    --threshold 0.5 `
    --min-pixels 10 `
    --html-report `
    --require-ground-area `
    --output-dir '.\outputs\plume_centroids\model6_batch_02_review_v31'

Start-Process '.\outputs\plume_centroids\model6_batch_02_review_v31\report.html'

Import-Csv '.\outputs\plume_centroids\model6_batch_02_review_v31\plume_centroids.csv' |
    Select-Object scene_id,component_id,pixel_count,centroid_lat,centroid_lon,component_ground_area_m2 |
    Format-Table -AutoSize
```

**Expected, if the original prediction files are unchanged:** 8 processed scenes, 6 components; the original sample UUIDs remain in the CSV because the eight folders have unique MARS UUIDs. Original geometrical centroid coordinates and pixel counts should match the earlier export. The WGS84 ground-area column may differ from the old native projected area.

## Other usage

Single GeoTIFF:

```powershell
& $Python .\EarthRemoteSensingRapidResponse\plume_centroids.py `
    --input '.\outputs\plume_centroids\model6_real_01\model6_probability.tif' `
    --scene-id '00116765-cbfa-4eae-a9e8-54f137bc511b' `
    --threshold 0.5 --min-pixels 10 --html-report `
    --output-dir '.\outputs\plume_centroids\model6_real_01_review_v31'
```

A folder containing several arbitrary TIFF names (these must be **prediction probability rasters**, not raw multispectral satellite TIFFs):

```powershell
& $Python .\EarthRemoteSensingRapidResponse\plume_centroids.py `
    --input-dir 'C:\PATH\TO\PREDICTION_RASTERS' --pattern '*.tif' `
    --min-pixels 10 `
    --output-dir '.\outputs\plume_centroids\ad_hoc_batch_v31'
```

NumPy arrays still require the **exact** CRS/affine reference raster using `--reference-raster`. Arrays can be `(H,W)`, `(1,H,W)`, `(H,W,1)` or `(1,1,H,W)`, provided extra axes are singleton. Multi-channel arrays are rejected rather than silently selecting the wrong band.

## Safe publication and crash recovery

* A sibling directory named `.<outputname>.centroid.lock` is created **only during publishing**. If another export targets that name, it fails rather than waiting or overwriting. The lock contains `owner.json` with the PID and creation time. Do not delete a lock until checking whether its owner is still running. A crash can leave a stale lock that requires manual inspection/removal.
* V3.1 refuses replacement if the destination or one of its existing managed files changes while a new export is staged. It refuses symlinks, directories named like managed files, and non-managed content.
* On Windows and Linux the final new-directory rename refuses to replace a newly appearing destination; on other operating systems the cooperative lock and final check apply, but a non-cooperating process could theoretically race the rename.
* On an intentional `--overwrite`, the old managed-only directory is renamed to a unique sibling `.<outputname>.backup-*`, then the complete new stage is published. Caught failures attempt rollback. **An abrupt crash/power loss between renames is not an atomic transaction**. Inspect the hidden backup and pending directories manually before rerunning or deleting anything.
* Obsolete HTML is removed when a successful new run omits `--html-report`.
* CSV formula safety adds `'` to dangerous text cells; if a downstream parser needs literal labels, use GeoJSON or remove the documented CSV text-marker intentionally.
* Input SHA-256s and file stats are rechecked before publishing. These are reproducibility safeguards, not cryptographic proof of model-checkpoint lineage or protection against another program modifying the input *after* the export is finished.

## Testing performed and limits

* **58 tests passed**: inherited 35 V3 tests (two updated for intentional new behavior), plus 23 V3.1 regression test cases including parameterized cases. Tests cover overwrite safety, symlinks, live locks, late destinations, modified old outputs, Python-NumPy 1D singleton axes, duplicate batch IDs, spreadsheet-formula escaping, and prediction modification during reading, while processing a batch, and while staging export.
* **900 randomized scenarios and 9,615 matched components** were compared directly between original V3 and V3.1. All component numerical fields matched with no changes.
* No real Windows MARS GeoTIFF or trained Model6 checkpoint was available inside the test environment. Replay the eight-scene pilot above and independently compare all six coordinates before considering the new version validated on your system.
* The HTML report is an **offline static** pixel-space visualization; there is no real-time threshold slider, satellite RGB overlay or GTM viewer integration yet.
* TESSERA raster lookup still checks only supplied raster cells and their validity. It does **not** download a tile, prove genuine embeddings, extract 128-D vectors, or verify annual temporal appropriateness.
* The geometric centroid is a segmentation centroid, **not a confirmed methane source location** or methane-mass centroid.
