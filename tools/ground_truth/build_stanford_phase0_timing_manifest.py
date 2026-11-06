import csv
import hashlib
import json
import math
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MARS = Path(r"E:\MethData\s2_emit_pairs\publication-v1\external\MARS-S2L")
PHASE0 = Path(
    r"E:\MethData\ground_truth\stanford_2024_2025\files"
    r"\2024-2025 release data - all Phases\Phase 0 (Aug - Dec 2024)"
)
OUTPUT = REPO / "reports" / "ground_truth" / "local" / "stanford_phase0_timing_manifest.csv"
FIELDS = [
    "release_id", "stanford_nominal_utc", "stanford_ch4_kgh_mean",
    "stanford_ch4_kgh_sigma", "stanford_ci95_lower", "stanford_ci95_upper",
    "datatake_start_utc",
    "mars_tile_date_raw", "tile_sensing_utc", "tile_minus_stanford_seconds",
    "nearest_flow_utc", "flow_sample_gap_seconds", "measured_flow_kgh",
    "match_distance_m", "mars_product", "mars_split", "mars_isplume",
    "observability", "percentage_clear", "s2path", "image_exists",
    "cloudmaskpath", "cloudmask_exists", "metadata_json_url",
    "metadata_xml_url", "metadata_json_sha256", "metadata_xml_sha256",
    "evaluation_eligibility", "status", "detail",
]

def iso(dt):
    return dt.isoformat().replace("+00:00", "Z")

def parse_clock(date, clock):
    return datetime.fromisoformat(f"{date}T{clock.strip()}").replace(tzinfo=timezone.utc)

def haversine_km(lat1, lon1, lat2, lon2):
    a, c = math.radians(lat1), math.radians(lat2)
    h = math.sin((c-a)/2)**2 + math.cos(a)*math.cos(c) * math.sin(math.radians(lon2-lon1)/2)**2
    return 12742 * math.asin(min(1.0, math.sqrt(h)))

def download(url):
    req = urllib.request.Request(url, headers={"User-Agent": "meth-metadata-audit/1.0"})
    with urllib.request.urlopen(req, timeout=25) as response:
        return response.read()

def official_sensing_time(product, date):
    tile_match = re.search(r"_T(\d{2})([A-Z])([A-Z]{2})_", product)
    if not tile_match:
        raise ValueError("Cannot parse Sentinel-2 tile identifier")
    zone, band, square = tile_match.groups()
    yr, mo, day = (int(x) for x in date.split("-"))
    base = (f"https://roda.sentinel-hub.com/sentinel-s2-l1c/tiles/"
            f"{zone}/{band}/{square}/{yr}/{mo}/{day}")
    last_error = None
    for sequence in range(12):
        prefix = f"{base}/{sequence}"
        try:
            json_bytes = download(prefix + "/tileInfo.json")
            info = json.loads(json_bytes)
        except Exception as err:
            last_error = err
            continue
        if str(info.get("productName", "")).removesuffix(".SAFE") != product:
            continue
        xml_bytes = download(prefix + "/metadata.xml")
        xml = ET.fromstring(xml_bytes)
        values = [el.text.strip() for el in xml.iter()
                  if el.tag.split("}")[-1] == "SENSING_TIME" and el.text]
        if not values:
            raise ValueError("XML SENSING_TIME missing")
        xml_time = datetime.fromisoformat(values[0].replace("Z", "+00:00"))
        json_time = datetime.fromisoformat(info["timestamp"].replace("Z", "+00:00"))
        if abs((xml_time - json_time).total_seconds()) > 1:
            raise ValueError("XML and JSON tile sensing times disagree by over 1 second")
        return (xml_time, prefix + "/tileInfo.json", prefix + "/metadata.xml",
                hashlib.sha256(json_bytes).hexdigest(), hashlib.sha256(xml_bytes).hexdigest())
    raise ValueError(f"Exact-product tile metadata unavailable (last lookup: {last_error})")

def main():
    if not (MARS / "validated_images_all.csv").is_file() or not PHASE0.is_dir():
        raise FileNotFoundError("MARS metadata or Stanford Phase 0 directory not found")
    by_date_sat = defaultdict(list)
    with (MARS / "validated_images_all.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            date = (row.get("tile_date") or "")[:10]
            sat = (row.get("satellite") or "").strip()
            if not ("2024-08-01" <= date <= "2024-12-31" and sat in ("S2A", "S2B", "S2C")):
                continue
            try:
                row["_lat"] = float(row["lat"])
                row["_lon"] = float(row["lon"])
            except (KeyError, ValueError, TypeError):
                continue
            by_date_sat[(date, sat)].append(row)

    results = []
    for summary in sorted(PHASE0.rglob("*_summary.csv")):
        name = summary.parent.name
        try:
            stamp, sat = name.rsplit("_", 1)
            if sat not in ("S2A", "S2B", "S2C"):
                continue
            date = datetime.strptime(stamp, "%m%d%Y").strftime("%Y-%m-%d")
        except ValueError:
            continue

        result = {field: "" for field in FIELDS}
        result["release_id"] = name
        results.append(result)
        try:
            with summary.open(encoding="utf-8-sig", newline="") as f:
                records = list(csv.DictReader(f))
            if len(records) != 1:
                raise ValueError(f"Expected one summary row, found {len(records)}")
            source = records[0]
            nominal = parse_clock(date, source["time_UTC"])
            result["stanford_nominal_utc"] = iso(nominal)
            for key in ("ch4_kgh_mean", "ch4_kgh_sigma", "ci95_lower", "ci95_upper"):
                result["stanford_" + key] = source.get(key, "")
            lat, lon = float(source["lat"]), float(source["lon"])
            candidates = [
                (haversine_km(lat, lon, row["_lat"], row["_lon"]), row)
                for row in by_date_sat.get((date, sat), [])
            ]
            candidates = [(km, row) for km, row in candidates if km <= 1]
            if not candidates:
                result["status"] = "NO_NEARBY_MARS_IMAGE"
                continue
            km, best = min(candidates, key=lambda x: x[0])
            product = best["tile"]
            result["match_distance_m"] = round(km * 1000, 2)
            result["mars_product"] = product
            result["mars_tile_date_raw"] = best.get("tile_date", "")
            result["mars_split"] = best.get("split_name", "")
            result["mars_isplume"] = best.get("isplume", "")
            result["observability"] = best.get("observability", "")
            result["percentage_clear"] = best.get("percentage_clear", "")
            for column, exists_column in (("s2path", "image_exists"),
                                          ("cloudmaskpath", "cloudmask_exists")):
                relative = (best.get(column) or "").strip()
                result[column] = relative
                result[exists_column] = bool(relative) and (MARS / relative).is_file()

            stamp_match = re.search(r"_MSIL1C_(\d{8}T\d{6})_", product)
            if not stamp_match:
                raise ValueError("Cannot parse product datatake start")
            datatake = datetime.strptime(stamp_match.group(1), "%Y%m%dT%H%M%S")
            result["datatake_start_utc"] = iso(datatake.replace(tzinfo=timezone.utc))

            sensing, json_url, xml_url, json_sha, xml_sha = official_sensing_time(product, date)
            result["tile_sensing_utc"] = iso(sensing)
            result["metadata_json_url"] = json_url
            result["metadata_xml_url"] = xml_url
            result["metadata_json_sha256"] = json_sha
            result["metadata_xml_sha256"] = xml_sha
            result["tile_minus_stanford_seconds"] = round((sensing - nominal).total_seconds(), 3)

            release_file = summary.with_name(summary.name.replace("_summary.csv", "_releasedata.csv"))
            with release_file.open(encoding="utf-8-sig", newline="") as f:
                flow_rows = []
                for measurement in csv.DictReader(f):
                    try:
                        flow_rows.append((parse_clock(date, measurement["time_UTC"]),
                                          float(measurement["methaneflow_KGperHR"])))
                    except (KeyError, ValueError, TypeError):
                        pass
            if not flow_rows:
                raise ValueError("No readable Stanford release measurements")
            observed, flow = min(flow_rows, key=lambda x: abs((x[0] - sensing).total_seconds()))
            gap = abs((observed - sensing).total_seconds())
            result["nearest_flow_utc"] = iso(observed)
            result["flow_sample_gap_seconds"] = round(gap, 3)
            result["measured_flow_kgh"] = round(flow, 6)
            if abs((sensing - nominal).total_seconds()) > 30:
                result["status"] = "NOMINAL_TIME_MISMATCH_REVIEW"
            elif gap > 2:
                result["status"] = "NO_FLOW_SAMPLE_WITHIN_2_SECONDS"
            elif flow > 0:
                result["status"] = "ACTIVE_FLOW_NEAR_TILE_TIME"
                result["evaluation_eligibility"] = "PENDING_IMAGE_REFERENCE_LEAKAGE_AUDITS"
            else:
                result["status"] = "ZERO_FLOW_NEAR_TILE_TIME"
        except Exception as exc:
            result["status"] = "REQUIRES_REVIEW"
            result["detail"] = str(exc)
        print(f"{name}: {result['status']}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print("\nSummary:")
    for status, count in sorted(Counter(r["status"] for r in results).items()):
        print(f"  {status}: {count}")
    print("Manifest:", OUTPUT)
    print("Original Stanford and MARS files unchanged")

if __name__ == "__main__":
    main()
