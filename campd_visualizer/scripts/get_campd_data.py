import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

BUCKET_URL_BASE = "https://api.epa.gov/easey/bulk-files/"


def download_bulk_emissions_data(
    api_key_path,
    download_dir,
    cutoff_date=None,
):
    """
    Downloads emissions CSV files, converts each to a zip, and overwrites existing
    zip files only if the remote file is newer than the local file's mod time.
    If a cutoff_date is supplied, we also skip remote files whose lastUpdated <= cutoff_date.
    """
    with open(api_key_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    endpoint = "https://api.epa.gov/easey/camd-services/bulk-files"
    params = {"api_key": api_key}
    response = requests.get(endpoint, params=params)
    print("Status code:", response.status_code)

    if response.status_code >= 400:
        try:
            err_msg = response.json()["error"]["message"]
        except (ValueError, KeyError):
            err_msg = response.text
        sys.exit("Error message: " + err_msg)

    bulk_files = response.json()

    def parse_last_updated(raw):
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))

    filtered = []
    for f in bulk_files:
        meta = f["metadata"]
        if (
            meta.get("dataType") == "Emissions"
            and "quarter" in meta
            and meta.get("dataSubType") == "Hourly"
        ):
            remote_dt = parse_last_updated(f["lastUpdated"])
            if cutoff_date and remote_dt <= cutoff_date:
                continue
            filtered.append(f)

    print("Potential files to check:", len(filtered))
    os.makedirs(download_dir, exist_ok=True)

    downloaded_count = 0
    for f in filtered:
        remote_dt = parse_last_updated(f["lastUpdated"])
        url = BUCKET_URL_BASE + f["s3Path"]
        local_csv = os.path.join(download_dir, f["filename"])
        local_zip = local_csv.replace(".csv", ".zip")

        # Check modtime on local_zip if it exists
        if os.path.exists(local_zip):
            mod_ts = os.path.getmtime(local_zip)
            local_mod = datetime.fromtimestamp(mod_ts, tz=timezone.utc)
            if local_mod >= remote_dt:
                print(f"Skipping (local file up to date): {local_zip}")
                continue

        print(f"Downloading: {url}")
        resp = requests.get(url)
        with open(local_csv, "wb") as out_file:
            out_file.write(resp.content)

        if os.path.exists(local_zip):
            os.remove(local_zip)

        with zipfile.ZipFile(local_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(local_csv, arcname=f["filename"])
        os.remove(local_csv)
        downloaded_count += 1

    print(f"Done. Downloaded/updated: {downloaded_count} file(s).")


def download_facility_data(
    api_key_path,
    download_dir,
    cutoff_date=None,
):
    """
    Downloads facility CSV files, converts each to a zip, and overwrites existing
    zip files only if the remote file is newer than the local file's mod time.
    If a cutoff_date is supplied, we skip remote files whose lastUpdated <= cutoff_date.
    """
    with open(api_key_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    endpoint = "https://api.epa.gov/easey/camd-services/bulk-files"
    params = {"api_key": api_key}
    response = requests.get(endpoint, params=params)
    print("Status code:", response.status_code)

    if response.status_code >= 400:
        try:
            err_msg = response.json()["error"]["message"]
        except (ValueError, KeyError):
            err_msg = response.text
        sys.exit("Error message: " + err_msg)

    bulk_files = response.json()

    def parse_last_updated(raw):
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))

    filtered = []
    for f in bulk_files:
        meta = f["metadata"]
        if meta.get("dataType") == "Facility":
            remote_dt = parse_last_updated(f["lastUpdated"])
            if cutoff_date and remote_dt <= cutoff_date:
                continue
            filtered.append(f)

    print("Potential files to check:", len(filtered))
    os.makedirs(download_dir, exist_ok=True)

    downloaded_count = 0
    for f in filtered:
        remote_dt = parse_last_updated(f["lastUpdated"])
        url = BUCKET_URL_BASE + f["s3Path"]
        local_csv = os.path.join(download_dir, f["filename"])
        local_zip = local_csv.replace(".csv", ".zip")

        if os.path.exists(local_zip):
            mod_ts = os.path.getmtime(local_zip)
            local_mod = datetime.fromtimestamp(mod_ts, tz=timezone.utc)
            if local_mod >= remote_dt:
                print(f"Skipping (local file up to date): {local_zip}")
                continue

        print(f"Downloading: {url}")
        resp = requests.get(url)
        with open(local_csv, "wb") as out_file:
            out_file.write(resp.content)

        if os.path.exists(local_zip):
            os.remove(local_zip)

        with zipfile.ZipFile(local_zip, "w") as zf:
            zf.write(local_csv, arcname=f["filename"])
        os.remove(local_csv)
        downloaded_count += 1

    print(f"Done. Downloaded/updated: {downloaded_count} file(s).")


if __name__ == "__main__":
    api_key = Path("../keys/camd_key")
    download_dir = Path("../data")

    # By default, no cutoff_date is used.
    # If you do want to filter server-side by only downloading data after a
    # certain cutoff_date, define it here:

    # cutoff_str = "2025-01-01"
    cutoff_str = None
    if cutoff_str:
        cutoff_date = datetime.strptime(cutoff_str, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
    else:
        cutoff_date = None

    download_bulk_emissions_data(api_key, download_dir, cutoff_date=cutoff_date)
    download_facility_data(api_key, download_dir, cutoff_date=cutoff_date)
