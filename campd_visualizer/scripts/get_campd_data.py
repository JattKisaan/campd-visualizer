import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

BUCKET_URL_BASE = "https://api.epa.gov/easey/bulk-files/"


def download_data(
    api_key_path,
    download_dir,
    filter_func,
):
    """
    Downloads CSV files according to `filter_func`, which decides whether
    or not a given file (represented by dict f) should be included. If included,
    the CSV is downloaded and converted to a zip. Overwrites only if the remote file
    is newer than the local file.

    :param api_key_path: Path to a file containing the API key.
    :param download_dir: Directory to which .zip files will be written.
    :param filter_func:  A function taking a file dict (f) from the bulk-files
                         response and returning True/False.
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

    # Filter out files using the provided function
    filtered = []
    for f in bulk_files:
        if filter_func(f):
            filtered.append(f)

    print("Potential files to check:", len(filtered))
    os.makedirs(download_dir, exist_ok=True)

    downloaded_count = 0
    for f in filtered:
        # We'll parse lastUpdated for timestamp comparisons
        remote_dt = datetime.fromisoformat(f["lastUpdated"].replace("Z", "+00:00"))
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

        with zipfile.ZipFile(local_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(local_csv, arcname=f["filename"])

        os.remove(local_csv)
        downloaded_count += 1

    print(f"Done. Downloaded/updated: {downloaded_count} file(s).")


if __name__ == "__main__":
    api_key = Path("../keys/camd_key")
    download_dir = Path("../data")

    def filter_emissions(f):
        """Keep only Emissions + Hourly + 'quarter' in metadata."""
        meta = f["metadata"]
        if (
            meta.get("dataType") == "Emissions"
            and meta.get("dataSubType") == "Hourly"
            and "quarter" in meta
        ):
            return True
        return False

    def filter_facility(f):
        """Keep only Facility data."""
        meta = f["metadata"]
        return meta.get("dataType") == "Facility"

    # Emissions
    download_data(
        api_key_path=api_key,
        download_dir=download_dir,
        filter_func=filter_emissions,
    )

    # Facilities
    download_data(
        api_key_path=api_key,
        download_dir=download_dir,
        filter_func=filter_facility,
    )
