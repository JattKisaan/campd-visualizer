import glob
import os
import zipfile

import duckdb
import pandas as pd


def concat_facility_zips_to_csv(zip_files, out_csv):
    if os.path.exists(out_csv):
        os.remove(out_csv)
    with open(out_csv, "wb") as out_f:
        header_written = False
        for zpath in zip_files:
            with zipfile.ZipFile(zpath, "r") as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    continue
                csv_inside = csv_names[0]
                with zf.open(csv_inside, "r") as f:
                    if not header_written:
                        out_f.write(f.read())
                        header_written = True
                    else:
                        lines = f.read().split(b"\n", 1)
                        if len(lines) > 1:
                            out_f.write(lines[1])

    df = pd.read_csv(out_csv)
    df.sort_values(by=["State", "Facility Name", "Unit ID", "Year"]).reset_index(
        drop=True
    ).to_csv(out_csv, index=False)
    print(f"Concatenated CSV written to: {out_csv}")


if __name__ == "__main__":
    DATA_DIR = "../data"
    facilities_zip_pattern = os.path.join(DATA_DIR, "facility*.zip")
    facilities_zip_files = sorted(glob.glob(facilities_zip_pattern))
    facilities_csv_out = os.path.join(DATA_DIR, "all_facilities.csv")

    print("Concatenating facility zip CSVs into one CSV...")
    concat_facility_zips_to_csv(facilities_zip_files, facilities_csv_out)
