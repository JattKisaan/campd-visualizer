#!/usr/bin/env python
import glob
import os
import shutil
import zipfile

import campd_visualizer.pkg.constants as constants
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq

os.environ["PYTHONBREAKPOINT"] = "IPython.embed"


def stream_csvs(list_of_year_files, convert_options):
    """
    Process each zip file individually rather than concatenating many files at once.
    For each zip file, read the CSV, add the Year column, and sort by Date and State.
    Yield one small table at a time.
    """
    for year_files in list_of_year_files:
        for zpath in year_files:
            print(f"Reading ZIP: {zpath}")
            with zipfile.ZipFile(zpath, "r") as zf:
                # Read the first CSV file from the zip.
                with zf.open(zf.namelist()[0]) as f:
                    csv_reader = pa.csv.open_csv(f, convert_options=convert_options)
                    for rb in csv_reader:
                        year_array = pc.year(rb.column("Date")).cast(pa.int32())
                        rb = rb.append_column("Year", year_array)
                        rb = rb.sort_by([("Date", "ascending"), ("State", "ascending")])
                        yield rb


def make_parquet_from_zip_files(
    zip_files,
    out_dir,
    partitioning=None,
):
    print("Converting Emissions CSVs to partitioned Parquet...")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # Assume zip_files and constants are defined elsewhere.
    # Compute years from the filenames.
    years = np.sort(np.unique([zipf.split("-")[2] for zipf in zip_files]))
    list_of_year_files = [
        [zipf for zipf in zip_files if year in zipf] for year in years
    ]

    # Update schema to include the Year column.
    schema = pa.schema(
        [
            (c[0], constants.TYPE_DICT["pyarrow"][c[1]])
            for c in constants.EMISSIONS_COLUMNS_AND_TYPES
        ]
        + [("Year", pa.int32())]
    )

    convert_opts = pa.csv.ConvertOptions(
        column_types={
            c[0]: constants.TYPE_DICT["pyarrow"][c[1]]
            for c in constants.EMISSIONS_COLUMNS_AND_TYPES
        }
    )

    ## No paritioning, but it works
    # with pq.ParquetWriter(
    #    out_dir,
    #    schema,
    # ) as writer:
    #    for table in stream_csvs(list_of_year_files, convert_opts):
    #        writer.write_table(
    #            table,
    #        )

    ##
    full_table = stream_csvs(list_of_year_files, convert_opts)
    ds.write_dataset(
        full_table,
        schema=schema,
        base_dir=out_dir,
        format="parquet",
        partitioning=[
            "Year",
            # "State",
        ],
        partitioning_flavor="hive",
        existing_data_behavior="overwrite_or_ignore",
    )
    print(f"Partitioned dataset written to: {out_dir}")


if __name__ == "__main__":
    DATA_DIR = "../data"

    emissions_zip_pattern = os.path.join(DATA_DIR, "emissions-hourly*.zip")
    emissions_zip_files = sorted(glob.glob(emissions_zip_pattern))
    emissions_out_dir_temp = os.path.join(DATA_DIR, "emissions_parquet_year_temp")

    make_parquet_from_zip_files(
        zip_files=emissions_zip_files,
        out_dir=emissions_out_dir_temp,
    )
