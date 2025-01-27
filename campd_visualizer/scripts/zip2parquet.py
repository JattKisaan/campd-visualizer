#!/usr/bin/env python
import glob
import os
import shutil
import zipfile

import campd_visualizer.pkg.constants as constants
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.dataset as ds

os.environ["PYTHONBREAKPOINT"] = "IPython.embed"


def make_parquet_from_zip_files(
    zip_files,
    out_dir,
    convert_opts,
    partitioning=None,
    extra_transform=None,
):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    for zpath in zip_files:
        print(f"Reading ZIP: {zpath}")
        with zipfile.ZipFile(zpath, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                print(f"  No .csv in {zpath}, skipping.")
                continue
            csv_inside = csv_names[0]
            print(f"  Found CSV: {csv_inside}")

            with zf.open(csv_inside, "r") as f:
                csv_reader = pcsv.open_csv(f, convert_options=convert_opts)
                overwrite_protect_counter = 0
                for rb in csv_reader:
                    table = pa.Table.from_batches([rb])
                    print(f"  Loaded {table.num_rows} rows")

                    if extra_transform is not None:
                        table = extra_transform(table)

                    ds.write_dataset(
                        data=table,
                        base_dir=out_dir,
                        basename_template=(
                            f"part-{csv_inside.split('.')[0]}-{overwrite_protect_counter}-"
                            + r"{i}.parquet"
                        ),
                        format="parquet",
                        partitioning=partitioning,
                        existing_data_behavior="overwrite_or_ignore",
                    )
                    overwrite_protect_counter += 1
    print(f"Partitioned dataset written to: {out_dir}")


def extra_transform_emissions(table):
    year_array = pc.year(table["Date"])
    return table.append_column("Year", year_array)


if __name__ == "__main__":
    DATA_DIR = "../data"

    emissions_zip_pattern = os.path.join(DATA_DIR, "emissions-hourly*.zip")
    emissions_zip_files = sorted(glob.glob(emissions_zip_pattern))
    emissions_out_dir_temp = os.path.join(DATA_DIR, "emissions_parquet_year_temp")
    emissions_out_dir = os.path.join(DATA_DIR, "emissions_parquet_year")

    emissions_convert_opts = pcsv.ConvertOptions(
        column_types={
            c[0]: constants.TYPE_DICT["pyarrow"][c[1]]
            for c in constants.EMISSIONS_COLUMNS_AND_TYPES
        }
    )
    emissions_partition_schema = pa.schema([pa.field("Year", pa.int32())])
    emissions_partitioning = ds.partitioning(emissions_partition_schema, flavor="hive")

    print("Converting Emissions CSVs to partitioned Parquet...")
    make_parquet_from_zip_files(
        zip_files=emissions_zip_files,
        out_dir=emissions_out_dir_temp,
        convert_opts=emissions_convert_opts,
        partitioning=emissions_partitioning,
        extra_transform=extra_transform_emissions,
    )
