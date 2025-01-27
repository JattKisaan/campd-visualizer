import gc
import os
import shutil

import campd_visualizer.pkg.constants as constants
import dask.dataframe as dd
import duckdb

os.environ["PYTHONBREAKPOINT"] = "IPython.embed"


def repartition_parquet_dataset(
    in_dir,
    out_dir,
    partition_on=None,
    partition_size="100MB",
    delete_input_files=False,
):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    df = dd.read_parquet(in_dir, engine="pyarrow")
    df_compacted = df.repartition(partition_size=partition_size)
    df_compacted.to_parquet(
        out_dir,
        partition_on=partition_on or [],
        engine="pyarrow",
        overwrite=True,
        write_index=False,
    )
    if delete_input_files and os.path.exists(in_dir):
        shutil.rmtree(in_dir)
    gc.collect()
    del df
    del df_compacted


def parted_repartition_parquet_dataset(
    in_dir,
    out_dir,
    partition_size="100MB",
    delete_input_files=False,
):
    """
    Reads each subdirectory (like Year=XXXX) in `in_dir` one at a time, calling
    `repartition_parquet_dataset` for each. This avoids loading the entire dataset
    at once, reducing memory usage.

    The only difference between subdir paths is that the input subdir path is
    derived from `in_dir` and the output subdir path is derived from `out_dir`.
    """
    if not os.path.exists(in_dir):
        print(f"Input directory {in_dir} does not exist.")
        return
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Find subdirectories containing .parquet (like Year=XXXX).
    subdirs = []
    for entry in os.scandir(in_dir):
        if entry.is_dir():
            # Check if there's at least one .parquet file inside
            any_parquet = any(
                f.name.endswith(".parquet")
                for f in os.scandir(entry.path)
                if f.is_file()
            )
            if any_parquet:
                subdirs.append(entry.name)

    breakpoint()
    # For each subdir, read from in_dir/subdir, repartition, and write to out_dir/subdir
    for sub in subdirs:
        part_in_dir = os.path.join(in_dir, sub)
        part_out_dir = os.path.join(out_dir, sub)
        print(f"Repartitioning {part_in_dir} -> {part_out_dir}")

        repartition_parquet_dataset(
            in_dir=part_in_dir,
            out_dir=part_out_dir,
            partition_size=partition_size,
            delete_input_files=delete_input_files,
        )
        gc.collect()


if __name__ == "__main__":
    DATA_DIR = "../data"

    emissions_out_dir_temp = os.path.join(DATA_DIR, "emissions_parquet_year_temp")
    emissions_out_dir = os.path.join(DATA_DIR, "emissions_parquet_year")

    parted_repartition_parquet_dataset(
        in_dir=emissions_out_dir_temp,
        out_dir=emissions_out_dir,
        partition_size="100_000MB",
        delete_input_files=False,
    )

    emissions_query = f"""
      SELECT DISTINCT "State", "Facility Name"
      FROM {constants.EMISSIONS_TABLE}
      WHERE "Year" = 2023
        AND "State" = 'AL'
        AND "Date" >= '2023-01-01'
      ORDER BY "Facility Name"
      LIMIT 50
    """
    print("Emissions DuckDB Query:\n", emissions_query)
    df_emissions = duckdb.query(emissions_query).to_df()
    print("Emissions Query result (first 10 rows):")
    print(df_emissions.head(10))
