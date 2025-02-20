#!/usr/bin/env python

"""
Enumerates each subdirectory in `in_dir` containing Parquet files,
and spawns a separate Python process for each, calling
repartition_one_subdir(...) in repartition_step.py.

Adjust `IN_DIR`, `OUT_DIR`, and `PARTITION_SIZE` as needed in the main().
"""

import os
import shutil
import subprocess
import sys


def repartition_whole_dataset(
    in_dir,
    out_dir,
    partition_size="5_000MB",
    delete_input=False,
):
    if not os.path.exists(in_dir):
        print(f"[Master] Input directory {in_dir} does not exist.")
        return

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    subdirs = []
    for entry in os.scandir(in_dir):
        if entry.is_dir():
            has_parquet = any(
                f.name.endswith(".parquet")
                for f in os.scandir(entry.path)
                if f.is_file()
            )
            if has_parquet:
                subdirs.append(entry.name)

    for sub in sorted(subdirs):
        part_in_dir = os.path.join(in_dir, sub)
        part_out_dir = os.path.join(out_dir, sub)
        print(f"[Master] Launching sub-process for {part_in_dir} -> {part_out_dir}")

        cmd_expr = (
            "import repartition_step;"
            f" repartition_step.repartition_one_subdir("
            f"'{part_in_dir}', "
            f"'{part_out_dir}', "
            f"'{partition_size}', "
            f"{delete_input}"
            ")"
        )
        cmd = [sys.executable, "-c", cmd_expr]
        subprocess.run(cmd, check=True)


def main():
    IN_DIR = "../data/emissions_parquet_year_temp"
    OUT_DIR = "../data/emissions_parquet_year"

    # e.g. "5_000MB" by default, or override to "100_000MB" if desired
    PARTITION_SIZE = "100_000MB"
    DELETE_INPUT = False

    repartition_whole_dataset(
        in_dir=IN_DIR,
        out_dir=OUT_DIR,
        partition_size=PARTITION_SIZE,
        delete_input=DELETE_INPUT,
    )


if __name__ == "__main__":
    main()
