import gc
import os
import shutil

import dask.dataframe as dd


def repartition_one_subdir(in_dir, out_dir, partition_size="100MB", delete_input=False):
    """
    Read Parquet files in `in_dir`, repartition to `partition_size`,
    write them to `out_dir`, optionally delete `in_dir`.
    Called as a separate process from repartion_master.py
    """
    if not os.path.exists(in_dir):
        print(f"[Child] Subdir {in_dir} does not exist.")
        return

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    print(
        f"[Child] Repartitioning {in_dir} -> {out_dir}, partition_size={partition_size}"
    )
    df = dd.read_parquet(in_dir, engine="pyarrow")
    df = df.repartition(npartitions=1)
    df.to_parquet(
        out_dir,
        engine="pyarrow",
        overwrite=True,
        write_index=False,
    )

    if delete_input and os.path.exists(in_dir):
        shutil.rmtree(in_dir)

    del df
    gc.collect()


print("[Child] repartition_step.py loaded.")
