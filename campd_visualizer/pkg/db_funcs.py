import csv

import oracledb
from sqlalchemy import create_engine


def create_sql_engine(DB_TYPE, DB_NAME):
    """
    You should supply DB_NAME. For SQLite this is the name of the db file at `../data/{DB_NAME}.db`.
    For Postgres or Oracle, it would be the name of the database, so probably "camdash", for Oracle at least.

    """
    if DB_TYPE == "postgresql":
        from campd_visualizer.keys.login import LOGIN_DETAILS

        DB_USER = LOGIN_DETAILS["postgresql"]["user"]
        DB_PASS = LOGIN_DETAILS["postgresql"]["pass"]
        DB_HOST = LOGIN_DETAILS["postgresql"]["host"]
        DB_PORT = LOGIN_DETAILS["postgresql"]["port"]
        cs = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        return create_engine(cs)

    elif DB_TYPE == "oracle":
        from campd_visualizer.keys.login import LOGIN_DETAILS

        oracledb.init_oracle_client()
        DB_USER = LOGIN_DETAILS["oracle"]["user"]
        DB_PASS = LOGIN_DETAILS["oracle"]["pass"]
        DB_HOST = LOGIN_DETAILS["oracle"]["host"]
        DB_PORT = LOGIN_DETAILS["oracle"]["port"]
        return create_engine(
            f"oracle+oracledb://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/?service_name={DB_NAME}"
        )

    else:
        raise ValueError(
            "Unsupported DB_TYPE. Use 'sqlite', 'postgresql', or 'oracle'. Or, adjust this function to support your DB_TYPE."
        )


def bulk_insert_csv_files(
    conn,
    table_name,
    columns,
    csv_files,
    chunksize=1_000_000,
):
    """
    Given a list of CSV file paths, loads them in chunks into `table_name`.
    The order of columns in `columns` must match the CSV column order.
    """
    # Prepare the insert statement with question marks
    insert_sql = f"INSERT INTO {table_name} VALUES ({','.join(['?'] * len(columns))})"

    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        with open(csv_file, "r", newline="") as f:
            # Skip header
            next(f)

            chunk_rows = []
            for i, line in enumerate(f, start=1):
                # Parse the CSV line into a row tuple
                row = list(csv.reader([line], delimiter=",", quotechar='"'))[0]
                chunk_rows.append(tuple(row))

                # If we reached the chunk size, do a bulk insert and reset
                if i % chunksize == 0:
                    conn.exec_driver_sql(insert_sql, chunk_rows)
                    chunk_rows.clear()

            # Insert any leftover rows at the end
            if chunk_rows:
                conn.exec_driver_sql(insert_sql, chunk_rows)


if __name__ == "__main__":
    pass
