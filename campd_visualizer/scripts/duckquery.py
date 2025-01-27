#!/usr/bin/env python

import campd_visualizer.pkg.constants as constants
import duckdb

if __name__ == "__main__":
    facilities_query = f"""
      SELECT DISTINCT "State", "Facility Name"
      FROM {constants.EMISSIONS_TABLE}
      WHERE "State" = 'AL'
      AND "Year" = '2023'
      ORDER BY "Facility Name"
    """
    print("Facilities DuckDB Query:\n", facilities_query)
    df_facilities = duckdb.query(facilities_query).to_df()
    print(df_facilities)
