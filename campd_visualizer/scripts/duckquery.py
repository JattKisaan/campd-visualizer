#!/usr/bin/env python

import campd_visualizer.pkg.constants as constants
import duckdb

if __name__ == "__main__":
    emissions_query = f"""
    COPY (
      SELECT "State", "Facility Name", "Facility ID", "Unit ID", "Year", "Date",
        "Hour", "Operating Time", "Gross Load (MW)", "NOx Mass (lbs)",
        "NOx Mass Measure Indicator", "NOx Rate (lbs/mmBtu)",
        "NOx Rate Measure Indicator", "Primary Fuel Type", "Secondary Fuel Type",
        "Unit Type", "NOx Controls"
      FROM {constants.EMISSIONS_TABLE}
      WHERE "State" = 'PA'
        AND "Operating Time" > 0
        AND "Year" = '2023'
      ORDER BY "Facility Name", "Unit ID", "Date", "Hour"
      LIMIT 10000
    )
    TO 'test.csv' (HEADER, DELIMITER ',');
    """
    print(
        f"""
    Emissions DuckDB Query:
    {emissions_query}

    The output of the SELECT statement is being written to test.csv
    """
    )
    duckdb.query(emissions_query)

    facilities_query = f"""
      SELECT *
      FROM {constants.FACILITIES_TABLE}
      WHERE "State" = 'PA'
        AND "Year" = '2023'
    """
    print("Facilities DuckDB Query:\n", facilities_query)
    df = duckdb.query(facilities_query).to_df()
    print(df)
