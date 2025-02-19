# Load required library
library(duckdb)

# Set up DuckDB connection
con <- dbConnect(duckdb::duckdb())

# Define the path to your Parquet files
data_dir <- "../data"
emissions_parquet_glob <- paste0(data_dir, "/emissions_parquet_year/*/*.parquet")
emissions_table <- paste0("read_parquet('", emissions_parquet_glob, "', hive_partitioning=TRUE)")

# Query
facilities_query <- sprintf("
  SELECT DISTINCT State, \"Facility Name\"
  FROM %s
  WHERE State = 'AL' AND Year = '2023'
  ORDER BY \"Facility Name\"
", emissions_table)

# Execute query and convert to a data frame
df_facilities <- dbGetQuery(con, facilities_query)

# Print the result
print(df_facilities)

# Disconnect
dbDisconnect(con)
