library(duckdb)

con <- duckdb::dbConnect(duckdb::duckdb())

emissions_query <- "
  COPY (
    SELECT \"State\", \"Facility Name\", \"Facility ID\", \"Unit ID\", \"Year\", \"Date\",
      \"Hour\", \"Operating Time\", \"Gross Load (MW)\", \"NOx Mass (lbs)\",
      \"NOx Mass Measure Indicator\", \"NOx Rate (lbs/mmBtu)\",
      \"NOx Rate Measure Indicator\", \"Primary Fuel Type\", \"Secondary Fuel Type\",
      \"Unit Type\", \"NOx Controls\"
    FROM read_parquet('../data/emissions_parquet_year/*/*.parquet', hive_partitioning=True)
    WHERE \"State\" = 'PA'
      AND \"Operating Time\" > 0
      AND \"Year\" = '2023'
    ORDER BY \"Facility Name\", \"Unit ID\", \"Date\", \"Hour\"
    LIMIT 10000
  )
  TO 'test.csv' (HEADER, DELIMITER ',');
"

dbGetQuery(con, emissions_query)

facilities_query <- "
  SELECT *
  FROM \"../data/all_facilities.csv\"
  WHERE \"State\" = 'PA'
    AND \"Year\" = '2023'
"

df <- dbGetQuery(con, facilities_query)
print(df)

# Disconnect
duckdb::dbDisconnect(con)
