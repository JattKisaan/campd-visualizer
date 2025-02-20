import os

import pyarrow as pa

DATA_DIR = "../data"
emissions_parquet_glob = os.path.join(
    DATA_DIR, "emissions_parquet_year", "*", "*.parquet"
)
EMISSIONS_TABLE = f"read_parquet('{emissions_parquet_glob}', hive_partitioning=True)"
FACILITIES_TABLE = f'''"{os.path.join(DATA_DIR, 'all_facilities.csv')}"'''

TYPE_DICT = {
    "pandas": {
        "text": "string",
        "int": "Int64",
        "float": "float64",
        "date": "object",
    },
    "pyarrow": {
        "text": pa.string(),
        "int": pa.int64(),
        "float": pa.float64(),
        "date": pa.date32(),
    },
    "sqlite": {
        "text": "TEXT",
        "int": "INTEGER",
        "float": "REAL",
        "date": "DATE",
    },
    "postgresql": {
        "text": "VARCHAR",
        "int": "INTEGER",
        "float": "DOUBLE PRECISION",
        "date": "DATE",
    },
    "oracle": {
        "text": "VARCHAR2(255)",
        "int": "NUMBER(10)",
        "float": "FLOAT",
        "date": "DATE",
    },
}

# fmt:off
EMISSIONS_COLUMNS_AND_TYPES = [
    ("State"                        , "text"),
    ("Facility Name"                , "text"),
    ("Facility ID"                  , "int"),
    ("Unit ID"                      , "text"),

    ("Associated Stacks"            , "text"),
    ("Date"                         , "date"),
    ("Hour"                         , "int"),
    ("Operating Time"               , "float"),

    ("Gross Load (MW)"              , "int"),
    ("Steam Load (1000 lb/hr)"      , "float"),
    ("SO2 Mass (lbs)"               , "float"),
    ("SO2 Mass Measure Indicator"   , "text"),

    ("SO2 Rate (lbs/mmBtu)"         , "float"),
    ("SO2 Rate Measure Indicator"   , "text"),
    ("CO2 Mass (short tons)"        , "float"),
    ("CO2 Mass Measure Indicator"   , "text"),

    ("CO2 Rate (short tons/mmBtu)"  , "float"),
    ("CO2 Rate Measure Indicator"   , "text"),
    ("NOx Mass (lbs)"               , "float"),
    ("NOx Mass Measure Indicator"   , "text"),

    ("NOx Rate (lbs/mmBtu)"         , "float"),
    ("NOx Rate Measure Indicator"   , "text"),
    ("Heat Input (mmBtu)"           , "float"),
    ("Heat Input Measure Indicator" , "text"),

    ("Primary Fuel Type"            , "text"),
    ("Secondary Fuel Type"          , "text"),
    ("Unit Type"                    , "text"),
    ("SO2 Controls"                 , "text"),

    ("NOx Controls"                 , "text"),
    ("PM Controls"                  , "text"),
    ("Hg Controls"                  , "text"),
    ("Program Code"                 , "text"),
]
# fmt:on

# fmt:off
FACILITIES_COLUMNS_AND_TYPES = [
    ("State"                                             , "text"),
    ("Facility Name"                                     , "text"),
    ("Facility ID"                                       , "int"),
    ("Unit ID"                                           , "text"),
     
    ("Associated Stacks"                                 , "text"),
    ("Year"                                              , "int"),
    ("Program Code"                                      , "text"),
    ("Primary Rep Info"                                  , "text"),
     
    ("EPA Region"                                        , "int"),
    ("NERC Region"                                       , "text"),
    ("County"                                            , "text"),
    ("County Code"                                       , "text"),
     
    ("FIPS Code"                                         , "text"),
    ("Source Category"                                   , "text"),
    ("Latitude"                                          , "float"),
    ("Longitude"                                         , "float"),
     
    ("Owner/Operator"                                    , "text"),
    ("SO2 Phase"                                         , "text"),
    ("NOx Phase"                                         , "text"),
    ("Unit Type"                                         , "text"),
     
    ("Primary Fuel Type"                                 , "text"),
    ("Secondary Fuel Type"                               , "text"),
    ("SO2 Controls"                                      , "text"),
    ("NOx Controls"                                      , "text"),
     
    ("PM Controls"                                       , "text"),
    ("Hg Controls"                                       , "text"),
    ("Commercial Operation Date"                         , "date"),
    ("Operating Status"                                  , "text"),
     
    ("Max Hourly HI Rate (mmBtu/hr)"                     , "float"),
    ("Associated Generators & Nameplate Capacity (MWe)"  , "text"),
]
# fmt:on
