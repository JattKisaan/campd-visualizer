import time
from datetime import datetime

import campd_visualizer.pkg.constants as constants
import duckdb
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def query(q, columns_and_types=None):
    conn = duckdb.connect()
    df = conn.execute(q).df().replace("", pd.NA)
    conn.close()

    if columns_and_types:
        columns_and_types = [c for c in columns_and_types if c[0] in df.columns]
        dtypes = {c[0]: constants.TYPE_DICT["pandas"][c[1]] for c in columns_and_types}
        df = df.astype(dtypes)

        date_cols = [c[0] for c in columns_and_types if c[1] == "date"]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])

    return df


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"{elapsed:.3f}s: {func.__name__}: ")
        return result

    return wrapper


MAX_PLOTLY_POINTS = 1_000_000

DISTINCT_STATES = np.sort(
    query(f'SELECT DISTINCT "State" FROM {constants.FACILITIES_TABLE}').values.squeeze(
        1
    )
)

DISTINCT_FUELS = np.sort(
    query(
        f"""SELECT DISTINCT "Primary Fuel Type"
              FROM {constants.FACILITIES_TABLE}
              WHERE "Primary Fuel Type" IS NOT NULL
           """
    ).values.squeeze(1)
)

n = len(DISTINCT_FUELS)
colors_list = mpl.colormaps["viridis"].resampled(n)
colors_list = [mcolors.to_hex(colors_list(i)) for i in range(n)]
FUEL_COLORS_DICT = dict(zip(DISTINCT_FUELS, colors_list))

POLLUTANT_OPTIONS = {
    "NOx Rate (lbs/mmBtu)": "NOx Rate (lbs/mmBtu)",
    "SO2 Rate (lbs/mmBtu)": "SO2 Rate (lbs/mmBtu)",
}


def insert_gaps(x_list, y_list, threshold):
    if not x_list:
        return x_list, y_list
    new_x, new_y = [x_list[0]], [y_list[0]]
    for i in range(1, len(x_list)):
        if x_list[i] - x_list[i - 1] > threshold:
            new_x.append(None)
            new_y.append(None)
        new_x.append(x_list[i])
        new_y.append(y_list[i])
    return new_x, new_y


def average_contiguous_points(x, y, window):
    L = len(x)
    if L == 0:
        return np.array([], dtype=x.dtype), np.array([], dtype=y.dtype)
    groups = np.arange(L) // window
    x_int = x.astype("int64")
    sum_x = np.bincount(groups, weights=x_int)
    count_x = np.bincount(groups)
    avg_x_int = sum_x / count_x
    avg_x = avg_x_int.astype("int64").view("datetime64[ns]")
    sum_y = np.bincount(groups, weights=y)
    count_y = np.bincount(groups)
    avg_y = sum_y / count_y
    return avg_x, avg_y


def average_segments(x_list, y_list, threshold, window):
    x = np.array(x_list, dtype="datetime64[ns]")
    y = np.array(y_list, dtype=float)
    if len(x) == 0:
        return x_list, y_list
    diffs = np.diff(x)
    gap_mask = diffs > np.timedelta64(threshold.value, "ns")
    gap_indices = np.nonzero(gap_mask)[0] + 1
    x_segments = np.split(x, gap_indices)
    y_segments = np.split(y, gap_indices)
    new_x, new_y = [], []
    for seg_x, seg_y in zip(x_segments, y_segments):
        avg_x, avg_y = average_contiguous_points(seg_x, seg_y, window)
        new_x.extend(avg_x)
        new_y.extend(avg_y)
        new_x.append(None)
        new_y.append(None)
    if new_x:
        new_x.pop()
        new_y.pop()
    return new_x, new_y


def _distribute_around_circle(group):
    Radius = 0.001
    N = group.shape[0]
    if N == 1:
        group["Plot Latitude"] = group["Latitude"]
        group["Plot Longitude"] = group["Longitude"]
    else:
        shifts = Radius * np.array(
            [
                [np.sin(theta), np.cos(theta)]
                for theta in np.linspace(0, 2 * np.pi, N + 1)
            ][:-1]
        )
        group["Plot Latitude"] = group["Latitude"] + shifts[:, 0]
        group["Plot Longitude"] = group["Longitude"] + shifts[:, 1]
    return group


def adjust_facility_markers(df):
    group_cols = ["State", "Facility Name", "Year", "Latitude", "Longitude"]
    return (
        df.groupby(group_cols)[df.columns]
        .apply(_distribute_around_circle)
        .reset_index(drop=True)
    )


def get_facilities(start_date, end_date):
    sy = datetime.strptime(start_date, "%Y-%m-%d").year
    ey = datetime.strptime(end_date, "%Y-%m-%d").year
    q = f"""
    SELECT "State","Facility Name","Unit ID","Primary Fuel Type","Latitude","Longitude","Year"
    FROM {constants.FACILITIES_TABLE}
    WHERE "Year" >= '{sy}' AND "Year" <= '{ey}'
      AND "Latitude" IS NOT NULL AND "Longitude" IS NOT NULL
    """
    df = query(q, columns_and_types=constants.FACILITIES_COLUMNS_AND_TYPES)
    group_cols = ["State", "Facility Name", "Unit ID", "Latitude", "Longitude"]
    df = (
        df.groupby(group_cols, as_index=False)[df.columns]
        .apply(lambda grp: grp[grp["Year"] == grp["Year"].max()])
        .reset_index(drop=True)
    )
    return df


def get_emissions(pollutant_col, state, plant, start_date, end_date):
    sy, ey = pd.to_datetime(start_date).year, pd.to_datetime(end_date).year
    plant_str = (
        f"""\n      AND "Facility Name" = '{plant}'"""
        if plant and plant != "ALL"
        else ""
    )
    q = f"""\
    SELECT "State","Facility Name","Unit ID","Date","Hour","{pollutant_col}"
    FROM {constants.EMISSIONS_TABLE}
    WHERE "State" = '{state}' AND "Year" >= {sy} AND "Year" <= {ey}
      AND "Operating Time" > 0.0
      AND "Date" >= '{start_date}' AND "Date" <= '{end_date}'
      AND "{pollutant_col}" IS NOT NULL {plant_str}
    """
    df = query(q, columns_and_types=constants.EMISSIONS_COLUMNS_AND_TYPES)
    df["Datetime"] = df["Date"] + pd.to_timedelta(df["Hour"], unit="h")
    df = df.sort_values(["Facility Name", "Unit ID", "Datetime"]).drop(
        columns=["Date", "Hour"]
    )
    return df


def get_geojson_data(start_date, end_date):
    df = get_facilities(start_date, end_date)
    adjusted_df = adjust_facility_markers(df)
    features = []
    for _, row in adjusted_df.iterrows():
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["Plot Longitude"], row["Plot Latitude"]],
            },
            "properties": {
                "state": row["State"],
                "facility_name": row["Facility Name"],
                "unit_id": row["Unit ID"],
                "primary_fuel": row["Primary Fuel Type"],
            },
        }
        features.append(feat)
    return {"type": "FeatureCollection", "features": features}
