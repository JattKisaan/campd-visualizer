#!/usr/bin/env python
import os
import time
from datetime import date, datetime

import campd_visualizer.pkg.constants as constants
import dash
import dash_leaflet as dl
import duckdb
import matplotlib as mpl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html
from dash_extensions.javascript import assign
from plotly import colors as pcolors

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


# ------------------------------------------------------------------------------
# BASIC QUERY & HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def query(q, columns_and_types=None):
    conn = duckdb.connect()
    df = conn.execute(q).df().replace("", pd.NA)
    conn.close()

    if columns_and_types:
        # This needs to be manually added because the raw EPA data doesn't
        # include this Year column, it's only added as a partition column
        # during the parquet dataset creation.
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


# ------------------------------------------------------------------------------
# GLOBAL CONSTANTS / CONFIG
# ------------------------------------------------------------------------------
MAX_PLOTLY_POINTS = 1_000_000

DISTINCT_STATES = np.sort(
    query(
        f'SELECT DISTINCT "State" FROM {constants.FACILITIES_TABLE}',
    ).values.squeeze(1)
)

DISTINCT_FUELS = np.sort(
    query(
        f"""SELECT DISTINCT "Primary Fuel Type"
        FROM {constants.FACILITIES_TABLE}
        WHERE "Primary Fuel Type" IS NOT NULL""",
    ).values.squeeze(1)
)
n = len(DISTINCT_FUELS)
colors_list = mpl.colormaps["viridis"].resampled(n)
colors_list = [colors_list(i) for i in range(n)]
FUEL_COLORS_DICT = dict(zip(DISTINCT_FUELS, colors_list))

POLLUTANT_OPTIONS = {
    "NOx Rate (lbs/mmBtu)": "NOx Rate (lbs/mmBtu)",
    "SO2 Rate (lbs/mmBtu)": "SO2 Rate (lbs/mmBtu)",
}

TILE = dl.TileLayer(
    url="https://tiles.stadiamaps.com/tiles/alidade_satellite/{z}/{x}/{y}{r}.jpg",
    maxZoom=20,
    attribution="""\
&copy; CNES, Distribution Airbus DS, © Airbus DS, © PlanetObserver (Contains Copernicus Data)
| &copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a>
&copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a>
&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors
""",
)


# ------------------------------------------------------------------------------
# PLOTTING HELPERS
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# MAP TILES, COLOR MAP FOR FUEL TYPES, ETC.
# ------------------------------------------------------------------------------


def get_facilities(start_date, end_date):
    sy = datetime.strptime(start_date, "%Y-%m-%d").year
    ey = datetime.strptime(end_date, "%Y-%m-%d").year
    q = f"""
    SELECT "State","Facility Name","Unit ID","Primary Fuel Type","Latitude","Longitude","Year"
    FROM {constants.FACILITIES_TABLE}
    WHERE "Year" >= '{sy}' AND "Year" <= '{ey}'
      AND "Latitude" IS NOT NULL AND "Longitude" IS NOT NULL
    """
    df = query(
        q,
        columns_and_types=constants.FACILITIES_COLUMNS_AND_TYPES,
    )
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
        f"""\
      AND "Facility Name" = '{plant}'"""
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
    df = query(
        q,
        columns_and_types=constants.EMISSIONS_COLUMNS_AND_TYPES,
    )
    df["Datetime"] = df["Date"] + pd.to_timedelta(df["Hour"], unit="h")
    df = df.sort_values(["Facility Name", "Unit ID", "Datetime"]).drop(
        columns=["Date", "Hour"]
    )
    return df


# Precompute color map from an example "2023"

plants_geojson_data = get_geojson_data("2023-01-01", "2023-12-31")

# ------------------------------------------------------------------------------
# JS logic for circle color, tooltip, etc.
# ------------------------------------------------------------------------------
point_to_layer = assign(
    """
    function(feature, latlng, context){
        const circleOptions = {...context.hideout.circleOptions};
        const colorMap = context.hideout.colorMap;
        const fuelType = feature.properties.primary_fuel;
        circleOptions.fillColor = colorMap[fuelType] || "#333333";
        return L.circleMarker(latlng, circleOptions);
    }
    """
)
on_each_feature = assign(
    """
    function(feature, layer){
        layer.bindTooltip(
            feature.properties.state + ", " +
            feature.properties.facility_name + ", " +
            feature.properties.unit_id +
            "<br>" +
            feature.properties.primary_fuel
        );
    }
    """
)

geojson = dl.GeoJSON(
    id="us-map-markers",
    data={},
    zoomToBounds=False,
    pointToLayer=point_to_layer,
    onEachFeature=on_each_feature,
    hideout={
        "circleOptions": {"fillOpacity": 1, "stroke": False, "radius": 8},
        "colorMap": FUEL_COLORS_DICT,
    },
)


# ------------------------------------------------------------------------------
# DASH APP & LAYOUT
# ------------------------------------------------------------------------------
app = Dash(__name__)
app.title = "CAMPD Visualizer"

app.layout = html.Div(
    className="main-container",
    children=[
        html.H1("CAMPD Visualizer", className="header"),
        dcc.Store(id="date-store", data={"start_date": None, "end_date": None}),
        dcc.Store(id="plant-change-store", data={"fromMarker": False}),
        dcc.Store(id="state-change-store", data={"fromMarker": False}),
        dl.Map(
            id="us-map",
            center=[35, -100],
            zoom=4,
            className="map-container",
            children=[
                TILE,
                geojson,
            ],
        ),
        html.Div(
            className="graph-row",
            children=[
                html.Div(
                    className="input-panel",
                    children=[
                        html.Div(
                            className="input-elem",
                            children=[
                                html.Label("Start Date"),
                                html.Br(),
                                dcc.DatePickerSingle(
                                    id="start-date-picker",
                                    date=date(2023, 11, 1),
                                ),
                            ],
                        ),
                        html.Div(
                            className="input-elem",
                            children=[
                                html.Label("End Date"),
                                html.Br(),
                                dcc.DatePickerSingle(
                                    id="end-date-picker",
                                    date=date(2023, 12, 31),
                                ),
                            ],
                        ),
                        html.Div(
                            className="input-elem",
                            children=[
                                html.Label("Pollutant"),
                                dcc.Dropdown(
                                    className="dropdown",
                                    id="pollutant-dropdown",
                                    options=[
                                        {"label": k, "value": v}
                                        for k, v in POLLUTANT_OPTIONS.items()
                                    ],
                                    value="NOx Rate (lbs/mmBtu)",
                                ),
                            ],
                        ),
                        html.Div(
                            className="input-elem",
                            children=[
                                html.Label("State"),
                                dcc.Dropdown(
                                    className="dropdown",
                                    id="state-dropdown",
                                    options=[
                                        {"label": st, "value": st}
                                        for st in DISTINCT_STATES
                                    ],
                                    value="AL",
                                ),
                            ],
                        ),
                        html.Div(
                            className="input-elem",
                            children=[
                                html.Label("Plant"),
                                dcc.Dropdown(
                                    className="dropdown",
                                    id="plant-dropdown",
                                    options=[],
                                    value=None,
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="graph-container",
                    children=dcc.Graph(id="emissions-graph"),
                ),
            ],
        ),
    ],
)


# ------------------------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------------------------
@app.callback(
    Output("us-map-markers", "data"),
    Output("us-map", "center"),
    Output("date-store", "data"),
    Output("plant-change-store", "data"),
    [
        Input("plant-dropdown", "value"),
        Input("start-date-picker", "date"),
        Input("end-date-picker", "date"),
    ],
    State("state-dropdown", "value"),
    State("date-store", "data"),
    State("plant-change-store", "data"),
)
@timeit
def update_map(
    plant,
    start_date,
    end_date,
    state,
    stored_dates,
    click_bool,
):
    if click_bool["fromMarker"]:
        new_center = dash.no_update
        new_click_bool = {"fromMarker": False}
    elif plant and plant != "ALL" and state:
        q = f"""
        SELECT Latitude, Longitude
        FROM {constants.FACILITIES_TABLE}
        WHERE "State"='{state}' AND "Facility Name"='{plant}'
        """
        df = query(
            q,
            columns_and_types=constants.FACILITIES_COLUMNS_AND_TYPES,
        )
        lat = df["Latitude"].mean()
        lon = df["Longitude"].mean()
        new_center = [lat, lon]
        new_click_bool = dash.no_update
    else:
        new_center = dash.no_update
        new_click_bool = dash.no_update

    dates_changed = (start_date != stored_dates.get("start_date")) or (
        end_date != stored_dates.get("end_date")
    )
    if start_date and end_date and dates_changed:
        plants_geojson_data = get_geojson_data(start_date, end_date)
        new_data = plants_geojson_data
        new_stored_dates = {"start_date": start_date, "end_date": end_date}
    else:
        new_data = dash.no_update
        new_stored_dates = dash.no_update

    return new_data, new_center, new_stored_dates, new_click_bool


@app.callback(
    Output("plant-dropdown", "options"),
    Output("plant-dropdown", "value"),
    Output("state-change-store", "data"),
    [
        Input("state-dropdown", "value"),
        Input("start-date-picker", "date"),
        Input("end-date-picker", "date"),
    ],
    State("state-change-store", "data"),
)
@timeit
def update_dropdown(
    state_val,
    start_date,
    end_date,
    click_bool,
):
    if state_val and start_date and end_date:
        df = get_facilities(start_date, end_date)
        df_state = df[df["State"] == state_val]
        plants = sorted(df_state["Facility Name"].dropna().unique())
        if plants:
            new_plant = plants[0]
            plant_opts = [{"label": p, "value": p} for p in (["ALL"] + plants)]
        else:
            new_plant = None
            plant_opts = []
    else:
        new_plant = dash.no_update
        plant_opts = dash.no_update

    if (
        "start-date-picker" == callback_context.triggered_id
        or "end-date-picker" == callback_context.triggered_id
    ):
        new_plant = dash.no_update

    if click_bool["fromMarker"]:
        new_plant = dash.no_update
        new_click_bool = {"fromMarker": False}
    else:
        new_click_bool = dash.no_update

    return plant_opts, new_plant, new_click_bool


@app.callback(
    Output("plant-dropdown", "value", allow_duplicate=True),
    Output("state-dropdown", "value"),
    Output("plant-change-store", "data", allow_duplicate=True),
    Output("state-change-store", "data", allow_duplicate=True),
    [
        Input("us-map-markers", "clickData"),
    ],
    prevent_initial_call=True,
)
@timeit
def update_on_click(click_data):
    new_state, new_plant = [
        click_data["properties"][key] for key in ["state", "facility_name"]
    ]
    click_bool_plant = {"fromMarker": True}
    state_bool_plant = {"fromMarker": True}
    return new_plant, new_state, click_bool_plant, state_bool_plant


@app.callback(
    Output("emissions-graph", "figure"),
    [
        Input("pollutant-dropdown", "value"),
        Input("state-dropdown", "value"),
        Input("plant-dropdown", "value"),
        Input("start-date-picker", "date"),
        Input("end-date-picker", "date"),
    ],
)
@timeit
def update_emissions_graph(
    pollutant_col,
    selected_state,
    plant,
    start_date,
    end_date,
):
    if not (
        selected_state
        and plant is not None
        and start_date
        and end_date
        and pollutant_col
    ):
        fig = go.Figure()
        fig.update_layout(autosize=False, height=600, template="plotly_dark")
        return fig

    df = get_emissions(pollutant_col, selected_state, plant, start_date, end_date)
    if df.empty:
        title = "No data for this selection!"
        fig = go.Figure()
        fig.update_layout(
            title=title, autosize=False, height=600, template="plotly_dark"
        )
        return fig

    if plant == "ALL" or not plant:
        title = f"{pollutant_col} in {selected_state}"
        grouping = "Facility Name"
        df = df.groupby([grouping, "Datetime"], as_index=False).agg(
            {pollutant_col: "mean"}
        )
    else:
        title = f"{pollutant_col} at {plant}"
        df["Facility Full"] = df["Facility Name"] + ", " + df["Unit ID"].astype(str)
        grouping = "Facility Full"

    threshold = df["Datetime"].diff().mode().squeeze()
    total_points = df.shape[0]
    avg_window = 1
    if total_points > MAX_PLOTLY_POINTS:
        avg_window = int(np.ceil(total_points / MAX_PLOTLY_POINTS))
        title += f" (Averaging Window: {avg_window} points)"

    unique_keys = sorted(df[grouping].unique())
    palette = pcolors.qualitative.Plotly
    color_map = {k: palette[i % len(palette)] for i, k in enumerate(unique_keys)}

    fig = go.Figure()
    for i, (key, grp) in enumerate(df.groupby(grouping)):
        grp = grp.sort_values("Datetime")
        hovertemplate = f"Facility: {key}<br>Date: %{{x|%Y-%m-%d %H:%M}}<br>{pollutant_col}: %{{y}}<extra></extra>"
        if avg_window > 1:
            new_x, new_y = average_segments(
                grp["Datetime"].tolist(),
                grp[pollutant_col].tolist(),
                threshold,
                avg_window,
            )
        else:
            new_x, new_y = insert_gaps(
                grp["Datetime"].tolist(),
                grp[pollutant_col].tolist(),
                threshold,
            )
        new_x = pd.Series(new_x)
        new_y = pd.Series(new_y)
        trace = go.Scattergl(
            mode="lines",
            x=new_x,
            y=new_y,
            name=key,
            legendgroup=key,
            showlegend=True,
            line=dict(color=color_map[key]),
            hovertemplate=hovertemplate,
        )
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        autosize=False,
        height=600,
        template="plotly_dark",
        hoverlabel={"namelength": -1},
    )
    fig.update_xaxes(type="date")
    return fig


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
