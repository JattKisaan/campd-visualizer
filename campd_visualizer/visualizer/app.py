# import cProfile
import os
import time
from datetime import date, datetime

import campd_visualizer.pkg.constants as constants
import dash
import dash_leaflet as dl
import duckdb
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html
from dash_extensions.javascript import assign
from plotly import colors as pcolors

os.environ["PYTHONBREAKPOINT"] = "IPython.embed"


# import pstats
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

DF_ALL_FAC = query(
    f"""
    SELECT "State","Facility Name","Unit ID","Primary Fuel Type","Latitude","Longitude","Year"
    FROM {constants.FACILITIES_TABLE}""",
    columns_and_types=constants.FACILITIES_COLUMNS_AND_TYPES,
)

q = """
SELECT DISTINCT "State"
FROM DF_ALL_FAC
"""
DISTINCT_STATES = np.sort(query(q).values.squeeze(1))

DISTINCT_FUELS = np.sort(
    query(
        """SELECT DISTINCT "Primary Fuel Type"
              FROM DF_ALL_FAC
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


def process_grp_vectorized(grp, avg_window, pollutant_col):
    # Convert threshold to nanoseconds (int64)
    # This is one hour in nanoseconds, as you can get from
    # grp["Datetime"].diff().mode().astype('int64').iloc[0]
    threshold_ns = 3600000000000

    _times = grp["Datetime"].to_numpy().astype("int64")
    _vals = grp[pollutant_col].to_numpy().astype(float)

    n_total = len(_times)
    diffs = np.diff(_times)

    gap_indices = np.concatenate(
        [
            [0],
            np.where(diffs > threshold_ns)[0] + 1,
            [n_total],
        ]
    )
    window_indices = [
        [int(g1), int(g2)] for g1, g2 in zip(gap_indices[:-1], gap_indices[1:])
    ]
    if avg_window > 1:
        break_indices = [
            list(range(i1, i2, avg_window)) + [i2] for i1, i2 in window_indices
        ]
        ## Just for reference, I tried this out too but there's no speed difference.
        # times = np.concatenate(
        #    [
        #        np.concatenate(
        #            [
        #                (
        #                    np.add.reduceat(
        #                        _times[indices[0] : indices[-1]].astype(np.float64),
        #                        (np.array(indices) - indices[0])[:-1],
        #                    )
        #                    / np.diff(indices)
        #                ).astype(np.int64),
        #                [np.nan],
        #            ]
        #        )
        #        for indices in break_indices
        #    ]
        # )[:-1]

        result = []
        for indices in break_indices:
            use_indices = np.array(indices) - indices[0]
            temp = _times[indices[0] : indices[-1]]
            temp = (
                np.add.reduceat(
                    temp.astype(np.float64),
                    use_indices[:-1],
                )
                / np.diff(use_indices)
            ).astype(np.int64)
            result.append(temp)
            result.append([pd.NaT])

        result.pop()
        times = pd.to_datetime(np.concatenate(result))

        result = []
        for indices in break_indices:
            use_indices = np.array(indices) - indices[0]
            temp = _vals[indices[0] : indices[-1]]
            temp = np.add.reduceat(
                temp,
                use_indices[:-1],
            ) / np.diff(use_indices)
            result.append(temp)
            result.append([np.nan])

        result.pop()
        vals = np.concatenate(result)

    else:
        insert_indices = gap_indices[1:-1]

        times = pd.to_datetime(
            np.insert(
                np.array(_times, dtype="datetime64[ns]"),
                insert_indices,
                [None] * len(insert_indices),
            )
        )
        vals = np.insert(_vals, insert_indices, [np.nan] * len(insert_indices))

    new_grp = pd.DataFrame({"Datetime": times, pollutant_col: vals})
    return new_grp


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


# @timeit
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
    FROM DF_ALL_FAC
    WHERE "Year" >= '{sy}'
      AND "Year" <= '{ey}'
      AND "Latitude" IS NOT NULL
      AND "Longitude" IS NOT NULL
    """
    df = query(q)
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
    SELECT "State", "Facility Name", "Unit ID", "Date", "Hour", "{pollutant_col}"
    FROM {constants.EMISSIONS_TABLE}
    WHERE "State" = '{state}'
      AND "Year" >= {sy}
      AND "Year" <= {ey}
      AND "Operating Time" > 0.0
      AND "Date" >= '{start_date}' AND "Date" <= '{end_date}'
      AND "{pollutant_col}" IS NOT NULL {plant_str}
    """
    df = query(q)
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


# Define the bare tile layer
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

# JS logic for circle color, tooltip, etc.
point_to_layer = assign(
    """
    function(feature, latlng, context){
        // Make a copy of the circle options
        const circleOptions = {...context.hideout.circleOptions};
        // Pull out the colorMap we passed in via "hideout"
        const colorMap = context.hideout.colorMap;
        const fuelType = feature.properties.primary_fuel;

        // If the fuel type is in our map, use that color; otherwise a fallback
        circleOptions.fillColor = colorMap[fuelType] || "#333333";

        // Return a Leaflet circle marker
        return L.circleMarker(latlng, circleOptions);
    }
    """
)

# For a tooltip, we can keep it simple:
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

# Empty data by default; will be populated in callbacks
geojson = dl.GeoJSON(
    id="us-map-markers",
    data=None,
    zoomToBounds=False,
    pointToLayer=point_to_layer,
    onEachFeature=on_each_feature,
    hideout={
        "circleOptions": {"fillOpacity": 1, "stroke": False, "radius": 8},
        "colorMap": FUEL_COLORS_DICT,
    },
)


app = Dash(__name__)
app.title = "CAMPD Visualizer"

app.layout = html.Div(
    className="main-container",
    children=[
        html.H1("CAMPD Visualizer", className="header"),
        # dcc.Store elements
        dcc.Store(id="date-store", data={"start_date": None, "end_date": None}),
        dcc.Store(id="plant-change-store", data={"fromMarker": False}),
        dcc.Store(id="state-change-store", data={"fromMarker": False}),
        # Map
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
                                    date=date(2022, 11, 1),
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
        FROM DF_ALL_FAC
        WHERE "State"='{state}' AND "Facility Name"='{plant}'
        """
        df = query(q)
        lat = df["Latitude"].mean()
        lon = df["Longitude"].mean()
        new_center = [lat, lon]
        new_click_bool = dash.no_update
    else:
        new_center = dash.no_update
        new_click_bool = dash.no_update

    # Map Children
    dates_changed = (start_date != stored_dates.get("start_date")) or (
        end_date != stored_dates.get("end_date")
    )

    if start_date and end_date and dates_changed:
        new_plants_geojson_data = get_geojson_data(start_date, end_date)
        new_stored_dates = {"start_date": start_date, "end_date": end_date}
    else:
        new_plants_geojson_data = dash.no_update
        new_stored_dates = dash.no_update

    print(
        f"""
    ### Inputs
    plant = {plant}
    start_date = {start_date}
    end_date = {end_date}
    state = {state}
    stored_dates = {stored_dates}
    click_bool = {click_bool}
    ### Outputs
    new_plants_geojson_data = TOO LONG
    new_center = {new_center}
    new_stored_dates = {new_stored_dates}
    new_click_bool = {new_click_bool}
    """
    )
    return new_plants_geojson_data, new_center, new_stored_dates, new_click_bool


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

    print(
        f"""
    ### Inputs
    state_val = {state_val}
    start_date = {start_date}
    end_date = {end_date}
    click_bool = {click_bool}
    ### Outputs
    plant-dropdown.options = TOO LONG
    plant-dropdown.value = {new_plant}
    state-change-store = {new_click_bool}
    """
    )
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
    print(
        f"""
    new_plant = {new_plant}
    new_state = {new_state}
    click_bool_plant = {click_bool_plant}
    state_bool_plant = {state_bool_plant}
    """
    )
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
    """
    An updated version of update_emissions_graph that, for each facility (or grouping),
    converts its sorted data into a NumPy array and then applies the vectorized segmentation
    and averaging logic via process_grp_vectorized.

    This approach reduces DataFrame overhead by minimizing per-row and per-group Python-level
    operations and should help bring the speed closer to (or even surpass) that of the first approach.
    """
    print(
        f"""
    pollutant_col = {pollutant_col}
    selected_state = {selected_state}
    plant = {plant}
    start_date = {start_date}
    end_date = {end_date}
    """
    )
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
    print(df.shape)
    if df.empty:
        title = "No data for this selection!"
        fig = go.Figure()
        fig.update_layout(
            title=title,
            autosize=False,
            height=600,
            template="plotly_dark",
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

    # Use the mode of the datetime differences as the threshold for a gap.
    total_points = df.shape[0]
    avg_window = 1
    if total_points > MAX_PLOTLY_POINTS:
        print(f"HAVE TO THIN DATA: {total_points} > {MAX_PLOTLY_POINTS}")
        avg_window = int(np.ceil(total_points / MAX_PLOTLY_POINTS))
        title += f" (Averaging Window: {avg_window} points)"

    unique_keys = sorted(df[grouping].unique())
    palette = pcolors.qualitative.Plotly
    color_map = {k: palette[i % len(palette)] for i, k in enumerate(unique_keys)}

    fig = go.Figure()
    counter = 0
    for key, grp in df.groupby(grouping):
        print(counter)
        grp = grp.sort_values("Datetime")
        hovertemplate = (
            f"Facility: {key}<br>Date: %{{x|%Y-%m-%d %H:%M}}<br>{pollutant_col}: %{{y}}"
            "<extra></extra>"
        )
        # Use our vectorized function to process the group's data.
        new_grp = process_grp_vectorized(grp, avg_window, pollutant_col)
        trace = go.Scattergl(
            mode="lines",
            x=new_grp["Datetime"],
            y=new_grp[pollutant_col],
            name=key,
            legendgroup=key,
            showlegend=True,
            line=dict(color=color_map[key]),
            hovertemplate=hovertemplate,
        )
        fig.add_trace(trace)
        counter += 1

    fig.update_layout(
        title=title,
        autosize=False,
        height=600,
        template="plotly_dark",
        hoverlabel={"namelength": -1},
    )
    fig.update_xaxes(type="date")
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
    # profiler = cProfile.Profile()
    # profiler.enable()

    # pollutant_col = "NOx Rate (lbs/mmBtu)"
    # selected_state = "CA"
    # plant = "ALL"
    # start_date = "1995-01-01"
    # end_date = "2025-12-31"

    # try:
    #    fig = update_emissions_graph(
    #        pollutant_col,
    #        selected_state,
    #        plant,
    #        start_date,
    #        end_date,
    #    )
    #    fig.show()

    # finally:
    #    pass
    #    profiler.disable()
    #    profiler.dump_stats("profile_results.prof")

    #    stats = pstats.Stats(profiler)
    #    stats.strip_dirs().sort_stats("cumulative").print_stats(50)
