import campd_visualizer.pkg.constants as constants
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, no_update
from plotly import colors as pcolors

from helpers import (MAX_PLOTLY_POINTS, average_segments, get_emissions,
                     get_facilities, get_geojson_data, insert_gaps, query,
                     timeit)


def register_callbacks(app):

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
            new_center = no_update
            new_click_bool = {"fromMarker": False}
        elif plant and plant != "ALL" and state:
            q = f"""
            SELECT Latitude, Longitude
            FROM {constants.FACILITIES_TABLE}
            WHERE "State"='{state}' AND "Facility Name"='{plant}'
            """
            df = query(q, columns_and_types=constants.FACILITIES_COLUMNS_AND_TYPES)
            lat = df["Latitude"].mean()
            lon = df["Longitude"].mean()
            new_center = [lat, lon]
            new_click_bool = no_update
        else:
            new_center = no_update
            new_click_bool = no_update

        dates_changed = (start_date != stored_dates.get("start_date")) or (
            end_date != stored_dates.get("end_date")
        )
        if start_date and end_date and dates_changed:
            new_data = get_geojson_data(start_date, end_date)
            new_stored_dates = {"start_date": start_date, "end_date": end_date}
        else:
            new_data = no_update
            new_stored_dates = no_update

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
            new_plant = no_update
            plant_opts = no_update

        if (
            "start-date-picker" == callback_context.triggered_id
            or "end-date-picker" == callback_context.triggered_id
        ):
            new_plant = no_update

        if click_bool["fromMarker"]:
            new_plant = no_update
            new_click_bool = {"fromMarker": False}
        else:
            new_click_bool = no_update

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
