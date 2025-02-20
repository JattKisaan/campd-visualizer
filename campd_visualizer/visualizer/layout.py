from datetime import date

import dash_leaflet as dl
from dash import dcc, html
from dash_extensions.javascript import assign

from helpers import DISTINCT_STATES, FUEL_COLORS_DICT, POLLUTANT_OPTIONS

# Define tile layer
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

layout = html.Div(
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
