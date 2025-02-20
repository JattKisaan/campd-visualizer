import os

from dash import Dash

from callbacks import register_callbacks
from layout import layout

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


app = Dash(__name__)
app.title = "CAMPD Visualizer"
app.layout = layout

# Register callbacks after the app is created and layout is set
register_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True)
