"""Main Dash application module."""

from dash import Dash
import dash_bootstrap_components as dbc
from resonance.layouts import create_layout
from resonance.callbacks import register_callbacks

# Initialize the Dash app
app = Dash(__name__, 
          external_stylesheets=[dbc.themes.DARKLY],
          suppress_callback_exceptions=True)

# Set up the app layout
app.layout = create_layout()

# Register all callbacks
register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)
