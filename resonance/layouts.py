"""Layout components for the Dash application."""

import dash_bootstrap_components as dbc
from dash import html, dcc

# Initial state
n_components = 2
seq_length = 100
frequencies = [1.0, 2.0]
phases = [0.0, 0.0]

def create_sidebar():
    """Create the sidebar layout with all settings controls."""
    return dbc.Col(
        [
            dbc.Card([
                dbc.CardHeader("Settings"),
                dbc.CardBody([
                    html.H4("Pattern Generator Settings"),
                    dbc.Form([
                        dbc.Label("Number of Components"),
                        dbc.Input(
                            type="number",
                            id="n-components-input",
                            value=n_components,
                            min=1,
                            max=5
                        ),
                        html.Br(),
                        dbc.Label("Sequence Length"),
                        dbc.Input(
                            type="number",
                            id="seq-length-input",
                            value=seq_length,
                            min=10,
                            max=1000
                        ),
                        html.Br(),
                        html.H5("Components"),
                        html.Div(id="component-controls"),
                    ])
                ])
            ]),
            html.Br(),
            dbc.Card([
                dbc.CardHeader("Network Settings"),
                dbc.CardBody([
                    dbc.Label("Number of Oscillators"),
                    dbc.Input(
                        type="number",
                        id="n-oscillators-input",
                        value=3,
                        min=1,
                        max=10
                    ),
                    html.Br(),
                    dbc.Label("Pattern History Length"),
                    dbc.Input(
                        type="number",
                        id="pattern-length-input",
                        value=10,
                        min=5,
                        max=50
                    ),
                    html.Br(),
                    dbc.Label("Random Seed"),
                    dbc.Input(
                        type="number",
                        id="random-seed-input",
                        value=42,
                        min=0
                    ),
                    html.Br(),
                    dbc.Label("Number of Epochs"),
                    dbc.Input(
                        type="number",
                        id="n-epochs-input",
                        value=10,
                        min=1,
                        max=100
                    ),
                    html.Br(),
                    dbc.Label("Learning Rate"),
                    dbc.Input(
                        type="number",
                        id="learning-rate-input",
                        value=0.01,
                        min=0.0001,
                        max=0.1,
                        step=0.0001
                    ),
                    html.Br(),
                    dbc.Button("Train Network", id="train-button", color="primary")
                ])
            ])
        ],
        width=3,
    )

def create_content():
    """Create the main content area with all visualization tabs."""
    return dbc.Col([
        dbc.Tabs([
            dbc.Tab([
                dcc.Graph(id='combined-pattern')
            ], label="Combined"),
            dbc.Tab([
                dcc.Graph(id='individual-outputs')
            ], label="Components"),
            dbc.Tab([
                dcc.Graph(id='training-progress'),
                dcc.Graph(id='prediction-comparison'),
                dcc.Graph(id='oscillator-outputs'),
                dcc.Graph(id='phase-space')
            ], label="Training Results")
        ])
    ], width=9)

def create_layout():
    """Create the main application layout."""
    return dbc.Container([
        dbc.Row([create_sidebar(), create_content()]),
        # Store for component parameters
        dcc.Store(id='component-params', data={
            'frequencies': frequencies,
            'phases': phases
        }),
        # Store for training results
        dcc.Store(id='training-results', data=None)
    ], fluid=True)
