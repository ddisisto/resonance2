"""Callback functions for the Dash application."""

import torch
import torch.optim as optim
import numpy as np
from dash import Input, Output, State, ALL
from resonance.generators import SineWaveGenerator
from resonance.models import ResonanceNetwork
from resonance.visualizations import (
    create_combined_signal_plot,
    create_component_signals_plot,
    create_training_progress_plot,
    create_prediction_comparison_plot,
    create_multi_phase_plot,
    create_oscillator_outputs_plot
)
import dash_bootstrap_components as dbc
from dash import html

def register_callbacks(app):
    """Register all callbacks with the Dash app."""
    
    @app.callback(
        Output("component-controls", "children"),
        [Input("n-components-input", "value")],
        [State("component-params", "data")]
    )
    def update_component_controls(n_comps, stored_params):
        if not n_comps:
            return []
        
        stored_freqs = stored_params.get('frequencies', [])
        stored_phases = stored_params.get('phases', [])
        
        controls = []
        for i in range(n_comps):
            freq_id = {'type': 'freq', 'index': i}
            phase_id = {'type': 'phase', 'index': i}
            
            # Use stored values if available, otherwise use defaults
            freq_value = stored_freqs[i] if i < len(stored_freqs) else float(i + 1)
            phase_value = stored_phases[i] if i < len(stored_phases) else 0.0
            
            row = dbc.Row([
                dbc.Col(html.Span(f"Component {i+1}"), width=4),
                dbc.Col(
                    dbc.Input(
                        type="number",
                        id=freq_id,
                        value=freq_value,
                        min=0.1,
                        max=10.0,
                        step=0.1,
                        placeholder="Frequency"
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Input(
                        type="number",
                        id=phase_id,
                        value=phase_value,
                        min=0.0,
                        max=2*np.pi,
                        step=0.1,
                        placeholder="Phase"
                    ),
                    width=4
                )
            ])
            controls.append(row)
            controls.append(html.Br())
        
        return controls

    @app.callback(
        Output('component-params', 'data'),
        [Input({'type': 'freq', 'index': ALL}, 'value'),
         Input({'type': 'phase', 'index': ALL}, 'value')],
        [State('component-params', 'data')]
    )
    def update_component_params(freqs, phases, stored_params):
        if not freqs or not phases:
            return stored_params
        
        # Get current stored values
        stored_freqs = stored_params.get('frequencies', [])
        stored_phases = stored_params.get('phases', [])
        
        # Update only changed values while preserving others
        new_freqs = list(stored_freqs)
        new_phases = list(stored_phases)
        
        # Extend lists if needed
        max_len = max(len(freqs), len(phases))
        while len(new_freqs) < max_len:
            new_freqs.append(float(len(new_freqs) + 1))
        while len(new_phases) < max_len:
            new_phases.append(0.0)
        
        # Update with new values
        for i, (freq, phase) in enumerate(zip(freqs, phases)):
            if freq is not None:
                new_freqs[i] = freq
            if phase is not None:
                new_phases[i] = phase
        
        return {
            'frequencies': new_freqs,
            'phases': new_phases
        }

    @app.callback(
        [Output("combined-pattern", "figure"),
         Output("individual-outputs", "figure")],
        [Input("n-components-input", "value"),
         Input("seq-length-input", "value"),
         Input("component-params", "data")]
    )
    def update_patterns(n_comps, seq_len, params):
        if not n_comps or not seq_len or not params:
            return {}, {}
        
        freqs = params['frequencies']
        phases = params['phases']
        
        # Generate patterns
        patterns = []
        valid_freqs = []
        
        for i in range(min(n_comps, len(freqs))):
            if freqs[i] is not None and phases[i] is not None:
                gen = SineWaveGenerator(seq_len, freqs[i], phases[i])
                patterns.append(gen.generate())
                valid_freqs.append(freqs[i])
        
        if not patterns:
            return {}, {}
        
        # Create combined pattern
        combined_pattern = sum(patterns)
        
        # Create figures
        combined_fig = create_combined_signal_plot(combined_pattern)
        components_fig = create_component_signals_plot(patterns, valid_freqs)
        
        return combined_fig, components_fig

    @app.callback(
        Output('training-results', 'data'),
        [Input('train-button', 'n_clicks')],
        [State('n-oscillators-input', 'value'),
         State('pattern-length-input', 'value'),
         State('random-seed-input', 'value'),
         State('n-epochs-input', 'value'),
         State('learning-rate-input', 'value'),
         State('n-components-input', 'value'),
         State('seq-length-input', 'value'),
         State('component-params', 'data')]
    )
    def train_network(n_clicks, n_oscillators, pattern_length, random_seed, 
                     n_epochs, learning_rate, n_comps, seq_len, params):
        if not n_clicks:
            return None
            
        # Generate patterns
        patterns = []
        for i in range(min(n_comps, len(params['frequencies']))):
            gen = SineWaveGenerator(seq_len, params['frequencies'][i], params['phases'][i])
            patterns.append(gen.generate())
        
        if not patterns:
            return None
            
        combined_pattern = sum(patterns)
        
        # Initialize network and optimizer
        network = ResonanceNetwork(
            input_size=1,
            n_oscillators=n_oscillators,
            pattern_length=pattern_length,
            seed=random_seed
        )
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        
        # Training loop
        epoch_losses = []
        states_history = []
        prev_states = torch.zeros(n_oscillators)
        
        for epoch in range(n_epochs):
            loss = network.train_step(combined_pattern, optimizer)
            epoch_losses.append(loss)
        
        # Generate predictions and collect states
        predictions = []
        prev_states = torch.zeros(n_oscillators)
        
        with torch.no_grad():
            for t in range(len(combined_pattern) - 1):
                x = combined_pattern[t].unsqueeze(0)
                _, new_states, pred = network(x, prev_states)
                predictions.append(pred.item())
                states_history.append(new_states.tolist())
                prev_states = new_states
        
        return {
            'loss': loss,
            'epoch_losses': epoch_losses,
            'predictions': predictions,
            'states_history': states_history,
            'n_oscillators': n_oscillators,
            'random_seed': random_seed
        }

    @app.callback(
        [Output('training-progress', 'figure'),
         Output('prediction-comparison', 'figure'),
         Output('oscillator-outputs', 'figure'),
         Output('phase-space', 'figure')],
        [Input('training-results', 'data'),
         Input('n-components-input', 'value'),
         Input('seq-length-input', 'value'),
         Input('component-params', 'data')]
    )
    def update_training_plots(results, n_comps, seq_len, params):
        if not results or not n_comps or not seq_len or not params:
            return {}, {}, {}, {}
        
        # Generate current pattern for comparison
        patterns = []
        for i in range(min(n_comps, len(params['frequencies']))):
            gen = SineWaveGenerator(seq_len, params['frequencies'][i], params['phases'][i])
            patterns.append(gen.generate())
        
        if not patterns:
            return {}, {}, {}, {}
            
        combined_pattern = sum(patterns)
        
        # Create training progress plot
        progress_fig = create_training_progress_plot(results['epoch_losses'])
        
        # Create prediction comparison plot
        prediction_fig = create_prediction_comparison_plot(
            combined_pattern[1:],
            results['predictions']
        )
        
        # Create oscillator outputs plot
        outputs_fig = create_oscillator_outputs_plot(
            results['states_history'],
            results['n_oscillators']
        )
        
        # Create phase space plot
        phase_fig = create_multi_phase_plot(
            results['states_history'],
            results['n_oscillators']
        )
        
        return progress_fig, prediction_fig, outputs_fig, phase_fig
