"""Training-related callbacks for the resonance application."""

import torch
import torch.optim as optim
from dash import Input, Output, State, html, dcc
from dash.exceptions import PreventUpdate
from resonance.generators import SineWaveGenerator
from resonance.models import ResonanceNetwork
from resonance.visualizations import (
    create_training_progress_plot,
    create_prediction_comparison_plot,
    create_oscillator_outputs_plot,
    create_multi_phase_plot
)

def register_training_callbacks(app):
    """Register training-related callbacks with the Dash app."""
    
    # Add interval for progress updates
    app.layout.children.append(dcc.Interval(
        id='training-interval',
        interval=100,
        disabled=True
    ))
    
    @app.callback(
        [Output('training-interval', 'disabled')],
        [Input('training-progress-store', 'data')]
    )
    def toggle_interval(progress_data):
        if not progress_data:
            return [True]
        return [not progress_data.get('is_training', False)]
    
    @app.callback(
        [Output('training-progress-bar', 'style'),
         Output('training-progress-text', 'style'),
         Output('training-progress-bar', 'value'),
         Output('training-progress-text', 'children')],
        [Input('training-interval', 'n_intervals'),
         Input('training-progress-store', 'data')]
    )
    def update_progress(n_intervals, progress_data):
        if not progress_data:
            raise PreventUpdate
            
        is_training = progress_data.get('is_training', False)
        progress = progress_data.get('progress', 0)
        
        if not is_training:
            return (
                {'height': '20px', 'visibility': 'hidden'}, 
                {'textAlign': 'center', 'marginTop': '5px', 'visibility': 'hidden'}, 
                0, 
                ''
            )
        
        return (
            {'height': '20px', 'visibility': 'visible'}, 
            {'textAlign': 'center', 'marginTop': '5px', 'visibility': 'visible'}, 
            progress, 
            f'Training Progress: {progress}%'
        )
    
    @app.callback(
        [Output('training-results', 'data'),
         Output('training-progress-store', 'data')],
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
            raise PreventUpdate
            
        # Generate patterns
        patterns = []
        for i in range(min(n_comps, len(params['frequencies']))):
            gen = SineWaveGenerator(seq_len, params['frequencies'][i], params['phases'][i])
            patterns.append(gen.generate())
        
        if not patterns:
            raise PreventUpdate
            
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
        
        # Start training
        progress_data = {'progress': 0, 'is_training': True}
        
        for epoch in range(n_epochs):
            loss = network.train_step(combined_pattern, optimizer)
            epoch_losses.append(loss)
            
            # Update progress
            progress_data = {
                'progress': int((epoch + 1) / n_epochs * 100),
                'is_training': True
            }
        
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
        
        training_results = {
            'loss': loss,
            'epoch_losses': epoch_losses,
            'predictions': predictions,
            'states_history': states_history,
            'n_oscillators': n_oscillators,
            'random_seed': random_seed
        }
        
        # Training complete
        progress_data = {'progress': 100, 'is_training': False}
        
        return training_results, progress_data

    @app.callback(
        [Output('training-progress-plot', 'figure'),
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
