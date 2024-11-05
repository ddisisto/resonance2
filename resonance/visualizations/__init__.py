"""Visualization utilities for resonance patterns and oscillator dynamics.

This package provides functions for creating various visualizations of oscillator
behavior, organized into three main categories:

1. Signal Plots - For visualizing wave patterns and signal components
2. Training Plots - For visualizing training progress and model predictions
3. Phase Plots - For visualizing oscillator phase space relationships
"""

from .signal_plots import (
    create_combined_signal_plot,
    create_component_signals_plot
)

from .training_plots import (
    create_training_progress_plot,
    create_prediction_comparison_plot
)

from .phase_plots import (
    create_oscillator_outputs_plot,
    create_multi_phase_plot
)

# Signal visualization functions
__all__ = [
    # Signal plots
    'create_combined_signal_plot',
    'create_component_signals_plot',
    
    # Training plots
    'create_training_progress_plot',
    'create_prediction_comparison_plot',
    
    # Output plots
    'create_oscillator_outputs_plot',
    'create_multi_phase_plot'
]
