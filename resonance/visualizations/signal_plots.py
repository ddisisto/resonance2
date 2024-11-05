"""Signal visualization utilities for resonance patterns.

This module provides functions for creating visualizations of signal patterns,
including combined signals and individual components.
"""

import plotly.graph_objects as go
import numpy as np
from typing import List, Union
import torch

def create_combined_signal_plot(signal: Union[torch.Tensor, np.ndarray]) -> go.Figure:
    """Create a plot of the combined signal pattern."""
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=signal,
        mode='lines',
        name='Combined Signal',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title="Combined Wave Pattern",
        xaxis_title="Time Step",
        yaxis_title="Amplitude",
        height=200,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_component_signals_plot(
    patterns: List[Union[torch.Tensor, np.ndarray]],
    frequencies: List[float]
) -> go.Figure:
    """Create a plot showing individual signal components."""
    fig = go.Figure()
    
    for i, pattern in enumerate(patterns):
        if isinstance(pattern, torch.Tensor):
            pattern = pattern.numpy()
            
        fig.add_trace(go.Scatter(
            y=pattern,
            mode='lines',
            name=f'f={frequencies[i]:.1f}Hz',
            line=dict(width=2)
        ))
        
    fig.update_layout(
        title="Individual Components",
        xaxis_title="Time Step",
        yaxis_title="Amplitude",
        height=200,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig
