"""Signal visualization utilities for resonance patterns.

This module provides functions for creating visualizations of signal patterns,
including combined signals and individual components.
"""

import plotly.graph_objects as go
import numpy as np
from typing import List, Union, Optional
import torch

def create_combined_signal_plot(
    signal: Union[torch.Tensor, np.ndarray],
    start_time: Optional[float] = None
) -> go.Figure:
    """Create a plot of the combined signal pattern.
    
    Args:
        signal: The signal data to plot
        start_time: Optional starting time point for continuous visualization
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()
    
    # Generate time points
    if start_time is not None:
        time = np.linspace(start_time, start_time + 2*np.pi, len(signal))
    else:
        time = np.arange(len(signal))
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=signal,
        mode='lines',
        name='Combined Signal',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title="Combined Wave Pattern",
        xaxis_title="Time (radians)" if start_time is not None else "Time Step",
        yaxis_title="Amplitude",
        height=200,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Add phase markers if using continuous time
    if start_time is not None:
        phase_points = np.arange(
            np.ceil(start_time / (2*np.pi)) * 2*np.pi,
            time[-1],
            2*np.pi
        )
        for phase_point in phase_points:
            fig.add_vline(
                x=phase_point,
                line_dash="dash",
                line_color="gray",
                opacity=0.5
            )
    
    return fig

def create_component_signals_plot(
    patterns: List[Union[torch.Tensor, np.ndarray]],
    frequencies: List[float],
    start_time: Optional[float] = None
) -> go.Figure:
    """Create a plot showing individual signal components.
    
    Args:
        patterns: List of signal patterns to plot
        frequencies: List of frequencies corresponding to each pattern
        start_time: Optional starting time point for continuous visualization
    """
    fig = go.Figure()
    
    # Generate time points
    if start_time is not None:
        time = np.linspace(start_time, start_time + 2*np.pi, len(patterns[0]))
    else:
        time = np.arange(len(patterns[0]))
    
    for i, pattern in enumerate(patterns):
        if isinstance(pattern, torch.Tensor):
            pattern = pattern.numpy()
            
        fig.add_trace(go.Scatter(
            x=time,
            y=pattern,
            mode='lines',
            name=f'f={frequencies[i]:.1f}Hz',
            line=dict(width=2)
        ))
        
    fig.update_layout(
        title="Individual Components",
        xaxis_title="Time (radians)" if start_time is not None else "Time Step",
        yaxis_title="Amplitude",
        height=200,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Add phase markers if using continuous time
    if start_time is not None:
        phase_points = np.arange(
            np.ceil(start_time / (2*np.pi)) * 2*np.pi,
            time[-1],
            2*np.pi
        )
        for phase_point in phase_points:
            fig.add_vline(
                x=phase_point,
                line_dash="dash",
                line_color="gray",
                opacity=0.5
            )
    
    return fig
