"""Phase space visualization utilities for resonance patterns.

This module provides functions for creating phase space visualizations,
including 3D and multi-oscillator phase plots.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Optional, Tuple

def create_oscillator_outputs_plot(states_history: List[List[float]], n_oscillators: int) -> go.Figure:
    """Create a plot showing individual oscillator outputs over time."""
    fig = go.Figure()
    
    # Extract states for each oscillator
    for i in range(n_oscillators):
        oscillator_states = [states[i] for states in states_history]
        
        fig.add_trace(go.Scatter(
            y=oscillator_states,
            mode='lines',
            name=f'Oscillator {i+1}',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        xaxis_title="Time Step",
        yaxis_title="State",
        height=200,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_multi_phase_plot(
    states_history: List[List[float]],
    n_oscillators: int,
    show_colorbar: bool = True
) -> go.Figure:
    """Create a grid of 2D phase space plots for all oscillator pairs."""
    # Calculate grid dimensions
    n_plots = (n_oscillators * (n_oscillators - 1)) // 2
    grid_size = int(np.ceil(np.sqrt(n_plots)))
    
    # Create subplot titles with more context
    subplot_titles = [
        f'Phase Space: Osc {i+1} vs {j+1}'
        for i in range(n_oscillators-1)
        for j in range(i+1, n_oscillators)
    ]
    
    fig = make_subplots(
        rows=grid_size,
        cols=grid_size,
        subplot_titles=subplot_titles
    )
    
    # Create normalized time values for color gradient
    time_values = np.linspace(0, 1, len(states_history))
    
    # Plot each oscillator pair
    plot_idx = 1
    for i in range(n_oscillators-1):
        for j in range(i+1, n_oscillators):
            # Calculate grid position
            row = ((plot_idx-1) // grid_size) + 1
            col = ((plot_idx-1) % grid_size) + 1
            
            # Extract states for the two oscillators
            osc1_states = [states[i] for states in states_history]
            osc2_states = [states[j] for states in states_history]
            
            # Add solid color line for trajectory
            fig.add_trace(
                go.Scatter(
                    x=osc1_states,
                    y=osc2_states,
                    mode='lines',
                    line=dict(
                        width=3,
                        color='rgba(100, 100, 100, 0.3)'  # Semi-transparent gray
                    ),
                    showlegend=False
                ),
                row=row,
                col=col
            )
            
            # Add markers with time-based color gradient
            fig.add_trace(
                go.Scatter(
                    x=osc1_states,
                    y=osc2_states,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=time_values,
                        colorscale='Viridis',
                        showscale=show_colorbar and col == grid_size and row == 1,
                        cmin=0,
                        cmax=1,
                        colorbar=dict(
                            title="Time Progression",
                            thickness=10,
                            len=0.3,
                            x=1.1 if col == grid_size else None,
                            y=0.5 if row == 1 else None,
                            ticktext=['Start', 'End'],
                            tickvals=[0, 1]
                        ) if show_colorbar and col == grid_size and row == 1 else None
                    ),
                    showlegend=False
                ),
                row=row,
                col=col
            )
            
            plot_idx += 1
    
    # Update layout
    fig.update_layout(
        height=200 * grid_size,
        width=200 * grid_size,
        showlegend=False,
        margin=dict(l=0, r=100 if show_colorbar else 0, t=40, b=0)
    )
    
    # Update all subplot axes to have the same range
    for i in range(1, plot_idx):
        fig.update_xaxes(range=[-1.1, 1.1], row=((i-1)//grid_size)+1, col=((i-1)%grid_size)+1)
        fig.update_yaxes(range=[-1.1, 1.1], row=((i-1)//grid_size)+1, col=((i-1)%grid_size)+1)
    
    return fig
