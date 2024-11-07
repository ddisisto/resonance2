"""Training visualization utilities for resonance patterns.

This module provides functions for creating visualizations of training progress
and model predictions.
"""

import plotly.graph_objects as go
import numpy as np
from typing import List, Union
import torch

def create_training_progress_plot(losses: List[float]) -> go.Figure:
    """Create a plot showing training loss over time."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=losses,
        mode='lines',
        name='Loss',
        line=dict(color='#2ca02c', width=2)
    ))
    
    fig.update_layout(
        title=f"Training Loss Over Time (Final: {losses[-1]:.5f})",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",  # Set y-axis to logarithmic scale
        height=250,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_prediction_comparison_plot(
    actual: Union[torch.Tensor, np.ndarray],
    predicted: List[float]
) -> go.Figure:
    """Create a plot comparing actual vs predicted signals."""
    if isinstance(actual, torch.Tensor):
        actual = actual.numpy()
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        y=predicted,
        mode='lines',
        name='Predicted',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.update_layout(
        title="Model Predictions vs Actual Signal",
        xaxis_title="Time Step",
        yaxis_title="Amplitude",
        height=250,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig
