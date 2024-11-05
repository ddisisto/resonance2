"""
Utility functions for visualization and testing of resonance networks.

This module provides helper functions for visualizing patterns and testing
the behavior of pattern generators and resonance networks.
"""

import matplotlib.pyplot as plt
from typing import Optional
import torch
from .generators import PatternGenerator

def plot_pattern(pattern: torch.Tensor, title: Optional[str] = None) -> None:
    """Plot a pattern sequence.
    
    Args:
        pattern (torch.Tensor): The pattern to plot
        title (Optional[str]): Optional title for the plot
    """
    plt.figure(figsize=(10, 4))
    plt.plot(pattern.detach().numpy())
    if title:
        plt.title(title)
    plt.grid(True)
    plt.show()

def test_generator(generator: PatternGenerator, title: Optional[str] = None) -> None:
    """Test a pattern generator by generating and plotting a pattern.
    
    Args:
        generator (PatternGenerator): The pattern generator to test
        title (Optional[str]): Optional title for the plot
    """
    pattern = generator.generate()
    plot_pattern(pattern, title or f"Pattern from {generator.__class__.__name__}")
