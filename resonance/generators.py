"""
Pattern generators for resonance networks.

This module provides base and concrete implementations of various pattern generators
that can be used to create input sequences for resonance networks.
"""

import torch
import numpy as np
from typing import Optional

class PatternGenerator:
    """Base class for pattern generators.
    
    Provides a common interface for generating sequences of patterns that can be
    used as input to resonance networks.
    
    Args:
        seq_length (int): The length of sequences to generate
    """
    
    def __init__(self, seq_length: int):
        self.seq_length = seq_length
        self.current_time = 0.0  # Track current time point
    
    def generate(self) -> torch.Tensor:
        """Generate a pattern sequence.
        
        Returns:
            torch.Tensor: A tensor of shape (seq_length,) containing the generated pattern
        """
        raise NotImplementedError("Subclasses must implement generate()")

class SineWaveGenerator(PatternGenerator):
    """Generates sinusoidal wave patterns.
    
    Creates sine wave patterns with configurable frequency and phase.
    Maintains continuity across multiple generations.
    
    Args:
        seq_length (int): The length of sequences to generate
        frequency (float): The frequency of the sine wave
        phase (float, optional): Initial phase offset of the wave. Defaults to 0.0
    """
    
    def __init__(self, seq_length: int, frequency: float, phase: float = 0.0):
        super().__init__(seq_length)
        self.frequency = frequency
        self.phase = phase
    
    def generate(self) -> torch.Tensor:
        """Generate a sine wave pattern continuing from the last time point.
        
        Returns:
            torch.Tensor: A tensor of shape (seq_length,) containing the sine wave pattern
        """
        # Generate time points continuing from current_time
        t = torch.linspace(
            self.current_time,
            self.current_time + 2*np.pi,
            self.seq_length
        )
        # Update current_time for next generation
        self.current_time += 2*np.pi
        
        return torch.sin(self.frequency * t + self.phase)
