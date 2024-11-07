"""Neural network models for resonance-based pattern processing.

This module implements neural network architectures designed to process and generate
patterns through resonance mechanisms.
"""

import torch
import torch.nn as nn
from collections import deque
from typing import List, Tuple, Deque

class ResonanceNetwork(nn.Module):
    """A neural network model that processes patterns through resonant oscillators.
    
    This network combines input patterns with internal oscillator states to process
    temporal patterns. It maintains a history of oscillator states and uses them
    to influence future predictions through emergent feedback dynamics.
    
    Args:
        input_size (int): Size of the input vector
        n_oscillators (int): Number of internal oscillators
        pattern_length (int): Length of pattern history to maintain
        hidden_size (int, optional): Size of hidden layers. Defaults to 32
        dropout_rate (float, optional): Dropout probability. Defaults to 0.0
        noise_level (float, optional): Standard deviation of noise during training. Defaults to 0.0
        seed (int, optional): Random seed for reproducibility. Defaults to None
    """
    
    def __init__(
        self,
        input_size: int,
        n_oscillators: int,
        pattern_length: int,
        hidden_size: int = 32,
        dropout_rate: float = 0.0,
        noise_level: float = 0.0,
        seed: int = None
    ):
        super().__init__()
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            
        self.n_oscillators = n_oscillators
        self.pattern_length = pattern_length
        self.seed = seed
        self.noise_level = noise_level
        
        # Initialize learnable scaling factors with noise
        self.oscillator_scales = nn.Parameter(
            torch.ones(n_oscillators) + 0.1 * torch.randn(n_oscillators)
        )
        
        # Initialize pattern history buffers for each oscillator
        self.pattern_buffers: List[Deque[float]] = [
            deque(maxlen=pattern_length) for _ in range(n_oscillators)
        ]
        
        # Initialize buffers with seeded random values
        self._reset_pattern_buffers()
        
        self.hidden_size = hidden_size
        
        # Oscillator state generation - now with dropout
        self.oscillator_net = nn.Sequential(
            nn.Linear(input_size + n_oscillators, self.hidden_size),
            # nn.Linear(input_size + n_oscillators + n_oscillators, self.hidden_size), # Added resonance context
            nn.Dropout(dropout_rate),
            nn.Tanh(),
            nn.Linear(self.hidden_size, n_oscillators),
            nn.Dropout(dropout_rate),
            nn.Tanh()
        )
        
        # Prediction network with resonance context
        self.prediction_net = nn.Sequential(
            nn.Linear(n_oscillators, self.hidden_size),
            # nn.Linear(n_oscillators + n_oscillators, self.hidden_size),  # Added resonance context
            nn.Dropout(dropout_rate),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )

    def _reset_pattern_buffers(self):
        """Reset pattern buffers to initial state with small random values."""
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
            
        for buffer in self.pattern_buffers:
            buffer.clear()
            for _ in range(self.pattern_length):
                buffer.append(torch.rand(1, generator=generator).item() * 0.1 - 0.05)

    def get_resonance_context(self) -> torch.Tensor:
        """Get the current resonance context from the pattern buffers.
        
        Returns:
            torch.Tensor: Tensor containing resonance information from pattern history
        """
        context = []
        for buffer in self.pattern_buffers:
            # Calculate phase-like relationships from history
            buffer_list = list(buffer)
            
            # Recent trajectory (last few states)
            recent = buffer_list[-3:]
            if len(recent) < 3:
                recent = [0.0] * (3 - len(recent)) + recent
                
            # Oscillation indicators
            diffs = [b - a for a, b in zip(recent[:-1], recent[1:])]
            sign_changes = sum(1 for a, b in zip(diffs[:-1], diffs[1:]) if a * b < 0)
            
            context.append(sum(recent) / len(recent))  # Average recent state
            context.append(sign_changes)  # Oscillation indicator
            
        return torch.tensor(context[::2], dtype=torch.float)  # Return only the averages for now

    def forward(self, x: torch.Tensor, prev_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process input through the network.
        
        The forward pass now forces reliance on oscillator dynamics by:
        1. Only allowing predictions based on oscillator states
        2. Maintaining simpler state transitions
        3. Using resonance context to influence state generation
        
        Args:
            x (torch.Tensor): Input tensor
            prev_states (torch.Tensor): Previous oscillator states
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - Resonance context
                - New oscillator states
                - Predicted next signal value
        """
        # Add noise during training if noise_level > 0
        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
            state_noise = torch.randn_like(prev_states) * self.noise_level
            prev_states = prev_states + state_noise
        
        # Generate new oscillator states based on input and previous states
        combined = torch.cat([x, prev_states])
        new_states = self.oscillator_net(combined)
        
        # Apply learnable scaling factors
        new_states = new_states * self.oscillator_scales
        
        # Get resonance context
        context = self.get_resonance_context()
        
        # Prediction can only use oscillator states - forcing reliance on resonance
        prediction = self.prediction_net(new_states)
        
        # Update pattern history buffers
        for i, value in enumerate(new_states):
            self.pattern_buffers[i].append(value.item())
            
        return context, new_states, prediction

    def train_step(self, signal: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Perform a single training step on a signal sequence.
        
        Training now emphasizes the importance of resonance by:
        1. Only allowing predictions through oscillator states
        2. Using longer sequences to establish resonance
        3. Including additional loss terms for resonance quality
        
        Args:
            signal (torch.Tensor): Input signal sequence
            optimizer (torch.optim.Optimizer): The optimizer to use
            
        Returns:
            float: The loss value for this training step
        """
        optimizer.zero_grad()
        
        # Initialize states with seeded random values if seed is set
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        prev_states = torch.rand(self.n_oscillators, generator=generator) * 0.1 - 0.05
        
        total_loss = 0.0
        seq_length = min(len(signal) - 1, max(50, self.pattern_length * 2))
        
        # Training loop
        for t in range(seq_length):
            x = signal[t].unsqueeze(0)
            target = signal[t + 1]
            
            # Forward pass
            context, new_states, prediction = self(x, prev_states)
            
            # Prediction loss
            loss = nn.functional.mse_loss(prediction.squeeze(), target)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Update states
            prev_states = new_states.detach()
        
        # Update weights
        optimizer.step()
        
        return total_loss / seq_length
