"""
Neural network models for resonance-based pattern processing.

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
        seed (int, optional): Random seed for reproducibility. Defaults to None
    """
    
    def __init__(
        self,
        input_size: int,
        n_oscillators: int,
        pattern_length: int,
        hidden_size: int = 32,
        seed: int = None
    ):
        super().__init__()
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            
        self.n_oscillators = n_oscillators
        self.pattern_length = pattern_length
        self.seed = seed
        
        # Initialize pattern history buffers for each oscillator
        # Increased buffer size to maintain longer history
        self.pattern_buffers: List[Deque[float]] = [
            deque(maxlen=pattern_length * 2) for _ in range(n_oscillators)
        ]
        
        # Initialize buffers with seeded random values
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
            
        for buffer in self.pattern_buffers:
            for _ in range(pattern_length * 2):  # Initialize full buffer
                buffer.append(torch.rand(1, generator=generator).item() * 0.1 - 0.05)
        
        self.hidden_size = hidden_size
        
        # Enhanced oscillator state generation
        self.oscillator_net = nn.Sequential(
            nn.Linear(input_size + n_oscillators + n_oscillators * 3, self.hidden_size),  # Adjusted for context size
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, n_oscillators),
            nn.Tanh()
        )
        
        # Prediction network with resonance context
        self.prediction_net = nn.Sequential(
            nn.Linear(n_oscillators + n_oscillators * 3, self.hidden_size),  # Adjusted for context size
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, 1)
        )

    def get_resonance_context(self) -> torch.Tensor:
        """Get the current resonance context from the pattern buffers.
        
        Returns:
            torch.Tensor: Tensor containing resonance information from pattern history
        """
        context = []
        for buffer in self.pattern_buffers:
            buffer_list = list(buffer)
            
            # Enhanced resonance analysis using full buffer
            if len(buffer_list) >= 4:
                # Compute rate of change (velocity)
                velocities = [b - a for a, b in zip(buffer_list[:-1], buffer_list[1:])]
                
                # Compute acceleration
                accelerations = [v2 - v1 for v1, v2 in zip(velocities[:-1], velocities[1:])]
                
                # Detect oscillation characteristics
                zero_crossings = sum(1 for v1, v2 in zip(velocities[:-1], velocities[1:]) 
                                   if v1 * v2 < 0)
                
                # Compute momentum (weighted sum of recent states)
                momentum = sum(v * (i + 1) for i, v in enumerate(reversed(buffer_list[-4:]))) / 10
                
                context.extend([
                    momentum,  # Recent trajectory with momentum
                    zero_crossings / len(velocities),  # Normalized oscillation frequency
                    sum(accelerations) / len(accelerations)  # Average acceleration
                ])
            else:
                # Fallback for insufficient history
                context.extend([0.0, 0.0, 0.0])
            
        return torch.tensor(context, dtype=torch.float)

    def forward(self, x: torch.Tensor, prev_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process input through the network.
        
        The forward pass now incorporates richer resonance dynamics by:
        1. Using extended resonance context
        2. Maintaining longer state history
        3. Including acceleration and momentum in state transitions
        
        Args:
            x (torch.Tensor): Input tensor
            prev_states (torch.Tensor): Previous oscillator states
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - Resonance context
                - New oscillator states
                - Predicted next signal value
        """
        # Get resonance context
        context = self.get_resonance_context()
        
        # Ensure input tensor has correct shape
        if len(x.shape) == 0:
            x = x.unsqueeze(0)
        
        # Generate new oscillator states based on input, previous states, and context
        combined = torch.cat([x, prev_states, context])
        new_states = self.oscillator_net(combined)
        
        # Prediction using both states and context
        pred_input = torch.cat([new_states, context])
        prediction = self.prediction_net(pred_input)
        
        # Update pattern history buffers
        for i, value in enumerate(new_states):
            self.pattern_buffers[i].append(value.item())
            
        return context, new_states, prediction

    def train_step(self, signal: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Perform a single training step on a signal sequence.
        
        Training now emphasizes continuous signal processing by:
        1. Using full sequence length
        2. Incorporating momentum and acceleration
        3. Maintaining state continuity
        
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
        
        # Process full sequence for better continuity
        seq_length = len(signal) - 1
        
        # Training loop
        for t in range(seq_length):
            x = signal[t].unsqueeze(0)
            target = signal[t + 1]
            
            # Forward pass
            context, new_states, prediction = self(x, prev_states)
            
            # Prediction loss
            pred_loss = nn.functional.mse_loss(prediction.squeeze(), target)
            
            # Resonance quality loss
            res_loss = 0.0
            if t > 0:
                # Encourage dynamic oscillator behavior
                state_diff = torch.mean((new_states - prev_states) ** 2)
                res_loss += 0.1 * torch.exp(-state_diff)  # Penalize static states
                
                # Encourage diverse oscillator behaviors
                state_cors = torch.corrcoef(torch.stack([new_states, prev_states]))
                if len(state_cors) > 1:
                    res_loss += 0.1 * torch.mean(torch.abs(state_cors[0, 1:]))
            
            # Combined loss
            loss = pred_loss + res_loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Update states
            prev_states = new_states.detach()
        
        # Update weights
        optimizer.step()
        
        return total_loss / seq_length
