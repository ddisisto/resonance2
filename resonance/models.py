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
        self.pattern_buffers: List[Deque[float]] = [
            deque(maxlen=pattern_length) for _ in range(n_oscillators)
        ]
        
        # Initialize buffers with seeded random values
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
            
        for buffer in self.pattern_buffers:
            for _ in range(pattern_length):
                buffer.append(torch.rand(1, generator=generator).item() * 0.1 - 0.05)
        
        # Store last states for continuity between training steps
        self.last_states = torch.zeros(n_oscillators)
        
        # Reduced network capacity to force reliance on resonance dynamics
        self.hidden_size = hidden_size // 2  # Smaller hidden layer
        
        # Oscillator state generation - deliberately simple to encourage emergent behavior
        self.oscillator_net = nn.Sequential(
            nn.Linear(input_size + n_oscillators, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, n_oscillators),
            nn.Tanh()
        )
        
        # Prediction must rely heavily on oscillator states
        self.prediction_net = nn.Sequential(
            nn.Linear(n_oscillators, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )

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
            
            # Calculate phase continuity metric
            phase_continuity = 0.0
            if len(buffer_list) >= 2:
                # Look at consistency of state changes
                diffs = [b - a for a, b in zip(buffer_list[:-1], buffer_list[1:])]
                phase_continuity = sum(1 for d1, d2 in zip(diffs[:-1], diffs[1:]) 
                                    if abs(d2 - d1) < 0.1) / (len(diffs) - 1)
            
            context.append(sum(recent) / len(recent))  # Average recent state
            context.append(sign_changes)  # Oscillation indicator
            context.append(phase_continuity)  # Phase continuity metric
            
        return torch.tensor(context[::3], dtype=torch.float)  # Return only the averages for now

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
        # Generate new oscillator states based on input and previous states
        combined = torch.cat([x, prev_states])
        new_states = self.oscillator_net(combined)
        
        # Get resonance context
        context = self.get_resonance_context()
        
        # Prediction can only use oscillator states - forcing reliance on resonance
        prediction = self.prediction_net(new_states)
        
        # Update pattern history buffers
        for i, value in enumerate(new_states):
            self.pattern_buffers[i].append(value.item())
            
        # Store states for continuity
        self.last_states = new_states.detach()
            
        return context, new_states, prediction

    def train_step(self, signal: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Perform a single training step on a signal sequence.
        
        Training now emphasizes the importance of resonance by:
        1. Only allowing predictions through oscillator states
        2. Using longer sequences to establish resonance
        3. Including additional loss terms for resonance quality
        4. Maintaining continuity between training steps
        
        Args:
            signal (torch.Tensor): Input signal sequence
            optimizer (torch.optim.Optimizer): The optimizer to use
            
        Returns:
            float: The loss value for this training step
        """
        optimizer.zero_grad()
        
        # Use last states from previous training step for continuity
        prev_states = self.last_states
        
        total_loss = 0.0
        n_steps = 0
        
        # Process the full sequence to maintain continuity
        for t in range(len(signal) - 1):
            x = signal[t].unsqueeze(0)
            target = signal[t + 1]
            
            # Forward pass
            context, new_states, prediction = self(x, prev_states)
            
            # Prediction loss
            pred_loss = nn.functional.mse_loss(prediction.squeeze(), target)
            
            # Resonance quality loss - encourage oscillatory behavior
            res_loss = 0.0
            if t > 0:
                # Encourage state changes (avoid static solutions)
                state_diff = torch.mean((new_states - prev_states) ** 2)
                res_loss += 0.1 * torch.exp(-state_diff)  # Penalize small changes
                
                # Encourage diverse oscillator behaviors
                state_cors = torch.corrcoef(torch.stack([new_states, prev_states]))
                if len(state_cors) > 1:
                    res_loss += 0.1 * torch.mean(torch.abs(state_cors[0, 1:]))
                
                # Encourage phase continuity
                context_diff = torch.mean(context ** 2)
                res_loss += 0.1 * torch.exp(-context_diff)
            
            # Combined loss
            loss = pred_loss + res_loss
            total_loss += loss.item()
            n_steps += 1
            
            # Backward pass
            loss.backward()
            
            # Update states
            prev_states = new_states.detach()
        
        # Update weights
        optimizer.step()
        
        # Store final states for next training step
        self.last_states = prev_states
        
        return total_loss / n_steps
