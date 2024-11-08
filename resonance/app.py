import streamlit as st
import torch
import torch.optim as optim
import numpy as np
from resonance.generators import SineWaveGenerator
from resonance.models import ResonanceNetwork
from resonance.visualizations import (
    create_combined_signal_plot,
    create_component_signals_plot,
    create_training_progress_plot,
    create_prediction_comparison_plot,
    create_multi_phase_plot,
    create_oscillator_outputs_plot
)

# Configure the page to use wide mode
st.set_page_config(layout="wide")

st.title("Resonance Pattern Generator")

# Initialize session state for generators if not exists
if 'generators' not in st.session_state:
    st.session_state.generators = []

# Sidebar controls
st.sidebar.header("Pattern Generator Settings")

seq_length = st.sidebar.slider("Sequence Length", min_value=10, max_value=1000, value=100)

# Multiple generator controls
n_components = st.sidebar.number_input("Number of Components", min_value=1, max_value=5, value=2)

# Create or update generators
if len(st.session_state.generators) != n_components:
    st.session_state.generators = []
    for i in range(n_components):
        st.session_state.generators.append(SineWaveGenerator(seq_length, i + 1.0, 0.0))

# Generator controls
with st.sidebar.expander('Components'):
    for i, gen in enumerate(st.session_state.generators):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            freq = st.number_input(f"Frequency {i+1}", 0.1, 10.0, i + 1.0, key=f"freq_{i}")
        with col2:
            phase = st.number_input(f"Phase {i+1}", 0.0, 2*np.pi, 0.0, key=f"phase_{i}")
        
        gen.frequency = freq
        gen.phase = phase

# Generate patterns using persisted generators
patterns = [gen.generate() for gen in st.session_state.generators]
combined_pattern = sum(patterns)

# Initialize session state for training results if not exists
if 'training_results' not in st.session_state:
    st.session_state.training_results = None

# Signal Visualization section (full width)
# st.subheader("Signal Visualization")

# Two column layout
col1, col2 = st.columns([1, 2])

with col1:
    # Combined signal plot
    sig_tab1, sig_tab2 = st.tabs(["Combined", "Components"])

    fig = create_combined_signal_plot(combined_pattern)
    sig_tab1.plotly_chart(fig, use_container_width=True)

    frequencies = [gen.frequency for gen in st.session_state.generators]
    fig = create_component_signals_plot(patterns, frequencies)
    sig_tab2.plotly_chart(fig, use_container_width=True)


# Network settings
st.sidebar.header("Resonance Network Settings")
n_oscillators = st.sidebar.slider("Number of Oscillators", min_value=1, max_value=10, value=3)
pattern_length = st.sidebar.slider("Pattern History Length", min_value=5, max_value=50, value=10)

# Random seed input with integer validation
# use_random_seed = st.sidebar.checkbox("Use Random Seed", value=False, 
#                                     help="Enable to set a specific seed for reproducible results")
# random_seed = None
# if use_random_seed:
random_seed = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1,
                                    help="Integer seed value for reproducible results")

# Training settings
st.sidebar.header("Training Settings")
n_epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=100, value=10)
learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.01, format="%.4f")

if st.sidebar.button("Train Network"):
    # Initialize network and optimizer
    network = ResonanceNetwork(
        input_size=1,
        n_oscillators=n_oscillators,
        pattern_length=pattern_length,
        seed=random_seed # if use_random_seed else None
    )
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    
    # Training progress
    progress_bar = st.progress(0)
    epoch_losses = []
    
    # Store oscillator states during training
    states_history = []
    prev_states = torch.zeros(n_oscillators)
    
    # Training loop
    for epoch in range(n_epochs):
        loss = network.train_step(combined_pattern, optimizer)
        epoch_losses.append(loss)
        progress_bar.progress((epoch + 1) / n_epochs)
    
    # Generate predictions and collect states
    predictions = []
    prev_states = torch.zeros(n_oscillators)
    
    with torch.no_grad():
        for t in range(len(combined_pattern) - 1):
            x = combined_pattern[t].unsqueeze(0)
            _, new_states, pred = network(x, prev_states)
            predictions.append(pred.item())
            states_history.append(new_states.tolist())
            prev_states = new_states
    
    # Store results in session state
    st.session_state.training_results = {
        'loss': loss,
        'epoch_losses': epoch_losses,
        'predictions': predictions,
        'states_history': states_history,
        'n_oscillators': n_oscillators,
        'random_seed': random_seed # if use_random_seed else None
    }

# Split layout for results
if st.session_state.training_results is not None:

    n_oscillators = st.session_state.training_results['n_oscillators']
    states_history = st.session_state.training_results['states_history']

    with col1:

        final_loss = st.session_state.training_results['epoch_losses'][-1]
        st.subheader(f"Training Progress (Loss: {final_loss:.5f})")
        fig = create_training_progress_plot(st.session_state.training_results['epoch_losses'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Prediction Results")
        fig = create_prediction_comparison_plot(
            combined_pattern[1:],
            st.session_state.training_results['predictions']
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Individual Oscillator Outputs")
        fig = create_oscillator_outputs_plot(states_history, n_oscillators)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        
        st.subheader("Multi-Oscillator Analysis")
        fig_multi = create_multi_phase_plot(states_history, n_oscillators)
        st.plotly_chart(fig_multi, use_container_width=True)