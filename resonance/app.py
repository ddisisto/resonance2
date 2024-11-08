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

# Initialize session state
if 'generators' not in st.session_state:
    st.session_state.generators = []
    st.session_state.n_components = 2
    st.session_state.seq_length = 100
    st.session_state.frequencies = [1.0, 2.0]
    st.session_state.phases = [0.0, 0.0]

# Callback for updating generators
def update_generators():
    st.session_state.generators = []
    for i in range(st.session_state.n_components):
        st.session_state.generators.append(
            SineWaveGenerator(
                st.session_state.seq_length,
                st.session_state.frequencies[i],
                st.session_state.phases[i]
            )
        )

# Callback for number of components change
def on_n_components_change():
    n = st.session_state.n_components_input
    # Update frequencies and phases arrays
    if n > len(st.session_state.frequencies):
        # Add new components
        for i in range(len(st.session_state.frequencies), n):
            st.session_state.frequencies.append(float(i + 1))
            st.session_state.phases.append(0.0)
    else:
        # Remove extra components
        st.session_state.frequencies = st.session_state.frequencies[:n]
        st.session_state.phases = st.session_state.phases[:n]
    
    st.session_state.n_components = n
    update_generators()

# Callback for sequence length change
def on_seq_length_change():
    st.session_state.seq_length = st.session_state.seq_length_input
    update_generators()

# Callback for frequency/phase changes
def on_param_change(param_type, index):
    if param_type == 'freq':
        st.session_state.frequencies[index] = st.session_state[f'freq_{index}']
    else:  # phase
        st.session_state.phases[index] = st.session_state[f'phase_{index}']
    update_generators()

# Sidebar - Pattern Generator Settings (with live updates)
st.sidebar.header("Pattern Generator Settings")

st.sidebar.number_input("Number of Components", min_value=1, max_value=5, 
                       value=st.session_state.n_components,
                       key='n_components_input', 
                       on_change=on_n_components_change)

st.sidebar.slider("Sequence Length", min_value=10, max_value=1000, 
                 value=st.session_state.seq_length,
                 key='seq_length_input', 
                 on_change=on_seq_length_change)

with st.sidebar.expander('Components (Freq/Phase)', expanded=True):
    for i in range(st.session_state.n_components):
        col1, col2, col3 = st.columns([1,2,2])
        with col1:
            st.write(i)
        with col2:
            st.number_input(f"Frequency {i+1}", 0.1, 10.0,
                          st.session_state.frequencies[i],
                          key=f'freq_{i}',
                          on_change=on_param_change,
                          label_visibility='collapsed',
                          args=('freq', i))
        with col3:
            st.number_input(f"Phase {i+1}", 0.0, 2*np.pi,
                          st.session_state.phases[i],
                          key=f'phase_{i}',
                          on_change=on_param_change,
                          label_visibility='collapsed',
                          args=('phase', i))

# Sidebar form - Network and Training Settings
with st.sidebar.form("training_settings_form"):
    st.header("Resonance Network Settings")
    n_oscillators = st.slider("Number of Oscillators", min_value=1, max_value=10, value=3)
    pattern_length = st.slider("Pattern History Length", min_value=5, max_value=50, value=10)
    random_seed = st.number_input("Random Seed", min_value=0, value=42, step=1,
                                help="Integer seed value for reproducible results")

    st.header("Training Settings")
    n_epochs = st.slider("Number of Epochs", min_value=1, max_value=100, value=10)
    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, 
                                  value=0.01, format="%.4f")

    # Submit button
    submitted = st.form_submit_button("Train Network")

# Generate patterns
if not st.session_state.generators:
    update_generators()

patterns = [gen.generate() for gen in st.session_state.generators]
combined_pattern = sum(patterns)

# Initialize session state for training results if not exists
if 'training_results' not in st.session_state:
    st.session_state.training_results = None

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

if submitted:
    # Initialize network and optimizer
    network = ResonanceNetwork(
        input_size=1,
        n_oscillators=n_oscillators,
        pattern_length=pattern_length,
        seed=random_seed
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
        'random_seed': random_seed
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
