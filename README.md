# Resonance Pattern Generator

A neural network-based system that explores emergent resonance patterns through feedback dynamics. Rather than explicitly programming oscillatory behaviors, this system creates conditions where complex temporal patterns can emerge naturally through the interaction of simple components.

## Core Concepts

### Resonance Networks

The system uses a neural network architecture designed to discover and generate resonant patterns. Instead of explicitly encoding oscillatory behaviors, it creates conditions where resonance can emerge through:

1. **Feedback Loops**: Network outputs affect future states, creating a continuous cycle of influence
2. **Pattern History**: Temporal context that shapes resonance behaviors
3. **Oscillator Interaction**: States influence each other through network dynamics

### Emergent Behavior

The key philosophy is that complex patterns emerge from simple interactions rather than being explicitly programmed. This emergence happens through:

- Rich input context including pattern histories
- Continuous feedback where outputs affect future states
- The network learning to generate patterns that help with its task
- Oscillators potentially falling into phase relationships naturally

## Key Parameters

### Number of Oscillators

This parameter determines how many internal oscillators the system maintains. Rather than being explicitly coupled, oscillators interact through:

- Shared input processing
- Pattern history context
- The network learning to generate outputs that create useful interactions

More oscillators allow for richer possible interactions and more complex emergent patterns.

### Pattern History Length

This determines how much temporal context is maintained for each oscillator. The history:

- Acts as context that helps shape resonance behaviors
- Allows patterns to persist and influence future states
- Enables the network to discover useful temporal relationships

Longer histories provide more context for patterns to emerge, while shorter histories create more immediate feedback loops.

## Components

### Pattern Generator

Generates input signals that the network learns to process. Currently supports:
- Sine wave patterns with configurable frequency and phase
- Multiple component signals that can be combined

### Resonance Network

The core neural network that:
- Processes input signals
- Maintains oscillator states
- Generates predictions based on learned patterns
- Creates conditions for resonance to emerge

### Interactive UI

A Streamlit-based interface that allows:
- Configuring signal generation parameters
- Adjusting network settings
- Visualizing the input signals, network responses, and predictions
- Training the network and observing emergent behaviors

## Usage

1. Configure the input signal parameters (frequency, phase, number of components)
2. Adjust network settings (number of oscillators, pattern history length)
3. Train the network and observe:
   - How different oscillators interact
   - What patterns emerge in the network response
   - How well predictions match the input signal
   - How changing parameters affects the emergent behaviors

## Implementation Philosophy

The implementation focuses on creating conditions for emergence rather than explicitly encoding behaviors:

1. **Simple Components**: Each part of the system is relatively simple, with complexity emerging from their interaction
2. **Feedback Loops**: Continuous cycles where outputs affect future states
3. **Rich Context**: Sufficient information (through pattern histories and oscillator states) for complex patterns to emerge
4. **Learning Freedom**: The network can discover what patterns are useful rather than being constrained to specific behaviors

This approach allows for surprising and complex behaviors to emerge from relatively simple underlying mechanisms.
