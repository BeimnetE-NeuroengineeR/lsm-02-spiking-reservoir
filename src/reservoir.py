from brian2 import (
    start_scope,
    defaultclock,
    ms,
    mV,
    NeuronGroup,
    Synapses,
    SpikeMonitor,
    PopulationRateMonitor,
    rand
)
import numpy as np

def build_spiking_reservoir(n_neurons=200, duration=10.0, dt=0.1):
    """Constructs a Leaky Integrate-and-Fire spiking reservoir in Brian2."""
    start_scope()
    defaultclock.dt = dt * ms

    # Define equations with explicit parameter values to avoid namespace resolution errors
    eqs = '''
    dv/dt = (-(v - (-65*mV)) + I_syn + I_ext) / (20*ms) : volt (unless refractory)
    dI_syn/dt = -I_syn / (5*ms) : volt
    dI_ext/dt = -I_ext / (2*ms) : volt
    '''

    # Neuron group
    neurons = NeuronGroup(
        n_neurons, 
        eqs, 
        threshold='v > -50*mV', 
        reset='v = -70*mV', 
        refractory=5*ms, 
        method='exact'
    )
    neurons.v = -65*mV + (-50*mV - (-65*mV)) * rand(n_neurons)

    # Recurrent connections with explicit weight variable 'w'
    synapses_rec = Synapses(neurons, neurons, model='w : 1', on_pre='I_syn += w * mV')
    synapses_rec.connect(p=0.1)
    
    # Assign weights using NumPy based on presynaptic source index (i)
    n_exc = int(0.8 * n_neurons)
    sources = synapses_rec.i[:]
    weights = np.where(sources < n_exc, np.random.rand(len(sources)) * 0.8, -np.random.rand(len(sources)) * 1.5)
    synapses_rec.w = weights

    # Spike and state monitors
    spike_monitor = SpikeMonitor(neurons)
    rate_monitor = PopulationRateMonitor(neurons)

    # Return synapses_rec as well so Brian2 keeps it alive in the network scope
    return neurons, synapses_rec, spike_monitor, rate_monitor