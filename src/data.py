import numpy as np

def generate_input_signal(duration=10.0, dt=0.1, cutoff=5.0, seed=42):
    """Generates a smooth bandlimited input signal for the spiking reservoir."""
    np.random.seed(seed)
    time = np.arange(0, duration, dt)
    n_steps = len(time)
    
    # White noise filtered to create smooth temporal variations
    raw_noise = np.random.randn(n_steps)
    from scipy.signal import butter, lfilter
    b, a = butter(2, cutoff / (0.5 * (1000.0 / dt)), btype='low')
    signal = lfilter(b, a, raw_noise)
    
    # Normalize to [-1, 1]
    signal = 2.0 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1.0
    return time, signal