import os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import run, ms, second, TimedArray, Synapses

from src.data import generate_input_signal
from src.reservoir import build_spiking_reservoir
from src.readout import train_readout

def main():
    print("Initializing Project 2: Spiking Reservoir (Brian2)...")
    
    # 1. Generate Input
    duration = 5.0 # seconds
    dt = 0.1 # ms
    time, signal = generate_input_signal(duration=duration, dt=dt)
    
    # 2. Build Reservoir
    n_neurons = 200
    neurons, synapses_rec, spike_monitor, rate_monitor = build_spiking_reservoir(
        n_neurons=n_neurons, duration=duration, dt=dt
    )
    
    print(f"Running Brian2 simulation for {duration} seconds...")
    run(duration * second)
    
    # 3. Process Spike Data & Train Readout
    bin_size_ms = 10.0
    time_bins = np.arange(0, duration * 1000, bin_size_ms)
    
    spike_times = spike_monitor.t / ms
    spike_indices = spike_monitor.i
    
    hist, _ = np.histogramdd(
        sample=(spike_times, spike_indices),
        bins=(time_bins, np.arange(n_neurons + 1))
    )
    
    from scipy.signal import resample
    target_resampled = resample(signal, len(hist))
    
    model, predictions, mse = train_readout(hist, target_resampled)
    print(f"Readout training complete. MSE: {mse:.5f}")
    
    # 4. Plot & Save Results
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(spike_monitor.t / ms, spike_monitor.i, '.k', markersize=1)
    plt.ylabel("Neuron ID")
    plt.title("Spiking Reservoir Raster Plot (Project 2)")
    
    plt.subplot(2, 1, 2)
    t_axis = np.linspace(0, duration, len(target_resampled))
    plt.plot(t_axis, target_resampled, 'b-', label="Target Signal", alpha=0.7)
    plt.plot(t_axis, predictions, 'r--', label="Spiking Readout", alpha=0.9)
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig("figures/main_result.png", dpi=300)
    print("Saved main result plot to figures/main_result.png")

if __name__ == "__main__":
    main()