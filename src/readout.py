import numpy as np
from sklearn.linear_model import Ridge

def train_readout(spike_rates, target_signal, alpha=1e-3):
    """Trains a Ridge regression readout on smoothed population spike rates."""
    from scipy.ndimage import gaussian_filter1d
    
    # Smooth population rates to create continuous firing rate traces
    smoothed_rates = gaussian_filter1d(spike_rates, sigma=5, axis=0)
    
    # Align shapes
    X = smoothed_rates
    y = target_signal[:len(X)]

    model = Ridge(alpha=alpha)
    model.fit(X, y)
    predictions = model.predict(X)
    
    mse = np.mean((y - predictions) ** 2)
    return model, predictions, mse