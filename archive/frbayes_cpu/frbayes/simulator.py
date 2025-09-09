import numpy as np
import matplotlib.pyplot as plt
from frbayes.models import get_model
from frbayes.settings import Settings

def simulate_data(model_name, theta, t_min=0, t_max=5, num_points=100):
    """
    Simulate data for a given model and parameters.

    Args:
        model_name (str): Name of the model (e.g., 'emg', 'exponential').
        theta (array): Model parameters.
        t_min (float): Start time for the simulation.
        t_max (float): End time for the simulation.
        num_points (int): Number of data points to simulate.

    Returns:
        dict: Dictionary containing 't' (time array) and 'pp' (simulated pulse profile).
    """
    settings = Settings()
    settings.set("max_peaks", 2)  # Set to match the number of peaks in theta
    settings.set("fit_pulses", False)
    model = get_model(model_name, settings)

    t = np.linspace(t_min, t_max, num_points)
    pp = np.zeros_like(t)

    # Sum contributions from each peak
    for i in range(model.max_peaks):
        if model_name == 'emg':
            pp += model.model_function(t, theta, i)
        elif model_name == 'exponential':
            pp += model.model_function(t, theta, i)
        elif model_name == 'periodic_exponential':
            pp += model.model_function(t, theta, i)

    # Add Gaussian noise
    sigma_index = model.get_sigma_param_index()
    sigma = theta[sigma_index]
    noise = np.random.normal(0, sigma, size=t.shape)
    pp += noise

    return {'t': t, 'pp': pp}

def main():
    """
    Example usage of simulate_data function.
    """
    # Choose the model
    model_name = 'emg'  # Change this to the desired model
    settings = Settings()
    settings.set("max_peaks", 2)
    settings.set("fit_pulses", False)
    model = get_model(model_name, settings)

    # Define model parameters theta for EMG model with 2 peaks
    # For EMGModel with 2 peaks:
    # A1, A2 (amplitudes)
    # tau1, tau2 (decay times)
    # u1, u2 (arrival times)
    # w1, w2 (widths)
    # sigma (noise)
    theta = np.array([
        0.8, 0.5,             # A1, A2
        1.0, 0.5,             # tau1, tau2
        1.0, 3.0,             # u1, u2
        0.1, 0.1,             # w1, w2
        0.05                  # sigma
    ])

    # Simulate data
    data = simulate_data(model_name, theta, t_min=0, t_max=5, num_points=200)

    # Plot and save the simulated data
    plt.figure(figsize=(10, 6))
    plt.plot(data['t'], data['pp'], 'k.', label='Simulated Data')
    plt.xlabel('Time')
    plt.ylabel('Pulse Profile')
    plt.title(f'Simulated Data for {model_name.upper()} Model')
    plt.legend()
    plt.grid(True)
    plt.savefig('simulated_data.png')
    plt.close()
    
    print(f"Simulated data saved to simulated_data.png")
    return data  # Return the data for testing

if __name__ == "__main__":
    main() 