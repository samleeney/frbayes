import yaml


# Default prior ranges for all models
DEFAULT_PRIOR_RANGES = {
    # Common priors
    "amplitude": {"min": 0.001, "max": 1.0},
    "tau": {"min": 0.0001, "max": 4.0},
    "u": {"min": 0.001, "max": 4.0},
    "sigma": {"min": 0.001, "max": 1.0},
    
    # Model-specific priors
    "emg": {
        "width": {"min": 0.01, "max": 1.0}
    },
    "exponential": {
        "amplitude": {"min": 0.001, "max": 0.1}  # Override common amplitude
    },
    "periodic": {
        "u0": {"min": 0.001, "max": None},  # max will be calculated based on max_peaks
        "period": {"min": 0.001, "max": None}  # max will be calculated based on max_peaks
    }
}


class Settings:
    """
    Class to manage global settings loaded from a YAML configuration file.

    Attributes:
        _config_file (str): Path to the YAML configuration file.
        settings (dict): Dictionary of settings loaded from the YAML file.
    """

    def __init__(self, config_file="settings.yaml"):
        """
        Initialize the Settings class by loading settings from the configuration file.

        Args:
            config_file (str): Path to the YAML configuration file.
        """
        self._config_file = config_file
        self.settings = None
        self.load_settings()

    def load_settings(self):
        """
        Load settings from the YAML configuration file.
        """
        with open(self._config_file, "r") as file:
            self.settings = yaml.safe_load(file)

    def get(self, key):
        """
        Retrieve a value from the settings dictionary.

        Args:
            key (str): The key of the setting to retrieve.

        Returns:
            The value associated with the given key, or None if the key is not found.
        """
        return self.settings.get(key, None)
        
    def get_prior_range(self, model_type, param_type):
        """
        Get the prior range for a specific parameter type and model.
        
        Args:
            model_type (str): The model type (e.g., 'emg', 'exponential', 'periodic')
            param_type (str): The parameter type (e.g., 'amplitude', 'tau', 'u', 'sigma', 'width')
            
        Returns:
            dict: A dictionary with 'min' and 'max' values for the prior range
        """
        # Check if prior ranges are defined in settings
        prior_ranges = self.settings.get("prior_ranges", {})
        
        # Check for model-specific override
        if model_type in prior_ranges and param_type in prior_ranges[model_type]:
            return prior_ranges[model_type][param_type]
        
        # Check for common parameter
        if param_type in prior_ranges:
            return prior_ranges[param_type]
        
        # Fall back to defaults
        if model_type in DEFAULT_PRIOR_RANGES and param_type in DEFAULT_PRIOR_RANGES[model_type]:
            return DEFAULT_PRIOR_RANGES[model_type][param_type]
        
        if param_type in DEFAULT_PRIOR_RANGES:
            return DEFAULT_PRIOR_RANGES[param_type]
        
        # If no range is found, return None
        return None
    
    def set(self, key, value):
        """
        Set a value in the settings dictionary.

        Args:
            key (str): The key of the setting to set.
            value: The value to associate with the key.
        """
        self.settings[key] = value


# Global settings instance
global_settings = Settings()
