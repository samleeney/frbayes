import os
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
        _settings_file (str): Path to the YAML configuration file.
        _settings (dict): Dictionary of settings loaded from the YAML file.
    """

    def __init__(self, settings_file=None):
        """
        Initialize the Settings class by loading settings from the configuration file.

        Args:
            settings_file (str): Path to the YAML configuration file.
        """
        if settings_file:
            self.settings_file = settings_file
        else:
            env_settings_file = os.environ.get("FRBAYES_SETTINGS_FILE")
            if env_settings_file:
                self.settings_file = env_settings_file
            else:
                self.settings_file = "settings.yaml" # Default fallback

        self._settings = self._load_settings()

    def _load_settings(self):
        """
        Load settings from the YAML configuration file.
        """
        with open(self.settings_file, "r") as file:
            settings = yaml.safe_load(file) or {} # Ensure settings is a dictionary even if file is empty
        
        # Set default for use_paper_preprocessing if not present
        if "preprocessing" not in settings:
            settings["preprocessing"] = {}
        if "use_paper_preprocessing" not in settings["preprocessing"]:
            settings["preprocessing"]["use_paper_preprocessing"] = False
        
        # Set default for process_raw_no_downsample if not present
        if "process_raw_no_downsample" not in settings["preprocessing"]:
            settings["preprocessing"]["process_raw_no_downsample"] = False
        
        return settings

    def get(self, key, default=None):
        """
        Retrieve a value from the settings dictionary.

        Args:
            key (str): The key of the setting to retrieve.
            default: The default value to return if the key is not found.

        Returns:
            The value associated with the given key, or default if the key is not found.
        """
        return self._settings.get(key, default)
        
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
        prior_ranges = self._settings.get("prior_ranges", {})
        
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
        self._settings[key] = value


# Global settings instance
global_settings = Settings()
