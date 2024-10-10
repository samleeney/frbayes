import yaml


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
