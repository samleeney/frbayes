import yaml


class Settings:
    def __init__(self, config_file="settings.yaml"):
        self._config_file = config_file
        self.settings = None
        self.load_settings()

    def load_settings(self):
        with open(self._config_file, "r") as file:
            self.settings = yaml.safe_load(file)

    def get(self, key):
        return self.settings.get(key, None)

    def set(self, key, value):
        self.settings[key] = value


# Global settings instance
global_settings = Settings()
