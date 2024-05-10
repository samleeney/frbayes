import yaml
from frbayes import data
from frbayes import analysis
from frbayes import sample
import importlib

importlib.reload(analysis)
importlib.reload(data)


def load_settings(yaml_file):
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


settings = load_settings("settings.yaml")

data.preprocess_data(settings)
analysis.plot_inputs(settings)
sample
