import yaml
from frbayes.analysis import plot_inputs


def load_settings(yaml_file):
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


settings = load_settings("settings.yaml")

# Plot inputs
plot_inputs(settings)

#


print("Done")
