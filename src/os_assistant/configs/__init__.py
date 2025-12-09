import os

import yaml

# Load the configuration file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)
