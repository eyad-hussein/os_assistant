import os
import yaml

# Load the configuration file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

# Expose specific sections for easier access
CODE_AGENT = CONFIG.get("CODE_AGENT", {})

# This file makes the configs directory a Python module.
