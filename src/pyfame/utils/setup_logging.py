import logging
import logging.config
import yaml
import os

# Function to initialize loggers
def setup_logging(default_path:str = "config/log_config.yaml"):
    if os.getenv("PYTEST_RUNNING") == "1":
        logging.basicConfig(level=logging.CRITICAL)
        return

    with open(default_path, "rt") as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)