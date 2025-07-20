import json
import os
from datetime import datetime
from pyfame.layer.layer import Layer
from pyfame.utilities.exceptions import *

def write_experiment_log(layers:list[Layer], input_file:str) -> None:
    if os.getenv("PYTEST_RUNNING") == "1":
        return
    else:
        cwd = os.getcwd()
        test_path = os.path.join(cwd, "data")
        if not os.path.isdir(test_path):
            raise FileWriteError(message="Unable to locate the input 'data/' directory. Please call make_output_paths() to initialise the working directory.")
        
        # Creating a unique file identifier
        timestamp = datetime.now().isoformat(timespec='seconds')
        output_path = os.path.join(cwd, "data", "logs")

        log_data = {
            "timestamp":timestamp,
            "input_file":input_file,
            "layers": [l.get_layer_parameters() for l in layers]
        }

        file_id = timestamp.replace(":","-")
        filename = os.path.join(output_path, f"{file_id}.json")
        with open(filename, "w") as f:
            json.dump(log_data, f, indent=2)