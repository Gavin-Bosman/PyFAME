import json
import os
from datetime import datetime
from pyfame.layer.layer import Layer
from pyfame.utilities.exceptions import *

def write_experiment_log(layers:list[Layer], input_file:str, root_directory_path:str) -> None:
    if os.getenv("PYTEST_RUNNING") == "1":
        return
    else:
        if not os.path.isdir(root_directory_path):
            raise FileWriteError(message=f"Unable to locate the input {os.path.basename(root_directory_path)} directory. Please call make_output_paths() to initialise the working directory.")
        
        # Creating a unique file identifier
        timestamp = datetime.now().isoformat(timespec='seconds')
        output_path = os.path.join(root_directory_path, "logs")

        layer_dict = {}
        for layer in layers:
            layer_type = type(layer).__name__
            layer_dict[layer_type] = layer.get_layer_parameters()

        log_data = {
            "timestamp":timestamp,
            "input_file":input_file,
            "layers": layer_dict
        }

        file_id = timestamp.replace(":","-")
        filename = os.path.join(output_path, f"{file_id}.json")
        with open(filename, "w") as f:
            json.dump(log_data, f, indent=2)