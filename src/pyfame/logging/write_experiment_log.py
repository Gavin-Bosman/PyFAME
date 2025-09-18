import json
import os
from datetime import datetime
from pyfame.layer.layer import Layer
from pyfame.utilities.general_utilities import get_roi_name
from tqdm import tqdm

def write_experiment_log(layers:list[Layer], input_file:str, working_directory_path:str) -> None:
    if os.getenv("PYTEST_RUNNING") == "1":
        return
    else:
        pb = tqdm(total=8, desc="Compiling log file:", bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', leave=False)
        if not os.path.isdir(working_directory_path):
            raise OSError(message=f"Unable to locate the input {os.path.basename(working_directory_path)} directory. Please call make_output_paths() to initialise the working directory.")
        pb.update(1)

        # Creating a unique file identifier
        timestamp = datetime.now().isoformat(timespec='seconds')
        output_path = os.path.join(working_directory_path, "logs")
        pb.update(1)

        layer_dict = {}
        for layer in layers:
            layer_type = type(layer).__name__
            parameters = layer.get_layer_parameters()

            if parameters.get("region_of_interest") is not None:
                parameters.update({"region_of_interest":get_roi_name(parameters.get("region_of_interest"))})
            
            if parameters.get("timing_function") is not None:
                parameters.update({"timing_function":parameters.get("timing_function").__name__})

            layer_dict[layer_type] = parameters
        pb.update(2)

        log_data = {
            "timestamp":timestamp,
            "input file":input_file,
            "layers": layer_dict
        }
        pb.update(1)

        file_id = timestamp.replace(":","-")
        filename = os.path.join(output_path, f"{file_id}.json")
        pb.update(1)

        with open(filename, "w") as f:
            json.dump(log_data, f, indent=2)
        pb.update(2)
        pb.close()