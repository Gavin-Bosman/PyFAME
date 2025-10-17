import json
import os
from datetime import datetime
from importlib.resources import files
import jsonschema
from jsonschema import ValidationError
from pyfame.layer.layer import Layer
from pyfame.utilities.general_utilities import get_landmark_names

def write_experiment_log(layers:list[Layer], input_file:str, working_directory_path:str) -> None:
    if os.getenv("PYTEST_RUNNING") == "1":
        return
    else:
        if not os.path.isdir(working_directory_path):
            raise OSError(message=f"Unable to locate the input {os.path.basename(working_directory_path)} directory. Please call make_output_paths() to initialise the working directory.")

        # Creating a unique file identifier
        timestamp = datetime.now().isoformat(timespec='seconds')
        output_path = os.path.join(working_directory_path, "logs")

        layer_dict = {}
        for layer in layers:
            layer_type = type(layer).__name__
            parameters = layer.get_layer_parameters()

            if parameters.get("landmark_paths") is not None:
                parameters.update({"landmark_paths":get_landmark_names(parameters.get("landmark_paths"))})
            
            if parameters.get("timing_function") is not None:
                parameters.update({"timing_function":parameters.get("timing_function").__name__})

            layer_dict[layer_type] = parameters

        log_data = {
            "schema_version":"1.0",
            "timestamp":timestamp,
            "file":input_file,
            "layers": layer_dict
        }

        # Attempt to validate json against schema
        try:
            schema_path = files("pyfame.schema").joinpath("manipulation_log.v1.schema.json")
            schema = json.load(open(schema_path))
            jsonschema.validate(instance=log_data, schema=schema)
        except ValidationError as e:
            raise ValueError(f"Experiment log failed to validate: {e.message}.")

        file_id = timestamp.replace(":","-")
        filename = os.path.join(output_path, f"{file_id}.json")

        with open(filename, "w") as f:
            json.dump(log_data, f, indent=2)