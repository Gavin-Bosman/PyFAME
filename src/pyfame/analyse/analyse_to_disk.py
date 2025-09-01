import os
import json
import pandas as pd
from datetime import datetime
from pyfame.file_access.checks import *
from pyfame.file_access.file_access_directories import create_output_directory

def analyse_to_disk(analysis_dictionary:dict[str, pd.DataFrame], working_directory_path:str, analysis_type:str) -> None:

    if not os.path.isdir(working_directory_path):
        raise OSError("Analyse_to_disk: Invalid directory path provided.")

    # Get a unique folder identifier for this analysis session
    output_root = os.path.join(working_directory_path, "analysis")
    timestamp = datetime.now().isoformat(timespec='seconds')
    folder_name = timestamp.replace(":","-")
    folder_path = create_output_directory(output_root, folder_name)

    for filename, df in analysis_dictionary.items():

        # Format the output file path
        file_path = os.path.join(folder_path, f"{filename}.json")
        analysis_dict = df.to_dict('index')

        # Format the output data
        output_dict = {
            "timestamp":timestamp,
            "filename":filename,
            "analysis":analysis_type,
            "results":analysis_dict
        }

        # Serialize to Json
        with open(file_path, "w") as f:
            json.dump(output_dict, f, indent=2)