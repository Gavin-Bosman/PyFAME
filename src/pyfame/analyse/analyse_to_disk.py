import os
import pandas as pd
from datetime import datetime
from pyfame.utilities.checks import *
from pyfame.file_access.file_access_directories import create_output_directory

def analyse_to_disk(analysis_dictionary:dict[str, pd.DataFrame]) -> None:

    # Get a unique folder identifier for this analysis session
    output_root = os.path.join(os.getcwd(), "data", "analysis")
    timestamp = datetime.now().isoformat(timespec='seconds')
    folder_name = timestamp.replace(":","-")
    file_path = create_output_directory(output_root, folder_name)

    for filename, df in analysis_dictionary.items():

        # Format the output file path
        file_path = os.path.join(file_path, f"{filename}.json")

        df.to_json(file_path, orient="records", lines=True, index=False)