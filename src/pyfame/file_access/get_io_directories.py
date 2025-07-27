import os
import pandas as pd
from pyfame.utilities.checks import *
from pathlib import Path

def make_paths(root_path:str = None, exclude_directories:list[str] | None = None) -> pd.DataFrame:
    # The standard folder names for Pyfame input and output files. 
    dir_names = ["raw", "processed", "logs"]
    exclude_directories = set(exclude_directories or [])

    if root_path is not None:
        # If a root_path is provided, check that it is a valid path that exists in the CWD.
        check_type(root_path, [str])
        check_valid_path(root_path)
        check_is_dir(root_path)
    else:
        # If no root_path is provided, the data/ folder will become the root for all of Pyfame's i/o.
        root_path = os.path.join(os.getcwd(), "data")
        # Check if the directory path already exists; if not then create it. 
        if not os.path.isdir(root_path):
            os.mkdir(root_path)

    # For each directory name in dir_names, check if it already exists. 
    # If not, create the directory at the specified path.
    for dir in dir_names:
        path = os.path.join(root_path, dir)
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            print(f"Directory {dir} already exists within {root_path}.")
    
    full_file_paths = []
    rel_file_paths = []

    for path, dirs, files in os.walk(root_path, topdown=True):
        # Exclude unwanted subdirectories
        dirs[:] = [d for d in dirs if d not in exclude_directories]

        for file in files:
            full_path = os.path.join(path, file)
            rel_path = os.path.relpath(full_path, root_path)
            rel_path = os.path.join(os.path.basename(root_path), rel_path)
            
            full_file_paths.append(full_path)
            rel_file_paths.append(rel_path)

    df1 = pd.DataFrame({
        "Absolute Path":full_file_paths,
        "Relative Path":rel_file_paths,
    })

    return df1

def get_directory_walk(input_directory:str) -> pd.DataFrame:
        full_file_paths = []
        rel_file_paths = []

        for path, dirs, files in os.walk(input_directory, topdown=True):
            for file in files:
                full_path = os.path.join(path, file)
                rel_path = os.path.relpath(full_path, input_directory)
                rel_path = os.path.join(os.path.basename(input_directory), rel_path)
                
                full_file_paths.append(full_path)
                rel_file_paths.append(rel_path)

        df1 = pd.DataFrame({
            "Absolute Path":full_file_paths,
            "Relative Path":rel_file_paths,
        })

        return df1

def get_sub_directories_relative_to_path(file_path:str, anchor_directory:str) -> Path:
    path = Path(file_path).resolve()

    try:
        parts = path.parts
        anchor_idx = parts.index(anchor_directory)
        relative_parts = parts[anchor_idx + 1:-1]
        return Path(*relative_parts)
    except ValueError:
        raise ValueError(f"Anchor directory '{anchor_directory}' not found in path: {file_path}")

def contains_sub_directories(directory_path:str) -> bool:
    # Perform parameter checks
    check_valid_path(directory_path)
    check_is_dir(directory_path)

    # Performs a top-down os walk, beginning at the directory path provided.
    # Ignores everything in the first hierarchal level, and only checks if nested 
    # directories exist. Upon the first encountered subdirectory, return true.
    for root, dirs, files in os.walk(directory_path, topdown=True):
        if root == directory_path:
            continue

        if dirs:
            return True
    
    return False
    
def create_output_directory(root_path:str, directory_name:str) -> str:
    # Perform parameter checks
    check_type(root_path, [str])
    check_valid_path(root_path)
    check_is_dir(root_path)

    check_type(directory_name, [str])
    
    # If the provided path + directory does not exist, create it
    if not os.path.isdir(os.path.join(root_path, directory_name)):
        os.mkdir(os.path.join(root_path, directory_name))
    
    # Return the newly combined path string
    return os.path.join(root_path, directory_name)
    
def map_directory_structure(input_directory:str, output_directory:str) -> None:
    """ Maps the subdirectory structure of one directory into another.

    Parameters
    ----------

    input_directory: str
        The path string to the directory whose structure will be copied.
    
    output_directory: str
        The path string to the directory where the copied structure will be written.
    
    Raises
    ------

    TypeError

    OSError
    
    """

    # Type checking parameters
    check_type(input_directory, [str])
    check_valid_path(input_directory)
    check_is_dir(input_directory)

    check_type(output_directory, [str])
    check_valid_path(output_directory)
    check_is_dir(output_directory)

    dir_names = []
    
    for path, dirs, files in os.walk(input_directory, topdown=True):
        for dir in dirs:
            if path == input_directory:
                dir_names.append(dir)
            else:
                sub_dir = os.path.basename(path)
                dir_names.append(os.path.join(sub_dir, dir))

    if dir_names is not None:
        for dir in dir_names:
            create_output_directory(output_directory, dir)