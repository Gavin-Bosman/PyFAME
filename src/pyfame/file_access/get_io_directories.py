import os
import logging
from pyfame.utilities.checks import *

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def make_paths(root_path:str = None) -> None:
    # The standard folder names for Pyfame input and output files. 
    dir_names = ["raw", "processed", "analysis", "logs"]

    if root_path is not None:
        # If a root_path is provided, check that it is a valid path that exists in the CWD.
        check_valid_path(root_path)
        check_is_dir(root_path)

        # For each directory name in dir_names, check if it already exists. 
        # If not, create the directory at the specified path.
        for dir in dir_names:
            path = os.path.join(root_path, dir)
            if not os.path.exists(path):
                os.mkdir(path)
            else:
                print(f"Directory {dir} already exists within {root_path}.")

    else:
        # If no root_path is provided, the data/ folder will become the root for all of Pyfame's i/o.
        root_path = os.getcwd()
        root_path = os.path.join(root_path, "data")

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

def get_directory_walk(root_path:str, with_sub_dirs:bool = False) -> list[str] | tuple[list[str],list[str]]:
    # return as pd dataframe
    # two columns; one for absolute path, one for relative path
    
    """ Takes the provided directory path and returns a full directory walk as a list of str.

    Parameters
    ----------

    root_path: str
        A path string to the root directory where the os walk will begin.
    
    with_sub_dirs: bool
        A boolean flag indicating if the root directory contains subdirectories.
    
    raises
    ------

    TypeError
    
    returns
    -------

    dirs: list[str]
        A list of subdirectory path strings.

    files: list[str]
        A list of file path strings.
    """

    # Type checking parameters
    check_type(root_path, [str])
    check_valid_path(root_path)
    check_is_dir(root_path)

    check_type(with_sub_dirs, [bool])
    
    files_to_process = []
    sub_dirs = []
    single_file = False

    if os.path.isfile(root_path):
        single_file = True

    if single_file:
        files_to_process.append(root_path)
    elif not with_sub_dirs:
        files_to_process = [os.path.join(root_path, file) for file in os.listdir(root_path)]
    else: 
        for path, dirs, files in os.walk(root_path, topdown=True):
            for dir in dirs:
                full_path = os.path.join(path, dir)
                rel_path = os.path.relpath(full_path, root_path)
                sub_dirs.append(rel_path)
            for file in files:
                files_to_process.append(os.path.join(path, file))

    if with_sub_dirs:
        return (sub_dirs, files_to_process)
    else:
        return files_to_process
    
def map_directory_structure(input_directory:str, output_directory:str, with_sub_dirs:bool = False) -> None:
    """ Maps the subdirectory structure of one directory into another.

    Parameters
    ----------

    input_directory: str
        The path string to the directory whose structure will be copied.
    
    output_directory: str
        The path string to the directory where the copied structure will be written.
    
    with_sub_dirs: bool
        A boolean flag indicating if the input directory contains subdirectories.
    
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

    check_type(with_sub_dirs, [bool])

    dir_names = None
    if with_sub_dirs:
        (dir_names, files) = get_directory_walk(input_directory, with_sub_dirs)
    else:
        files = get_directory_walk(input_directory, with_sub_dirs)

    if dir_names is not None:
        for dir in dir_names:
            create_output_directory(output_directory, dir)