from pyfame.utilities.checks import *
import os

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