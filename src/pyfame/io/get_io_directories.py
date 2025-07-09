import os
import logging
from pyfame.util.util_checks import *

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def make_output_paths(root_path:str = None) -> None:
    dir_names = ["raw", "results", "analysis", "logs", "sample_data"]

    if root_path is not None:
        check_valid_path(root_path)
        check_is_dir(root_path)

        for dir in dir_names:
            path = os.path.join(root_path, dir)
            os.mkdir(path)

    else:
        root_path = os.getcwd()

        for dir in dir_names:
            path = os.path.join(root_path, dir)
            os.mkdir(path)
    
def create_output_directory(root_path:str, dir_name:str) -> str:
    """ Creates a new directory appended to the root path provided, then returns the appended path string.

    Parameters
    ----------
    
    root_path: str
        A path string to the root folder location where the new directory will be created.
    
    dir_name: str
        The name of the newly created directory.
    
    Raises
    ------

    TypeError

    OSError

    ValueError
    
    Returns
    -------

    combined_path: str
        The new output directory name appended to the provided root_path.
    """

    logger.info("Now entering function create_output_directory().")
    logger.info(f"Input Parameters: root_path = {root_path}, dir_name = {dir_name}.")

    check_type(root_path, [str])
    check_valid_path(root_path)
    check_is_dir(root_path)

    check_type(dir_name, [str])
    
    if not os.path.isdir(root_path + f"\\{dir_name}"):
        os.mkdir(root_path + f"\\{dir_name}")
        
    return root_path + f"\\{dir_name}"

def get_directory_walk(root_path:str, with_sub_dirs:bool = False) -> list[str] | tuple[list[str],list[str]]:
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
        A list of subsirectory path strings.

    files: list[str]
        A list of file path strings.
    """

    logger.info("Now entering function get_directory_walk().")
    logger.info(f"Input Parameters: root_path = {root_path}, with_sub_dirs = {with_sub_dirs}.")

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
        files_to_process = [root_path + "\\" + file for file in os.listdir(root_path)]
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
    
def map_directory_structure(input_dir:str, output_dir:str, with_sub_dirs:bool = False) -> None:
    """ Maps the subdirectory structure of one directory into another.

    Parameters
    ----------

    input_dir: str
        The path string to the directory whose structure will be copied.
    
    output_dir: str
        The path string to the directory where the copied structure will be written.
    
    with_sub_dirs: bool
        A boolean flag indicating if the input directory contains subdirectories.
    
    Raises
    ------

    TypeError

    OSError
    
    """

    logger.info("Now entering function map_directory_structure().")
    logger.info(f"Input Parameters: input_dir = {input_dir}, output_dir = {output_dir}, with_sub_dirs = {with_sub_dirs}.")

    # Type checking parameters
    check_type(input_dir, [str])
    check_valid_path(input_dir)
    check_is_dir(input_dir)

    check_type(output_dir, [str])
    check_valid_path(output_dir)
    check_is_dir(output_dir)

    check_type(with_sub_dirs, [bool])

    dir_names = None
    if with_sub_dirs:
        (dir_names, files) = get_directory_walk(input_dir, with_sub_dirs)
    else:
        files = get_directory_walk(input_dir, with_sub_dirs)

    if dir_names is not None:
        for dir in dir_names:
            create_output_directory(output_dir, dir)