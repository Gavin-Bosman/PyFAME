import os
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

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

    if not isinstance(root_path, str):
        logger.error("Function encountered a TypeError for input parameter root_path."
                    "Message: parameter root_path expects a str.")
        raise TypeError("Create_output_directory: parameter root_path expects a str.")
    elif not os.path.exists(root_path):
        logger.error("Function encountered an OSError for input parameter root_path."
                    "Message: parameter root_path is not a valid path string.")
        raise OSError("Create_output_directory: parameter root_path must be a valid directory path in the current scope.")
    elif os.path.isfile(root_path):
        logger.error("Function encountered a ValueError for input parameter root_path."
                    "Message: parameter root_path must be a path string to a directory.")
        raise ValueError("Create_output_directory: parameter root_path must contain a path string to a directory.")
    
    if not isinstance(dir_name, str):
        logger.error("Function encountered a TypeError for input parameter dir_name."
                    "Message: parameter dir_name expects a str.")
        raise TypeError("Create_output_directory: parameter dir_name expects a str.")
    elif len(dir_name) == 0:
        logger.error("Function encountered a ValueError for input parameter dir_name."
                    "Message: parameter dir_name must be a path string to a directory.")
        raise ValueError("Create_output_directory: parameter dir_name cannot be an empty str.")
    
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
    if not isinstance(root_path, str):
        logger.error("Function encountered a TypeError for input parameter root_path."
                    "Message: parameter root_path expects a str.")
        raise TypeError("Get_directory_walk: parameter root_path expects a str.")
    elif not os.path.exists(root_path):
        logger.error("Function encountered an OSError for input parameter root_path."
                    "Message: parameter root_path is not a valid path string.")
        raise OSError("Get_directory_walk: parameter root_path must be a valid directory path in the current scope.")
    elif os.path.isfile(root_path):
        logger.error("Function encountered a ValueError for input parameter root_path."
                    "Message: parameter root_path must be a path string to a directory.")
        raise ValueError("Get_directory_walk: parameter root_path must contain a path string to a directory.")
    
    if not isinstance(with_sub_dirs, bool):
        logger.error("Function encountered a TypeError for input parameter with_sub_dirs."
                    "Message: parameter with_sub_dirs expects a bool.")
        raise TypeError("Get_directory_walk: parameter with_sub_dirs expects a bool.")

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
    if not isinstance(input_dir, str):
        logger.error("Function encountered a TypeError for input parameter input_dir."
                    "Message: parameter input_dir expects a str.")
        raise TypeError("Map_directory_structure: parameter input_dir expects a str.")
    elif not os.path.exists(input_dir):
        logger.error("Function encountered an OSError for input parameter input_dir."
                    "Message: parameter input_dir is not a valid path string.")
        raise OSError("Map_directory_structure: parameter input_dir must be a valid directory path in the current scope.")
    elif os.path.isfile(input_dir):
        logger.error("Function encountered a ValueError for input parameter input_dir."
                    "Message: parameter input_dir must be a path string to a directory.")
        raise ValueError("Map_directory_structure: parameter input_dir must contain a path string to a directory.")
    
    if not isinstance(output_dir, str):
        logger.error("Function encountered a TypeError for input parameter output_dir."
                    "Message: parameter output_dir expects a str.")
        raise TypeError("Map_directory_structure: parameter output_dir expects a str.")
    elif not os.path.exists(output_dir):
        logger.error("Function encountered an OSError for input parameter output_dir."
                    "Message: parameter output_dir is not a valid path string.")
        raise OSError("Map_directory_structure: parameter output_dir must be a valid directory path in the current scope.")
    elif os.path.isfile(output_dir):
        logger.error("Function encountered a ValueError for input parameter output_dir."
                    "Message: parameter output_dir must be a path string to a directory.")
        raise ValueError("Map_directory_structure: parameter output_dir must contain a path string to a directory.")
    
    if not isinstance(with_sub_dirs, bool):
        logger.error("Function encountered a TypeError for input parameter with_sub_dirs."
                    "Message: parameter with_sub_dirs expects a bool.")
        raise TypeError("Map_directory_structure: parameter with_sub_dirs expects a bool.")

    dir_names = None
    if with_sub_dirs:
        (dir_names, files) = get_directory_walk(input_dir, with_sub_dirs)
    else:
        files = get_directory_walk(input_dir, with_sub_dirs)

    if dir_names is not None:
        for dir in dir_names:
            create_output_directory(output_dir, dir)