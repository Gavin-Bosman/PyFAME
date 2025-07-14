from pyfame.utilities.util_constants import *
from pyfame.mesh import *
from pyfame.file_access import *
from pyfame.utilities.util_general_utilities import get_variable_name
from pyfame.utilities.util_exceptions import *
import os
import cv2 as cv
import numpy as np
from skimage.util import *
import itertools
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def generate_shuffled_block_array(file_path:str, shuffle_method:int = FRAME_SHUFFLE_RANDOM, rand_seed:int|None = None, block_duration:int = 1000) -> tuple:
    """ This function takes a given file path and shuffling method, and returns a tuple containing the block_order array and the block size parameter used
    in shuffle_frame_order(). The output of generate_shuffled_block_array can be directly fed into shuffle_frame_order as the block_order parameter.

    Parameters
    ----------

    file_path: str
        A path string to a video file in your current working directory tree.

    shuffle_method: int
        An integer flag indicating the shuffling method used. For a full list of available options please see 
        pyfame_utils.display_shuffle_method_options().

    rand_seed: int | None
        A seed to numpy's random number generator.
    
    block_duration: int
        The precise time duration in milliseconds of each block of frames.

    Raises
    ------

    OSError: given invalid input file paths.
    TypeError: given invalid parameter types.
    ValueError: given unrecognized values for input parameters.
    """ 

    logger.info("Now entering function generate_shuffled_block_array().")

    # Input parameter checks
    if not isinstance(file_path, str):
        logger.warning("Function encountered a TypeError for input parameter file_path. "
                       "Message: invalid type for parameter file_path, expected str.")
        raise TypeError("Generate_shuffled_block_array: invalid type for parameter file_path.")
    elif not os.path.exists(file_path):
        logger.warning("Function encountered an OSError for input parameter file_path. "
                       "Message: file_path is not a valid path in your current working tree.")
        raise OSError("Generate_shuffled_block_array: file_path is not a valid path, or path does not exist.")
    elif not os.path.isfile(file_path):
        logger.warning("Function encountered a ValueError for input parameter file_path. "
                       "Message: file_path is not a valid path to a file in your current working tree.")
        raise ValueError("Generate_shuffled_block_array: file_path must be a valid path to a video file.")
    
    if not isinstance(shuffle_method, int):
        logger.warning("Function encountered a TypeError for input parameter shuffle_method. "
                       "Message: invalid type for parameter shuffle_method, expected int.")
        raise TypeError("Generate_shuffled_block_array: parameter running_mode must be an integer.")
    if shuffle_method not in FRAME_SHUFFLE_METHODS:
        logger.warning("Function has encountered a ValueError for input parameter shuffle_method. "
                       "Message: unrecognized value for parameter shuffle_method. For a full list of "
                       "accepted values, please see pyfame_utils.display_shuffle_method_options().")
        raise ValueError("Shuffle_frame_order: unrecognized value for parameter shuffle_method.")

    if rand_seed != None:
        if not isinstance(rand_seed, int):
            logger.warning("Function encountered a TypeError for input parameter rand_seed. "
                       "Message: invalid type for parameter rand_seed, expected int.")
            raise TypeError("Generate_shuffled_block_array: parameter rand_seed must be an integer.")
    
    if not isinstance(block_duration, int):
        logger.warning("Function encountered a TypeError for input parameter block_duration. "
                       "Message: invalid type for parameter block_duration, expected int.")
        raise TypeError("Generate_shuffled_block_array: parameter block_size must be an integer")
    elif block_duration < 34:
        logger.warning("Function encountered a ValueError for input parameter block_duration. "
                       "Message: block_duration must be greater than 34ms (the time duration of a single frame).")
        raise ValueError("Generate_shuffled_block_array: parameter block_duration must be greater than 34ms (the time duration of a single frame).")
    
    # Logging input parameters
    shuffle_method_name = get_variable_name(shuffle_method, globals())
    logger.info(f"Input Parameters: file_path = {file_path}, shuffle_method = {shuffle_method_name}, rand_seed = {rand_seed}, "
                f"block_duration = {block_duration}.")

    # Initialize videocapture object
    capture = get_video_capture(file_path=file_path)
    
    # Initialize numpy random number generator
    rng = None
    if rand_seed != None:
        rng = np.random.default_rng(seed=rand_seed)
    else:
        rng = np.random.default_rng()

    # Retrieve relevant capture properties to compute block_size
    num_frames = capture.get(cv.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv.CAP_PROP_FPS)
    video_duration = (num_frames/fps) * 1000
    num_blocks = (video_duration / block_duration)
    block_size = int(num_frames//num_blocks)

    num_blocks = int(np.ceil(num_blocks))
    block_order = []

    for i in range(num_blocks):
        block_order.append(i)

    match shuffle_method:
        case 34:
            rng.shuffle(block_order)

        case 35:
            block_order = rng.choice(block_order, size=num_blocks, replace=True)
        
        case 36:
            block_order.reverse()

    logger.info(f"Function execution completed, returning ({block_order}, {block_size}).")
    capture.release()
    return (block_order, block_size)

def apply_frame_shuffle(input_directory:str, output_directory:str, shuffle_method:int = FRAME_SHUFFLE_RANDOM, rand_seed:int|None = None, block_order:tuple|None = None, 
                        block_duration:int = 1000, drop_last_block:bool = True, with_sub_dirs:bool = False) -> None:
    """For each video file contained within input_dir, randomly shuffles the frame order by shuffling blocks of frames. Use utility function generate_shuffled_block_array()
    to pre-generate the block_order list. The output of generate_shuffled_block_array() can be directly passed as the value of input parameter block_order. Alternatively, 
    simply specify the shuffle_method of choice and the block_duration, and shuffle_frame_order() will invoke generate_shuffled_block_array() internally. After shuffling the 
    block order, the function writes the processed file to output_dir. 
    
    Parameters
    ----------
    input_directory: str
        A path string to the directory containing input video files.

    output_directory: str
        A path string to the directory where outputted video files will be saved.

    shuffle_method: int
        An integer flag indicating the functions running mode. For a full list of available options,
        please see pyfame_utils.display_shuffle_method_options().
    
    rand_seed: int
        The seed number provided to the numpy random generator instance.
    
    block_order: tuple or None
        A tuple (as output of generate_shuffled_block_array()) containing a (list, int), where the list is the block ordering
        and the integer specifies the number of frames per block. If None, the tuple contents will be computed internally.

    block_duration: int
        The time duration (in milliseconds) of each block of frames.

    drop_last_block: bool
        A boolean flag indicating if the uneven block of remaining frames should be dropped from the output.
    
    with_sub_dirs: bool
        A boolean flag indicating if the input directory contains subdirectories.
    
    Raises
    ------

    TypeError: given invalid parameter types.
    OSError: given invalid file paths to input_dir or output_dir.
    """

    logger.info("Now entering function shuffle_frame_order().")

    # Performing checks on function parameters
    if not isinstance(input_directory, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Shuffle_frame_order: invalid type for parameter input_dir.")
    elif not os.path.exists(input_directory):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path in your current working tree.")
        raise OSError("Shuffle_frame_order: input directory path is not a valid path, or the directory does not exist.")

    if not isinstance(output_directory, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Shuffle_frame_order: parameter output_dir must be a str.")
    elif not os.path.exists(output_directory):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path in your current working tree.")
        raise OSError("Shuffle_frame_order: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_directory):
        logger.warning("Function encountered a ValueError for input parameter output_dir. "
                       "Message: output_dir is not a valid path to a directory in your current working tree.")
        raise ValueError("Shuffle_frame_order: output_dir must be a valid path to a directory.")
    
    if not isinstance(shuffle_method, int):
        logger.warning("Function encountered a TypeError for input parameter shuffle_method. "
                       "Message: invalid type for parameter shuffle_method, expected int.")
        raise TypeError("Shuffle_frame_order: parameter running_mode must be an integer.")
    if shuffle_method not in FRAME_SHUFFLE_METHODS:
        logger.warning("Function has encountered a ValueError for input parameter shuffle_method. "
                       "Message: unrecognized value for parameter shuffle_method. For a full list of "
                       "accepted values, please see pyfame_utils.display_shuffle_method_options().")
        raise ValueError("Shuffle_frame_order: unrecognized value for parameter shuffle_method.")

    if rand_seed != None:
        if not isinstance(rand_seed, int):
            logger.warning("Function encountered a TypeError for input parameter rand_seed. "
                       "Message: invalid type for parameter rand_seed, expected int.")
            raise TypeError("Shuffle_frame_order: parameter rand_seed must be an integer.")
    
    if not isinstance(block_duration, int):
        logger.warning("Function encountered a TypeError for input parameter block_duration. "
                       "Message: invalid type for parameter block_duration, expected int.")
        raise TypeError("Shuffle_frame_order: parameter block_size must be an integer")
    elif block_duration < 34:
        logger.warning("Function encountered a ValueError for input parameter block_duration. "
                       "Message: block_duration must be greater than 34ms (the time duration of a single frame).")
        raise ValueError("Shuffle_frame_order: parameter block_duration must be greater than 34ms (the time duration of a single frame).")
    
    if block_order != None:
        if not isinstance(block_order, tuple):
            logger.warning("Function encountered a TypeError for input parameter block_order. "
                           "Message: invalid type for parameter block_order, expected tuple.")
            raise TypeError("Shuffle_frame_order: parameter block_order must be a tuple(list, int).")
        elif not isinstance(block_order[0], list) or not isinstance(block_order[1], int):
            logger.warning("Function encountered a ValueError for input parameter block_order. "
                           "Message: Unrecognized value of parameter block_order, expected a tuple of (list, int).")
            raise ValueError("Shuffle_frame_order: parameter block_order must be a tuple of (list, int).")
    
    if not isinstance(drop_last_block, bool):
        logger.warning("Function Encountered a TypeError for input parameter drop_last_block. "
                       "Message: invalid type for parameter drop_last_block, expected bool.")
        raise TypeError("Shuffle_frame_order: parameter drop_last_block must be a bool.")

    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function Encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Shuffle_frame_order: parameter with_sub_dirs must be a bool.")
    
    # Logging input parameters
    shuffle_method_name = get_variable_name(shuffle_method, globals())
    logger.info(f"Input Parameters: shuffle_method = {shuffle_method_name}, rand_seed = {rand_seed}, block_order = {block_order}, "
                f"block_duration = {block_duration}, drop_last_block = {drop_last_block}, with_sub_dirs = {with_sub_dirs}.")
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = get_directory_walk(input_directory, with_sub_dirs)
        
    logger.info(f"Function has read in {len(files_to_process)} files from input directory {input_directory}.")
    
    # Creating named output directories for video output
    output_directory = create_output_directory(output_directory, "Frame_Shuffled")

    for file in files_to_process:
            
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        block_size = None
        dir_file_path = output_directory + f"\\{filename}_frame_shuffled{extension}"

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "mp4v"
            case ".mov":
                codec = "mp4v"
            case _:
                logger.error("Function has encountered an unparseable file type, Function exiting with status 1. " 
                             "Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                   "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                raise UnrecognizedExtensionError()
      
        capture = get_video_capture(file)
        size = (int(capture.get(3)), int(capture.get(4)))

        result = get_video_writer(dir_file_path, size, video_codec=codec)

        if block_order is None:
            block_order, block_size = generate_shuffled_block_array(file, shuffle_method, rand_seed, block_duration)
        else:
            block_order, block_size = block_order
        
        match shuffle_method:
            case 34 | 35 | 36:
                shuffled_frames = {}
                counter = 0
                cur_block = []

                # Read in and store all frames
                while ret := capture.read():
                    success, frame = ret
                    if success:
                        cur_block.append(frame)
                        if len(cur_block) == block_size:
                            shuffled_frames.update({counter:cur_block.copy()})
                            cur_block = []
                            counter += 1
                    elif len(cur_block) > 0:
                        shuffled_frames.update({counter:cur_block.copy()})
                        break
                    else:
                        break
                
                if drop_last_block:
                    key = len(shuffled_frames) - 1 
                    shuffled_frames.pop(key)

                    largest_key_idx = block_order.index(max(block_order))
                    del block_order[largest_key_idx]

                original_keys = list(shuffled_frames.keys())
                ref_dict = shuffled_frames.copy()

                for old_key,new_key in zip(original_keys,block_order):
                    new_block = ref_dict[new_key]
                    shuffled_frames.update({old_key:new_block})
                    
                for key in original_keys:
                    block = shuffled_frames.get(key)
                    for out_frame in block:
                        result.write(out_frame)
            
            case 37:
                shuffled_frames = {}
                counter = 0
                cur_block = []

                # Read in and store all frames
                while ret := capture.read():
                    success, frame = ret
                    if success:
                        cur_block.append(frame)
                        if len(cur_block) == block_size:
                            shuffled_frames.update({counter:cur_block.copy()})
                            cur_block = []
                            counter += 1
                    elif len(cur_block) > 0:
                        shuffled_frames.update({counter:cur_block.copy()})
                        break
                    else:
                        break
                
                if drop_last_block:
                    key = len(shuffled_frames) - 1 
                    shuffled_frames.pop(key)
                
                original_keys = list(shuffled_frames.keys())

                # Perform right cyclic shift
                for key in original_keys:
                    block = shuffled_frames.get(key)
                    new_block = [block[-1]] + block[:-1]
                    shuffled_frames.update({key:new_block})
                
                for key in original_keys:
                    block = shuffled_frames.get(key)
                    for out_frame in block:
                        result.write(out_frame)
            
            case 38:
                shuffled_frames = {}
                counter = 0
                cur_block = []

                # Read in and store all frames
                while ret := capture.read():
                    success, frame = ret
                    if success:
                        cur_block.append(frame)
                        if len(cur_block) == block_size:
                            shuffled_frames.update({counter:cur_block.copy()})
                            cur_block = []
                            counter += 1
                    elif len(cur_block) > 0:
                        shuffled_frames.update({counter:cur_block.copy()})
                        break
                    else:
                        break
                
                if drop_last_block:
                    key = len(shuffled_frames) - 1 
                    shuffled_frames.pop(key)
                
                original_keys = list(shuffled_frames.keys())
                # Perform left cyclic shift
                for key in original_keys:
                    block = shuffled_frames.get(key)
                    new_block = block[1:] + [block[0]]
                    shuffled_frames.update({key:new_block})
                    
                for key in original_keys:
                    block = shuffled_frames.get(key)
                    for out_frame in block:
                        result.write(out_frame)
            
            case 39:
                shuffled_frames = {}
                counter = 0
                cur_block = []

                # Read in and store all frames
                while ret := capture.read():
                    success, frame = ret
                    if success:
                        cur_block.append(frame)
                        if len(cur_block) == block_size:
                            shuffled_frames.update({counter:cur_block.copy()})
                            cur_block = []
                            counter += 1
                    elif len(cur_block) > 0:
                        shuffled_frames.update({counter:cur_block.copy()})
                        break
                    else:
                        break
                
                if drop_last_block:
                    key = len(shuffled_frames) - 1 
                    shuffled_frames.pop(key)
                
                shuffled_palindrome_frames = {}
                original_keys = list(shuffled_frames.keys())
                counter = 0

                # Perform palindrome stutter
                for key in original_keys:
                    block = shuffled_frames.get(key)
                    rev_block = block[::-1]
                    shuffled_palindrome_frames.update({counter:block})
                    shuffled_palindrome_frames.update({counter+1:rev_block})
                    counter += 2
                
                palindrome_keys = list(shuffled_palindrome_frames.keys())
                for key in palindrome_keys:
                    block = shuffled_palindrome_frames.get(key)
                    for out_frame in block:
                        result.write(out_frame)

            case 40:
                shuffled_frames = {}
                counter = 0
                cur_block = []

                # Read in and store all frames
                while ret := capture.read():
                    success, frame = ret
                    if success:
                        cur_block.append(frame)
                        if len(cur_block) == block_size:
                            shuffled_frames.update({counter:cur_block.copy()})
                            cur_block = []
                            counter += 1
                    elif len(cur_block) > 0:
                        shuffled_frames.update({counter:cur_block.copy()})
                        break
                    else:
                        break

                shuffled_frames.pop(len(shuffled_frames)-1)

                # perform interleaving of frames
                interleaved_frames = list(itertools.chain(*zip(*(shuffled_frames.values()))))

                for frame in interleaved_frames:
                    result.write(frame)
        
        capture.release()
        result.release()
        logger.info(f"Function execution completed successfully, view outputted file(s) at {dir_file_path}.")