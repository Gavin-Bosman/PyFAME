from pyfame.utilities.constants import *
from pyfame.landmark import *
from pyfame.file_access import *
from pyfame.utilities.exceptions import *
import os
import cv2 as cv
import numpy as np
from skimage.util import *
import itertools

def generate_shuffled_block_array(file_path:str, shuffle_method:int = TEMPORAL_SHUFFLE_RANDOM, rand_seed:int|None = None, block_duration:int = 1000) -> tuple:
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

    capture.release()
    return (block_order, block_size)

def apply_frame_shuffle(input_directory:str, output_directory:str, shuffle_method:int = TEMPORAL_SHUFFLE_RANDOM, rand_seed:int|None = None, 
                        block_order:tuple|None = None, block_duration:int = 1000, drop_last_block:bool = True) -> None:
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
    
    # Creating a list of file path strings to iterate through when processing
    file_paths = get_directory_walk(input_directory)
    files_to_process = file_paths["Absolute Path"]
    
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