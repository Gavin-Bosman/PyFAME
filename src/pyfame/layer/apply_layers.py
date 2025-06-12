from pyfame.io import map_directory_structure, get_video_capture, get_video_writer, get_directory_walk
from pyfame.util.util_exceptions import *
from pyfame.timing.timing_curves import *
from pyfame.mesh.get_mesh_landmarks import *
from pyfame.layer import layer, layer_pipeline
from typing import Callable
import cv2 as cv
import os


def apply_layers(layers:list[layer], input_dir:str, output_dir:str, onset_t:int = 0.0, offset_t:int = 0.0, timing_func:Callable[...,float] = timing_linear,
                 roi:list[list[tuple]] = [FACE_OVAL_PATH], with_sub_dirs:bool = False, **timing_kwargs):
    
    # Map the input dir structure to the output dir
    map_directory_structure(input_dir, output_dir, with_sub_dirs)
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)

    # Initialize the processing pipeline
    pipeline = layer_pipeline()
    for layer in layers:
        pipeline.add_layer(layer)

    for file in files_to_process:
        filename, extension = os.path.splitext(os.path.basename(file))
        static_image_mode = False
        codec = "mp4v"
        capture = None
        result = None
        cap_duration = 0
        dir_file_path = output_dir + f"\\{filename}_processed{extension}"
        
        if not static_image_mode:
            capture = get_video_capture(file)
            size = (int(capture.get(3)), int(capture.get(4)))
            result = get_video_writer(dir_file_path, size, codec)

            # Getting the video duration for weight calculations
            frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv.CAP_PROP_FPS)
            cap_duration = float(frame_count)/float(fps)

            if offset_t == 0.0:
                offset_t = cap_duration - 1.0
        
        # Loop over the current file until completion; (single iteration for static images)
        while(True):
            if static_image_mode:
                frame = cv.imread(file)
                if frame is None:
                    raise FileReadError()
            else:
                success, frame = capture.read()
                if not success:
                    break
            
            # declaring variables so they maintain their larger scope
            dt = None
            output_frame = frame.copy()

            if not static_image_mode:
                # Getting the current video timestamp
                dt = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                timing_kwargs.update({"start":onset_t, "end":offset_t})
                weight = timing_func(dt, **timing_kwargs)

                if dt < onset_t:
                    output_frame = pipeline.apply_layers(output_frame, 0.0, roi)
                elif dt < offset_t:
                    output_frame = pipeline.apply_layers(output_frame, weight, roi)
                else:
                    inv_dt = cap_duration - (dt-onset_t)
                    weight = timing_func(inv_dt, **timing_kwargs)
                    output_frame = pipeline.apply_layers(output_frame, weight, roi)
                
                result.write(output_frame)
                
            else:
                output_frame = pipeline.apply_layers(output_frame, 1.0, roi)
                success = cv.imwrite(dir_file_path, output_frame)
                if not success:
                    raise FileWriteError()
                
                break
        
        if not static_image_mode:
            capture.release()
            result.release()