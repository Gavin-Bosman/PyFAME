from pyfame.io import map_directory_structure, get_video_capture, get_video_writer, get_directory_walk
from pyfame.mesh import get_mesh
from pyfame.util.util_exceptions import *
from .apply_timing_function import *
from typing import Callable
import cv2 as cv
from cv2.typing import MatLike
import os
import mediapipe as mp
import inspect

class layer:
    def __init__(self, func:Callable[...,MatLike], roi:list[list[tuple]], magnitude:float = 0.0, **kwargs):
        self.func = func
        self.roi = roi
        self.magnitude = magnitude
        self.kwargs = kwargs

    def apply_layer(self, frame:MatLike, weight:float, face_mesh:mp.solutions.face_mesh.FaceMesh):
        # Using the inspect module allows handling of different function types that may normally be incompatible
        sig = inspect.signature(self.func)
        if "weight" in sig.parameters:
            return self.func(frame, face_mesh, self.roi, weight, self.magnitude, **self.kwargs)
        else:
            return self.func(frame, face_mesh, self.roi, **self.kwargs)

class layered_pipeline:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer:layer):
        self.layers.append(layer)
    
    def apply_layers(self, frame:MatLike, weight:float, face_mesh:mp.solutions.face_mesh.FaceMesh) -> MatLike:
        for layer in self.layers:
            frame = layer.apply_layer(frame, weight, face_mesh)
        return frame

def layer_manipulations(layers:list[layer], input_dir:str, output_dir:str, onset_t:int = 0.0, offset_t:int = 0.0, timing_func:Callable[...,float] = timing_linear,
                        with_sub_dirs:bool = False, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **timing_kwargs):
    
    # Map the input dir structure to the output dir
    map_directory_structure(input_dir, output_dir, with_sub_dirs)
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
    # Initialize the processing pipeline
    pipeline = layered_pipeline()
    for layer in layers:
        pipeline.add_layer(layer)

    for file in files_to_process:
        filename, extension = os.path.splitext(os.path.basename(file))
        static_image_mode = False
        codec = None
        capture = None
        result = None
        face_mesh = None
        cap_duration = 0
        dir_file_path = output_dir + f"\\{filename}_processed{extension}"

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "mp4v"
                face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, static_image_mode)
            case ".mov":
                codec = "mp4v"
                face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, static_image_mode)
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                static_image_mode = True
                face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, static_image_mode)
            case _:
                raise UnrecognizedExtensionError()
        
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
        
        while(True):
            if static_image_mode:
                frame = cv.imread(file)
                if frame is None:
                    raise FileReadError()
            else:
                success, frame = capture.read()
                if not success:
                    break
            
            dt = None
            output_frame = frame.copy()
            if not static_image_mode:
                # Getting the current video timestamp
                dt = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                timing_kwargs.update({"start":onset_t, "end":offset_t})
                weight = timing_func(dt, **timing_kwargs)

                if dt < onset_t:
                    output_frame = pipeline.apply_layers(output_frame, 0.0, face_mesh)
                elif dt < offset_t:
                    output_frame = pipeline.apply_layers(output_frame, weight, face_mesh)
                else:
                    inv_dt = cap_duration - (dt-onset_t)
                    weight = timing_func(inv_dt, **timing_kwargs)
                    output_frame = pipeline.apply_layers(output_frame, weight, face_mesh)
                
                result.write(output_frame)
            else:
                output_frame = pipeline.apply_layers(output_frame, 1.0, face_mesh)
                success = cv.imwrite(dir_file_path, output_frame)
                if not success:
                    raise FileWriteError()
                
                break
        
        if not static_image_mode:
            capture.release()
            result.release()