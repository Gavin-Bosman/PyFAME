from pyfame.utilities.exceptions import *
import cv2 as cv
from typing import Any
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import mediapipe as mp
from pyfame.file_access.file_access_paths import get_landmarker_task_path
from pyfame.landmark.facial_landmarks import *

def get_face_landmarker(running_mode:str = "image", num_faces:int = 1, min_face_detection_confidence:float = 0.4, 
                        min_face_presence_confidence:float = 0.7, min_tracking_confidence:float = 0.7, 
                        output_face_blendshapes:bool = False, output_transform_matrixes:bool = False):
    task_path = get_landmarker_task_path()

    match running_mode.lower():
        case "image":
            running_mode = VisionTaskRunningMode.IMAGE
        case "video":
            running_mode = VisionTaskRunningMode.VIDEO
        case _:
            raise ValueError("Unrecognized value passed to parameter running_mode. Expects one of 'image' or 'video'.")

    baseOptions = python.BaseOptions(model_asset_path=task_path)
    options = vision.FaceLandmarkerOptions(
        base_options = baseOptions,
        running_mode = running_mode,
        num_faces = num_faces,
        min_face_detection_confidence = min_face_detection_confidence,
        min_face_presence_confidence = min_face_presence_confidence,
        min_tracking_confidence = min_tracking_confidence,
        output_face_blendshapes = output_face_blendshapes,
        output_facial_transformation_matrixes = output_transform_matrixes
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    return detector

def get_pixel_coordinates(frame_rgb:cv.typing.MatLike, face_landmarker:Any, timestamp_msec:float | None = None) -> tuple[tuple[list[tuple[int,int]]], list[Any]] | tuple[list[tuple[int,int]]]:
    
    # Save the orignal dimensions for determining padding
    original_h, original_w = frame_rgb.shape[:2]

    # Pad to square dimensions before face landmarking
    if original_h > original_w:
        pad = (original_h - original_w) // 2
        padded_frame = cv.copyMakeBorder(frame_rgb, 0, 0, pad, pad, cv.BORDER_CONSTANT, value=(0,0,0))

        vert_pad, horiz_pad = 0, pad
    elif original_w > original_h:
        pad = (original_w - original_h) // 2
        padded_frame = cv.copyMakeBorder(frame_rgb, pad, pad, 0, 0, cv.BORDER_CONSTANT, value=(0,0,0))

        vert_pad, horiz_pad = pad, 0
    else:
        padded_frame = frame_rgb
        vert_pad, horiz_pad = 0, 0
    
    # Get the new image dimensions after padding
    padded_h, padded_w = padded_frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=padded_frame)

    if face_landmarker._running_mode == VisionTaskRunningMode.IMAGE:
        lm_results = face_landmarker.detect(mp_image)
    elif face_landmarker._running_mode == VisionTaskRunningMode.VIDEO:
        if timestamp_msec is None:
            raise ValueError("Face_landmarker requires the current frame's timestamp when operating in video mode.")
        # Video running mode requires a timestamp in u-seconds
        lm_results = face_landmarker.detect_for_video(mp_image, int(timestamp_msec * 1000))
    
    if lm_results.face_landmarks:
        pixel_coords = []
        for lm in lm_results.face_landmarks[0]:
            x_pad = int(lm.x * padded_w)
            y_pad = int(lm.y * padded_h)

            x_pix = x_pad - horiz_pad
            y_pix = y_pad - vert_pad

            pixel_coords.append((x_pix, y_pix))
    else:
        raise FaceNotFoundError()
    
    if lm_results.face_blendshapes:
        return (pixel_coords, lm_results.face_blendshapes[0])
    return pixel_coords

def get_pixel_coordinates_from_landmark(landmarker_coordinates:Any, landmark_path:list[tuple]) -> list[tuple[int,int]]:
    path_screen_coords = []

    if landmark_path in CONCAVE_LANDMARKS:
        landmark_path = get_concave_landmark_coordinates(landmark_path)

    for i, (cur_source, cur_target) in enumerate(landmark_path):
        if i == 0:
            source = landmarker_coordinates[cur_source]
            path_screen_coords.append((source[0], source[1]))
       
        target = landmarker_coordinates[cur_target]
        path_screen_coords.append((target[0], target[1]))
    
    return path_screen_coords

def get_concave_landmark_coordinates(concave_path) -> list[tuple[int,int]]:

    lc_path = create_landmark_path(LEFT_CHEEK_IDX)
    rc_path = create_landmark_path(RIGHT_CHEEK_IDX)
    chin_path = create_landmark_path(CHIN_IDX)
    lips_path = create_landmark_path(LIPS_IDX)
    output_path = []

    match concave_path:
        # Both Cheeks
        case [(0,)]:
            output_path.extend(lc_path)
            output_path.extend(rc_path)
        
        # Left Cheek Only
        case [(1,)]:
            output_path.extend(lc_path)
        
        # Right Cheek Only
        case [(2,)]:
            output_path.extend(rc_path)

        # Cheeks and Nose
        case [(3,)]:
            output_path.extend(lc_path)
            output_path.extend(rc_path) 
            output_path.extend(LANDMARK_NOSE)
        
        # Both eyes
        case [(4,)]:
            output_path.extend(LANDMARK_LEFT_EYE_REGION)
            output_path.extend(LANDMARK_RIGHT_EYE_REGION)

        # Face Skin
        case [(5,)]:
           output_path.extend(LANDMARK_FACE_OVAL)
           output_path.extend(LANDMARK_LEFT_EYE_REGION)
           output_path.extend(LANDMARK_RIGHT_EYE_REGION)
           output_path.extend(LANDMARK_LIPS)
        # Chin
        case [(6,)]:    
            output_path.extend(chin_path)
        
        case [(7,)]:
            output_path.extend(lips_path)
        
        case [(8,)]:
            output_path.extend(LANDMARK_LEFT_EYE, LANDMARK_RIGHT_EYE)
        
        case [(9,)]:
            output_path.extend(LANDMARK_LEFT_IRIS, LANDMARK_RIGHT_IRIS)

        case _:
            raise ValueError("The concave path you have passed is either unrecognized or incompatible with get_concave_path_coords().")
    
    return output_path