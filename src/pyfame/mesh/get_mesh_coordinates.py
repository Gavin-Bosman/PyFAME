from pyfame.utilities.exceptions import *
import cv2 as cv
import numpy as np
from typing import Any
import mediapipe as mp

def get_mesh(min_tracking_confidence:float, min_detection_confidence:float, static_image_mode:bool, max_num_faces:int = 1):
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces = max_num_faces, 
        static_image_mode = static_image_mode, 
        min_detection_confidence = min_detection_confidence, 
        min_tracking_confidence = min_tracking_confidence
    )

def get_mesh_coordinates(frame_rgb:cv.typing.MatLike, face_mesh:Any) -> list[dict]:

    # Save the orignal dimensions for determining padding
    original_h, original_w = frame_rgb.shape[:2]

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

    face_mesh_results = face_mesh.process(frame_rgb)
    landmark_screen_coords = []

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:

            # Convert normalised landmark coordinates to x-y pixel coordinates
            for id,lm in enumerate(face_landmarks.landmark):
                # Extract padded coordinates 
                px = int(lm.x * padded_w)
                py = int(lm.y * padded_h)

                # Map back to original dim space
                orig_x = px - horiz_pad
                orig_y = py - vert_pad

                orig_x = np.clip(orig_x, 0, original_w - 1)
                orig_y = np.clip(orig_y, 0, original_h - 1)

                landmark_screen_coords.append({'id':id, 'x':orig_x, 'y':orig_y})
    else:
        raise FaceNotFoundError()
    
    return landmark_screen_coords

def get_mesh_coordinates_from_path(frame_rgb:cv.typing.MatLike, face_mesh:Any, landmark_path:list[tuple]) -> list[tuple]:

    lm_screen_coords = get_mesh_coordinates(frame_rgb, face_mesh)
    path_screen_coords = []

    for cur_source, cur_target in landmark_path:
        source = lm_screen_coords[cur_source]
        target = lm_screen_coords[cur_target]
        path_screen_coords.append((source.get('x'), source.get('y')))
        path_screen_coords.append((target.get('x'), target.get('y')))
    
    return path_screen_coords
