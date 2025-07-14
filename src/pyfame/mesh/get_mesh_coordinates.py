from pyfame.utilities.util_exceptions import *
import cv2 as cv
import mediapipe as mp
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def get_mesh(min_tracking_confidence:float, min_detection_confidence:float, static_image_mode:bool, max_num_faces:int = 1) -> mp.solutions.face_mesh.FaceMesh:
    return mp.solutions.face_mesh.FaceMesh(max_num_faces = max_num_faces, static_image_mode = static_image_mode, 
                                           min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)

def get_mesh_coordinates(frame_rgb:cv.typing.MatLike, face_mesh:mp.solutions.face_mesh.FaceMesh) -> list[dict]:

    face_mesh_results = face_mesh.process(frame_rgb)
    landmark_screen_coords = []

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:

            # Convert normalised landmark coordinates to x-y pixel coordinates
            for id,lm in enumerate(face_landmarks.landmark):
                ih, iw, ic = frame_rgb.shape
                x,y = int(lm.x * iw), int(lm.y * ih)
                landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
    else:
        logger.error("Face mesh detection error, function exiting with status 1.")
        debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process() on the current frame.")
        raise FaceNotFoundError()
    
    return landmark_screen_coords

def get_mesh_coordinates_from_path(frame_rgb:cv.typing.MatLike, face_mesh:mp.solutions.face_mesh.FaceMesh, landmark_path:list[tuple]) -> list[tuple]:

    lm_screen_coords = get_mesh_coordinates(frame_rgb, face_mesh)
    path_screen_coords = []

    for cur_source, cur_target in landmark_path:
        source = lm_screen_coords[cur_source]
        target = lm_screen_coords[cur_target]
        path_screen_coords.append((source.get('x'), source.get('y')))
        path_screen_coords.append((target.get('x'), target.get('y')))
    
    return path_screen_coords
