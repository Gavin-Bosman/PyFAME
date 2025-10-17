from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, NonNegativeInt, PositiveFloat
from typing import Union, Tuple, Optional
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.blendshape_smoother import EyeBlendshapeSmoother
from pyfame.landmark.get_landmark_coordinates import get_pixel_coordinates_from_landmark
from pyfame.file_access import *
from pyfame.utilities import compute_rotation_angle, compute_slope
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask.mask_from_landmarks import mask_from_landmarks
from pyfame.utilities.exceptions import FileReadError
import cv2 as cv
import numpy as np
from pathlib import Path
import os
from collections import deque

_OVERLAY_DIR = Path(__file__).parent / "overlay_images"
# (127, 6) for right cheek scaling, center lm 119
# (6, 356) for left cheek scaling, center 348
# (127, 356) for facial-width scaling, center lm 6
_OVERLAY_MAPPING = {
    "sunglasses": {
        "path": _OVERLAY_DIR / "sunglasses.png",
        "anchor_landmarks": (127, 356),
        "scale_factor": None
    },
    "glasses": {
        "path": _OVERLAY_DIR / "glasses.png",
        "anchor_landmarks": (127, 356),
        "scale_factor": None
    },
    "teardrop_short_1": {
        "path": _OVERLAY_DIR / "teardrops" / "teardrop_short_1.png",
        "anchor_landmarks": (6, 356),
        "scale_factor": 0.2
    }
}

_OVERLAY_CACHE: dict[str, cv.typing.MatLike] = {}

def compute_scale(landmarker_coordinates:list[dict], anchor_landmarks:tuple[int,int], scale_factor:float = 1.0) -> float:

    if scale_factor is None:
        scale_factor = 1.0
    
    if anchor_landmarks is None:
        anchor_landmarks = (127, 356)
    
    p1 = np.array([
        landmarker_coordinates[anchor_landmarks[0]][0],
        landmarker_coordinates[anchor_landmarks[0]][1]
    ])

    p2 = np.array([
        landmarker_coordinates[anchor_landmarks[1]][0],
        landmarker_coordinates[anchor_landmarks[1]][1]
    ])
    
    return np.linalg.norm(p1-p2) * scale_factor

def load_overlay(overlay_name:str, landmarker_coordinates:list[tuple[int,int]], anchor_landmarks:tuple[int, int] | None = None, 
                 scale_factor:float | None = None) -> tuple[cv.typing.MatLike, tuple, tuple, float]:

    global _OVERLAY_CACHE

    # Pre-defined overlay type
    config = _OVERLAY_MAPPING[overlay_name]
    file_path = str(config["path"])
    scale_factor = config["scale_factor"]

    # lazy-loading the image
    if overlay_name not in _OVERLAY_CACHE:
        _OVERLAY_CACHE[overlay_name] = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        if _OVERLAY_CACHE[overlay_name] is None:
            raise FileReadError("Error reading in file.")
    
    img = _OVERLAY_CACHE[overlay_name]

    if anchor_landmarks is None:
        anchor_landmarks = config["anchor_landmarks"]
    
    cx = int((landmarker_coordinates[anchor_landmarks[0]][0] + landmarker_coordinates[anchor_landmarks[1]][0])/2.0)
    cy = int((landmarker_coordinates[anchor_landmarks[0]][1] + landmarker_coordinates[anchor_landmarks[1]][1])/2.0)
    center_point = (cx, cy)
    
    if scale_factor is None:
        scale = compute_scale(landmarker_coordinates, anchor_landmarks)
    else:
        scale = compute_scale(landmarker_coordinates, anchor_landmarks, scale_factor)
    
    return (img, anchor_landmarks, center_point, scale)
        
class OverlayParameters(BaseModel):
    overlay_type:Union[NonNegativeInt, str]
    overlay_bounding_landmarks:Optional[Tuple[int,int]]
    overlay_scale_factor:Optional[float] = None
    y_offset:int
    pupil_scale_factor:PositiveFloat

    @field_validator("overlay_type", mode="before")
    @classmethod
    def check_accepted_value(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if isinstance(value, str):
            value = str.lower(value)
            if value in {"sunglasses", "glasses", "teardrop_short_1", "pupil_dilation"}:
                return value
            
            elif os.path.isfile(value):
                return value
            
            raise ValueError(f"Unrecognized value or invalid file path provided to parameter {field_name}.")
            
        elif isinstance(value, int):
            if value not in {43,44,45,46}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            
            mapping = {43: "sunglasses", 44:"glasses", 45:"teardrop_short_1", 46:"pupil_dilation"}
            return mapping.get(value)
        
        raise TypeError(f"Invalid type for parameter {field_name}. Expected int or str.")
    
    @field_validator("overlay_bounding_landmarks")
    @classmethod
    def check_landmarks_in_range(cls, value, info:ValidationInfo):
        if value is not None:
            for lm in value:
                if not 0 <= lm <= 477:
                    raise ValueError("FaceLandmarker landmark ID's lie in the range [0-477].")
        
        return value
    
    @field_validator("pupil_scale_factor")
    @classmethod
    def check_normalised_range(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if not 0.0 < value <= 1.0:
            raise ValueError(f"Parameter {field_name} must lie in the normalised range (0.0-1.0].")
        
        return value
    
class LayerOverlay(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, overlay_parameters:OverlayParameters):

        self.time_config = timing_configuration
        self.overlay_params = overlay_parameters

        # Initialise superclass
        super().__init__(self.time_config)
        
        # Declare class parameters
        self.overlay_type = self.overlay_params.overlay_type
        self.anchor_landmarks = self.overlay_params.overlay_bounding_landmarks
        self.overlay_scale_factor = self.overlay_params.overlay_scale_factor
        self.y_offset = self.overlay_params.y_offset
        self.pupil_scale_factor = self.overlay_params.pupil_scale_factor
        
        # For eye related overlays
        self.eye_blendshape_smoother = EyeBlendshapeSmoother(frame_window_size=1)
        self.left_pupil_radius_deque = deque(maxlen=5)
        self.right_pupil_radius_deque = deque(maxlen=5)

        # Snapshot of initial state
        self._snapshot_state()

    def supports_weight(self):
        return False

    def get_layer_parameters(self) -> dict:
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.overlay_params.model_dump())
        self._layer_parameters["time_onset"] = self.onset_t
        self._layer_parameters["time_offset"] = self.offset_t
        return dict(self._layer_parameters)
    
    def apply_layer(self, landmarker_coordinates, frame, dt, blendshapes):

        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        
        overlayed_frame = frame.copy()
        left_eye_open, right_eye_open = self.eye_blendshape_smoother.update(blendshapes)

        if self.overlay_type == "pupil_dilation":

            left_iris_arr = np.array(get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_LEFT_IRIS))
            right_iris_arr = np.array(get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_RIGHT_IRIS))

            li_xs = left_iris_arr[:, 0]
            li_ys = left_iris_arr[:, 1]
            ri_xs = right_iris_arr[:, 0]
            ri_ys = right_iris_arr[:, 1]

            # Use the centroid of the iris landmarks to center the pupil overlay
            centroid_left_pupil = (int(round(np.mean(li_xs))), int(round(np.mean(li_ys)) + 3))
            centroid_right_pupil = (int(round(np.mean(ri_xs))), int(round(np.mean(ri_ys)) + 3))

            # left and right cantha of the left eye: landmarks 362, 263
            p1 = landmarker_coordinates[362]
            p2 = landmarker_coordinates[263]
            left_eye_canthal_width = np.linalg.norm(np.array(p1) - np.array(p2))

            # left and right cantha of the right eye: landmarks 33, 133
            p1 = landmarker_coordinates[33]
            p2 = landmarker_coordinates[133]
            right_eye_canthal_width = np.linalg.norm(np.array(p1) - np.array(p2))

            # Scale pupil radius by the eye width, according to average 
            # eye-width to pupil diameter ratios of adults found in the literature. 
            # Typical adult eye width is 27-32mm, while typical diameter of a dilated pupil is 4-8mm.
            # Thus, typical pupil diameter:eye-width ratios in adults [0.130-0.300]
            self.left_pupil_radius_deque.append(int((left_eye_canthal_width * self.pupil_scale_factor)/2))
            self.right_pupil_radius_deque.append(int((right_eye_canthal_width * self.pupil_scale_factor)/2))
            left_pupil_radius = int(np.mean(self.left_pupil_radius_deque))
            right_pupil_radius = int(np.mean(self.right_pupil_radius_deque))
            pupil_radius = int((left_pupil_radius + right_pupil_radius)/2)

            pupil_overlay = frame.copy()

            if left_eye_open:
                sclera_mask = mask_from_landmarks(frame, LANDMARK_LEFT_EYE, landmarker_coordinates)
                pupil_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv.circle(pupil_mask, centroid_left_pupil, pupil_radius, (255,255,255), -1)

                # Overlay the pupil only where it lies in the visible sclera
                masked_pupil = cv.bitwise_and(pupil_mask, sclera_mask)
                pupil_overlay[masked_pupil > 0] = (0,0,0)

            if right_eye_open:
                sclera_mask = mask_from_landmarks(frame, LANDMARK_RIGHT_EYE, landmarker_coordinates)
                pupil_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv.circle(pupil_mask, centroid_right_pupil, pupil_radius, (255,255,255), -1)

                # Overlay the pupil only where it lies in the visible sclera
                masked_pupil = cv.bitwise_and(pupil_mask, sclera_mask)
                pupil_overlay[masked_pupil > 0] = (0,0,0)
            
            overlayed_frame = cv.addWeighted(pupil_overlay, 0.7, overlayed_frame, 0.3, 0)

        else:
            overlay, anchor_lms, center_point, scale = load_overlay(
                overlay_name=self.overlay_type,
                landmarker_coordinates=landmarker_coordinates,
                anchor_landmarks=self.anchor_landmarks,
                scale_factor=self.overlay_scale_factor
            )

            # Rescaling the overlay to match 
            scaling_factor = 1/(overlay.shape[1]/(scale))
            overlay = cv.resize(src=overlay, dsize=None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)

            # Save the overlay img dimensions for later
            overlay_width = overlay.shape[1]
            overlay_height = overlay.shape[0]
            
            # Compute the angle from the x axis 
            p1 = landmarker_coordinates[anchor_lms[0]]
            p2 = landmarker_coordinates[anchor_lms[1]]
            cur_slope = compute_slope(p1, p2)
            rotation_angle = compute_rotation_angle(slope_1=cur_slope)
                
            # Add transparent padding prior to rotation
            diag_size = int(np.ceil(np.sqrt(overlay_height**2 + overlay_width**2)))
            pad_h = (diag_size-overlay_height)//2
            pad_w = (diag_size-overlay_width)//2
            padded = np.zeros((diag_size, diag_size, 4), dtype=np.uint8)
            padded[pad_h:pad_h+overlay_height, pad_w:pad_w + overlay_width] = overlay

            # Get center point of padded overlay
            padded_height = padded.shape[0]
            padded_width = padded.shape[1]
            padded_center = (padded_width//2, padded_height//2)

            # Rotate the overlay to match the angle of inclination of the head
            rot_mat = cv.getRotationMatrix2D(padded_center, rotation_angle, 1)
            overlay = cv.warpAffine(padded, rot_mat, (padded_width, padded_height), 
                                    flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0,0,0,0))

            # Generate a binary mask of the overlay for addition onto original frame
            overlay_img = overlay[:,:,:3]
            overlay_mask = overlay[:,:,3] / 255.0
            overlay_mask = overlay_mask[:,:,np.newaxis]

            x_pos = center_point[0] - padded_width//2
            y_pos = center_point[1] - padded_height//2 + self.y_offset

            roi = frame[y_pos:y_pos + padded_height, x_pos:x_pos + padded_width]
            blended = (1.0 - overlay_mask) * roi + overlay_mask * overlay_img

            overlayed_frame[y_pos:y_pos + padded_height, x_pos:x_pos + padded_width] = blended.astype(np.uint8)

        return overlayed_frame
        
def layer_overlay(timing_configuration:TimingConfiguration | None = None, overlay_type:int|str = "sunglasses", overlay_bounding_landmarks:tuple[int,int] | None = None, 
                  overlay_scale_factor:float | None = None, y_offset:int = 20, pupil_scale_factor:float = 0.25) -> LayerOverlay:
    
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = OverlayParameters(
            overlay_type=overlay_type,
            overlay_bounding_landmarks=overlay_bounding_landmarks, 
            overlay_scale_factor=overlay_scale_factor, 
            y_offset = y_offset,
            pupil_scale_factor=pupil_scale_factor
        )
        
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerOverlay.__name__}: {e}")
    
    return LayerOverlay(time_config, params)