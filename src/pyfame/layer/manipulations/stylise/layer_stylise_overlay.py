from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, NonNegativeInt
from typing import Union, Tuple, Optional, Any
from pyfame.mesh import *
from pyfame.file_access import *
from pyfame.utilities import compute_rot_angle
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.utilities.exceptions import FileReadError
import cv2 as cv
import numpy as np
from pathlib import Path
import os

_OVERLAY_DIR = Path(__file__).parent / "overlay_images"

_OVERLAY_MAPPING = {
    "sunglasses": {
        "path": _OVERLAY_DIR / "sunglasses.png",
        "anchor_landmarks": (227, 447),
        "centre_landmark": 6,
        "scale": None
    },
    "glasses": {
        "path": _OVERLAY_DIR / "glasses.png",
        "anchor_landmarks": (227, 447),
        "centre_landmark": 6,
        "scale": None
    },
    "teardrop": {
        "path": _OVERLAY_DIR / "teardrop.png",
        "anchor_landmarks": (127, 6),
        "centre_landmark": 119,
        "scale": 0.25
    }
}

_OVERLAY_CACHE: dict[str, cv.typing.MatLike] = {}

def compute_scale(landmark_coordinates:list[dict], anchor_landmarks:tuple[int,int], scale_factor:float = 1.0) -> float:

    if landmark_coordinates is None:
        raise ValueError("Compute_scale requires a landmark screen coordinates mapping in order to compute the overlay image scale.")

    if scale_factor is None:
        scale_factor = 1.0
    
    if anchor_landmarks is None:
        anchor_landmarks = (227, 447)
    
    p1 = np.array([
        landmark_coordinates[anchor_landmarks[0]].get('x'),
        landmark_coordinates[anchor_landmarks[0]].get('y')
    ])

    p2 = np.array([
        landmark_coordinates[anchor_landmarks[1]].get('x'),
        landmark_coordinates[anchor_landmarks[1]].get('y')
    ])
    
    return np.linalg.norm(p1-p2) * scale_factor

def load_overlay(name_or_path:str, landmark_coordinates:list[dict], anchor_landmarks:tuple[int, ...] | None = None, 
                 centre_point:tuple[int,int] = None, scale_factor:float | None = None) -> tuple[cv.typing.MatLike, tuple, tuple, float]:

    global _OVERLAY_CACHE

    if landmark_coordinates is None:
        raise ValueError("load_overlay requires a landmark screen coordinate mapping to properly compute the overlay scale.")

    # Pre-defined overlay type
    if name_or_path in _OVERLAY_MAPPING:
        config = _OVERLAY_MAPPING[name_or_path]
        file_path = str(config["path"])
        scale_factor = config["scale"]

        # lazy-loading the image
        if name_or_path not in _OVERLAY_CACHE:
            _OVERLAY_CACHE[name_or_path] = cv.imread(file_path, cv.IMREAD_UNCHANGED)
            if _OVERLAY_CACHE[name_or_path] is None:
                raise FileReadError("Error reading in file.")
        
        img = _OVERLAY_CACHE[name_or_path]

        if anchor_landmarks is None:
            anchor_landmarks = config["anchor_landmarks"]
        
        if centre_point is None:
            centre = config["centre_landmark"]
            centre_point = (landmark_coordinates[centre].get('x'), landmark_coordinates[centre].get('y'))
        
        if scale_factor is None:
            scale = compute_scale(landmark_coordinates, anchor_landmarks)
        else:
            scale = compute_scale(landmark_coordinates, anchor_landmarks, scale_factor)
        
        return (img, anchor_landmarks, centre_point, scale)
    
    else:
        if not Path(name_or_path).exists():
            raise FileNotFoundError(f"Overlay image could not be found at '{name_or_path}'")
        if scale_factor is None or centre_point is None:
            raise ValueError("User-defined overlays require a 'centre_point' and a 'scale_factor'")
        
        if anchor_landmarks is None:
            anchor_landmarks = (227, 447)
        
        # Lazy-loading the image
        if name_or_path not in _OVERLAY_CACHE:
            _OVERLAY_CACHE[name_or_path] = cv.imread(name_or_path, cv.IMREAD_UNCHANGED)
            if _OVERLAY_CACHE[name_or_path] is None:
                raise FileReadError("Error reading in file.")
        
        scale = compute_scale(landmark_coordinates, anchor_landmarks, scale_factor)

        return (_OVERLAY_CACHE[name_or_path], anchor_landmarks, centre_point, scale)
        
class OverlayParameters(BaseModel):
    overlay_name_or_path:Union[NonNegativeInt, str]
    overlay_centre_point:Optional[Tuple[NonNegativeInt, NonNegativeInt]] = None
    overlay_scale_factor:Optional[float] = None

    @field_validator("overlay_name_or_path", mode="before")
    @classmethod
    def check_accepted_value(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if isinstance(value, str):
            value = str.lower(value)
            if value in {"sunglasses", "glasses", "teardrop", "teardrop_2"}:
                return value
            
            elif os.path.isfile(value):
                return value
            
            raise ValueError(f"Unrecognized value or invalid file path provided to parameter {field_name}.")
            
        elif isinstance(value, int):
            if value not in {43,44,45}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            
            mapping = {43: "sunglasses", 44:"glasses", 45:"teardrop"}
            return mapping.get(value)
        
        raise TypeError(f"Invalid type for parameter {field_name}. Expected int or str.")
    
    @field_validator("overlay_scale_factor")
    @classmethod
    def check_not_none(cls, value, info:ValidationInfo):
        field_name = info.field_name
        params = info.data

        # Check that an overlay type was passed, and that it is a custom type
        if params.get("overlay_name_or_path") and (params.get("overlay_name_or_path") not in {"sunglasses", "glasses", "teardrop", 43, 44, 45}):
            if value is None:
                raise ValueError(f"{field_name} is a required parameter when passing custom overlay paths.")
            elif not (0.0 < value <= 1.0):
                raise ValueError(f"{field_name} must be a float in the normal range 0.0 - 1.0.")
    
    @field_validator("overlay_centre_point")
    @classmethod
    def check_valid_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        params = info.data

        # Check that an overlay type was passed, and that it is a custom type
        if params.get("overlay_name_or_path") and (params.get("overlay_name_or_path") not in {"sunglasses", "glasses", "teardrop", 43, 44, 45}):
            if value is None:
                raise ValueError(f"{field_name} is a required parameter when passing custom overlay paths.")

class LayerStyliseOverlay(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, overlay_parameters:OverlayParameters):

        self.time_config = timing_configuration
        self.overlay_params = overlay_parameters

        # Initialise superclass
        super().__init__(self.time_config)
        
        # Declare class parameters
        self.overlay_name_or_path = self.overlay_params.overlay_name_or_path
        self.overlay_centre_point = self.overlay_params.overlay_centre_point
        self.overlay_scale_factor = self.overlay_params.overlay_scale_factor
        self.min_tracking_confidence = self.time_config.min_tracking_confidence
        self.min_detection_confidence = self.time_config.min_detection_confidence
        self.static_image_mode = False

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
    
    def apply_layer(self, face_mesh:Any, frame:cv.typing.MatLike, dt:float):

        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            # Get the pixel coordinates of the full face and face-oval
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            landmark_screen_coords = get_mesh_coordinates(frame_rgb, face_mesh)
            overlayed_frame = frame.copy()

            overlay, anchor_lms, centre_point, scale = load_overlay(
                name_or_path=self.overlay_name_or_path,
                landmark_coordinates=landmark_screen_coords,
                centre_point=self.overlay_centre_point,
                scale_factor=self.overlay_scale_factor
            )

            # Rescaling the overlay to match 
            overlay_width = overlay.shape[1]
            overlay_height = overlay.shape[0]
            scaling_factor = 1/(overlay_width/(scale))
            overlay = cv.resize(src=overlay, dsize=None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)
            # Reassign after scaling
            overlay_width = overlay.shape[1]
            overlay_height = overlay.shape[0]
            
            # Compute the angle from the x axis 
            p1 = landmark_screen_coords[anchor_lms[0]]
            p2 = landmark_screen_coords[anchor_lms[1]]
            cur_slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))
            rotation_angle = compute_rot_angle(slope_1=cur_slope)
                
            # Add transparent padding prior to rotation
            diag_size = int(np.ceil(np.sqrt(overlay_height**2 + overlay_width**2)))
            pad_h = (diag_size-overlay_height)//2
            pad_w = (diag_size-overlay_width)//2
            padded = np.zeros((diag_size, diag_size, 4), dtype=np.uint8)
            padded[pad_h:pad_h+overlay_height, pad_w:pad_w + overlay_width] = overlay

            # Get center point of padded overlay
            padded_height = padded.shape[0]
            padded_width = padded.shape[1]
            padded_center = (overlay_width//2, overlay_height//2)

            # Rotate the overlay to match the angle of inclination of the head
            rot_mat = cv.getRotationMatrix2D(padded_center, rotation_angle, 1)
                
            overlay = cv.warpAffine(padded, rot_mat, (padded_width, padded_height), flags=cv.INTER_LINEAR)

            # Generate a binary mask of the overlay for addition onto original frame
            overlay_img = overlay[:,:,:3]
            overlay_mask = overlay[:,:,3] / 255.0
            overlay_mask = overlay_mask[:,:,np.newaxis]
            overlay_width = overlay.shape[1]
            overlay_height = overlay.shape[0]

            x_pos = centre_point[0] - padded_width//2
            y_pos = centre_point[1] - padded_height//2

            roi = frame[y_pos:y_pos + padded_height, x_pos:x_pos + padded_width]
            blended = (1.0 - overlay_mask) * roi + overlay_mask * overlay_img

            overlayed_frame[y_pos:y_pos + padded_height, x_pos:x_pos + padded_width] = blended.astype(np.uint8)

            return overlayed_frame
        
def layer_stylise_overlay(timing_configuration:TimingConfiguration | None = None, overlay_name_or_path:int|str = "sunglasses", 
                          overlay_centre_point:tuple[int,int] | None = None, overlay_scale_factor:float | None = None) -> LayerStyliseOverlay:
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = OverlayParameters(
            overlay_name_or_path=overlay_name_or_path, 
            overlay_centre_point=overlay_centre_point, 
            overlay_scale_factor=overlay_scale_factor
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerStyliseOverlay.__name__}: {e}")
    
    return LayerStyliseOverlay(time_config, params)