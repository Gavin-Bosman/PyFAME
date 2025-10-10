from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, NonNegativeInt
from typing import Union, Tuple, Optional
from pyfame.landmark.facial_landmarks import *
from pyfame.file_access import *
from pyfame.utilities import compute_rot_angle, compute_slope
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.utilities.exceptions import FileReadError
import cv2 as cv
import numpy as np
from pathlib import Path
import os

_OVERLAY_DIR = Path(__file__).parent / "overlay_images"
# (6, 356) for left cheek scaling, center 348
# (127, 6) for right cheek scaling, center lm 119
# (127, 356) for facial-width scaling, center lm 6
_OVERLAY_MAPPING = {
    "sunglasses": {
        "path": _OVERLAY_DIR / "sunglasses.png",
        "anchor_landmarks": (127, 356),
        "scale": None
    },
    "glasses": {
        "path": _OVERLAY_DIR / "glasses.png",
        "anchor_landmarks": (127, 356),
        "scale": None
    },
    "tear_short_1": {
        "path": _OVERLAY_DIR / "tears" / "tear_short_1.png",
        "anchor_landmarks": (6, 356),
        "scale": 0.2
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

def load_overlay(name_or_path:str, landmarker_coordinates:list[tuple[int,int]], anchor_landmarks:tuple[int, ...] | None = None, 
                 scale_factor:float | None = None) -> tuple[cv.typing.MatLike, tuple, tuple, float]:

    global _OVERLAY_CACHE

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
        anchor_landmarks = config["anchor_landmarks"]
        
        cx = int((landmarker_coordinates[anchor_landmarks[0]][0] + landmarker_coordinates[anchor_landmarks[1]][0])/2.0)
        cy = int((landmarker_coordinates[anchor_landmarks[0]][1] + landmarker_coordinates[anchor_landmarks[1]][1])/2.0)
        center_point = (cx, cy)
        
        if scale_factor is None:
            scale = compute_scale(landmarker_coordinates, anchor_landmarks)
        else:
            scale = compute_scale(landmarker_coordinates, anchor_landmarks, scale_factor)
        
        return (img, anchor_landmarks, center_point, scale)
    else:
        if not Path(name_or_path).exists():
            raise FileNotFoundError(f"Overlay image could not be found at '{name_or_path}'")
        if scale_factor is None:
            raise ValueError("User-defined overlays require a 'scale_factor'.")
        
        if anchor_landmarks is None:
            anchor_landmarks = (127, 356)
        
        cx = int((landmarker_coordinates[anchor_landmarks[0]][0] + landmarker_coordinates[anchor_landmarks[1]][0])/2.0)
        cy = int((landmarker_coordinates[anchor_landmarks[0]][1] + landmarker_coordinates[anchor_landmarks[1]][1])/2.0)
        center_point = (cx, cy)
        
        # Lazy-loading the image
        if name_or_path not in _OVERLAY_CACHE:
            _OVERLAY_CACHE[name_or_path] = cv.imread(name_or_path, cv.IMREAD_UNCHANGED)
            if _OVERLAY_CACHE[name_or_path] is None:
                raise FileReadError("Error reading in file.")
        
        scale = compute_scale(landmarker_coordinates, anchor_landmarks, scale_factor)

        return (_OVERLAY_CACHE[name_or_path], anchor_landmarks, center_point, scale)
        
class OverlayParameters(BaseModel):
    overlay_name_or_path:Union[NonNegativeInt, str]
    overlay_bounding_landmarks:Tuple[int,int]
    overlay_scale_factor:Optional[float] = None
    y_offset:int

    @field_validator("overlay_name_or_path", mode="before")
    @classmethod
    def check_accepted_value(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if isinstance(value, str):
            value = str.lower(value)
            if value in {"sunglasses", "glasses", "tear_short_1"}:
                return value
            
            elif os.path.isfile(value):
                return value
            
            raise ValueError(f"Unrecognized value or invalid file path provided to parameter {field_name}.")
            
        elif isinstance(value, int):
            # 46,47,... not currently actual constants
            if value not in {43,44,45}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            
            mapping = {43: "sunglasses", 44:"glasses", 45:"tear_short_1"}
            return mapping.get(value)
        
        raise TypeError(f"Invalid type for parameter {field_name}. Expected int or str.")
    
    @field_validator("overlay_bounding_landmarks")
    @classmethod
    def check_landmarks_in_range(cls, value, info:ValidationInfo):

        for lm in value:
            if not 0 <= lm <= 477:
                raise ValueError(f"FaceLandmarker landmark ID's lie in the range [0-477].")
        
        return value
    
    @field_validator("overlay_scale_factor")
    @classmethod
    def check_scale_not_none(cls, value, info:ValidationInfo):
        field_name = info.field_name
        params = info.data

        # Check that an overlay type was passed, and that it is a custom type
        if params.get("overlay_name_or_path") and (params.get("overlay_name_or_path") not in {"sunglasses", "glasses", "tear_short_1", 43, 44, 45}):
            if value is None:
                raise ValueError(f"{field_name} is a required parameter when passing custom overlay paths.")
            elif not (0.0 < value <= 1.0):
                raise ValueError(f"{field_name} must be a float in the normal range 0.0 - 1.0.")
            
        return value
    
class LayerStyliseOverlay(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, overlay_parameters:OverlayParameters):

        self.time_config = timing_configuration
        self.overlay_params = overlay_parameters

        # Initialise superclass
        super().__init__(self.time_config)

        # Intra-frame tracking
        self.overlay_img = None
        self.overlay_scale = None
        
        # Declare class parameters
        self.overlay_name_or_path = self.overlay_params.overlay_name_or_path
        self.anchor_landmarks = self.overlay_params.overlay_bounding_landmarks
        self.overlay_scale_factor = self.overlay_params.overlay_scale_factor
        self.y_offset = self.overlay_params.y_offset

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
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float):

        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            overlayed_frame = frame.copy()

            overlay, anchor_lms, center_point, scale = load_overlay(
                name_or_path=self.overlay_name_or_path,
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

            # Uncomment for debugging overlay positioning
            # cv.circle(overlayed_frame, center_point, 3, (0,0,255), -1)
            # cv.circle(overlayed_frame, landmarker_coordinates[anchor_lms[0]], 3, (0,0,255), -1)
            # cv.circle(overlayed_frame, landmarker_coordinates[anchor_lms[1]], 3, (0,0,255), -1)
            # cv.circle(overlayed_frame, (x_pos, y_pos), 3, (255,0,0), -1)
            # cv.circle(overlayed_frame, (x_pos+padded_width, y_pos+padded_height), 3, (255,0,0), -1)
            # cv.imshow("overlay", overlayed_frame)
            # cv.waitKey(0)

            return overlayed_frame
        
def layer_stylise_overlay(timing_configuration:TimingConfiguration | None = None, overlay_name_or_path:int|str = "sunglasses", 
                          overlay_bounding_landmarks:tuple[int,int] = (127, 356), overlay_scale_factor:float | None = None, y_offset:int = 20) -> LayerStyliseOverlay:
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = OverlayParameters(
            overlay_name_or_path=overlay_name_or_path,
            overlay_bounding_landmarks=overlay_bounding_landmarks, 
            overlay_scale_factor=overlay_scale_factor, 
            y_offset = y_offset
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerStyliseOverlay.__name__}: {e}")
    
    return LayerStyliseOverlay(time_config, params)