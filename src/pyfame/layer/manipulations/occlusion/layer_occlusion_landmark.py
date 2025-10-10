from pydantic import BaseModel, field_validator, ValidationInfo, ValidationError
from typing import Union, List, Tuple
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.get_landmark_coordinates import get_pixel_coordinates_from_landmark
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.utilities.constants import *
import cv2 as cv
import numpy as np

### Note: possibly expand fill_method options to include colour presets in the future

class LandmarkOcclusionParameters(BaseModel):
    fill_method:Union[int,str]
    landmark_paths:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]

    @field_validator("fill_method", mode="before")
    @classmethod
    def check_accepted_value(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if isinstance(value, str):
            value = str.lower(value)
            if value not in {"black", "mean"}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        elif isinstance(value, int):
            if value not in {8,9}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        raise TypeError(f"Invalid type provided for {field_name}. Must be one of int or str.")
        

class LayerOcclusionLandmark(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, occlusion_parameters:LandmarkOcclusionParameters):

        self.time_config = timing_configuration
        self.occlude_params = occlusion_parameters

        # Initialise superclass
        super().__init__(self.time_config)

        # Declaring class parameters
        self.fill_method = self.occlude_params.fill_method
        self.landmark_paths = self.occlude_params.landmark_paths

        # Snapshot of initial state
        self._snapshot_state()

    def supports_weight(self):
        return False
    
    def get_layer_parameters(self) -> dict:
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.occlude_params.model_dump())
        self._layer_parameters["time_onset"] = self.onset_t
        self._layer_parameters["time_offset"] = self.offset_t
        return dict(self._layer_parameters)
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float = None):
        
        weight = super().compute_weight(dt, self.supports_weight())
        
        if weight == 0.0:
            return frame
        else:
            # Mask out the region of interest
            mask = mask_from_landmarks(frame, self.landmark_paths, landmarker_coordinates)
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

            match self.fill_method:
                case 8 | "black":
                    bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    occluded = np.where(mask == 255, (0,0,0), frame)
                    return occluded
                
                case 9 | "mean":
                    fo_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_FACE_OVAL)

                    # Creating boolean masks for the facial landmarks 
                    bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    bool_mask = cv.fillConvexPoly(bool_mask, np.array(fo_coords), 1)
                    bool_mask = bool_mask.astype(bool)

                    # Extracting the mean pixel value of the face
                    bin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    bin_mask[bool_mask] = 255
                    mean = cv.mean(frame, bin_mask)

                    # Fill occlusion regions with facial mean
                    mean_img = np.zeros_like(frame, dtype=np.uint8)
                    mean_img[:] = mean[:3]
                    occluded = np.where(mask == 255, mean_img, frame)
                    return occluded

def layer_occlusion_landmark(timing_configuration:TimingConfiguration | None = None, landmark_paths:list[list[tuple[int,...]]] | list[tuple[int,...]]=LANDMARK_FACE_OVAL, fill_method:int|str = OCCLUSION_FILL_BLACK) -> LayerOcclusionLandmark:
    
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = LandmarkOcclusionParameters(
            fill_method=fill_method, 
            landmark_paths=landmark_paths
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerOcclusionLandmark.__name__}: {e}")

    return LayerOcclusionLandmark(time_config, params)