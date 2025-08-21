from pydantic import BaseModel, field_validator, ValidationInfo, ValidationError
from typing import Union, List, Tuple
from pyfame.mesh import *
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_path
from pyfame.utilities.constants import *
import cv2 as cv
import numpy as np

### Note: possibly expand fill_method options to include colour presets in the future

class PathOcclusionParameters(BaseModel):
    fill_method:Union[int,str]
    region_of_interest:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]

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
        

class LayerOcclusionPath(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, occlusion_parameters:PathOcclusionParameters):

        self.time_config = timing_configuration
        self.occlude_params = occlusion_parameters

        # Initialise superclass
        super().__init__(self.time_config)

        # Declaring class parameters
        self.fill_method = self.occlude_params.fill_method
        self.region_of_interest = self.occlude_params.region_of_interest
        self.min_tracking_confidence = self.time_config.min_tracking_confidence
        self.min_detection_confidence = self.time_config.min_detection_confidence
        self.static_image_mode = False

        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.occlude_params.model_dump())

    def supports_weight(self):
        return False
    
    def get_layer_parameters(self):
        return dict(self._layer_parameters)
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):
        
        # Update the faceMesh when switching between image and video processing
        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        weight = super().compute_weight(dt, self.supports_weight())
        
        if weight == 0.0:
            return frame
        else:
            # Mask out the region of interest
            face_mesh = super().get_face_mesh()
            mask = mask_from_path(frame, self.region_of_interest, face_mesh)
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

            match self.fill_method:
                case 8 | "black":
                    occluded = np.where(mask == 255, self.fill_method, frame)
                    return occluded
                
                case 9 | "mean":
                    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    # Uses the tight-path to avoid any background inclusion in the mean colour sampling
                    fo_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, FACE_OVAL_TIGHT_PATH)

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

def layer_occlusion_path(timing_configuration:TimingConfiguration | None = None, region_of_interest:list[list[tuple[int,...]]] | list[tuple[int,...]]=FACE_OVAL_PATH, fill_method:int|str = OCCLUSION_FILL_BLACK) -> LayerOcclusionPath:
    
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = PathOcclusionParameters(
            fill_method=fill_method, 
            region_of_interest=region_of_interest
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerOcclusionPath.__name__}: {e}")

    return LayerOcclusionPath(time_config, params)