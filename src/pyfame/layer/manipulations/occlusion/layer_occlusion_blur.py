from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, PositiveInt
from typing import Union, List, Tuple, Any
from pyfame.mesh import *
from pyfame.utilities.constants import *
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_path
import cv2 as cv
import numpy as np

class BlurringParameters(BaseModel):
    blur_method:Union[str, int]
    kernel_size:Tuple[PositiveInt, PositiveInt]
    region_of_interest:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]

    @field_validator("blur_method", mode="before")
    @classmethod
    def check_compatible_value(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if isinstance(value, str):
            value = str.lower(value)
            if value not in {"average", "gaussian", "median"}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        elif isinstance(value, int):
            if value not in {11, 12, 13}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        raise TypeError(f"{field_name} provided an invalid type. Must be one of int, str.")

    @field_validator("kernel_size")
    @classmethod
    def check_odd_dims(cls, value, info:ValidationInfo):
        field_name = info.field_name

        # Ensures kernels provided are square, odd and greater or equal to (3,3)
        if not (value[0] % 2 == 1 and value[1] % 2 == 1 and value[0] >= 3 and value[0] == value[1]):
            raise ValueError(f"{field_name} expects odd, square kernel dimensions >= (3,3).")
        
        return value

class LayerOcclusionBlur(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, blurring_parameters:BlurringParameters):

        self.time_config = timing_configuration
        self.blur_params = blurring_parameters

        # Initialise superclass
        super().__init__(self.time_config)
        
        # Define class parameters
        self.blur_method = self.blur_params.blur_method
        self.kernel_size = self.blur_params.kernel_size
        self.region_of_interest = self.blur_params.region_of_interest
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
        self._layer_parameters.update(self.blur_params.model_dump())
        self._layer_parameters["time_onset"] = self.onset_t
        self._layer_parameters["time_offset"] = self.offset_t
        return dict(self._layer_parameters)
    
    def apply_layer(self, face_mesh:Any, frame:cv.typing.MatLike, dt:float = None):

        # Blurring does not support weight, so weight will always be 0.0 or 1.0
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            # Mask out region of interest
            mask = mask_from_path(frame, self.region_of_interest, face_mesh)
            output_frame = np.zeros_like(frame, dtype=np.uint8)

            # Blur the input frame depending on user-specified blur method
            match self.blur_method:
                case "average" | 11:
                    frame_blurred = cv.blur(frame, self.kernel_size)
                    output_frame = np.where(mask == 255, frame_blurred, frame)
                
                case "gaussian" | 12:
                    frame_blurred = cv.GaussianBlur(frame, self.kernel_size, 0)
                    output_frame = np.where(mask == 255, frame_blurred, frame)
                
                case "median" | 13:
                    frame_blurred = cv.medianBlur(frame, self.kernel_size[0])
                    output_frame = np.where(mask == 255, frame_blurred, frame)
            
            return output_frame

def layer_occlusion_blur(timing_configuration:TimingConfiguration | None = None, blur_method:str|int = "gaussian", region_of_interest:list[list[tuple[int,...]]] | list[tuple[int,...]] = FACE_OVAL_PATH, kernel_size:tuple[int,int] = (15,15)) -> LayerOcclusionBlur:
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    try:
        params = BlurringParameters(
            blur_method=blur_method, 
            kernel_size=kernel_size, 
            region_of_interest=region_of_interest
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerOcclusionBlur.__name__}: {e}")

    return LayerOcclusionBlur(time_config, params) 