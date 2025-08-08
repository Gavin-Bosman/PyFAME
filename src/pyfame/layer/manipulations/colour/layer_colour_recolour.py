from pydantic import BaseModel, NonNegativeFloat, field_validator, ValidationInfo, ValidationError
from typing import Union, List, Tuple
from pyfame.layer.manipulations.mask import mask_from_path
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.mesh.mesh_landmarks import FACE_OVAL_PATH
from pyfame.utilities.exceptions import *
from pyfame.utilities.checks import *
from pyfame.utilities.constants import *
import cv2 as cv
import numpy as np 

class RecolourParameters(BaseModel):
    region_of_interest:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]
    focus_colour:Union[str, int]
    magnitude:NonNegativeFloat

    @field_validator('focus_colour')
    @classmethod
    def check_compatible_value(cls, value, info:ValidationInfo):
        field_name = info.field_name
        if isinstance(value, int):
            if value not in [4,5,6,7]:
                raise ValueError(f"{field_name} has been provided an unrecognized value.")
        elif isinstance(value, str):
            if value not in ["red", "green", "blue", "yellow"]:
                raise ValueError(f"{field_name} has been provided an unrecognized value.")

class LayerColourRecolour(Layer):

    def __init__(self, timing_configuration:TimingConfiguration, recolour_parameters:RecolourParameters):
        
        self.time_config = timing_configuration
        self.colour_params = recolour_parameters

        # Initialise the superclass
        super().__init__(configuration=self.time_config)
        self.static_image_mode = False
        
        # Define class parameters
        self.region_of_interest = self.colour_params.region_of_interest
        self.focus_colour = self.colour_params.focus_colour
        self.magnitude = self.colour_params.magnitude
        self.min_detection_confidence = self.time_config.min_detection_confidence
        self.min_tracking_confidence = self.time_config.min_tracking_confidence
        
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.colour_params.model_dump())
    
    def supports_weight(self) -> bool:
        return True
    
    def get_layer_parameters(self) -> dict:
        return dict(self._layer_parameters)
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):
        weight = None

        # Update the faceMesh when switching between image and video processing
        if self.static_image_mode != static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        # Since a static image is essentially one frame, perform the manipulation at maximal weight
        if self.static_image_mode:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())
        
        # Occurs when the current dt is less than the onset_time, or greater than the offset_time
        if weight == 0.0:
            return frame
        else:
            # Get a mask of our region of interest
            face_mesh = super().get_face_mesh()
            mask = mask_from_path(frame, self.region_of_interest, face_mesh)

            # Convert input image to CIE La*b* color space (perceptually uniform space)
            img_LAB = cv.cvtColor(frame, cv.COLOR_BGR2LAB).astype(np.float32)
            # Split the image into individual channels for precise colour manipulation
            l,a,b = cv.split(img_LAB)

            # Shift the various colour channels according to the user-specified focus_colour
            match self.focus_colour:
                case "red" | 4:
                    a = np.where(mask==255, a + (weight * self.magnitude), a)
                    np.clip(a, -128, 127)
                case "blue" | 5:
                    b = np.where(mask==255, b - (weight * self.magnitude), b)
                    np.clip(b, -128, 127)
                case "green" | 6:
                    a = np.where(mask==255, a - (weight * self.magnitude), a)
                    np.clip(a, -128, 127)
                case "yellow" | 7:
                    b = np.where(mask==255, b + (weight * self.magnitude), b)
                    np.clip(b, -128, 127)
            
            # After shifting the colour channels, merge the individual channels back into one image
            img_LAB = cv.merge([l,a,b])

            # Convert CIE La*b* back to BGR
            result = cv.cvtColor(img_LAB.astype(np.uint8), cv.COLOR_LAB2BGR)
            return result

def layer_colour_recolour(timing_configuration:TimingConfiguration | None = None, region_of_interest:list[list[tuple[int,...]]] | list[tuple[int,...]] = FACE_OVAL_PATH,  focus_colour:str|int = "red", magnitude:float = 10.0) -> LayerColourRecolour:
    # Populate with defaults if None
    config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = RecolourParameters(region_of_interest=region_of_interest, focus_colour=focus_colour, magnitude=magnitude)
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerColourRecolour.__name__}: {e}")
    
    return LayerColourRecolour(config, params)