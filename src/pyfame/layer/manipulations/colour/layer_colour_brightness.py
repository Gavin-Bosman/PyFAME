from pydantic import BaseModel, field_validator, ValidationInfo, ValidationError
from typing import Union, List, Tuple
from pyfame.mesh.mesh_landmarks import FACE_OVAL_PATH
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_path
from pyfame.utilities.constants import *
import cv2 as cv
import numpy as np

class BrightnessParameters(BaseModel):
    region_of_interest:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]
    magnitude:float

    @field_validator("magnitude")
    @classmethod
    def check_value_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        if not (-25.0 <= value <= 25.0):
            raise ValueError(f"{field_name} must lie between -25.0 and 25.0.")

class LayerColourBrightness(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, brightness_parameters:BrightnessParameters):
        
        self.time_config = timing_configuration
        self.bright_params = brightness_parameters

        # Initialise the superclass
        super().__init__(self.time_config)
        self.static_image_mode = False

        # Define class parameters
        self.region_of_interest = self.bright_params.region_of_interest
        self.magnitude = self.bright_params.magnitude
        self.min_detection_confidence = self.time_config.min_detection_confidence
        self.min_tracking_confidence = self.time_config.min_tracking_confidence

        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.bright_params.model_dump())
    
    def supports_weight(self):
        return True
    
    def get_layer_parameters(self) -> dict:
        return dict(self._layer_parameters)

    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):
        weight = None

        # Update the faceMesh when switching between image and video processing
        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        if self.static_image_mode:
            weight = 1.0    
        else:
            weight = super().compute_weight(dt, self.supports_weight())
        
        # Occurs when the dt < onset_time, or > offset_time
        if weight == 0.0:
            return frame
        else:
            # Mask out the region of interest
            face_mesh = super().get_face_mesh()
            mask = mask_from_path(frame, self.region_of_interest, face_mesh)
            # Reshape the mask for compatibility with cv2.convertScaleAbs()
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

            # Otsu thresholding to seperate foreground and background
            grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
            thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

            # Adding a temporary image border to allow for correct floodfill behaviour
            bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
            floodfilled = bordered_thresholded.copy()
            cv.floodFill(floodfilled, None, (0,0), 255)

            # Removing temporary border and creating foreground mask
            floodfilled = floodfilled[10:-10, 10:-10]
            floodfilled = cv.bitwise_not(floodfilled)
            foreground = cv.bitwise_or(thresholded, floodfilled)

            # Within the masked region, upscale the brightness according to the current weight
            img_brightened = np.where(mask == 255, cv.convertScaleAbs(src=frame, alpha=1, beta=(weight * self.magnitude)), frame)
            img_brightened[foreground == 0] = frame[foreground == 0]
            return img_brightened

def layer_colour_brightness(timing_configuration:TimingConfiguration | None = None, region_of_interest:list[list[tuple[int,...]]] | list[tuple[int,...]] = FACE_OVAL_PATH, magnitude:float = 20.0) -> LayerColourBrightness:
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = BrightnessParameters(region_of_interest, magnitude)
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerColourBrightness.__name__}: {e}")
        

    return LayerColourBrightness(time_config, params)