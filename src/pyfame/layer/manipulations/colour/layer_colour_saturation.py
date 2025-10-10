from pydantic import BaseModel, field_validator, ValidationInfo, ValidationError
from typing import Union, List, Tuple
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.landmark.facial_landmarks import LANDMARK_FACE_OVAL
from pyfame.utilities.constants import *
import cv2 as cv
import numpy as np

class SaturationParameters(BaseModel):
    landmark_paths:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]
    magnitude:float

    @field_validator("magnitude")
    @classmethod
    def check_value_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        if not (-25.0 <= value <= 25.0):
            raise ValueError(f"{field_name} must lie between -25.0 and 25.0.")
        
        return value

class LayerColourSaturation(Layer):

    def __init__(self, timing_configuration:TimingConfiguration, saturation_parameters:SaturationParameters):

        self.time_config = timing_configuration
        self.sat_params = saturation_parameters

        # Initialise the superclass
        super().__init__(self.time_config)
        self.static_image_mode = False
        
        # Define class parameters
        self.landmark_paths = self.sat_params.landmark_paths
        self.magnitude = self.sat_params.magnitude

        # Snapshot of initial state
        self._snapshot_state()
    
    def supports_weight(self) -> bool:
        return True
    
    def get_layer_parameters(self) -> dict:
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.sat_params.model_dump())
        self._layer_parameters["time_onset"] = self.onset_t
        self._layer_parameters["time_offset"] = self.offset_t
        return dict(self._layer_parameters)
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float = None):
        
        weight = super().compute_weight(dt, self.supports_weight())
        
        # Occurs when the dt < onset_time, or > offset_time
        if weight == 0.0:
            return frame
        else:
            # Mask out our region of interest
            mask = mask_from_landmarks(frame, self.landmark_paths, landmarker_coordinates)

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

            # Convert the image into the HSV space so we can manipulate the saturation
            img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)
            # Split the image channels so only the saturation can be shifted
            h,s,v = cv.split(img_hsv)
            s = np.where(mask == 255, s + (weight * self.magnitude), s)
            np.clip(s,0,255)
            img_hsv = cv.merge([h,s,v])
            
            # Convert the HSV image back to BGR before returning the processed image
            img_bgr = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
            img_bgr[foreground == 0] = frame[foreground == 0]
            return img_bgr
        
def layer_colour_saturation(timing_configuration:TimingConfiguration|None = None, landmark_paths:list[list[tuple[int,...]]] | list[tuple[int,...]] = LANDMARK_FACE_OVAL, magnitude:float = -12.0) -> LayerColourSaturation:
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = SaturationParameters(
            landmark_paths=landmark_paths, 
            magnitude=magnitude
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerColourSaturation.__name__}: {e}")

    return LayerColourSaturation(time_config, params)