from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo
from typing import Union, List, Tuple
from pyfame.utilities.constants import *
from pyfame.layer.manipulations.mask import mask_from_path
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.mesh.mesh_landmarks import FACE_OVAL_PATH
import numpy as np
import cv2 as cv

class MaskingParameters(BaseModel):
    region_of_interest:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]
    background_colour:Tuple[int,int,int]

    @field_validator("background_colour")
    @classmethod
    def check_in_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        for elem in value:
            if not (0 <= elem <= 255):
                raise ValueError(f"{field_name} values must lie between 0 and 255.")

class LayerMask(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, masking_parameters:MaskingParameters):
        
        self.time_config = timing_configuration
        self.mask_params = masking_parameters

        # Initialise the superclass
        super().__init__(self.time_config)

        # Define class parameters
        self.static_image_mode = False
        self.region_of_interest = self.mask_params.region_of_interest
        self.background_colour = self.mask_params.background_colour
        self.min_detection_confidence = self.time_config.min_detection_confidence
        self.min_tracking_confidence = self.time_config.min_tracking_confidence

        # Dump pydantic models to get full param list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.mask_params.model_dump())
    
    def supports_weight(self):
        return False
    
    def get_layer_parameters(self) -> dict:
        return dict(self._layer_parameters)
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):

        # Update the faceMesh when switching between image and video processing
        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        # Masking does not support weight, so weight will always be 0.0 or 1.0
        weight = super().compute_weight(dt, self.supports_weight())

        # Occurs when the dt is less than the onset_time, or greater than the offset_time
        if weight == 0.0:
            return frame
        else:
            # Mask out the region of interest
            face_mesh = super().get_face_mesh()
            mask = mask_from_path(frame, self.region_of_interest, face_mesh)

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

            # Remove unwanted background inclusions in the masked area
            masked_frame = cv.bitwise_and(mask, foreground)
            masked_frame = np.reshape(masked_frame, (masked_frame.shape[0], masked_frame.shape[1], 1))
            masked_frame = np.where(masked_frame == 255, frame, self.background_colour)
            masked_frame = masked_frame.astype(np.uint8)
            return masked_frame

def layer_mask(timing_configuration:TimingConfiguration | None = None, region_of_interest:list[list[tuple[int,...]]] | list[tuple[int,...]]= FACE_OVAL_PATH, background_colour:tuple[int,int,int] = (0,0,0)) -> LayerMask:
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = MaskingParameters(
            region_of_interest=region_of_interest, 
            background_colour=background_colour
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerMask.__name__}: {e}")

    return LayerMask(time_config, params)