from pydantic import BaseModel, NonNegativeFloat, field_validator, ValidationInfo, ValidationError
from typing import Union, List, Tuple
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.landmark.facial_landmarks import *
from pyfame.utilities.constants import *
from pyfame.utilities.general_utilities import get_landmark_names
from pyfame.landmark.blendshape_smoother import EyeBlendshapeSmoother
import cv2 as cv
import numpy as np

class RecolourParameters(BaseModel):
    landmark_paths:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]
    focus_colour:Union[str, int]
    magnitude:NonNegativeFloat

    @field_validator('focus_colour')
    @classmethod
    def check_compatible_value(cls, value, info:ValidationInfo):
        field_name = info.field_name
        if isinstance(value, int):
            if value not in SHIFT_COLOUR_OPTIONS:
                raise ValueError(f"{field_name} has been provided an unrecognized value.")
        elif isinstance(value, str):
            if value not in ["red", "green", "blue", "yellow"]:
                raise ValueError(f"{field_name} has been provided an unrecognized value.")
        
        return value

class LayerColourRecolour(Layer):

    def __init__(self, timing_configuration:TimingConfiguration, recolour_parameters:RecolourParameters):
        
        self.time_config = timing_configuration
        self.colour_params = recolour_parameters

        # Initialise the superclass
        super().__init__(configuration=self.time_config)
        
        # Define class parameters
        self.landmark_paths = self.colour_params.landmark_paths
        self.focus_colour = self.colour_params.focus_colour
        self.magnitude = self.colour_params.magnitude
        self.eye_blendshape_smoother = EyeBlendshapeSmoother(frame_window_size=1) 
        self.colour_left_sclera = None
        self.colour_right_sclera = None
        self.adjusted_landmark_paths = self.colour_params.landmark_paths.copy()

        # Snapshot of initial state
        self._snapshot_state()
    
    def supports_weight(self) -> bool:
        return True
    
    def get_layer_parameters(self) -> dict:
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.colour_params.model_dump())
        self._layer_parameters["time_onset"] = self.onset_t
        self._layer_parameters["time_offset"] = self.offset_t
        return dict(self._layer_parameters)
    
    def _adjust_landmark_paths_eye_landmarks(self):
        if self.colour_left_sclera is None or self.colour_right_sclera is None:
            # Check if scleral colouring is required
            landmark_names = get_landmark_names(self.landmark_paths)

            if isinstance(landmark_names, list):
                for name in landmark_names:
                    if name == 'LANDMARK_BOTH_EYES':
                        self.colour_left_sclera = True
                        self.colour_right_sclera = True

                        if isinstance(self.adjusted_landmark_paths[0], list):
                            self.adjusted_landmark_paths.pop(self.adjusted_landmark_paths.index(LANDMARK_BOTH_EYES))
                        else:
                            self.adjusted_landmark_paths = []
                    
                    elif name == 'LANDMARK_LEFT_EYE':
                        self.colour_left_sclera = True
                        self.colour_right_sclera = False if self.colour_right_sclera is None else self.colour_right_sclera

                        if isinstance(self.adjusted_landmark_paths[0], list):
                            self.adjusted_landmark_paths.pop(self.adjusted_landmark_paths.index(LANDMARK_LEFT_EYE))
                        else:
                            self.adjusted_landmark_paths = []
                    
                    elif name == 'LANDMARK_RIGHT_EYE':
                        self.colour_left_sclera = False if self.colour_left_sclera is None else self.colour_left_sclera
                        self.colour_right_sclera = True

                        if isinstance(self.adjusted_landmark_paths[0], list):
                            self.adjusted_landmark_paths.pop(self.adjusted_landmark_paths.index(LANDMARK_RIGHT_EYE))
                        else:
                            self.adjusted_landmark_paths = []

            else:
                if landmark_names == 'LANDMARK_BOTH_EYES':
                        self.colour_left_sclera = True
                        self.colour_right_sclera = True

                        if isinstance(self.adjusted_landmark_paths[0], list):
                            self.adjusted_landmark_paths.pop(self.adjusted_landmark_paths.index(LANDMARK_BOTH_EYES))
                        else:
                            self.adjusted_landmark_paths = []
                    
                elif landmark_names == 'LANDMARK_LEFT_EYE':
                    self.colour_left_sclera = True
                    self.colour_right_sclera = False if self.colour_right_sclera is None else self.colour_right_sclera

                    if isinstance(self.adjusted_landmark_paths[0], list):
                        self.adjusted_landmark_paths.pop(self.adjusted_landmark_paths.index(LANDMARK_LEFT_EYE))
                    else:
                        self.adjusted_landmark_paths = []
                
                elif landmark_names == 'LANDMARK_RIGHT_EYE':
                    self.colour_left_sclera = False if self.colour_left_sclera is None else self.colour_left_sclera
                    self.colour_right_sclera = True

                    if isinstance(self.adjusted_landmark_paths[0], list):
                        self.adjusted_landmark_paths.pop(self.adjusted_landmark_paths.index(LANDMARK_RIGHT_EYE))
                    else:
                        self.adjusted_landmark_paths = []
    
    def apply_layer(self, landmarker_coordinates, frame, dt, blendshapes):

        weight = super().compute_weight(dt, self.supports_weight())
        
        # Occurs when the current dt is less than the onset_time, or greater than the offset_time
        if weight == 0.0:
            return frame

        self._adjust_landmark_paths_eye_landmarks()

        # Get a mask of our region of interest
        if len(self.adjusted_landmark_paths) > 0:
            mask = mask_from_landmarks(frame, self.adjusted_landmark_paths, landmarker_coordinates)
        else:
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        left_eye_open, right_eye_open = self.eye_blendshape_smoother.update(blendshapes)
        le_mask = mask_from_landmarks(frame, LANDMARK_LEFT_EYE, landmarker_coordinates)
        re_mask = mask_from_landmarks(frame, LANDMARK_RIGHT_EYE, landmarker_coordinates)

        # Convert input image to CIE La*b* color space (perceptually uniform space)
        img_LAB = cv.cvtColor(frame, cv.COLOR_BGR2LAB).astype(np.float32)
        # Split the image into individual channels for precise colour manipulation
        l,a,b = cv.split(img_LAB)

        # Shift the various colour channels according to the user-specified focus_colour
        match self.focus_colour:
            case "red" | 4:
                a = np.where(mask==255, a + (weight * self.magnitude), a)

                if self.colour_left_sclera and left_eye_open:
                    a = np.where(le_mask==255, a + (weight * self.magnitude), a)
                if self.colour_right_sclera and right_eye_open:
                    a = np.where(re_mask==255, a + (weight * self.magnitude), a)

                np.clip(a, -128, 127)

            case "blue" | 5:
                b = np.where(mask==255, b - (weight * self.magnitude), b)

                if self.colour_left_sclera and left_eye_open:
                    b = np.where(le_mask==255, b - (weight * self.magnitude), b)
                if self.colour_right_sclera and right_eye_open:
                    b = np.where(re_mask==255, b - (weight * self.magnitude), b)

                np.clip(b, -128, 127)

            case "green" | 6:
                a = np.where(mask==255, a - (weight * self.magnitude), a)

                if self.colour_left_sclera and left_eye_open:
                    a = np.where(le_mask==255, a - (weight * self.magnitude), a)
                if self.colour_right_sclera and right_eye_open:
                    a = np.where(re_mask==255, a - (weight * self.magnitude), a)

                np.clip(a, -128, 127)

            case "yellow" | 7:
                b = np.where(mask==255, b + (weight * self.magnitude), b)

                if self.colour_left_sclera and left_eye_open:
                    b = np.where(le_mask==255, b + (weight * self.magnitude), b)
                if self.colour_right_sclera and right_eye_open:
                    b = np.where(re_mask==255, b + (weight * self.magnitude), b)

                np.clip(b, -128, 127)
        
        # After shifting the colour channels, merge the individual channels back into one image
        img_LAB = cv.merge([l,a,b])

        # Convert CIE La*b* back to BGR
        result = cv.cvtColor(img_LAB.astype(np.uint8), cv.COLOR_LAB2BGR)

        return result

def layer_colour_recolour(timing_configuration:TimingConfiguration | None = None, landmark_paths:list[list[tuple[int,...]]] | list[tuple[int,...]] = LANDMARK_FACE_OVAL,  focus_colour:str|int = "red", magnitude:float = 10.0) -> LayerColourRecolour:
    # Populate with defaults if None
    config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = RecolourParameters(
            landmark_paths=landmark_paths, 
            focus_colour=focus_colour, 
            magnitude=magnitude
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerColourRecolour.__name__}: {e}")
    
    return LayerColourRecolour(config, params)