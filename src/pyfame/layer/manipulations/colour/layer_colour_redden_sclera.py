from pydantic import BaseModel, ValidationError, NonNegativeFloat
from pyfame.layer.layer import Layer, TimingConfiguration
import cv2 as cv
import numpy as np
from collections import deque
from mediapipe.tasks.python.vision.face_landmarker import Blendshapes
from pyfame.layer.manipulations.mask.mask_from_landmarks import mask_from_landmarks
from pyfame.landmark.facial_landmarks import *

class EyeBlendshapeSmoother:
    def __init__(self, frame_window_size=5):
        self.left_blink_scores = deque(maxlen=frame_window_size)
        self.right_blink_scores = deque(maxlen=frame_window_size)
        self.prev_left_state = True
        self.prev_right_state = True
    
    def update(self, blendshapes):
        left_blink_score = self._get_blendshape_score(blendshapes, Blendshapes.EYE_BLINK_LEFT)
        right_blink_score = self._get_blendshape_score(blendshapes, Blendshapes.EYE_BLINK_RIGHT)

        self.left_blink_scores.append(left_blink_score)
        self.right_blink_scores.append(right_blink_score)

        smoothed_left = np.mean(self.left_blink_scores)
        smoothed_right = np.mean(self.right_blink_scores)

        self.prev_left_state = self._is_eye_open(smoothed_left, self.prev_left_state)
        self.prev_right_state = self._is_eye_open(smoothed_right, self.prev_right_state)

        return self.prev_left_state, self.prev_right_state
    
    @staticmethod
    def _get_blendshape_score(blendshapes, name):
        for bs in blendshapes:
            if bs.category_name == name:
                return bs.score
        return 0.0
    
    @staticmethod
    def _is_eye_open(score, prev_state):
        if score >= 0.55:
            return False
        elif score <= 0.35:
            return True
        else:
            return prev_state

class ColourScleraParameters(BaseModel):
    magnitude:NonNegativeFloat

class LayerColourReddenSclera(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, sclera_colour_parameters:ColourScleraParameters):
        self.time_config = timing_configuration
        self.sclera_params = sclera_colour_parameters

        # Initialise the superclass
        super().__init__(configuration=self.time_config)
        
        # Initialise class parameters
        self.blendshape_smoother = EyeBlendshapeSmoother()
        self.magnitude = self.sclera_params.magnitude

        self._snapshot_state()

    def supports_weight(self) -> bool:
        return True
    
    def get_layer_parameters(self) -> dict:
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.sclera_params.model_dump())
        self._layer_parameters["time_onset"] = self.onset_t
        self._layer_parameters["time_offset"] = self.offset_t
        return dict(self._layer_parameters)
    
    def _apply_sclera_overlay(self, frame, eye_mask, weight):
        # Convert input image to CIE La*b* color space (perceptually uniform space)
        img_LAB = cv.cvtColor(frame, cv.COLOR_BGR2LAB).astype(np.float32)
        # Split the image into individual channels for precise colour manipulation
        l,a,b = cv.split(img_LAB)

        a = np.where(eye_mask==255, a + (weight * self.magnitude), a)
        np.clip(a, -128, 127)

        img_LAB = cv.merge([l,a,b])
        frame_BGR = cv.cvtColor(img_LAB.astype(np.uint8), cv.COLOR_LAB2BGR)

        return frame_BGR

    def apply_layer(self, landmarker_coordinates, frame, dt, blendshapes):
        
        weight = super().compute_weight(dt, self.supports_weight())
        if weight == 0.0:
            return frame
        
        left_eye_open, right_eye_open = self.blendshape_smoother.update(blendshapes)
        le_mask = mask_from_landmarks(frame, LANDMARK_LEFT_EYE, landmarker_coordinates)
        re_mask = mask_from_landmarks(frame, LANDMARK_RIGHT_EYE, landmarker_coordinates)

        if left_eye_open:
            frame = self._apply_sclera_overlay(frame, le_mask, weight)
        if right_eye_open:
            frame = self._apply_sclera_overlay(frame, re_mask, weight)
        
        return frame
        
def layer_colour_redden_sclera(timing_configuration:TimingConfiguration | None = None, magnitude:float = 12.0) -> LayerColourReddenSclera:
    # Populate with defaults if None
    config = timing_configuration or TimingConfiguration()

    try:
        params = ColourScleraParameters(
            magnitude=magnitude
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerColourReddenSclera.__name__}: {e}")
    
    return LayerColourReddenSclera(config, params)