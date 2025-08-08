from pydantic import BaseModel, NonNegativeFloat, field_validator, ValidationInfo, ConfigDict
from typing import Callable, Optional
from abc import ABC, abstractmethod
from cv2.typing import MatLike
from pyfame.layer.timing_curves import timing_linear
from pyfame.utilities.checks import *
from pyfame.mesh.get_mesh_coordinates import get_mesh

class TimingConfiguration(BaseModel):
    time_onset:Optional[NonNegativeFloat] = None
    time_offset:Optional[NonNegativeFloat] = None
    timing_function:Callable[...,float] = timing_linear
    rise_duration:NonNegativeFloat = 0.5
    fall_duration:NonNegativeFloat = 0.5
    min_tracking_confidence:NonNegativeFloat = 0.5
    min_detection_confidence:NonNegativeFloat = 0.5
    # kwargs-like dict of extra arguments stored in model_extra dictionary
    model_config = ConfigDict(extra="allow")

    @field_validator('min_tracking_confidence', 'min_detection_confidence')
    @classmethod
    def check_normal_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{field_name} must lie in the range [0.0, 1.0].")

class Layer(ABC): 
    """ An abstract base class to be extended by pyfame's manipulation layer classes. """

    def __init__(self, configuration:TimingConfiguration|None = None):
        
        # if config is none, populate with defaults
        self.config = configuration or TimingConfiguration()

        self.face_mesh = get_mesh(self.config.min_tracking_confidence, self.config.min_detection_confidence, False)
        self.onset_t = self.config.time_onset
        self.offset_t = self.config.time_offset
        self.timing = self.config.timing_function
        self.rise = self.config.rise_duration
        self.fall = self.config.fall_duration
        self.time_kwargs = self.config.model_extra
        
    def compute_weight(self, dt:float, supports_weight:bool) -> float:
        if supports_weight:
            return self.timing(dt, self.onset_t, self.offset_t, self.rise, self.fall, **self.time_kwargs)
        else:
            return 1.0
    
    def get_face_mesh(self):
        return self.face_mesh
    
    def set_face_mesh(self, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, static_image_mode:bool = False):
        self.face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, static_image_mode)

    @abstractmethod
    def supports_weight(self) -> bool:
        pass

    @abstractmethod
    def get_layer_parameters(self) -> dict:
        pass

    @abstractmethod
    def apply_layer(self, frame:MatLike, dt:float, static_image_mode:bool = False) -> MatLike:
        pass