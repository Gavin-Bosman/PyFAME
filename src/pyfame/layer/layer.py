from pydantic import BaseModel, NonNegativeFloat, field_validator, ValidationInfo, ConfigDict
from typing import Callable, Optional, Any
from abc import ABC, abstractmethod
from cv2.typing import MatLike
from pyfame.layer.timing_curves import timing_linear
from pyfame.file_access.checks import *
from pyfame.mesh.get_mesh_coordinates import get_mesh
import copy

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

        self.onset_t = self.config.time_onset
        self.offset_t = self.config.time_offset
        self.timing = self.config.timing_function
        self.rise = self.config.rise_duration
        self.fall = self.config.fall_duration
        self.time_kwargs = self.config.model_extra
        self.face_mesh = None
    
    def _snapshot_state(self):
        self._initial_state = copy.deepcopy(self.__dict__)
    
    def _reset_state(self):
        init_state = copy.deepcopy(self._initial_state)
        init_state["_initial_state"] = self._initial_state
        init_state["face_mesh"] = None
        self.__dict__ = init_state
        
    def compute_weight(self, dt:float, supports_weight:bool) -> float:
        if dt is None:
            return 0.0
        elif supports_weight:
            return self.timing(dt, self.onset_t, self.offset_t, self.rise, self.fall, **self.time_kwargs)
        else:
            return 1.0
    
    def get_face_mesh(self, static_image_mode=False):
        if self.face_mesh is not None:
            return self.face_mesh
        else:
            # Only instantiate when necessary (after state reset)
            # otherwise lazy-load
            self.face_mesh = get_mesh(
                min_tracking_confidence=self.config.min_tracking_confidence, 
                min_detection_confidence=self.config.min_detection_confidence,
                static_image_mode=static_image_mode,
                max_num_faces=1
            )
            return self.face_mesh

    @abstractmethod
    def supports_weight(self) -> bool:
        pass

    @abstractmethod
    def get_layer_parameters(self) -> dict:
        pass

    @abstractmethod
    def apply_layer(self, face_mesh:Any, frame:MatLike, dt:float) -> MatLike:
        pass