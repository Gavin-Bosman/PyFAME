from pydantic import BaseModel, NonNegativeFloat, ConfigDict
from typing import Callable, Optional
from abc import ABC, abstractmethod
from cv2.typing import MatLike
from pyfame.layer.timing_curves import timing_linear
import copy

class TimingConfiguration(BaseModel):
    time_onset:Optional[NonNegativeFloat] = None
    time_offset:Optional[NonNegativeFloat] = None
    timing_function:Callable[...,float] = timing_linear
    rise_duration:NonNegativeFloat = 0.5
    fall_duration:NonNegativeFloat = 0.5
    # kwargs-like dict of extra arguments stored in model_extra dictionary
    model_config = ConfigDict(extra="allow")

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
    
    def _snapshot_state(self):
        self._initial_state = copy.deepcopy(self.__dict__)
    
    def _reset_state(self):
        init_state = copy.deepcopy(self._initial_state)
        init_state["_initial_state"] = self._initial_state
        self.__dict__ = init_state
        
    def compute_weight(self, dt:float, supports_weight:bool) -> float:
        if dt is None:
            return 0.0
        elif supports_weight:
            return self.timing(dt, self.onset_t, self.offset_t, self.rise, self.fall, **self.time_kwargs)
        else:
            return 1.0
    
    # def get_frame_as_mp_image(frame:MatLike):
    #     # Save the orignal dimensions for determining padding
    #     original_h, original_w = frame.shape[:2]

    #     # Pad to square dimensions before face landmarking
    #     if original_h > original_w:
    #         pad = (original_h - original_w) // 2
    #         padded_frame = cv.copyMakeBorder(frame, 0, 0, pad, pad, cv.BORDER_CONSTANT, value=(0,0,0))
    #     elif original_w > original_h:
    #         pad = (original_w - original_h) // 2
    #         padded_frame = cv.copyMakeBorder(frame, pad, pad, 0, 0, cv.BORDER_CONSTANT, value=(0,0,0))
    #     else:
    #         padded_frame = frame
        
    #     return mp.Image(image_format=mp.ImageFormat.SRGB, data=padded_frame)

    @abstractmethod
    def supports_weight(self) -> bool:
        pass

    @abstractmethod
    def get_layer_parameters(self) -> dict:
        pass

    @abstractmethod
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:MatLike, dt:float) -> MatLike:
        pass