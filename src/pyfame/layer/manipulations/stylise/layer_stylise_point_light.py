from pydantic import BaseModel, field_validator, ValidationInfo, ValidationError, PositiveFloat, NonNegativeInt
from typing import Union, List, Tuple, Any
from pyfame.mesh import *
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_path
from pyfame.utilities.constants import *
import cv2 as cv
import numpy as np
from skimage.util import *

class PointLightParameters(BaseModel):
    region_of_interest:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]
    point_density:PositiveFloat
    point_colour:tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]
    display_history_vectors:bool
    history_method:Union[int,str] = SHOW_HISTORY_ORIGIN
    history_window_msec:NonNegativeInt = 500
    history_vector_colour:tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]
    maintain_background:bool

    @field_validator("point_colour", "history_vector_colour")
    @classmethod
    def check_in_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        for elem in value:
            if not (0 <= elem <= 255):
                raise ValueError(f"{field_name} values must lie between 0 and 255.")
        
        return value
    
    @field_validator("point_density")
    @classmethod
    def check_normal_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Invalid value for parameter {field_name}. Must lie in the range 0.0 - 1.0.")
        
        return value
    
    @field_validator("history_method", mode="before")
    @classmethod
    def check_accepted_value(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if isinstance(value, str):
            if value not in {"origin", "relative"}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        elif isinstance(value, int):
            if value not in {32, 33}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        raise TypeError(f"Invalid type for parameter {field_name}. Expected int or str.")

class LayerStylisePointLight(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, point_light_parameters:PointLightParameters):

        self.time_config = timing_configuration
        self.pl_params = point_light_parameters

        # Initialise superclass
        super().__init__(self.time_config)

        # Declare class parameters
        self.frame_history = []
        self.prev_points = None
        self.idx_to_display = np.array([], dtype=np.uint8)
        self.point_density = self.pl_params.point_density
        self.point_colour = self.pl_params.point_colour
        self.maintain_background = self.pl_params.maintain_background
        self.display_history_vectors = self.pl_params.display_history_vectors
        self.history_method = self.pl_params.history_method
        self.history_window_msec = self.pl_params.history_window_msec
        self.history_colour = self.pl_params.history_vector_colour
        self.region_of_interest = self.pl_params.region_of_interest
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
        self._layer_parameters.update(self.pl_params.model_dump())
        self._layer_parameters["time_onset"] = self.onset_t
        self._layer_parameters["time_offset"] = self.offset_t
        return dict(self._layer_parameters)

    def apply_layer(self, face_mesh:Any, frame:cv.typing.MatLike, dt:float):
        
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            # Get the screen pixel coordinates of the frame/image
            landmark_screen_coords = get_mesh_coordinates(cv.cvtColor(frame, cv.COLOR_BGR2RGB), face_mesh)
            mask = np.zeros_like(frame, dtype=np.uint8)
            output_img = None
            frame_history_count = round(30 * (self.history_window_msec/1000))

            if self.maintain_background:
                output_img = frame.copy()
            else:
                output_img = np.zeros_like(frame, dtype=np.uint8)

            if self.idx_to_display.size == 0:
                for lm_path in self.region_of_interest:
                    lm_mask = mask_from_path(frame, [lm_path], face_mesh)
                    lm_mask = lm_mask.astype(bool)

                    # Use the generated bool mask to get valid indicies
                    for lm in landmark_screen_coords:
                        x = lm.get('x')
                        y = lm.get('y')
                        if lm_mask[y,x] == True:
                            self.idx_to_display = np.append(self.idx_to_display, lm.get('id'))
            
            if self.point_density != 1.0:
                # Pad and reshape idx array to slices of size 10
                new_lm_idx = self.idx_to_display.copy()
                pad_size = len(new_lm_idx)%10
                append_arr = np.full(10-pad_size, -1)
                new_lm_idx = np.append(new_lm_idx, append_arr)
                new_lm_idx = new_lm_idx.reshape((-1, 10))

                bin_idx_mask = np.zeros((new_lm_idx.shape[0], new_lm_idx.shape[1]))

                for i,_slice in enumerate(new_lm_idx):
                    num_ones = round(np.floor(10*self.point_density))

                    # Generate normal distribution around center of slice
                    mean = 4.5
                    std_dev = 1.67
                    normal_idx = np.random.normal(loc=mean, scale=std_dev, size=num_ones)
                    normal_idx = np.clip(normal_idx, 0, 9).astype(int)

                    new_bin_arr = np.zeros(10)
                    for idx in normal_idx:
                        new_bin_arr[idx] = 1
                    
                    # Ensure the correct proportion of ones are present
                    while new_bin_arr.sum() < num_ones:
                        add_idx = np.random.choice(np.where(new_bin_arr == 0)[0])
                        new_bin_arr[add_idx] = 1
                    
                    bin_idx_mask[i] = new_bin_arr
                
                bin_idx_mask = bin_idx_mask.reshape((-1,))
                new_lm_idx = new_lm_idx.reshape((-1,))
                self.idx_to_display = np.where(bin_idx_mask == 1, new_lm_idx, -1)

            cur_points = []
            history_mask = np.zeros_like(frame, dtype=np.uint8)

            # Store the x,y coords of every idx_to_display
            for id in self.idx_to_display:
                if id != -1:
                    point = landmark_screen_coords[id]
                    cur_points.append((point.get('x'), point.get('y')))
            
            if self.prev_points == None or self.display_history_vectors == False:
                self.prev_points = cur_points.copy()

                # Draw a circle (point) over each of the current_points
                for point in cur_points:
                    x1, y1 = point
                    if x1 > 0 and y1 > 0:
                        output_img = cv.circle(output_img, (x1, y1), 3, self.point_colour, -1)

            elif self.history_method == SHOW_HISTORY_ORIGIN:
                # If show_history is true, display vector paths of all points;
                # On top of the points themselves
                for (old, new) in zip(self.prev_points, cur_points):
                    x0, y0 = old
                    x1, y1 = new
                    mask = cv.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), self.history_colour, 2)
                    output_img = cv.circle(output_img, (int(x1), int(y1)), 3, self.point_colour, -1)

                self.prev_points = cur_points.copy()
                output_img = cv.add(output_img, mask)
                mask = np.zeros_like(frame, dtype=np.uint8)

            else:
                # If show_history is true, display vector paths of all points
                for (old, new) in zip(self.prev_points, cur_points):
                    x0, y0 = old
                    x1, y1 = new
                    mask = cv.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), self.history_colour, 2)
                    output_img = cv.circle(output_img, (int(x1), int(y1)), 3, self.point_colour, -1)

                # Relative vector history only displays up to history_window_msec seconds of history
                if len(self.frame_history) < frame_history_count:
                    self.frame_history.append(mask)
                    for img in self.frame_history:
                        history_mask = cv.bitwise_or(history_mask, img)
                else:
                    self.frame_history.append(mask)
                    self.frame_history.pop(0)
                    for img in self.frame_history:
                        history_mask = cv.bitwise_or(history_mask, img)

                self.prev_points = cur_points.copy()
                output_img = cv.add(output_img, history_mask)
                mask = np.zeros_like(frame, dtype=np.uint8)

            return output_img
        
def layer_stylise_point_light(timing_configuration:TimingConfiguration | None = None, region_of_interest:list[list[tuple[int,int]]] | list[tuple[int,int]]=FACE_OVAL_PATH, 
                              point_density:float = 1.0, point_colour:tuple[int,int,int] = (255,255,255), display_history_vectors:bool = False, 
                              history_method:int|str = SHOW_HISTORY_ORIGIN, history_vector_colour:tuple[int,int,int] = (0,0,255), maintain_background:bool = True):
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input params
    try:
        params = PointLightParameters(
            region_of_interest=region_of_interest, 
            point_density=point_density, 
            point_colour=point_colour, 
            display_history_vectors=display_history_vectors, 
            history_method=history_method, 
            history_vector_colour=history_vector_colour, 
            maintain_background=maintain_background
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerStylisePointLight.__name__}: {e}")
    
    return LayerStylisePointLight(time_config, params)