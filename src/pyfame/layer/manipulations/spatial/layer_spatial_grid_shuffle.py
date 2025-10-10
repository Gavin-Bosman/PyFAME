from pydantic import BaseModel, field_validator, ValidationInfo, ValidationError, PositiveInt
from typing import Union, List, Tuple, Optional
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.get_landmark_coordinates import get_pixel_coordinates_from_landmark
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.utilities.constants import * 
from pyfame.utilities.general_utilities import compute_rot_angle  
import cv2 as cv
import numpy as np
from operator import itemgetter

class GridShuffleParameters(BaseModel):
    random_seed:Optional[int]
    shuffle_method:Union[int,str]
    shuffle_threshold:PositiveInt
    grid_square_size:PositiveInt
    landmark_paths:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]

    @field_validator("shuffle_method", mode="before")
    @classmethod
    def check_accepted_value(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if isinstance(value, str):
            value = str.lower(value)
            if value not in {"full random", "semi random"}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        elif isinstance(value, int):
            if value not in {27, 28}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value

        raise TypeError(f"Invalid type for parameter {field_name}, Must be one of int or str.")
    

class LayerSpatialGridShuffle(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, shuffle_parameters:GridShuffleParameters):

        self.time_config = timing_configuration
        self.shuffle_params = shuffle_parameters

        # Initialise superclass
        super().__init__(self.time_config)

        # Declare class parameters
        self.rand_seed = self.shuffle_params.random_seed
        self.shuffle_method = self.shuffle_params.shuffle_method
        self.shuffle_threshold = self.shuffle_params.shuffle_threshold
        self.grid_square_size = self.shuffle_params.grid_square_size
        self.landmark_paths = self.shuffle_params.landmark_paths

        # Snapshot of initial state
        self._snapshot_state()

    def supports_weight(self):
        return False
    
    def get_layer_parameters(self) -> dict:
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.shuffle_params.model_dump())
        self._layer_parameters["time_onset"] = self.onset_t
        self._layer_parameters["time_offset"] = self.offset_t
        return dict(self._layer_parameters)
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float) -> cv.typing.MatLike:
        
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            # Mask out the roi
            mask = mask_from_landmarks(frame, self.landmark_paths, landmarker_coordinates)
            bin_mask = mask.astype(bool)

            fo_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_FACE_OVAL)
            output_frame = np.zeros_like(frame, dtype=np.uint8)

            rng = None

            if self.rand_seed is not None:
                rng = np.random.default_rng(self.rand_seed)
            else:
                rng = np.random.default_rng()

            # Get x and y bounds of the face oval
            max_x = max(fo_screen_coords, key=itemgetter(0))[0]
            min_x = min(fo_screen_coords, key=itemgetter(0))[0]

            max_y = max(fo_screen_coords, key=itemgetter(1))[1]
            min_y = min(fo_screen_coords, key=itemgetter(1))[1]

            height = max_y-min_y
            width = max_x-min_x

            # Calculate x and y padding, ensuring integer
            x_pad = self.grid_square_size - (width % self.grid_square_size)
            y_pad = self.grid_square_size - (height % self.grid_square_size)

            if x_pad % 2 !=0:
                min_x -= np.floor(x_pad/2)
                max_x += np.ceil(x_pad/2)
            else:
                min_x -= x_pad/2
                max_x += x_pad/2
            
            if y_pad % 2 !=0:
                min_y -= np.floor(y_pad/2)
                max_y += np.ceil(y_pad/2)
            else:
                min_y -= y_pad/2
                max_y += y_pad/2
                
            # Ensure integer
            min_x = int(min_x)
            max_x = int(max_x)
            min_y = int(min_y)
            max_y = int(max_y)

            height = max_y-min_y
            width = max_x-min_x
            cols = int(width/self.grid_square_size)
            rows = int(height/self.grid_square_size)

            grid_squares = {}

            # Populate the grid_squares dict with segments of the frame
            for i in range(rows):
                for j in range(cols):
                    grid_squares.update({(i,j):frame[min_y:min_y + self.grid_square_size, min_x:min_x + self.grid_square_size]})
                    min_x += self.grid_square_size
                min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
                min_y += self.grid_square_size
            
            keys = list(grid_squares.keys())

            # Shuffle the keys of the grid_squares dict
            if self.shuffle_method == LOW_LEVEL_GRID_SHUFFLE or self.shuffle_method == "semi_random":
                shuffled_keys = keys.copy()
                shuffled_keys = np.array(shuffled_keys, dtype="i,i")
                shuffled_keys = shuffled_keys.reshape((rows, cols))

                ref_keys = keys.copy()
                ref_keys = np.array(ref_keys, dtype="i,i")
                ref_keys = ref_keys.reshape((rows, cols))

                visited_keys = set()

                # Localised threshold based shuffling of the grid
                for y in range(rows):
                    for x in range(cols):
                        if (x,y) in visited_keys:
                            continue
                        else:
                            x_min = max(0, x - self.shuffle_threshold)
                            x_max = min(cols-1, x + self.shuffle_threshold)
                            y_min = max(0, y - self.shuffle_threshold)
                            y_max = min(rows - 1, y + self.shuffle_threshold)

                            valid_new_positions = [
                                (new_x, new_y)
                                for new_x in range(x_min, x_max + 1)
                                for new_y in range(y_min, y_max + 1)
                                if (new_x, new_y) not in visited_keys
                            ]

                            if valid_new_positions:
                                new_x, new_y = rng.choice(valid_new_positions)

                                # Perform the positional swap
                                shuffled_keys[new_y,new_x] = ref_keys[y,x]
                                shuffled_keys[y,x] = ref_keys[new_y, new_x]
                                visited_keys.add((new_x, new_y))
                                visited_keys.add((x,y))
                            else:
                                visited_keys.add((x,y))

                shuffled_keys = list(shuffled_keys.reshape((-1,)))
                # Ensure tuple(int) 
                shuffled_keys = [tuple([int(x), int(y)]) for (x,y) in shuffled_keys]
            else:
                # Fully-random shuffling of the keys of the grid_squares dict
                shuffled_keys = keys.copy()
                rng.shuffle(shuffled_keys)
            
            # Populate the scrambled grid dict
            shuffled_grid_squares = {}

            for old_key, new_key in zip(keys, shuffled_keys):
                square = grid_squares.get(new_key)
                shuffled_grid_squares.update({old_key:square})

            # Fill the output frame with scrambled grid segments
            for i in range(rows):
                for j in range(cols):
                    cur_square = shuffled_grid_squares.get((i,j))
                    output_frame[min_y:min_y+self.grid_square_size, min_x:min_x+self.grid_square_size] = cur_square
                    min_x += self.grid_square_size
                min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
                min_y += self.grid_square_size
            
            min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
            min_y = int(min(fo_screen_coords, key=itemgetter(1))[1])

            # Calculate the slope of the connecting line & angle to the horizontal
            # landmarks 162, 389 form a paralell line to the x-axis when the face is vertical
            p1 = landmarker_coordinates[162]
            p2 = landmarker_coordinates[389]
            slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))

            if slope > 0:
                angle_from_x_axis = (-1)*compute_rot_angle(slope_1=slope)
            else:
                angle_from_x_axis = compute_rot_angle(slope_1=slope)
            cx = min_x + (width/2)
            cy = min_y + (height/2)

            # Using the rotation angle, generate a rotation matrix and apply affine transform
            rot_mat = cv.getRotationMatrix2D((cx,cy), (angle_from_x_axis), 1)
            output_frame = cv.warpAffine(output_frame, rot_mat, (frame.shape[1], frame.shape[0]))

            # Ensure grid only overlays the face oval
            masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
            masked_frame[bin_mask] = 255
            output_frame = np.where(masked_frame == 255, output_frame, frame)
            output_frame = output_frame.astype(np.uint8)

            return output_frame
        
def layer_spatial_grid_shuffle(timing_configuration:TimingConfiguration | None = None, landmark_paths:list[list[tuple[int,...]]] | list[tuple[int,...]]=LANDMARK_FACE_OVAL, 
                               random_seed:int|None = None, shuffle_method:int|str = "full random", shuffle_threshold:int = 2, grid_square_size:int = 40) -> LayerSpatialGridShuffle:
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input params
    try:
        params = GridShuffleParameters(
            random_seed=random_seed, 
            shuffle_method=shuffle_method, 
            shuffle_threshold=shuffle_threshold, 
            grid_square_size=grid_square_size, 
            landmark_paths=landmark_paths
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerSpatialGridShuffle.__name__}: {e}")
    
    return LayerSpatialGridShuffle(time_config, params)