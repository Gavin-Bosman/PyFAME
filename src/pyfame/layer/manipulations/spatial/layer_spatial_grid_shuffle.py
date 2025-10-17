from pydantic import BaseModel, field_validator, ValidationInfo, ValidationError, PositiveInt
from typing import Union, Optional
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.get_landmark_coordinates import get_pixel_coordinates_from_landmark
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.utilities.constants import * 
from pyfame.utilities.general_utilities import compute_rotation_angle, compute_slope
import cv2 as cv
import numpy as np
from operator import itemgetter

class GridShuffleParameters(BaseModel):
    random_seed:Optional[int]
    shuffle_method:Union[int,str]
    shuffle_threshold:PositiveInt
    grid_square_size:PositiveInt

    @field_validator("shuffle_method", mode="before")
    @classmethod
    def check_accepted_value(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if isinstance(value, str):
            value = str.lower(value)
            if value not in {"random", "semi random"}:
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

        self.prev_grid_size = None
        self.shuffled_keys = None

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
    
    def shuffle_keys(self, keys:list[tuple[int,int]], grid_size:tuple[int,int]):
        rows, cols = grid_size
        self.prev_grid_size = grid_size
        shuffled_keys = np.array(keys.copy(), dtype="i,i").reshape((rows, cols))
        visited_keys = set()
        rng = np.random.default_rng(self.rand_seed)

        # Shuffle the keys of the grid_squares dict
        if self.shuffle_method == SEMI_RANDOM_GRID_SHUFFLE or self.shuffle_method == "semi random":
            # Localised threshold based shuffling of the grid
            for r in range(rows):
                for c in range(cols):
                    if (r,c) in visited_keys:
                        continue
                    else:
                        r_min = max(0, r - self.shuffle_threshold)
                        r_max = min(rows-1, r + self.shuffle_threshold)
                        c_min = max(0, c - self.shuffle_threshold)
                        c_max = min(cols-1, c + self.shuffle_threshold)

                        valid_new_positions = [
                            (new_row, new_col)
                            for new_row in range(r_min, r_max + 1)
                            for new_col in range(c_min, c_max + 1)
                            if (new_row, new_col) not in visited_keys
                        ]

                        if not valid_new_positions:
                            visited_keys.add((r,c))
                            continue

                        # Randomly select a new grid position from the list of valid positions
                        new_row, new_col = rng.choice(valid_new_positions)

                        # Perform the positional swap
                        shuffled_keys[r,c], shuffled_keys[new_row,new_col] = (
                            shuffled_keys[new_row, new_col], shuffled_keys[r,c]
                        )
                        
                        visited_keys.add((r,c))
                        visited_keys.add((new_row, new_col))
                        
            shuffled_keys = shuffled_keys.reshape((-1)).tolist()
            # Ensure tuple(int) 
            shuffled_keys = [tuple([int(x), int(y)]) for (x,y) in shuffled_keys]
        else:
            # Fully-random shuffling of the keys of the grid_squares dict
            shuffled_keys = keys.copy()
            rng.shuffle(shuffled_keys)

        return shuffled_keys
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float) -> cv.typing.MatLike:
        
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        
        # Mask out the landmarks provided
        mask = mask_from_landmarks(frame, LANDMARK_FACE_OVAL, landmarker_coordinates)
        bin_mask = mask.astype(bool)

        fo_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_FACE_OVAL)
        output_frame = np.zeros_like(frame, dtype=np.uint8)

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
            min_x -= int(np.floor(x_pad/2))
            max_x += int(np.ceil(x_pad/2))
        else:
            min_x -= int(x_pad/2)
            max_x += int(x_pad/2)
        
        if y_pad % 2 !=0:
            min_y -= int(np.floor(y_pad/2))
            max_y += int(np.ceil(y_pad/2))
        else:
            min_y -= int(y_pad/2)
            max_y += int(y_pad/2)

        padded_height = max_y-min_y
        padded_width = max_x-min_x
        cols = int(padded_width/self.grid_square_size)
        rows = int(padded_height/self.grid_square_size)

        grid_squares = {}
        x_iter = min_x
        y_iter = min_y

        # Populate the grid_squares dict with segments of the frame
        for i in range(rows):
            for j in range(cols):
                grid_squares.update({(i,j):frame[y_iter:y_iter + self.grid_square_size, x_iter:x_iter + self.grid_square_size]})
                x_iter += self.grid_square_size
            x_iter = min_x
            y_iter += self.grid_square_size
        
        keys = list(grid_squares.keys())
        cur_grid_size = (rows, cols)

        # Only reshuffle the keys if the image grid has changed size
        if self.shuffled_keys is None or cur_grid_size != self.prev_grid_size:
            shuffled_keys = self.shuffle_keys(keys, (rows, cols))
        
        # Populate the scrambled grid dict
        shuffled_grid_squares = {}
        x_iter = min_x
        y_iter = min_y

        for old_key, new_key in zip(keys, shuffled_keys):
            square = grid_squares.get(new_key)
            shuffled_grid_squares.update({old_key:square})

        # Fill the output frame with shuffled grid segments
        for i in range(rows):
            for j in range(cols):
                cur_square = shuffled_grid_squares.get((i,j))
                output_frame[y_iter:y_iter+self.grid_square_size, x_iter:x_iter+self.grid_square_size] = cur_square
                x_iter += self.grid_square_size
            x_iter = min_x
            y_iter += self.grid_square_size

        # Calculate the slope of the connecting line & angle to the horizontal
        # landmarks 162, 389 form a paralell line to the x-axis when the face is vertical
        p1 = landmarker_coordinates[162]
        p2 = landmarker_coordinates[389]
        slope = compute_slope(p1, p2)
        rot_angle = compute_rotation_angle(slope_1=slope)

        cx = min_x + (padded_width/2)
        cy = min_y + (padded_height/2)

        # Using the rotation angle, generate a rotation matrix and apply affine transform
        rot_mat = cv.getRotationMatrix2D((cx,cy), (rot_angle), 1)
        output_frame = cv.warpAffine(output_frame, rot_mat, (frame.shape[1], frame.shape[0]))

        # Ensure grid only overlays the face oval
        masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
        masked_frame[bin_mask] = 255
        output_frame = np.where(masked_frame == 255, output_frame, frame)
        output_frame = output_frame.astype(np.uint8)

        return output_frame
        
def layer_spatial_grid_shuffle(timing_configuration:TimingConfiguration | None = None, random_seed:int|None = None, shuffle_method:int|str = "full random", shuffle_threshold:int = 2, grid_square_size:int = 40) -> LayerSpatialGridShuffle:
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input params
    try:
        params = GridShuffleParameters(
            random_seed=random_seed, 
            shuffle_method=shuffle_method, 
            shuffle_threshold=shuffle_threshold, 
            grid_square_size=grid_square_size
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerSpatialGridShuffle.__name__}: {e}")
    
    return LayerSpatialGridShuffle(time_config, params)