from pyfame.utilities.constants import *
from pyfame.mesh import *
from pyfame.utilities.general_utilities import compute_rot_angle, sanitize_json_value, get_roi_name
from pyfame.utilities.checks import *
from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
from pyfame.layer.manipulations.mask import mask_from_path 
import cv2 as cv
import numpy as np
from operator import itemgetter
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerSpatialGridShuffle(Layer):
    def __init__(self, rand_seed:int|None, method:int|str=HIGH_LEVEL_GRID_SHUFFLE, shuffle_threshold:int=2, grid_square_size:int=40, 
                 onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, roi:list[list[tuple]] | list[tuple]=FACE_OVAL_PATH, 
                 rise_duration:int=500, fall_duration:int=500, min_tracking_confidence:float=0.5, min_detection_confidence:float=0.5, **kwargs):
        # Initialise superclass
        super().__init__(onset_t, offset_t, timing_func, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        self.static_image_mode = False
        self._pre_attrs = []
        self._pre_attrs = set(self.__dict__) # snapshot of just the superclass parameters
        
        # Perform input parameter checks
        check_type(rand_seed, [int, type(None)])
        check_type(method, [int, str])
        check_value(method, [LOW_LEVEL_GRID_SHUFFLE, HIGH_LEVEL_GRID_SHUFFLE, "fully_random", "semi_random"])
        check_type(shuffle_threshold, [int])
        check_value(shuffle_threshold, min=1, max=5)
        check_type(grid_square_size, [int])

        # Declare class parameters
        self.rand_seed = rand_seed
        self.shuffle_method = method
        self.shuffle_threshold = shuffle_threshold
        self.grid_square_size = grid_square_size
        self.time_onset = onset_t
        self.time_offset = offset_t
        self.timing_function = timing_func
        self.region_of_interest = roi
        self.rise_duration_msec = rise_duration
        self.fall_duration_msec = fall_duration
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.timing_kwargs = kwargs

        self._capture_init_params()
    
    def _capture_init_params(self):
        # Extracting total parameter list post init
        post_attrs = set(self.__dict__.keys())

        # Getting only the subclass parameters
        new_attrs = post_attrs - self._pre_attrs

        # Store only subclass level params; ignore self
        params = {attr: getattr(self, attr) for attr in new_attrs}

        # Handle non serializable types
        if "region_of_interest" in params:
            params["region_of_interest"] = get_roi_name(params["region_of_interest"])

        # Sanitize values to be JSON compatible for logging
        self._layer_parameters = {
            k: sanitize_json_value(v) for k, v in params.items()
        }

    def supports_weight(self):
        return False
    
    def get_layer_parameters(self):
        return dict(self._layer_parameters)
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float, static_image_mode:bool = False):

        # Update the faceMesh when switching between image and video processing
        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            # Mask out the roi
            face_mesh = super().get_face_mesh()
            mask = mask_from_path(frame, self.region_of_interest, face_mesh)
            output_frame = np.zeros_like(frame, dtype=np.uint8)

            # Get the pixel coordinates of the face oval
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            fo_screen_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, FACE_OVAL_PATH)
            fo_mask = mask.astype(bool)

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
            landmark_screen_coords = get_mesh_coordinates(cv.cvtColor(frame, cv.COLOR_BGR2RGB), self.face_mesh)
            p1 = landmark_screen_coords[162]
            p2 = landmark_screen_coords[389]
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
            masked_frame[fo_mask] = 255
            output_frame = np.where(masked_frame == 255, output_frame, frame)
            output_frame = output_frame.astype(np.uint8)

            return output_frame
        
def layer_spatial_grid_shuffle(rand_seed:int|None, method:int|str = "fully_random", shuffle_threshold:int = 2, grid_square_size:int = 40, 
                               time_onset:float=None, time_offset:float=None, timing_function:Callable[...,float]=timing_linear, 
                               region_of_interest:list[list[tuple]] | list[tuple]=FACE_OVAL_PATH, rise_duration:int=500, fall_duration:int = 500, 
                               min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerSpatialGridShuffle:
    
    return LayerSpatialGridShuffle(rand_seed, method, shuffle_threshold, grid_square_size, time_onset, time_offset, 
                                   timing_function, region_of_interest, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)