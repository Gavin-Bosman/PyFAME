from pyfame.utilities.constants import *
from pyfame.mesh import *
from pyfame.utilities.checks import *
from pyfame.utilities.general_utilities import sanitize_json_value, get_roi_name
from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
from pyfame.layer.manipulations.mask import mask_from_path
import cv2 as cv
import numpy as np
from skimage.util import *
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerStylisePointLight(Layer):
    def __init__(self, point_density:float = 1.0, point_colour:tuple[int] = (255,255,255), maintain_background:bool = True, display_history_vectors:bool = False, 
                 history_method:int|str = SHOW_HISTORY_ORIGIN, history_window_msec:int = 500, history_vec_colour:tuple[int] = (0,0,255), onset_t:float=None, 
                 offset_t:float=None, timing_func:Callable[...,float]=timing_linear, roi:list[list[tuple]] | list[tuple]=FACE_OVAL_PATH, rise_duration:int=500,
                 fall_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        # Initialise superclass
        super().__init__(onset_t, offset_t, timing_func, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        self.static_image_mode = False
        self.frame_history = []
        self.prev_points = None
        self.idx_to_display = np.array([], dtype=np.uint8)
        self._pre_attrs = []
        self._pre_attrs = set(self.__dict__) # snapshot of just the superclass parameters
        
        # Perform input parameter checks
        check_type(point_density, [float])
        check_type(point_colour, [tuple])
        check_type(point_colour, [int], iterable=True)
        for i in point_colour:
            check_value(i, min=0, max=255)
        check_type(maintain_background, [bool])
        check_type(display_history_vectors, [bool])
        check_type(history_method, [int, str])
        check_value(history_method, [SHOW_HISTORY_ORIGIN, SHOW_HISTORY_RELATIVE, "origin", "relative"])
        check_type(history_window_msec, [int])
        check_value(history_window_msec, min=0)
        check_type(history_vec_colour, [tuple])
        check_type(history_vec_colour, [int], iterable=True)
        for i in history_vec_colour:
            check_value(i, min=0, max=255)

        # Declare class parameters
        self.point_density = point_density
        self.point_colour = point_colour
        self.maintain_background = maintain_background
        self.display_history_vectors = display_history_vectors
        self.history_method = history_method
        self.history_window_msec = history_window_msec
        self.history_colour = history_vec_colour
        self.time_onset = onset_t
        self.time_offset = offset_t
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

        self._layer_parameters = {
            k: sanitize_json_value(v) for k, v in params.items()
        }
    
    def supports_weight(self):
        return False
    
    def get_layer_parameters(self):
        return dict(self._layer_parameters)

    def apply_layer(self, frame:cv.typing.MatLike, dt:float, static_image_mode:bool = False):

        # Update the faceMesh when switching between image and video proccessing
        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            # Get the screen pixel coordinates of the frame/image
            face_mesh = super().get_face_mesh()
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
        
def layer_stylise_point_light(point_density:float = 1.0, point_colour:tuple[int] = (255,255,255), maintain_background:bool = True, display_history_vectors:bool = False, 
                 history_method:int|str = SHOW_HISTORY_ORIGIN, history_window_msec:int = 500, history_colour:tuple[int] = (0,0,255), time_onset:float=None, 
                 time_offset:float=None, timing_function:Callable[...,float]=timing_linear, region_of_interest:list[list[tuple]] | list[tuple]=FACE_OVAL_PATH, 
                 rise_duration:int=500, fall_duration:int=500, min_tracking_confidence:float=0.5, min_detection_confidence:float=0.5, **kwargs):
    
    return LayerStylisePointLight(point_density, point_colour, maintain_background, display_history_vectors, history_method, history_window_msec, 
                                  history_colour, time_onset, time_offset, timing_function, region_of_interest, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)