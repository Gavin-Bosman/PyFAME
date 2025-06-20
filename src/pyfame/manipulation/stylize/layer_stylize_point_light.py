from pyfame.util.util_constants import *
from pyfame.mesh import *
from pyfame.util.util_exceptions import *
from pyfame.layer import Layer
import cv2 as cv
import mediapipe as mp
import numpy as np
from skimage.util import *
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

### Add on/off check in layer.apply_layers(...)

class layer_stylize_point_light(Layer):
    def __init__(self, face_mesh:mp.solutions.face_mesh.FaceMesh, point_density:float, point_color:tuple[int] = (255,255,255), 
                 maintain_background:bool = True, display_history_vectors:bool = False, history_method:int = SHOW_HISTORY_ORIGIN, 
                 history_window_msec:int = 500, history_vec_color:tuple[int] = (0,0,255)):
        self.face_mesh = face_mesh
        self.density = point_density
        self.point_color = point_color
        self.maintain_background = maintain_background
        self.history = display_history_vectors
        self.history_method = history_method
        self.window_msec = history_window_msec
        self.history_color = history_vec_color
        self.frame_history = []
        self.prev_points = None
    
    def apply_layer(self, frame:cv.typing.MatLike, weight:float, roi:list[list[tuple]] = [FACE_OVAL_PATH]):
        landmark_screen_coords = get_mesh_coordinates(cv.cvtColor(frame, cv.COLOR_BGR2RGB), self.face_mesh)
        lm_idx_to_display = np.array([], dtype=np.uint8)
        mask = np.zeros_like(frame, dtype=np.uint8)
        output_img = None
        frame_history_count = round(30 * (self.window_msec/1000))

        if self.maintain_background:
            output_img = frame.copy()
        else:
            output_img = np.zeros_like(frame, dtype=np.uint8)

        for lm_path in roi:
            lm_mask = get_mask_from_path(frame, [lm_path], self.face_mesh)
            lm_mask = lm_mask.astype(bool)

            # Use the generated bool mask to get valid indicies
            for lm in landmark_screen_coords:
                x = lm.get('x')
                y = lm.get('y')
                if lm_mask[y,x] == True:
                    lm_idx_to_display = np.append(lm_idx_to_display, lm.get('id'))
        
        if self.density != 1.0:
            # Pad and reshape idx array to slices of size 10
            new_lm_idx = lm_idx_to_display.copy()
            pad_size = len(new_lm_idx)%10
            append_arr = np.full(10-pad_size, -1)
            new_lm_idx = np.append(new_lm_idx, append_arr)
            new_lm_idx = new_lm_idx.reshape((-1, 10))

            bin_idx_mask = np.zeros((new_lm_idx.shape[0], new_lm_idx.shape[1]))

            for i,_slice in enumerate(new_lm_idx):
                num_ones = round(np.floor(10*self.density))

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
            lm_idx_to_display = np.where(bin_idx_mask == 1, new_lm_idx, -1)

        cur_points = []
        history_mask = np.zeros_like(frame, dtype=np.uint8)

        # Get current landmark screen coords
        for id in lm_idx_to_display:
            if id != -1:
                point = landmark_screen_coords[id]
                cur_points.append((point.get('x'), point.get('y')))
        
        if self.prev_points == None or self.history == False:
            self.prev_points = cur_points.copy()

            for point in cur_points:
                x1, y1 = point
                if x1 > 0 and y1 > 0:
                    output_img = cv.circle(output_img, (x1, y1), 3, self.point_color, -1)

        elif self.history_method == SHOW_HISTORY_ORIGIN:
            # If show_history is true, display vector paths of all points
            for (old, new) in zip(self.prev_points, cur_points):
                x0, y0 = old
                x1, y1 = new
                mask = cv.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), self.history_color, 2)
                output_img = cv.circle(output_img, (int(x1), int(y1)), 3, self.point_color, -1)

            self.prev_points = cur_points.copy()
            output_img = cv.add(output_img, mask)
            mask = np.zeros_like(frame, dtype=np.uint8)

        else:
            # If show_history is true, display vector paths of all points
            for (old, new) in zip(self.prev_points, cur_points):
                x0, y0 = old
                x1, y1 = new
                mask = cv.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), self.history_color, 2)
                output_img = cv.circle(output_img, (int(x1), int(y1)), 3, self.point_color, -1)

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