from pyfame.util.util_constants import *
from pyfame.mesh import *
from pyfame.util.util_general_utilities import compute_rot_angle
from pyfame.util.util_exceptions import *
from pyfame.layer import layer
import cv2 as cv
import mediapipe as mp
import numpy as np
from operator import itemgetter
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class layer_spatial_grid_shuffle(layer):
    def __init__(self, face_mesh:mp.solutions.face_mesh.FaceMesh, rand_seed:int|None, method:int = HIGH_LEVEL_GRID_SHUFFLE,
                 shuffle_threshold:int = 2, grid_square_size:int = 40):
        self.face_mesh = face_mesh
        self.rand_seed = rand_seed
        self.method = method
        self.threshold = shuffle_threshold
        self.size = grid_square_size
    
    def apply_layer(self, frame, weight, roi):
        mask = get_mask_from_path(frame, roi, self.face_mesh)
        output_frame = np.zeros_like(frame, dtype=np.uint8)

        # Computing shuffled grid positions
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        fo_screen_coords = get_mesh_coordinates_from_path(frame_rgb, self.face_mesh, FACE_OVAL_PATH)
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
        x_pad = self.size - (width % self.size)
        y_pad = self.size - (height % self.size)

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
        cols = int(width/self.size)
        rows = int(height/self.size)

        grid_squares = {}

        # Populate the grid_squares dict with segments of the frame
        for i in range(rows):
            for j in range(cols):
                grid_squares.update({(i,j):frame[min_y:min_y + self.size, min_x:min_x + self.size]})
                min_x += self.size
            min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
            min_y += self.size
        
        keys = list(grid_squares.keys())

        # Shuffle the keys of the grid_squares dict
        if self.method == LOW_LEVEL_GRID_SHUFFLE:
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
                        x_min = max(0, x - self.threshold)
                        x_max = min(cols-1, x + self.threshold)
                        y_min = max(0, y - self.threshold)
                        y_max = min(rows - 1, y + self.threshold)

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
            # Scramble the keys of the grid_squares dict
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
                output_frame[min_y:min_y+self.size, min_x:min_x+self.size] = cur_square
                min_x += self.size
            min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
            min_y += self.size
        
        min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
        min_y = int(min(fo_screen_coords, key=itemgetter(1))[1])

        # Calculate the slope of the connecting line & angle to the horizontal
        # landmarks 162, 389 form a paralell line to the x-axis when the face is vertical
        landmark_screen_coords = get_mesh_coordinates(cv.cvtColor(frame, cv.COLOR_BGR2RGB), self.face_mesh)
        p1 = landmark_screen_coords[162]
        p2 = landmark_screen_coords[389]
        slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))

        if slope > 0:
            angle_from_x_axis = (-1)*compute_rot_angle(slope1=slope)
        else:
            angle_from_x_axis = compute_rot_angle(slope1=slope)
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