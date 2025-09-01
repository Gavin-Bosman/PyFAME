from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, NonNegativeInt, PositiveInt, PositiveFloat, NonNegativeFloat
from typing import List, Tuple, Optional
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_path
from pyfame.mesh import *
from pyfame.utilities.exceptions import *
import cv2 as cv
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mpcolors
from .draw_optical_flow_legend import draw_legend

def draw_scaled_flow_arrows(frame, origin_point, flow_vector, colour, arrow_length:int = 10, line_thickness:int = 1) -> cv.typing.MatLike:

    x, y = origin_point
    dx, dy = flow_vector

    magnitude = np.sqrt(dx**2 + dy**2)

    if magnitude > 1e-3:    # Avoid divide by zero errors 
        dx /= magnitude
        dy /= magnitude
    else:
        dx, dy = 0, 0
    
    # Scale end_pt to constant arrow length
    end_point = (int(x + dx * arrow_length), int(y + dy * arrow_length))

    # Draw the arrows 
    cv.arrowedLine(frame, (int(x), int(y)), end_point, colour, line_thickness, tipLength=0.3)
    return frame


class SparseFlowParameters(BaseModel):
    landmarks_to_track:Optional[List[NonNegativeInt]]
    max_points:PositiveInt
    point_quality_threshold:PositiveFloat
    min_point_distance:NonNegativeInt = 7
    pixel_neighborhood_size:Tuple[NonNegativeInt, NonNegativeInt] = (5,5)
    search_window_size:Tuple[NonNegativeInt, NonNegativeInt] = (15,15)
    max_pyramid_level:NonNegativeInt = 2
    gaussian_deviation:NonNegativeFloat = 1.2
    max_iterations:PositiveInt
    flow_accuracy_threshold:PositiveFloat
    point_colour:Tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]
    point_radius:PositiveInt
    vector_colour:Tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]
    vector_line_width:PositiveInt
    legend:bool = True

    @field_validator("point_quality_threshold", "flow_accuracy_threshold")
    @classmethod
    def check_normal_range(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if not (0.0 < value <= 1.0):
            raise ValueError(f"Parameter {field_name} must lie in the normalised range 0.0-1.0.")
        
        return value

    @field_validator("point_colour", "vector_colour")
    @classmethod
    def check_in_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        for elem in value:
            if not (0 <= elem <= 255):
                raise ValueError(f"{field_name} values must lie between 0 and 255.")
        
        return value

class LayerStyliseOpticalFlowSparse(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, flow_parameters:SparseFlowParameters):

        self.time_config = timing_configuration
        self.flow_params = flow_parameters

        # Initialise superclass
        super().__init__(self.time_config)
        self.initial_points = []
        self.previous_grey_frame = None
        self.vector_mask = None
        self.loop_counter = 1
        self.last_arrows = None
        self.cmap = cm.get_cmap("viridis")
        self.norm = mpcolors.Normalize(vmin=0.0, vmax=20.0)
        self.magnitude_max = 20.0

        # Declare class parameters
        self.landmark_idx_to_track = self.flow_params.landmarks_to_track
        self.max_points = self.flow_params.max_points
        self.point_quality_threshold = self.flow_params.point_quality_threshold
        self.min_point_distance = self.flow_params.min_point_distance
        self.pixel_neighborhood_size = self.flow_params.pixel_neighborhood_size
        self.search_window_size = self.flow_params.search_window_size
        self.max_pyramid_level = self.flow_params.max_pyramid_level
        self.gaussian_deviation = self.flow_params.gaussian_deviation
        self.max_iterations = self.flow_params.max_iterations
        self.flow_accuracy_threshold = self.flow_params.flow_accuracy_threshold
        self.point_colour = self.flow_params.point_colour
        self.point_radius = self.flow_params.point_radius
        self.vector_colour = self.flow_params.vector_colour
        self.vector_line_width = self.flow_params.vector_line_width
        self.legend = self.flow_params.legend
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
        self._layer_parameters.update(self.flow_params.model_dump())
        self._layer_parameters["time_onset"] = self.onset_t
        self._layer_parameters["time_offset"] = self.offset_t
        return dict(self._layer_parameters)
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float, static_image_mode:bool = False):

        # Update the faceMesh when switching between image and video processing
        if static_image_mode == True:
            raise UnrecognizedExtensionError(message="Sparse optical flow does not support static image files.")

        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            # Defining persistent loop params
            output_img = None

            # Parameters for lucas kanade optical flow
            lk_params = dict(winSize  = self.search_window_size,
                maxLevel = self.max_pyramid_level,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, self.max_iterations, self.flow_accuracy_threshold))

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            face_mesh = super().get_face_mesh()
            landmark_screen_coords = get_mesh_coordinates(frame_rgb, face_mesh)
            
            # Create face oval image mask
            face_mask = mask_from_path(frame, FACE_OVAL_PATH, face_mesh)

            # Main Processing loop
            
            if self.loop_counter == 1:
                self.vector_mask = np.zeros_like(frame)
                self.previous_grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # If landmarks were provided 
                if self.landmark_idx_to_track is not None:
                    self.initial_points = np.array([[lm.get('x'), lm.get('y')] for lm in landmark_screen_coords if lm.get('id') in self.landmark_idx_to_track], dtype=np.float32)
                    self.initial_points = self.initial_points.reshape(-1,1,2)
                else:
                    self.initial_points = cv.goodFeaturesToTrack(self.previous_grey_frame, self.max_points, self.point_quality_threshold, self.min_point_distance, self.pixel_neighborhood_size, mask=face_mask)
                
                self.loop_counter += 1
                return frame

            if self.loop_counter > 1:
                grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # Calculate optical flow
                cur_points, st, err = cv.calcOpticalFlowPyrLK(self.previous_grey_frame, grey_frame, self.initial_points, None, **lk_params)

                if cur_points is None:
                    return frame
                
                # Select good points
                good_new_points = cur_points[st==1]
                good_old_points = self.initial_points[st==1]
                
                arrows = []
                for (x0,y0), (x1,y1) in zip(good_old_points, good_new_points):
                    dx, dy = (x1 - x0, y1 - y0)
                    mag = np.sqrt(dx**2 + dy**2)
                    arrows.append((int(x0), int(y0), int(dx), int(dy), float(mag)))
                
                self.last_arrows = arrows
                # Update previous frame and points
                self.previous_grey_frame = grey_frame.copy()
                self.initial_points = good_new_points.reshape(-1, 1, 2)

                output_img = frame.copy()

                # Scale the arrow length to the facial width
                y_whites, x_whites = np.where(face_mask > 0)
                x_min = x_whites.min()
                x_max = x_whites.max()
                arrow_length = int((x_max - x_min) * 0.05)

                if self.last_arrows is not None:
                    for (x0, y0, dx, dy, mag) in self.last_arrows:
                        colour = self.cmap(self.norm(mag))[:3]   # RGB in [0,1]
                        colour_bgr = tuple(int(255*c) for c in colour[::-1])
                        output_img = draw_scaled_flow_arrows(output_img, (x0, y0), (dx, dy), colour_bgr, arrow_length, self.vector_line_width)

                self.loop_counter += 1

                if self.legend:
                    draw_legend(output_img, self.magnitude_max)
                    
                return output_img
                
def layer_stylise_optical_flow_sparse(timing_configuration:TimingConfiguration | None = None, landmarks_to_track:list[int] | None = None, max_points:int = 20, 
                                      point_quality_threshold:float = 0.3, max_iterations:int = 10, flow_accuracy_threshold:float = 0.03, point_colour:tuple[int,int,int] = (0,0,191), 
                                      point_radius:int = 5, vector_colour:tuple[int,int,int] = (0,0,191), vector_line_width:int = 2, legend:bool = True) -> LayerStyliseOpticalFlowSparse:
    
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input params
    try:
        params = SparseFlowParameters(
            landmarks_to_track=landmarks_to_track, 
            max_points=max_points, 
            point_quality_threshold=point_quality_threshold,
            max_iterations=max_iterations, 
            flow_accuracy_threshold=flow_accuracy_threshold, 
            point_colour=point_colour,
            point_radius=point_radius, 
            vector_colour=vector_colour, 
            vector_line_width=vector_line_width,
            legend=legend
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerStyliseOpticalFlowSparse.__name__}: {e}")
    
    return LayerStyliseOpticalFlowSparse(time_config, params)