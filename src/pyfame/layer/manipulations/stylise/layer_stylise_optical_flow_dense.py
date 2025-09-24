from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, PositiveInt, NonNegativeInt, PositiveFloat, NonNegativeFloat
from typing import Any
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.utilities.exceptions import UnrecognizedExtensionError
import cv2 as cv
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mpcolors
import pandas as pd
from pyfame.layer.manipulations.stylise.draw_optical_flow_legend import draw_legend
from pyfame.analyse.analyse_optical_flow_sparse import analyse_optical_flow_sparse

class DenseFlowParameters(BaseModel):
    pixel_neighborhood_size:PositiveInt
    search_window_size:PositiveInt
    max_pyramid_level:NonNegativeInt
    pyramid_scale:PositiveFloat
    max_iterations:PositiveInt
    gaussian_deviation:NonNegativeFloat
    legend:bool
    legend_position:str="top-left"
    precise_colour_scale:bool = True

    @field_validator("pyramid_scale")
    @classmethod
    def check_normal_range(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if not (0.0 < value <= 1.0):
            raise ValueError(f"Parameter {field_name} must lie in the normalised range 0.0-1.0.")
        
        return value
    
    @field_validator("legend_position")
    @classmethod
    def check_accepted_value(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if value not in {"top-left", "bottom-left", "top-right", "bottom-right"}:
            raise ValueError(
                f"Unrecognized value provided for parameter {field_name}."
                f" {field_name} must be one of: 'top-left', 'bottom-left', 'top-right', 'bottom-right'."
            )
        
        return value

class LayerStyliseOpticalFlowDense(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, flow_parameters:DenseFlowParameters):
        
        self.time_config = timing_configuration
        self.flow_params = flow_parameters

        super().__init__(self.time_config)

        # Intra-frame tracking parameters
        self.loop_counter = 1
        self.previous_grey_frame = None
        self.cmap = cm.get_cmap("viridis")

        # Dynamic magnitude scaling params
        self.global_mags = []
        self.norm = None
        self.magnitude_min = None
        self.magnitude_max = None

        # Declare class parameters
        self.pixel_neighborhood_size = self.flow_params.pixel_neighborhood_size
        self.search_window_size = self.flow_params.search_window_size
        self.max_pyramid_level = self.flow_params.max_pyramid_level
        self.pyramid_scale = self.flow_params.pyramid_scale
        self.max_iterations = self.flow_params.max_iterations
        self.gaussian_deviation = self.flow_params.gaussian_deviation
        self.legend = self.flow_params.legend
        self.legend_position = self.flow_params.legend_position
        self.precise_colour_scale = self.flow_params.precise_colour_scale
        self.min_tracking_confidence = self.time_config.min_tracking_confidence
        self.min_detection_confidence = self.time_config.min_detection_confidence

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
    
    def precompute_colour_scale(self, file_path:str) -> None:
        if self.norm is not None:
            return

        results = analyse_optical_flow_sparse(
            pd.DataFrame({"Absolute Path" : [file_path]}),
            max_points=10,
            output_detail_level="full",
            frame_step=15
        )

        result_df = list(results.values())[0]
        mean_magnitude = result_df["magnitude"].mean()
        std_magnitude = result_df["magnitude"].std()

        self.magnitude_min = float(max(0.0, mean_magnitude - std_magnitude))
        self.magnitude_max = float(mean_magnitude + std_magnitude)
        self.norm = mpcolors.Normalize(vmin=self.magnitude_min, vmax=self.magnitude_max)
    
    def apply_layer(self, face_mesh:Any, frame:cv.typing.MatLike, dt:float, file_path:str|None = None) -> cv.typing.MatLike:
        
        # If precise_colour_scale is True, precompute the norm using the full analysis results
        if self.precise_colour_scale and self.norm is None:
            if not file_path:
                raise ValueError("File_path must be provided to apply_layer() when precise_colour_scale = True.")
            self.precompute_colour_scale(file_path)

        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        
        if self.loop_counter == 1:
            self.previous_grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            self.loop_counter += 1
            viridis = self.cmap(np.zeros(frame.shape[:2], dtype=float))
            viridis_rgb = viridis[:, :, :3]
            viridis_bgr = viridis_rgb[:, :, ::-1]
            output_img = (viridis_bgr * 255).astype(np.uint8)
            return output_img

        if self.loop_counter > 1:
            # Recompute the max magnitude every 1000 msec or ~30 frames
            timestamp_msec = dt*1000
            rolling_time_window = 1000

            grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Calculate dense optical flow
            flow = cv.calcOpticalFlowFarneback(
                self.previous_grey_frame, 
                grey_frame,
                None, 
                self.pyramid_scale, 
                self.max_pyramid_level, 
                self.search_window_size, 
                self.max_iterations, 
                self.pixel_neighborhood_size, 
                self.gaussian_deviation, 
                0
            )

            # Get vector magnitudes and angles
            magnitudes, _ = cv.cartToPolar(flow[...,0],flow[...,1])

            # Collect global mags
            self.global_mags.extend(magnitudes.flatten())

            if not self.precise_colour_scale:
                if self.magnitude_max is None or self.magnitude_min is None or timestamp_msec >= rolling_time_window:
                    self.magnitude_min = float(np.min(self.global_mags))
                    self.magnitude_max = float(np.max(self.global_mags))
                    self.norm = mpcolors.Normalize(vmin=self.magnitude_min, vmax=self.magnitude_max)
                    rolling_time_window += rolling_time_window

            # Normalise magnitudes to [0,1]
            normal_mags = self.norm(magnitudes)

            # Map magnitudes to viridis colour scale
            viridis = self.cmap(normal_mags)
            viridis_rgb = viridis[:, :, :3]
            viridis_bgr = viridis_rgb[:, :, ::-1]
            output_img = (viridis_bgr * 255).astype(np.uint8)

            self.previous_grey_frame = grey_frame.copy()

            if self.legend:
                draw_legend(
                    frame=output_img, 
                    vmin=self.magnitude_min, 
                    vmax=self.magnitude_max, 
                    legend_position=self.legend_position
                )

            return output_img

def layer_stylise_optical_flow_dense(timing_configuration:TimingConfiguration | None = None, pixel_neighborhood_size:int = 5, search_window_size:int = 15, 
                                     max_pyramid_level:int = 2, pyramid_scale:float = 0.5, max_iterations:int = 10, gaussian_deviation:float = 1.2, 
                                     legend:bool = True, legend_position:str = "top-left", precise_colour_scale:bool = True) -> LayerStyliseOpticalFlowDense:
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = DenseFlowParameters(
            pixel_neighborhood_size=pixel_neighborhood_size, 
            search_window_size=search_window_size, 
            max_pyramid_level=max_pyramid_level, 
            pyramid_scale=pyramid_scale, 
            max_iterations=max_iterations, 
            gaussian_deviation=gaussian_deviation, 
            legend=legend,
            legend_position=legend_position,
            precise_colour_scale=precise_colour_scale
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerStyliseOpticalFlowDense.__name__}: {e}")

    return LayerStyliseOpticalFlowDense(time_config, params)