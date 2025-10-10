from .analyse_facial_colour_means import analyse_facial_colour_means
from .analyse_optical_flow_sparse import analyse_optical_flow_sparse
from .analyse_optical_flow_dense import analyse_optical_flow_dense

# U.S. spelling function name aliasing
analyze_facial_color_means = analyse_facial_colour_means
analyze_optical_flow_dense = analyse_optical_flow_dense
analyze_optical_flow_sparse = analyse_optical_flow_sparse

__all__ = [
    "analyse_facial_colour_means", "analyse_optical_flow_dense", "analyse_optical_flow_sparse",
    "analyze_facial_color_means", "analyze_optical_flow_dense", "analyze_optical_flow_sparse"
]