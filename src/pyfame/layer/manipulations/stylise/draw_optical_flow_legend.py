import cv2 as cv
import numpy as np
import matplotlib.cm as cm
import numbers

def _resolve_position(position, frame_h, frame_w) -> tuple[int,int]:
    if not (isinstance(position, tuple) and len(position) == 2):
        raise ValueError("Legend position must be a tuple of (x,y).")

    x,y = position

    if type(x) is not type(y):
        raise ValueError("Legend position requires both (x,y) coordinates to be of the same type.")

    # Relative coordinates (fractions)
    if isinstance(x, numbers.Real) and isinstance(y, numbers.Real):
        if isinstance(x, float) and 0 <= x <= 1 and 0 <= y <= 1:
            return int(x * frame_w), int(y * frame_h)
        else:
            return int(x), int(y)
    
    raise TypeError("Legend position must be floats in [0,1] or integers (pixels).")

def draw_legend(frame, max_magnitude, position=None):
    h, w, _ = frame.shape

    if position is None:
        # 1% of height and width padding from the frame border
        position = _resolve_position((0.01, 0.01), h, w)
    else:
        position = _resolve_position(position, h, w)

    x, y = position

    # Scale to preferred aspect ratio
    ref_height = 720
    scale = h / ref_height # Uniform scaling factor

    # Scale base legend dimensions
    width = int(150 * scale)
    height = int(200 * scale)
    bar_width = int(30 * scale)
    padding = int(10 * scale)

    # Create a vertical gradient colour mapped to viridis
    gradient = np.linspace(0, 1, height)[:, None]
    cmap = cm.get_cmap("viridis")
    colours = (cmap(gradient)[:, :, :3] * 255).astype(np.uint8)

    legend_bar = cv.resize(colours, (bar_width, height - padding))
    legend_bar = cv.cvtColor(legend_bar, cv.COLOR_RGB2BGR)

    # Enclosing box
    cv.rectangle(frame, (x,y), (x + width + padding, y + height + int(60*scale)), (255,255,255), -1)
    # Legend border
    cv.rectangle(frame, (x - 2, y - 2), (x + width + padding + 2, y + height + int(60*scale) + 2), (0,0,0), 2)
    # Fill in the viridis colour bar
    frame[y+int(55*scale):y+height+int(45*scale), x+padding:x+bar_width+padding] = legend_bar

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * scale
    thickness = max(1, int(1 * scale))

    # Legend title
    cv.putText(frame, "Motion Vector", (x + padding, y + int(20*scale)), font, font_scale, (0,0,0), thickness, cv.LINE_AA)
    cv.putText(frame, "Magnitudes", (x + bar_width, y + int(40*scale)), font, font_scale, (0,0,0), thickness, cv.LINE_AA)

    # Axis labels
    cv.putText(frame, "0", (x + bar_width + int(30*scale), y + int(60*scale)),
               font, 0.5*scale, (0,0,0), thickness, cv.LINE_AA)
    cv.putText(frame, "10.0", (x + bar_width + int(30*scale), y + int(155*scale)), 
               font, 0.5*scale, (0,0,0), thickness, cv.LINE_AA)
    cv.putText(frame, f"{max_magnitude:.1f}", (x + bar_width + int(30*scale), y + height + int(50*scale)),
               font, 0.5*scale, (0,0,0), thickness, cv.LINE_AA)

    # Axis ticks
    cv.line(frame, (x + bar_width + int(15*scale), y + int(55*scale)), (x + bar_width + int(25*scale), y + int(55*scale)), (0,0,0), thickness, cv.LINE_AA)
    cv.line(frame, (x + bar_width + int(15*scale), y + int(150*scale)), (x + bar_width + int(25*scale), y + int(150*scale)), (0,0,0), thickness, cv.LINE_AA)
    cv.line(frame, (x + bar_width + int(15*scale), y + height + int(45*scale)), (x + bar_width + int(25*scale), y + height + int(45*scale)), (0,0,0), thickness, cv.LINE_AA)

    return frame