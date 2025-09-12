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

def draw_legend(frame, vmin:float = 0.0, vmax:float = 1.0, legend_position:str = "top-left"):
    h, w, _ = frame.shape
    ref_height = 720
    scale = h / ref_height # Uniform scaling factor

    # Layout parameters
    width = int(150 * scale)        # legend box width
    height = int(200 * scale)       # legend box height
    bar_width = int(30 * scale)     # width of the colour bar
    padding = int(10 * scale)       # inner padding for text / bar
    title_offset = int(20 * scale)
    bottom_offset = int(50 * scale)
    font_scale = 0.5 * scale
    tick_font_scale = 0.4 * scale
    thickness = max(1, int(1 * scale))
    margin = int(0.01 * min(h, w))  # 1% padding from edges
    font = cv.FONT_HERSHEY_SIMPLEX

    # ---- Determine legend position ----
    if legend_position == "top-left":
        x, y = margin, margin
    elif legend_position == "top-right":
        x, y = w - width - bar_width - 2 * padding - margin, margin
    elif legend_position == "bottom-left":
        x, y = margin, h - height - bottom_offset - margin
    elif legend_position == "bottom-right":
        x, y = w - width - bar_width - 2 * padding - margin, h - height - bottom_offset - margin
    else:
        raise ValueError("legend_position must be one of: 'top-left', 'top-right', 'bottom-left', 'bottom-right'.")


    # Create a vertical gradient colour mapped to viridis
    gradient = np.linspace(0, 1, height)[:, None]
    cmap = cm.get_cmap("viridis")
    colours = (cmap(gradient)[:, :, :3] * 255).astype(np.uint8)
    legend_bar = cv.resize(colours, (bar_width, height - padding))
    legend_bar = cv.cvtColor(legend_bar, cv.COLOR_RGB2BGR)

    # Enclosing box and border
    cv.rectangle(frame, (x,y), (x + width + padding, y + height + bottom_offset), (255,255,255), -1)
    cv.rectangle(frame, (x - 2, y - 2), (x + width + padding + 2, y + height + bottom_offset + 2), (0,0,0), 2)
    # Fill in the viridis colour bar
    bar_top = y + int(45*scale)
    bar_bottom = y + height + int(35*scale)
    frame[bar_top:bar_bottom, x+padding:x+bar_width+padding] = legend_bar

    # Legend title
    cv.putText(frame, "Vector Magnitudes", (x + padding, y + title_offset), font, font_scale, (0,0,0), thickness, cv.LINE_AA)

    # Axis ticks
    # ---- Draw axis ticks ----
    ticks_y = [bar_top, (bar_top + bar_bottom)//2, bar_bottom]
    tick_values = [vmin, (vmin+vmax)/2, vmax]
    
    for ty, val in zip(ticks_y, tick_values):
        # tick line
        cv.line(frame, (x + bar_width + int(15 * scale), ty), (x + bar_width + int(25 * scale), ty), (0,0,0), thickness, cv.LINE_AA)
        # label 
        cv.putText(frame, f"{val:.1f}", (x + bar_width + int(30 * scale), ty + int(5 * scale)), font, tick_font_scale, (0, 0, 0), thickness, cv.LINE_AA)

    return frame