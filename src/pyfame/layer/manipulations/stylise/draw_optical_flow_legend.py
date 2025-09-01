import cv2 as cv
import numpy as np
import matplotlib.cm as cm

def draw_legend(frame, max_magnitude, width=150, height=200, bar_width=30, position=(10,10)):
    x, y = position

    # Create a vertical gradient colour mapped to viridis
    gradient = np.linspace(0, 1, height)[:, None]
    cmap = cm.get_cmap("viridis")
    colours = (cmap(gradient)[:, :, :3] * 255).astype(np.uint8)

    legend_bar = cv.resize(colours, (bar_width, height - 10))
    legend_bar = cv.cvtColor(legend_bar, cv.COLOR_RGB2BGR)

    # Start drawing the legend components, starting with the enclosing box
    cv.rectangle(frame, position, (x + width + 10, y + height + 60), (255,255,255), -1)
    # Draw the legend border
    cv.rectangle(frame, (x - 2, y - 2), (x + width + 12, y + height + 62), (0,0,0), 2)
    # Fill in the viridis colour bar
    frame[y+55:y+height+45, x+15:x+bar_width+15] = legend_bar

    font = cv.FONT_HERSHEY_SIMPLEX
    # Overlay the legend title
    cv.putText(frame, "Motion Vector", (x + 15, y + 20), font, 0.6, (0,0,0), 1, cv.LINE_AA)
    cv.putText(frame, "Magnitudes", (x + bar_width, y + 40), font, 0.6, (0,0,0), 1, cv.LINE_AA)

    # Draw axis labels
    cv.putText(frame, "0", (x + bar_width + 30, y + 60),
               font, 0.5, (0,0,0), 1, cv.LINE_AA)
    cv.putText(frame, "10.0", (x + bar_width + 30, y + 155), 
               font, 0.5, (0,0,0), 1, cv.LINE_AA)
    cv.putText(frame, f"{max_magnitude:.1f}", (x + bar_width + 30, y + height + 50),
               font, 0.5, (0,0,0), 1, cv.LINE_AA)
    
    # Draw axis ticks
    cv.line(frame, (x + bar_width + 15, y + 55), (x + bar_width + 25, y + 55), (0,0,0), 1, cv.LINE_AA)
    cv.line(frame, (x + bar_width + 15, y + 150), (x + bar_width + 25, y + 150), (0,0,0), 1, cv.LINE_AA)
    cv.line(frame, (x + bar_width + 15, y + height + 45), (x + bar_width + 25, y + height + 45), (0,0,0), 1, cv.LINE_AA)

    return frame