import pyfame as pf
from pyfame.manipulation.color import LayerColorSaturation
from pyfame.layer import *

in_dir = "C:\\Users\\gavin\\Desktop\\PyFAME\\data\\Video_Song_Actors_01-24\\Video_Song_Actor_01\\Actor_01\\01-02-02-01-01-02-01.mp4"
out_dir = "C:\\Users\\gavin\\Desktop\\PyFAME\\images\\"

# Notice how face_mesh no longer needs to be passed (now handled internally)
layer_sat = LayerColorSaturation(0.5, 3.0, magnitude=-15.0)
pf.apply_layers([layer_sat], in_dir, out_dir)

# The config and call are now reduced to only 2 lines of code!