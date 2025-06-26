import pyfame as pf
from pyfame.layer import *

in_dir = "C:\\Users\\gavin\\Desktop\\PyFAME\\data\\Video_Song_Actors_01-24\\Video_Song_Actor_01\\Actor_01\\01-02-02-01-01-02-01.mp4"
out_dir = "C:\\Users\\gavin\\Desktop\\PyFAME\\images\\"

 
face_mesh = pf.get_mesh(0.5,0.5,False)

point_light = LayerConfig(
    layer=pf.LayerStylizePointLight(1.0, maintain_background=False),
    face_mesh=face_mesh,
    roi=[pf.FACE_OVAL_TIGHT_PATH]
)

mask_config = LayerConfig(
    layer=pf.LayerMask(),
    face_mesh=face_mesh,
    roi=[pf.FACE_OVAL_PATH]
)

color_config = LayerConfig(
    layer=pf.LayerColor("red", 15.0),
    face_mesh=face_mesh,
    onset_t=0.5,
    offset_t=2.5,
    timing_func=pf.timing_sigmoid,
    roi=[pf.CHEEKS_PATH]
)

apply_layers([point_light], in_dir, out_dir, False)
# Internally, creates a LayerPipeline, which is used to sequentially apply the manipulations
# on a per-frame basis