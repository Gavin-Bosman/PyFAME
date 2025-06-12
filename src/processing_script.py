import cv2 as cv
import mediapipe as mp
import matplotlib.pyplot as plt
import pyfame as pf

in_dir = ".\\data\\Video_Song_Actors_01-24\\Video_Song_Actor_01\\Actor_01\\01-02-02-01-01-02-01.mp4"
out_dir = "C:\\Users\\gavin\\Desktop\\PyFAME\\images\\"

face_mesh = pf.get_mesh(0.5, 0.5, False, 1)
pld = pf.layer_stylize_point_light(face_mesh, 1.0, point_color=(0,0,255))
pf.apply_layers(layers=[pld], input_dir=in_dir, output_dir=out_dir)