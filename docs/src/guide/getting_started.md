---
layout: doc
title: Getting Started
prev: false
next: 
    title: 'Examples'
    link: '/guide/examples'
---

# Getting Started

## Installing Pyfame {#install}

PyFAME requires Python >= 3.9 to be installed on your system. You can find information on installing and setting up Python [here](https://wiki.python.org/moin/BeginnersGuide/Download). Once you have Python installed, PyFAME can be installed with pip via PyPi:

```sh
pip install pyfame
```

## Quick Example {#quick_example}

PyFAME includes a large library of classical facial psychology manipulations, such as masking, occlusion, landmark-shuffling, and color manipulation. Additionally, almost all of these manipulations can be layered. Below is a quick example on how you could desaturate, then occlude half of the face in the provided video file.

```python
# All of PyFAME's core functions are available via a top-level import
import pyfame as pf

# Define the input and output directory paths
input_directory = "./my/input/path/"
output_directory = "./my/output/path/"

# Applying a facial desaturation
pf.face_saturation_shift(input_directory, output_directory, shift_magnitude=-10.0)

# Every PyFAME operation has a unique output_directory name, for saturation it is Sat_Shifted/.
# In order to layer the manipulations, we pass the previous output directory as the input directory 
# to the next manipulation
input_directory = output_directory + "Sat_Shifted/"

# Applying a hemi-face occlusion
pf.occlude_face_region(output_directory, output_directory, landmarks_to_occlude=[HEMI_FACE_LEFT])
```

### Before:
<div style="display:flex; align-items:center; justify-content:center;">
    <img src="/Actor_08.png" width=400px />
</div>

### After:
<div style="display:flex; align-items:center; justify-content:center;">
    <img src="/Actor_08_sat_shifted_occluded.png" width=400px />
</div>
