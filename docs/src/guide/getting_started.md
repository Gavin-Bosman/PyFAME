---
layout: doc
title: Getting Started
prev: 
    title: 'What is PyFAME?'
    link: '/guide/intro'
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

If you encounter any issues with installation, you are strongly encouraged to raise an issue on the projects GitHub repository page, found [here](https://github.com/Gavin-Bosman/PyFAME/issues/new). User-feedback and suggestions are greatly appreciated, and we at the PyFAME team thank you for all of your support.

## PyFAME Basics {#pyfame_basics}

Illustrated below is the simplest method of instantiating and applying a manipulation pipeline in PyFAME. The first step in most PyFAME scripts is to call the `make_paths()` function, which will internally perform an OS-walk of the project folder (named 'data' by default; but can be user-specified). Additionally, `Make_paths()` will create a subdirectory structure internal to the project folder; namely the `raw` and `processed` subdirectories will contain all of the pre, and post-manipulation files, respectively. The function returns a pandas DataFrame containing all of the absolute and relative file paths of every file in the working folder. After retrieving the files to be processed, users have two options. Firstly, to instantiate manipulation layers and pass them along with the file paths to the `apply_layers()` function. Secondly, users may pass the file_paths directly to an analysis function (i.e. `analyse_optical_flow()`).

### Simple Example {#simple_example}
```python
# All of PyFAME's manipulations, functions and constants are 
# available via a single top-level import.
import pyfame as pf

# Recursively walk the working folder; '/data/raw' by default.
# Log files and processed output are ignored by default; custom
# excluded directory names can be passed as a list of strings to make_paths().
files_to_process = pf.make_paths()          # Returns an [Nx2] dataframe of [absolute paths, relative paths]

# How this dataframe of file paths is split up is left to the user.
# i.e. blue_files = files_to_process.iloc[:, [0]]

# Declaring and configuring layers.

# TimingConfig is an object that groups together all of the parameters
# that control the timing and weighting of the manipulation Layers.
# For ease of use, TimingConfiguration does not require any input parameters, 
# and as such, will be filled with reasonable default values if nothing is passed.
time_config = pf.TimingConfiguration(time_onset = 0.5, time_offset = 3.2)

# Layer class objects are not directly exposed to the users; instead the Layers
# can be instantiated dynamically with factory functions. 
# Similar to TimingConfiguration, each of the Layer factory functions do not
# require any input parameters to run as expected, as they will be filled 
# with reasonable defaults if no parameters are passed.
mask_layer = pf.layer_mask(timing_configuration = time_config)
occlude_bar_layer = pf.layer_occlusion_bar()        # This will still return a valid Layer

# Finally, apply the layers to your selected file paths.
# The processed files will be written to the '/data/processed' folder by default
pf.apply_layers(file_paths = files_to_process, layers = [mask_layer, occlude_bar_layer])
```

### Additional Help

If you would like to see more specific and in-depth examples or tutorials, check out the [Examples](./examples.md) page next!

If you are unfamiliar with any of PyFAME's dependent architecture, including NumPy or Pandas, or you are just getting started with Python, check out some of the helpful links below:

| Dependency | Reference | 
| ---------- | --------- |
| NumPy      | [Python Numpy](https://www.geeksforgeeks.org/numpy/python-numpy/) |
| Pandas     | [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) |
| OpenCV     | [OpenCV Python](https://www.geeksforgeeks.org/python/opencv-python-tutorial/) |