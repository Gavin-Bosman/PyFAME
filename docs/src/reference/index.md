---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "PyFAME"
  text: "The Python Facial Analysis and Manipulation Environment"
  tagline: An Innovative Facial Psychology Toolkit For Python
  image:
    src: ../images/output_grid.png
    alt: Some example function outputs
  actions:
    - theme: brand
      text: Source Code
      link: https://github.com/Gavin-Bosman/PyFAME
    - theme: alt
      text: API Reference
      link: /reference

features:
  - title: Multimedia Operability
    details: PyFAME can perform dynamic facial masking, occlusion and color-shifting over a variety of image and video file types. PyFAME is one of the first packages of it's kind, allowing users to directly manipulate and analyze video files, rather than only still images.
  - title: Temporal and Spatial Manipulations
    details: PyFAME provides a variety of novel features allowing users to perform temporal and spatial manipulations over video files. For example, the shuffle_frame_order function allows users to reorganize video frames using a variety of methods including cyclic shifts, interleaving, and palindrome. 
  - title: Multi-File Processing
    details: With just a single line of Python code, PyFAME allows you to process any number of files given an input directory. This directory can contain nested subdirectories, and varying file extensions, which will all be handled automatically. 
---

