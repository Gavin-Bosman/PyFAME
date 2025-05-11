---
layout: doc
title: What is PyFAME?
prev: false
next:
    title: 'Getting Started'
    link: '/guide/getting_started'
---
# What is PyFAME?

PyFAME is an open source Python package that leverages computer vision to apply a variety of classical facial psychology manipulations to both static images and videos. The manipulations provided include perception-based manipulations (color-shifting, occluding, blurring, masking), spatial manipulations (landmark shuffling, point-light display), and temporal manipulations (shuffle frame order, timing functions). PyFAME leverages MediaPipe's FaceMesh solution to identify and track 478 unique facial landmarks, allowing for precise positional tracking and regional application of perception-based manipulations.

<img src="/output_grid.png" width=500 style="display:block; margin:auto;"/>

## Statement Of Need

Currently, there are no available tools for performing these types of pixel-level operations over videos. Existing research has commonly used general image editing tools (such as Photoshop). However, these tools apply changes to the entire image, causing noticeable background artifacts. PyFAME provides users the ability to selectively modify specific regions of the face, for both static images and videos. Our package also seamlessly integrates temporal functions into it's video processing, allowing users to specify how and when pixel-level operations will be applied.

Furthermore, to our knowledge there are no existing software packages or tools that provide an equivalent breadth of features like PyFAME. An in-depth literature review of facial psychology was performed to identify any and all pertinent facial manipulations used. Currently, PyFAME provides access to facial masking, occlusion, blurring, noise-application, color-shifting, saturation-shifting, brightness-shifting, color-sampling, sparse optical-flow, dense optical-flow, landmark shuffling, grid-based shuffling, frame shuffling, point-light display, and more. Additionally, many of the above manipulations can be temporally controlled with several predefined timing functions; including linear, constant, sigmoid and Gaussian. 