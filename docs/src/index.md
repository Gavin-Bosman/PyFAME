---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "PyFAME"
  text: "The Python Facial Analysis and Manipulation Environment"
  tagline: An Innovative Facial Psychology Toolkit For Python
  image:
    src: ./pyfame_logo.png
    alt: Some example function outputs
  actions:
    - theme: brand
      text: Getting Started
      link: /examples/examples
    - theme: alt
      text: API Reference
      link: /reference/overview

features:
  - icon:
      light: 'icons/multimedia-light.svg'
      dark: 'icons/multimedia-dark.svg'
    title: Multimedia Operability
    details: Apply a library of 15+ classical facial manipulations to still images and movie stimuli (masking, occlusion, blurring, point-light display etc.)
  - icon:
      light: 'icons/layer.svg'
      dark: 'icons/layer.svg'
    title: Layer Manipulations
    details: Multiple facial manipulations can be layered/composed into a single stimuli (i.e. landmark shuffling + desaturation)
  - icon:
      light: 'icons/graph.svg'
      dark: 'icons/graph.svg'
    title: Temporal Manipulation
    details: Timing functions allow precise control over modulating the application of manipulation layers, as well as temporal shuffling of video frames 
---

