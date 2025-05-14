---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "PyFAME"
  tagline: The Python Facial Analysis and Manipulation Environment
  image:
    src: ./pyfame_logo.png
    alt: PyFAME Logo
  actions:
    - theme: brand
      text: What Is PyFAME?
      link: /guide/getting_started
    - theme: alt
      text: API Reference
      link: /reference/overview
    - theme: alt
      text: Source Code
      link: 'https://github.com/Gavin-Bosman/PyFAME'

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

