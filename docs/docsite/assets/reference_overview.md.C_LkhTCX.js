import{_ as o,c as i,o as t,ae as a}from"./chunks/framework.C5RTWoTo.js";const n="/PyFAME/pyfame_logo.png",f=JSON.parse('{"title":"Api Overview","description":"","frontmatter":{"layout":"doc","title":"Api Overview","prev":false,"next":{"text":"Occlusion","link":"/reference/occlusion"}},"headers":[],"relativePath":"reference/overview.md","filePath":"reference/overview.md"}'),s={name:"reference/overview.md"};function l(r,e,c,d,p,u){return t(),i("div",null,e[0]||(e[0]=[a('<h1 id="overview" tabindex="-1">Overview <a class="header-anchor" href="#overview" aria-label="Permalink to &quot;Overview&quot;">​</a></h1><p><img src="'+n+'" width="400px" style="float:center;"></p><p>PyFAME: The Python Facial Analysis and Manipulation Environment is a Python toolkit for performing a variety of classical facial psychology manipulations over both still images and videos. All of PyFAME&#39;s manipulation functions can be layered, and they are designed in such a way that users can apply several manipulations in succession easily. Thus, PyFAME can be used to perform individual facial psychology experiments, or to create novel facial psychology stimuli which themselves can be used in experiments or as inputs to train neural networks.</p><p>The PyFAME package is divided into two main submodules <code>core</code> and <code>utils</code>. Simply importing pyfame will expose both submodules and the entirety of the package&#39;s contents. However, most users will be focused on using the core functionalities. So, generally PyFAME will be imported as follows</p><div class="language-python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pyfame.core </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">as</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pf</span></span></code></pre></div><p>The <code>core</code> submodule contains several submodules itself, breaking up the package into specific functional groups, which is detailed below.</p><h2 id="module_analysis" tabindex="-1">Analysis <a class="header-anchor" href="#module_analysis" aria-label="Permalink to &quot;Analysis {#module_analysis}&quot;">​</a></h2><p>This module provides several functions for extracting color and motion information from both image and video files.</p><p><code>extract_face_color_means()</code> allows users to specify a sampling region of the face, a sampling frequency, and a color space to sample in. Only a single sample is performed over still images, while videos are sampled at periodic timestamps determined by the sampling frequency. Extracted color channel information is written out into a CSV file.</p><p><code>get_optical_flow()</code> provides users the ability to perform both sparse (Lucas-Kanadae&#39;s algorithm) and dense (Farneback&#39;s algorithm) optical flow. This function provides two output files for each file input; a video file visualizing the optical flow, and a csv file containing periodic samples (determined by sampling frequency parameter) of the optical flow vectors.</p><h2 id="module_coloring" tabindex="-1">Coloring <a class="header-anchor" href="#module_coloring" aria-label="Permalink to &quot;Coloring {#module_coloring}&quot;">​</a></h2><p>This module contains a variety of functions for manipulating color in the facial region over still images and video files.</p><p><code>face_color_shift()</code> provides users the ability to manipulate and shift various color channels over the face region. This function operates in the BGR color space, and can manipulate the colors red, green, blue and yellow over the facial region. These color manipulations can also be modulated dynamically using a timing function. PyFAME contains several predefined timing functions, namely <code>linear()</code>, <code>gaussian()</code>, <code>sigmoid()</code> and <code>constant()</code>, but also allows users to define their own timing functions (at the risk of unexpected results).</p><p><code>face_saturation_shift()</code> and <code>face_brightness_shift()</code> operate in a near-identical manner to <code>face_color_shift()</code>. They manipulate image or frame saturation (using the HSV color space) and brightness respectively. Additionally, both functions can be modulated with PyFAME&#39;s predefined timing functions, as seen with <code>face_color_shift()</code> above.</p><h2 id="module_occlusion" tabindex="-1">Occlusion <a class="header-anchor" href="#module_occlusion" aria-label="Permalink to &quot;Occlusion {#module_occlusion}&quot;">​</a></h2><p>This module contains several functions associated with occluding, obstructing or cutting out regions of the face. Again, all of these functions operate over still images and videos.</p><p><code>mask_face_region()</code> provides users the ability to dynamically mask individual, or several facial regions at once. For specific use cases (i.e. green/blue screen) a background_color parameter is provided, with the default being white. Much of <code>mask_face_region()</code>&#39;s functionality is utilised in almost every <code>Core</code> function, in order to allow dynamic regional application of the various manipulations.</p><p><code>occlude_face_region()</code> provides users access to several classical facial occlusion methods, including bubble occlusion, bar occlusion, and dynamic regional occlusion with custom fill colors. All occlusion types are set up to positionally lock onto, and track the face, making it simple to dynamically occlude the face in video files.</p><p><code>blur_face_region()</code> allows users to apply classical image and video blurring methods restricted over the facial region. This function can perform gaussian blurring, average blurring, and median blurring.</p><p><code>apply_noise()</code> functions similarly to <code>blur_face_region()</code>, but it encorporates more general noise methods and does not restrict the noise only to the facial region. This function can pixelate the face, apply gaussian noise, as well as salt and pepper noise. All of the noise methods are highly customizable, with input parameters such as <code>noise_prob</code>, <code>mean</code> and <code>standard_dev</code>.</p><h2 id="module_pld" tabindex="-1">Point_light_display <a class="header-anchor" href="#module_pld" aria-label="Permalink to &quot;Point_light_display {#module_pld}&quot;">​</a></h2><p>This module contains only one function, namely <code>generate_point_light_display()</code>. A point-light-display is a motion-perception paradigm that allows researchers to study how the brain perceives and interprets biological motion. This function focusses on the underlying face, overlaying pertinent landmarks with point-lights and tracking their position/velocity. Classically, point-light-displays have been created using motion capture software, which is both costly and requires physical labour. Alternatively, <code>generate_point_light_display()</code> is able to take any video containing a face, and overlay up to 468 unique points to generate a dynamic point-light-display.</p><p>One novel ability of <code>generate_point_light_display()</code> is it&#39;s ability to display historical displacement vectors. The function allows users to specify the history time window, as well as several methods of displaying the history vectors (relative positional history, relative to origin history).</p><h2 id="module_scrambling" tabindex="-1">Scrambling <a class="header-anchor" href="#module_scrambling" aria-label="Permalink to &quot;Scrambling {#module_scrambling}&quot;">​</a></h2><p>Again, this module only contains one function, namely the <code>facial_scramble()</code> function. However, this function is multimodal and leverages several distinct methods of shuffling/scrambling the facial features. The two main scrambling methods provided are <code>landmark_scramble</code> and <code>grid_scramble</code>. These methods shuffle the facial features by masking out specified landmarks and randomizing their positions, and breaking up the face into a grid then repositioning the grid-squares respectively.</p><h2 id="module_tt" tabindex="-1">Temporal_transforms <a class="header-anchor" href="#module_tt" aria-label="Permalink to &quot;Temporal_transforms {#module_tt}&quot;">​</a></h2><p>This module contains two related functions <code>generate_shuffled_block_array()</code> and <code>shuffle_frame_order()</code>. <code>shuffle_frame_order()</code> provides a variety of methods (i.e. palindrome, random sampling, cyclic shift) to temporally shift and restructure input video files. It performs the shuffling by breaking up the video frames into &#39;blocks&#39; of frames, for which the time duration is specified by the user. <code>generate_shuffled_block_array()</code> is a helper function that returns a specific tuple which can be fed directly as input into <code>shuffle_frame_order()</code>. Depending on input parameters, <code>generate_shuffled_block_array()</code> returns a tuple containing the <code>block_order</code> array, the <code>block_size</code> and <code>block_duration</code>.</p><p>The <code>utils</code> submodule also contains several submodules, each providing various utility functions, predefined constants, and any extra features not directly relevant to the core funtionality of the package.</p><table tabindex="0"><thead><tr><th style="text-align:left;">Submodule Name</th><th style="text-align:left;">Description</th></tr></thead><tbody><tr><td style="text-align:left;">Display_options</td><td style="text-align:left;">A group of functions that display parameter options to the terminal, i.e. <code>display_mask_type_options()</code>.</td></tr><tr><td style="text-align:left;">Landmarks</td><td style="text-align:left;">Predefined landmark regions for use with all of the core functions.</td></tr><tr><td style="text-align:left;">Predefined_constants</td><td style="text-align:left;">Evident from this submodules name, it contains a large set of predefined parameter values for use with all of the core functions.</td></tr><tr><td style="text-align:left;">Setup_logging</td><td style="text-align:left;">Provides access to a function <code>setup_logging()</code> which allows users to provide a custom logging config.yml if they want to define custom logging behaviour.</td></tr><tr><td style="text-align:left;">Timing_functions</td><td style="text-align:left;">A set of predefined timing functions, namely <code>constant()</code>, <code>linear()</code>, <code>sigmoid()</code> and <code>gaussian()</code>.</td></tr><tr><td style="text-align:left;">Utils</td><td style="text-align:left;">Any extra utilities and mathematical operations not part of the core functionality.</td></tr></tbody></table>',29)]))}const m=o(s,[["render",l]]);export{f as __pageData,m as default};
