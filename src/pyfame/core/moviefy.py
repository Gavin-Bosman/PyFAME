from pyfame.core.exceptions import *
from pyfame.utils.predefined_constants import *
import mediapipe as mp
import numpy as np
import cv2 as cv
import os

def normalize_image_sizes(input_dir:str, method:int = NORMALIZE_IMAGES_CROP, pad_color:tuple[int] = (255,255,255)) -> None:
    single_file = False
    # add error raise for single file

    # Creating a list of file path strings to iterate through when processing
    files_to_process = []
    files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]

    min_size = None
    max_size = None

    for file in files_to_process:
        frame = cv.imread(file)
        if frame is None:
            raise FileReadError()
        
        if min_size is None:
            h, w, ch = frame.shape
            min_size = [h, w, ch]
        
        if max_size is None:
            h, w, ch = frame.shape
            max_size = [h, w, ch]
        
        cur_size = [frame.shape[0], frame.shape[1], frame.shape[2]]
        if cur_size[0] < min_size[0]:
            min_size[0] = cur_size[0]
        if cur_size[1] < min_size[1]:
            min_size[1] = cur_size[1]
        if cur_size[0] > max_size[0]: 
            max_size[0] = cur_size[0]
        if cur_size[1] > max_size[1]:
            max_size[1] = cur_size[1]
    # resizing the images
    if method == NORMALIZE_IMAGES_CROP:
        for file in files_to_process:
            img = cv.imread(file)
            if img is None:
                raise FileReadError()
            
            h, w, ch = img.shape
            h2, w2, ch2 = min_size
            diff_h = abs(h-h2)
            diff_w = abs(w-w2)

            crop_margin_h = diff_h/2
            crop_margin_w = diff_w/2
            margin_l = 0
            margin_r = 0
            margin_t = 0
            margin_b = 0

            if isinstance(crop_margin_h, float):
                margin_t = int(np.ceil(crop_margin_h))
                margin_b = int(np.floor(crop_margin_h))
            else:
                margin_t = crop_margin_h
                margin_b = crop_margin_h

            if isinstance(crop_margin_w, float):
                margin_l = int(np.ceil(crop_margin_w))
                margin_r = int(np.floor(crop_margin_w))
            else:
                margin_l = crop_margin_w
                margin_r = crop_margin_w

            if diff_h > 0:
                img = img[margin_t:(h-margin_b), :]
            if diff_w > 0:    
                img = img[:, margin_l:(w-margin_r)]
                
            # writing output image
            success = cv.imwrite(file, img)
            if not success:
                raise FileWriteError()
    else:
        for file in files_to_process:
            img = cv.imread(file)
            if img is None:
                raise FileReadError()
            
            h, w, ch = img.shape
            h2, w2, ch2 = max_size
            diff_h = abs(h-h2)
            diff_w = abs(w-w2)
            
            pad_margin_h = diff_h/2
            pad_margin_w = diff_w/2
            margin_l = 0
            margin_r = 0
            margin_t = 0
            margin_b = 0

            if isinstance(pad_margin_h, float):
                margin_t = int(np.ceil(pad_margin_h))
                margin_b = int(np.floor(pad_margin_h))
            else:
                margin_t = pad_margin_h
                margin_b = pad_margin_h

            if isinstance(pad_margin_w, float):
                margin_l = int(np.ceil(pad_margin_w))
                margin_r = int(np.floor(pad_margin_w))
            else:
                margin_l = pad_margin_w
                margin_r = pad_margin_w

            if diff_h > 0:
                bordered = cv.copyMakeBorder(img, margin_t, margin_b, margin_t, margin_b, cv.BORDER_CONSTANT, value=pad_color)
                bordered = bordered.astype(np.uint8)
                h = h + margin_t + margin_b
                img = bordered[:, margin_t:w + margin_t]
            if diff_w > 0:
                bordered = cv.copyMakeBorder(img, margin_l, margin_r, margin_l, margin_r, cv.BORDER_CONSTANT, value=pad_color)
                bordered = bordered.astype(np.uint8)
                img = bordered[margin_l:h + margin_l, :]

            # writing output image
            success = cv.imwrite(file, img)
            if not success:
                raise FileWriteError()


def moviefy_images(input_dir:str, output_dir:str, output_filename:str, fps:int = 30, repeat_duration:int = 1000, 
            blend_images:bool = False, with_sub_dirs:bool = False) -> None:
    """ Takes a series of static images contained in input_dir, and converts them into a video sequence by repeating
    and interpolating frames. Output "movie" files will be written to output_dir.
    """
    single_file = False

    # Creating a list of file path strings to iterate through when processing
    files_to_process = []

    if single_file:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    #logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Moviefied"):
        os.mkdir(output_dir + "\\Moviefied")
        output_dir = output_dir + "\\Moviefied"
        #logger.info(f"Function created new output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\Moviefied"
    
    image_list = []
    im_size = None

    for file in files_to_process:
        frame = cv.imread(file)
        if frame is None:
            raise FileReadError()
        
        image_list.append(frame)

        if im_size is None:
            im_size = (frame.shape[0], frame.shape[1])
    
    # Before proceeding, need to check that all input images have the same size
    im_shape = None
    for image in image_list:
        if im_shape is None:
            im_shape = image.shape
        else:
            if im_shape != image.shape:
                raise ImageShapeError()
    
    # Instantiating our videowriter
    writer = cv.VideoWriter(output_dir + f"\\{output_filename}.mp4", cv.VideoWriter.fourcc(*"mp4v"), fps, im_size)

    if not writer.isOpened():
        raise FileWriteError()
    
    num_repeats = int(np.floor((repeat_duration/1000)*fps))
    blend_window = round(num_repeats * 0.2)
    blend_inc = 1/blend_window
    
    for i in range(len(image_list)):
        cur_inc = 0.0
        for j in range(num_repeats):
            if blend_images == True:
                if j > (num_repeats-blend_window) and (i < (len(image_list) - 1)):
                    # begin blending images
                    cur_inc += blend_inc
                    alpha = cur_inc
                    beta = (1-alpha)
                    blended_img = cv.addWeighted(image_list[i], alpha, image_list[i+1], beta, 0.0)
                    blended_img.astype(np.uint8)
                    writer.write(blended_img)
                else:
                    writer.write(image_list[i])
            else:
                writer.write(image_list[i])
    
    writer.release()