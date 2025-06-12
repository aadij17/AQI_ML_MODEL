import cv2
import numpy as np
import os
from pathlib import Path
import glob

#Prints all the image paths which will be used to resize the data

# get the path/directory
folder_dir = 'D:\Sayyam AQI Model\Dataset\TEST\POOR'
 
# iterate over files in that directory
image_paths = []
for images in glob.iglob(f'{folder_dir}/*'):
    
    # To consider all the type of images
    if (images.endswith(".png") or images.endswith(".jpg")
        or images.endswith(".jpeg")):
        image_paths.append(images)

# Insert all the image paths from the list created using the above method.
print(image_paths)
for path in image_paths:
    print(path)
    image = cv2.imread(path, 1)

    # Loading the image
    xshape = image.shape[1]
    yshape = image.shape[0]
    half = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
    bigger = cv2.resize(image, (xshape, yshape))

    stretch_near = cv2.resize(image, (512, 512),
                interpolation = cv2.INTER_LINEAR)

    image_name = os.path.basename(path).split('/')[-1]
    newdir = "D:\Sayyam AQI Model\Dataset\TEST\POOR"
    os.chdir(newdir)
    updated_image = cv2.imwrite(image_name, stretch_near)
