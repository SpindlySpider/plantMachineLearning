import cv2
import imghdr
import os
# this file is used to clean up image data, removing bad images (currupted/inaccessable)

data_dir = "data" # points image directory 
vaild_extention = ["jpeg","jpg","bmp","png"]

for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir,folder)
    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path,image)
        try:
            attempt_image_open = cv2.imread(image_path) # attempts to open file
            extention = imghdr.what(image_path) # retrives file type
            if extention not in vaild_extention:
                print(f"file : {image} is not of vaild file type ({vaild_extention})")
        except Exception as e:
            print(f"exception: {e} \n raised on image: {image} in {folder_path}")