import imghdr
import os
from PIL import Image

# this file is used to clean up image data, removing bad images (currupted/inaccessable) from downloaded data online


def image_cleanup(data_dir, width, height):
    vaild_extention = ["jpeg", "jpg", "png"]
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        for image in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image)
            try:
                attempt_image_open = Image.open(image_path)  # attempts to open file
                attempt_image_open.verify()
                attempt_image_open = Image.open(image_path)# have to reopen after using verify
                extention = imghdr.what(image_path)  # retrives file type
                if extention not in vaild_extention:
                    print(
                        f"file : {image} is not of vaild file type ({vaild_extention})"
                    )
                    os.remove(image_path)
                    attempt_image_open.close()
                if attempt_image_open.size != (width,height):
                    resize_image = attempt_image_open.resize((width,height),Image.LANCZOS)
                    resize_image.save(image_path,extention)
                attempt_image_open.close()
            except Exception as e:
                print(f"exception: {e} \n raised on image: {image} in {folder_path}")
                os.remove(image_path)
