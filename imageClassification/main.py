# this is entry point for the machine learning
from utlities import *

data_dir = "data" # for docker container as cwd is app/
model_dir = "models" # for docker container as cwd is app/
height = 250
width = 250

image_path = "imageClassification/flower.jpg"

enable_gpu()
clean_images()
train_model()
predict_image()

#keeps docker container alive
while(True):
    pass