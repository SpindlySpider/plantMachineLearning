# this is entry point for the machine learning
from utlities import *

data_dir = "data" # for docker container as cwd is app/
model_dir = "models" # for docker container as cwd is app/
filename= "test1"
docker_volume = "docker_data"
height = 250
width = 250
epochs = 15
image_path = "imageClassification/flower.jpg"

enable_gpu()
clean_images(data_dir,height,width)
train_model(data_dir,docker_volume,filename,epochs)
predict_image(docker_volume,filename,image_path,width,height)




#keeps docker container alive
# while(True):
#     pass