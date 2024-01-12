# this is entry point for the machine learning
from utlities import *

data_dir = "data" # for docker container as cwd is app/
model_dir = "models" # for docker container as cwd is app/
filename= "test1"
docker_volume = "docker_data"
height = 250
width = 250

image_path = "imageClassification/flower.jpg"

# utlities.enable_gpu()
# utlities.clean_images(data_dir)
# utlities.train_model(data_dir,model_dir,filename)
# utlities.predict_image(model_dir,filename,image_path,width,height)
enable_gpu()
clean_images(data_dir)
train_model(data_dir,docker_volume,filename)
predict_image(docker_volume,filename,image_path,width,height)




#keeps docker container alive
while(True):
    pass