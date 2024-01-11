from DataInitliser import data_initliser
from model import image_classifer
import imageClassification
import tensorflow as tf

def enable_gpu():
    #limmit memory use of the gpu, to prevent a out of memory error, GPU run out of VRAM
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Num GPUs Available: ", len(gpus))
    #cant be used on windows native as support ended in 2.10
    #unless you use wsl2, and pass it through a docker container
    pass

def clean_images(data_dir):
    """clean up images which maybe corrupted"""
    imageClassification.image_cleanup(data_dir)

def train_model(data_dir,model_dir,filename):
    """train model and save result"""
    path = "data" # for docker container as cwd is app/
    height = 250
    width = 250
    preprocesssing = data_initliser()
    preprocesssing.set_proprties(32,height,width,path)
    preprocesssing.train_data_load()
    preprocesssing.validation_data_load()
    data = preprocesssing.get_data()
    
    model = image_classifer(height,width)
    model.set_data(data[0],data[1])
    model.train_model()
    model.save_model("models",filename)
    
def predict_image(model_dir,filename,image_path,width,height):
    """predicts the type of flower, only supports one image"""
    model = image_classifer()
    model.load_model(model_dir,filename)
    img = data_initliser().get_processsed_img(image_path,width,height)
    prediction = model.predict_model(img)
    if type(prediction) == str:
        model.display_prediction(img,prediction)
    print(prediction)

    