# this is entry point for the machine learning 
import tensorflow as tf
from  DataInitliser import data_initliser
from model import image_classifer

#limmit memory use of the gpu, to prevent a out of memory error, GPU run out of VRAM
# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# print("Num GPUs Available: ", len(gpus))
#cant be used on windows as support ended in 2.10

tf.config.threading.set_inter_op_parallelism_threads(112)


path = "..\data"
height = 250
width = 250

# preprocesssing = data_initliser(32,height,width,path)
# preprocesssing.train_data_load()
# preprocesssing.validation_data_load()

# num_of_classes = len(preprocesssing.get_class_names())
# data = preprocesssing.get_data()
 
# model = image_classifer(height,width,num_of_classes)
# model.set_data(data[0],data[1])

# model.train_model()

# model.visulize_results()
# model.save_model("..\models","plantmodel.h5")
model = image_classifer().load_model("..\models","plantmodel.h5")
model.display_classes_training()