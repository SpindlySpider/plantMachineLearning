import tensorflow as tf
import keras
import numpy
import matplotlib.pyplot as plt

class data_initliser():
    def __init__(self,path:str,vaildation_split:float = 0.2,seed:int = numpy.random.randint(1,10000)):
        # note we handle image colour standardization in the actual model not here
        self.img_path = path
        self.seed = seed
        self.vaildation_split = vaildation_split
    
    def set_propaties(self,batch_size:int, width:int,height:int):
        self.batch_size = batch_size
        self.width = width
        self.height = height
    
    def train_data_load(self) -> list:
        train_data = keras.utils.image_dataset_from_directory(
            self.img_path,
            validation_split=self.vaildation_split,
            subset="training",
            seed=self.seed,
            batch_size= self.batch_size,
            image_size=(self.height,self.width))
        self.train_data = train_data
        
    def validation_data_load(self) -> list:
        #keras loads images from directory for validation
        validation_data = keras.utils.image_dataset_from_directory(
            self.img_path,
            validation_split=self.vaildation_split,
            subset="validation",
            seed=self.seed,
            batch_size= self.batch_size,
            image_size=(self.height,self.width))
        self.validation_data = validation_data
        
    def get_class_names(self):
        self.class_names = self.train_data.class_names
        return self.class_names
    
    def display_classes_training(self,width:int = 10,height:int=10):
        #makes a 4x4 grid
        #TODO make this dynamic
        # num_of_classes = len(self.get_class_names())
        plt.figure(figsize=(width,height))
        for images, labels in self.train_data.take(1):
            for i in range(16):
                ax = plt.subplot(4,4 ,i+ 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.get_class_names()[labels[i]])
                plt.axis("off")
        plt.show()
        
    def configure_performance(self):
        #this uses caching functions to keep dataset in memory during first epoch
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_data = self.train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) # might need to cache in a file
        self.validation_data = self.validation_data.cache().prefetch(buffer_size=AUTOTUNE)
        
    def set_data(self,img_path):
        self.img_path = img_path
        self.train_data_load()
        self.validation_data_load()
        
    def get_data(self):
        return [self.train_data,self.validation_data]
        pass
    

# path = "data"
# # path = "..\data"
# image_loader = data_initliser(32,180,180,path,seed=78987)
# image_loader.train_data_load()
# image_loader.validation_data_load()
    
# class_names = image_loader.get_class_names()
# print(class_names)
# image_loader.display_classes_training(10,10)