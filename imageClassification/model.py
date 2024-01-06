from keras import *
import matplotlib.pyplot as plt
import os
import numpy
from PIL import Image
class image_classifer():
    
    def set_architecture(self,height,width,num_of_classes):
        num_classes = num_of_classes
        data_augmentation = Sequential([
            layers.RandomFlip("horizontal",
                              input_shape=(height,width,3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1)])
        # this is the architecture of the mode, change here if you want to implement YOLO
        self.model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255,input_shape=(height,width,3)), # adapts colour depth to be between 0 -> 1 
        layers.Conv2D(16,3,padding="same",activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32,3,padding="same",activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,padding="same",activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.2), # reduces over fitting by setting a percetnage of outputs to 0 
        layers.Flatten(),
        layers.Dense(128,activation="relu"),
        layers.Dense(num_classes)
        ])
        self.model.compile(optimizer="adam",
                           loss=losses.SparseCategoricalCrossentropy(from_logits=True), 
                           #^ the loss function, from logit says that the model is not normalized
                           metrics=["accuracy"]) #we are watching the data metric
        
    def set_data(self,training_data,vailidation_data):
        self.training_data = training_data
        self.vailidation_data = vailidation_data
        
    def view_layers(self):
        self.model.summary()
        
    def train_model(self,epochs =10):
        self.epochs= epochs
        self.history = self.model.fit(
            self.training_data,
            validation_data=self.vailidation_data,
            epochs=epochs
        )
    
    def display_prediction(self,img:numpy.ndarray,label):
        #this only works with one image 
        image = numpy.squeeze(img)
        img = Image.fromarray(image)
        plt.imshow(img)
        plt.title(label)
        plt.show()
        pass
    def visulize_results(self):
        accuracy = self.history.history["accuracy"]
        validation_accuracy = self.history.history["val_accuracy"]
        
        loss = self.history.history["loss"]
        validation_loss = self.history.history["val_loss"]
        
        epochs_range = range(self.epochs)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
         
        plt.plot(epochs_range, accuracy, label='Training Accuracy')
        plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, validation_accuracy, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
    def save_model(self,path,filename):
        path_filename = os.path.join(path,filename)        
        self.model.save(path_filename)   
    def load_model(self,path,filename):
        path_filename = os.path.join(path,filename)       
        self.model = models.load_model(path_filename)
    def predict_model(self,img, class_names:list = None)-> str|list:
        #img is of type numpy.ndarray, and only works for one image at the moment
        results = self.model.predict(img)
        first_img = results[0] # must change this so it supports multiple images
        if class_names != None:
            index = 0
            guess = max(first_img)
            for result in first_img:
                if guess == result:
                    prediction_label =class_names[index]
                    return prediction_label
                
                else:
                    index +=1
            return results  
        return results        


