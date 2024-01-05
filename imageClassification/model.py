import tensorflow as tf
from keras import *
import matplotlib.pyplot as plt
class image_classifer():
    def __init__(self,height,width,num_of_classes):
        num_classes = num_of_classes
        # this is the architecture of the mode, change here if you want to implement YOLO
        self.model = Sequential([
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
        #
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
    def visulize_results(self):
        accuracy = self.history.history["accuracy"]
        validation_accuracy = self.history.history["val_accuracy"]
        
        loss = self.history.history["loss"]
        validation_loss = self.history.history["val_loss"]
        
        epochs_range = range(self.epochs)
         
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