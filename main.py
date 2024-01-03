# this is entry point for the machine l;earning 
import tensorflow as tf
import os

#limmit memory use of the gpu, to prevent a out of memory error, GPU run out of VRAM
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    