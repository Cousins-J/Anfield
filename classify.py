from IPython.display import Image, display
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from keras.preprocessing.image import load_img
from tensorflow import keras
from keras.models import load_model

model = load_model('/home/justin/MLflask/themodel.h5')


def classify(image_path):
  display(Image(image_path))
  image_data = tf.io.gfile.GFile(image_path, 'rb').read() # get this image file
  #image_data = image_data.reshape((147, 147))
  pred = model.predict(image_data)
  model.print_scores(pred=pred, k=10, only_first_name=True)