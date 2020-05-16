from keras.models import Model
from keras.layers import Input
from numpy import array
from numpy import loadtxt
from keras.models import model_from_json
import numpy as np
import tensorflow as tf

#Reading the model from JSON file
with open('keras/KERAS_gru_model.json', 'r') as json_file:
    json_savedModel= json_file.read()
#load the model architecture 
model_j = tf.keras.models.model_from_json(json_savedModel)
model_j.summary()

model_j.load_weights('keras/KERAS_gru_model_weights.h5')

input_layer = model_j.input
gru_layer = model_j.get_layer('gru_selu').output
intermediate_layer_model = tf.keras.models.Model(inputs=input_layer, outputs=gru_layer)

X = np.zeros([1, 20, 6])
predictions = (model_j.predict(X))
np.set_printoptions(suppress=True)
intermediate_output = intermediate_layer_model.predict(X)
print("GRU Layer output : ")
print(intermediate_output)
print("\n********************************************************************************\n")
print("Full model output : ")
print(predictions)