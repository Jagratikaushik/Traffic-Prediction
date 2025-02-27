# model.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from keras import layers
from keras import losses
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import load_model


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)
        self._optimizer = None # Initialize optimizer to None

    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        #Note that the model is not compiled here 
        return model

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)

    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states)

    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)

    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)

    def load_model(self, path):
        """
        Load a saved model from the specified path
        """
        model_path = os.path.join(path, 'trained_model.h5')
        if os.path.isfile(model_path):
            self._model = load_model(model_path)
            self._optimizer = Adam(learning_rate=self._learning_rate) # Create the optimizer after loading model
            self._model.compile(loss=losses.mean_squared_error, optimizer=self._optimizer) # Compile model after loading it
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"No model found at {model_path}. Using the initial model.")
            self._optimizer = Adam(learning_rate=self._learning_rate) # Creates the optimizer even if no saved model was found
            self._model.compile(loss=losses.mean_squared_error, optimizer=self._optimizer) # Compile model even if no saved model was found

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def batch_size(self):
        return self._batch_size
