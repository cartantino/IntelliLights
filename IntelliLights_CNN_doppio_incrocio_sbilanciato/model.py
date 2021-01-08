import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras import models


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)
 


    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        model = models.Sequential()
        model.add(layers.Conv2D(16, (2,2), activation='relu', strides=(1, 2), input_shape=(16, 133, 3)))
        model.add(layers.Conv2D(32, (2,2), activation='relu', strides=(1, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self._output_dim, activation='linear'))

        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        return model

    """
    Predict one gives in output a vector containing the Q-Values for each action of the current episode.
    """
    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        # print("Predict one, shape in input = " + str(state.shape))
        # state = np.reshape(state, [1, self._input_dim])
        state = state.reshape(1, 16, 133, 3)
        return self._model.predict(state)


    """
    Predict batch gives in output a matrix containing the Q-Values for each episode within the batch given the action.
    """
    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        states = states.reshape(self.batch_size, 16, 133, 3)
        return self._model.predict(states)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model_1(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        print(os.path.join(path, 'trained_model_1.h5'))
        self._model.save(os.path.join(path, 'trained_model_1.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure_1.png'), show_shapes=True, show_layer_names=True)


    def save_model_2(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model_2.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure_2.png'), show_shapes=True, show_layer_names=True)


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path, tl):
        self._input_dim = input_dim
        self._tl = tl
        if tl == 1:
            self._model = self._load_my_model_1(model_path)
        if tl == 2:
            self._model = self._load_my_model_2(model_path)


        


    def _load_my_model_1(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model_1.h5')

        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model 1 number not found")

    def _load_my_model_2(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model_2.h5')

        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model 2 number not found")


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        # state = np.reshape(state, [1, self._input_dim])
        state = state.reshape(1, 16, 133, 3)
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim