import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np
import sys
sys.path.append("..\\")
from delega import AgentBase


class Agent(AgentBase):
    def __init__(self, n_inputs, n_outputs):
        self.q_eval = keras.Sequential([
            keras.Input(shape=n_inputs),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(n_outputs, activation=None)
        ])

    def compile_model(self, lr):
        self.q_eval.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    
