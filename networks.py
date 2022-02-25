import numpy as np 
from tensorflow.keras import initializers, layers, models

def Actor(state_size, action_size):
    init = initializers.random_uniform(minval=-3e-3, maxval=3e-3)
    inputs = layers.Input(shape=(state_size, ))
    dense1 = layers.Dense(units=512, activation="relu")(inputs)
    dense2 = layers.Dense(units=512, activation="relu")(dense1)
    
    action = layers.Dense(units=action_size, activation="tanh", kernel_initializer=init)(dense2)

    model = models.Model(inputs=[inputs], outputs=[action])

    return(model)
def Critic(state_size, action_size):
    state_input = layers.Input(shape=(state_size, ))
    state_layer_1 = layers.Dense(units=32, activation="relu")(state_input)
    state_output = layers.Dense(units=64, activation="relu")(state_layer_1)

    action_input = layers.Input(shape=(action_size, ))
    action_output = layers.Dense(units=32, activation="relu")(action_input)

    concat = layers.Concatenate()([state_output, action_output])

    dense1 = layers.Dense(units=512, activation="relu")(concat)
    dense2 = layers.Dense(units=512, activation="relu")(dense1)

    q_value  = layers.Dense(units=1)(dense2)
    model = models.Model(inputs=[state_input, action_input], outputs=[q_value])
    return(model)

