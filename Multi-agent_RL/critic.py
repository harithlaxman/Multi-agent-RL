from tensorflow.keras import layers, models, optimizers, initializers
from tensorflow.keras import backend as K


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        #net_states = layers.Dense(units=32, activation='relu')(states)
        #net_states = layers.Dense(units=64, activation='relu')(net_states)
        net_states = layers.Dense(units=512, activation='relu')(states)

        # Add hidden layer(s) for action pathway
        #net_actions = layers.Dense(units=32, activation='relu')(actions)
        #net_actions = layers.Dense(units=64, activation='relu')(net_actions)

        # Combine state and action pathways
        #concat = layers.Concatenate()([net_states, net_actions])
        concat = layers.Concatenate()([net_states, actions])

        #net = layers.Multiply()([net_states, net_actions])
        net = layers.Dense(512, activation="relu")(concat)
        net = layers.Dense(512, activation="relu")(net)

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, activation="linear", name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients)
