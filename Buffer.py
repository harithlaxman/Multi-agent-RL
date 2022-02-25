import numpy as np 

class Buffer:
    def __init__(self, buffer_file, buffer_size, buffer_capacity, state_size, action_size):
        self.buffer_size = buffer_size
        self.buffer_capacity = buffer_capacity
        self.buffer_counter = 0
        self.state_size = state_size
        self.action_size = action_size

        if buffer_file is not None:
            f = np.load(buffer_file)
            self.state_buffer = f['state_buffer']
            self.next_state_buffer = f['next_state_buffer']
            self.action_buffer = f['action_buffer']
            self.reward_buffer = f['reward_buffer']
        else:
            self.state_buffer = np.zeros((self.buffer_capacity, self.state_size))
            self.next_state_buffer = np.zeros((self.buffer_capacity, self.state_size))
            self.action_buffer = np.zeros((self.buffer_capacity, self.action_size))
            self.reward_buffer = np.zeros((self.buffer_capacity, 1))

    def record(self, obs):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs[0]
        self.action_buffer[index] = obs[1]
        self.next_state_buffer[index] = obs[2]
        self.reward_buffer[index] = obs[3]

        self.buffer_counter += 1

    def save(self, filename):
        np.savez(filename, state_buffer = self.state_buffer,
        next_state_buffer = self.next_state_buffer,
        action_buffer = self.action_buffer,
        reward_buffer = self.reward_buffer)