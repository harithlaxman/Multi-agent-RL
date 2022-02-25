import numpy as np

class Botenv:
    def __init__(self, T, dt, ENV_SIZE, V):
        self.ENV_SIZE = ENV_SIZE
        self.V = V
        self.dt = dt
        self.T = T
        self.gamma = 1e-3
        self.reset()
    def reset(self):
        #   Agent Position
        self.x = np.random.randint(0, self.ENV_SIZE)
        self.y = np.random.randint(0, self.ENV_SIZE)
        self.theta = np.random.uniform(0, 2*np.pi)
        self.hit_boundary_x = 0
        self.hit_boundary_y = 0
        #   Target Position
        self.target_pos = np.random.rand(1, 2)[0]*self.ENV_SIZE
        """
        #   Obstacle Positions
        NUM_OBS = 3
        self.obs_pos = np.random.rand(2, NUM_OBS)*self.ENV_SIZE
        """
        #   Distance from Target
        self.target_dist = np.hypot(self.x - self.target_pos[0], self.y - self.target_pos[1])
        #   State Space
        self.state = np.array([self.x, self.y, self.theta, self.target_dist])
        self.T = 10
        self.done = False
    def action(self, w):
        self.T -= self.dt
        curr_state = self.state

        self.theta += w * self.dt * 10
        
        if self.theta > 2*np.pi:
            self.theta -= 2*np.pi
        
        if self.x > self.ENV_SIZE or self.x < 0:
            self.hit_boundary_x += 1
        if self.y > self.ENV_SIZE or self.y < 0:
            self.hit_boundary_y += 1
        
        self.vx = ((-1)**(self.hit_boundary_x)) * (self.V * np.cos(self.theta))
        self.vy = ((-1)**(self.hit_boundary_y)) * (self.V * np.sin(self.theta))
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        self.target_dist = np.hypot(self.x - self.target_pos[0], self.y - self.target_pos[1])
        #   Add Code for Obs distance here
        if(self.target_dist <= 0.5):
            self.done = True
            print(f"Agent hit target at timestep: {self.T//self.dt}")
        if(self.T <= 0):
            self.done = True
        #   Update State
        self.state = np.array([self.x, self.y, self.theta, self.target_dist])

        reward = self.get_reward(curr_state)
        return(self.state, reward, self.done)
        
    def get_reward(self, state):
        if state[3] <= 0.35:
            return(1000)
        elif self.T>0:
            reward = 1 - np.exp(0.05 * state[3])
            return(reward)
        else:
            return(-1000)
