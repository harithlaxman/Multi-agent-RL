import numpy as np
import matplotlib.pyplot as plt
# from physics_sim import PhysicsSim
from Get_Pose import Bot


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, runtime=20., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = Bot(init_pose, target_pos, runtime)
        self.state_size = 5
        self.action_size = 2

        #self.action_low = -1
        #self.action_high = 1
        #self.action_range = self.action_high - self.action_low

        self.x = np.array([3, 5, 8.5, 11, 13])
        self.y = np.array([3, 5, 8.5, 11, 13])

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([1.0, 1.0])
        self.diff = 0.0
        self.d = 0

        self.estep = -0.4

    def get_reward(self, reward, pose):
        """Uses current pose of sim to return reward."""
        bonus = 0
        penalty_obs = 0

        # calculate the coordinate distance from the target position
        dist_x = self.target_pos[0] - pose[0]
        dist_y = self.target_pos[1] - pose[1]

        # bonus
        self.diff = np.hypot(dist_x, dist_y)
        if self.diff < 0.35:
            bonus = 100

        # obstacle penalty
        x_obs = abs(self.sim.pose[0] - self.x)
        y_obs = abs(self.sim.pose[1] - self.y)
        d_ob = np.hypot(x_obs, y_obs)

        for i in range(len(d_ob)):
            if d_ob[i] < 0.35:
                penalty_obs = -10

        # calculate reward
        reward += self.estep + penalty_obs + bonus - (1.5 * (self.diff - self.d))
        self.d = self.diff
        return reward

    def step(self, state, action):
        """Uses action to obtain next state, reward, done."""
        reward = 1e-4
        done = self.sim.move_to_pose(action, self.x, self.y)  # update the sim pose
        next_state = self.sim.pose
        reward = self.get_reward(reward, state)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = self.sim.pose
        return state
