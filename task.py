import numpy as np
import matplotlib.pyplot as plt
# from physics_sim import PhysicsSim
from Get_Pose import Bot


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self):
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
        self.sim = Bot()
        self.state_size = 3
        self.action_size = 1

        self.diff = 0.0
        self.d = 0
        self.estep = -0.5

    def get_reward(self, reward, pose):
        """Uses current pose of sim to return reward."""

        # calculate the coordinate distance from the target position
        dist_x = self.sim.target_pos[0] - pose[0]
        dist_y = self.sim.target_pos[1] - pose[1]

        x_obs = abs(self.sim.pose[0] - self.sim.x_obs)
        y_obs = abs(self.sim.pose[1] - self.sim.y_obs)
        d_ob = np.hypot(x_obs, y_obs)

        # bonus
        self.diff = np.hypot(dist_x, dist_y)

        if d_ob.any() < 0.35:
            reward = -40
        elif self.diff < 0.35:
            reward = 20
        else:
            reward = self.estep

        # 1.8*(self.d-self.diff)
        self.d = self.diff

        return reward

    def step(self, state, action):
        """Uses action to obtain next state, reward, done."""
        reward = 1e-4
        done = self.sim.move_to_pose(action)  # update the sim pose
        next_state = self.sim.pose
        reward = self.get_reward(reward, state)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.d = 0
        self.sim.reset()
        state = self.sim.pose
        return state
