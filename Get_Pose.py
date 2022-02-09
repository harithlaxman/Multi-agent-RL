import matplotlib
import matplotlib.pyplot as plt
import numpy as np

show_animation = True

class Bot:
    def __init__(self):
        # simulation parameters
        self.runtime = 20
        self.kp = 1
        self.dt = 0.01
        self.x_bound = 20
        self.y_bound = 20

        self.reset()

    def reset(self):
        self.time = 0.0

        #   Reset Initial Position of Bot
        x_init = np.random.random()*20
        y_init = np.random.random()*20
        theta_init = np.random.uniform(0, 2*np.pi)
        init_pose = np.array([x_init, y_init, theta_init])
        self.pose = init_pose

        #    Reset Target Position
        self.target_pos = np.random.rand(1, 2)[0]*20

        #    Reset Obstacle Positions
        NUM_OBS = 4
        obs_pos = np.random.rand(2, NUM_OBS)*20
        self.x_obs = obs_pos[0]
        self.y_obs = obs_pos[1]

        self.done = False

    def move_to_pose(self, action):
    """
    rho is the distance between the robot and the goal position
    alpha is the angle to the goal relative to the heading of the robot
    beta is the angle between the robot's position and the goal position plus the goal angle
    Kp_rho*rho and Kp_alpha*alpha drive the robot along a line towards the goal
    Kp_beta*beta rotates the line so that it is parallel to the goal angle
    """
        x = self.pose[0]
        y = self.pose[1]
        theta = self.pose[2]

        x_goal = self.target_pos[0]
        y_goal = self.target_pos[1]

        x_diff = x_goal - x
        y_diff = y_goal - y
        rho = np.hypot(x_diff, y_diff)

        v = self.kp

        theta = theta + (action[0] * self.dt)
        x = x + (v * np.cos(theta) * self.dt)
        y = y + (v * np.sin(theta) * self.dt)

        x_o = abs(x - self.x_obs)
        y_o = abs(y - self.y_obs)
        d_obs = np.hypot(x_o, y_o)

        for i in range(len(d_obs)):
            if d_obs[i] < 0.35:
                self.done = True
                return self.done

        self.time += self.dt

        if x > self.x_bound:
            x = self.x_bound
            theta = -theta
        elif x < 0:
            x = 0
            theta = -theta

        if y > self.y_bound:
            y = self.y_bound
            theta = np.pi - theta
        elif y < 0:
            y = 0
            theta = np.pi - theta

        self.pose = np.array([x, y, theta])

        if self.time > self.runtime or rho < 0.35:
            self.done = True

        return self.done

    def plot_trajectory(self, x, y, theta, x_traj, y_traj):
        #   Clear
        plt.cla()
        plt.xlim(0, self.x_bound+1)
        plt.ylim(0, self.y_bound+1)
        #   Plot Agent
        agent_marker, scale = arrow_marker(theta)
        markersize = 25
        plt.scatter(agent.x, agent.y, marker=agent_marker, s=(markersize*scale)**2)
        #   Plot Obstacles
        plt.scatter(self.x_obs, self.y_obs, s=100, c='green', edgecolor='black', linewidth=1, alpha=0.75)
        #   Plot Target
        plt.scatter(self.target_pos[0], self.target_pos[1], c='yellow', edgecolor='black', s=100, linewidth=1, alpha=0.75)
        #   Plot Trajectory
        plt.plot(x_traj, y_traj, 'b--')

        plt.pause(self.dt)

    def arrow_marker(theta):
        arr = np.array([[0.1, 0.3], [0.1, -0.3], [1, 0]])
        rotation_matrix = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
            ])
        arr = np.matmul(arr, rotation_matrix)
        x0 = np.amin(arr[:,0])
        x1 = np.amax(arr[:,0])
        y0 = np.amin(arr[:,1])
        y1 = np.amax(arr[:,1])
        scale = np.amax(np.abs([x0, x1, y0, y1]))
        arrow_head_marker = matplotlib.path.Path(arr)
        return arrow_head_marker, scale
