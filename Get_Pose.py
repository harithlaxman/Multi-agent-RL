"""

Move to specified pose

Author: Daniel Ingram (daniel-s-ingram)
        Atsushi Sakai(@Atsushi_twi)

P. I. Corke, "Robotics, Vision & Control", Springer 2017, ISBN 978-3-319-54413-7

"""

import matplotlib.pyplot as plt
import numpy as np

show_animation = True


class Bot:
    def __init__(self, init_pose=None, target_pos=None, runtime=10.):
        self.init_pose = init_pose
        self.runtime = runtime
        self.target_pos = target_pos

        # simulation parameters
        self.kp = 1
        self.dt = 0.01
        self.x_bound = 20
        self.y_bound = 20

        self.reset()

    def reset(self):
        self.time = 0.0
        self.pose = np.array(([0.0, 0.0, 0.0, 0.0, 0.0])) if self.init_pose is None else np.copy(self.init_pose)
        self.done = False

    def move_to_pose(self, action, x_obs, y_obs):
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

        x_o = abs(x - x_obs)
        y_o = abs(y - y_obs)
        d_obs = np.hypot(x_o, y_o)

        for i in range(len(d_obs)):
            if d_obs[i] < 0.35:
                self.done = True
                return self.done

        self.time += self.dt

        if x > self.x_bound:
            x = self.x_bound
        elif x < 0:
            x = 0

        if y > self.y_bound:
            y = self.y_bound
        elif y < 0:
            y = 0

        self.pose = np.array([x, y, theta])

        if self.time > self.runtime or rho < 0.35:
            self.done = True

        return self.done

    def plot_trajectory(self, x, y, theta, x_traj, y_traj, x_obs, y_obs):
        # pragma: no cover
        # Corners of triangular vehicle when pointing to the right (0 radians)

        plt.cla()

        p1_i = np.array([0.5, 0, 1]).T
        p2_i = np.array([-0.5, 0.25, 1]).T
        p3_i = np.array([-0.5, -0.25, 1]).T
        T = self.transformation_matrix(x, y, theta)
        p1 = np.matmul(T, p1_i)
        p2 = np.matmul(T, p2_i)
        p3 = np.matmul(T, p3_i)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'k-')
        plt.plot([p3[0], p1[0]], [p3[1], p1[1]], 'k-')

        plt.style.use('seaborn')
        plt.scatter(x_obs, y_obs, s=100, c='green', edgecolor='black', linewidth=1, alpha=0.75)
        plt.plot(x_traj, y_traj, 'b--')

        plt.xlim(0, self.x_bound)
        plt.ylim(0, self.y_bound)

        plt.pause(self.dt)

    def transformation_matrix(self, x, y, theta):
        return np.array([[np.cos(theta), -np.sin(theta), x], [np.sin(theta), np.cos(theta), y], [0, 0, 1]])

# def main():

 #   for i in range(5):
  #      x_start = 20 * random()
   #     y_start = 20 * random()
    #    theta_start = 2 * np.pi * random() - np.pi
     #   x_goal = 20 * random()
      #  y_goal = 20 * random()
       # theta_goal = 2 * np.pi * random() - np.pi
        #print("Initial x: %.2f m\nInitial y: %.2f m\nInitial theta: %.2f rad\n" %
         #     (x_start, y_start, theta_start))
        #print("Goal x: %.2f m\nGoal y: %.2f m\nGoal theta: %.2f rad\n" %
         #     (x_goal, y_goal, theta_goal))
        #move_to_pose(x_start, y_start, theta_start, x_goal, y_goal, theta_goal)


#if __name__ == '__main__':
 #   main()
