import matplotlib
import matplotlib.pyplot as plt
import numpy as np 

#################################################################

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

#################################################################

def plot_trajectory(SIZE, agent, target_pos=None, obs=None, trajectory=None):
	plt.cla()
	#	Set Limits
	plt.xlim(0, SIZE+1)
	plt.ylim(0, SIZE+1)

	#	Agent Marker
	# agent_marker, scale = arrow_marker(agent.theta)
	# markersize = 25
	# plt.scatter(agent.x, agent.y, marker=agent_marker, s=(markersize*scale)**2)
	plt.scatter(agent.x, agent.y, c="blue")

	#	Target Position
	# if target_pos:
	plt.scatter(agent.target_pos[0], agent.target_pos[1], c="green")

	#	Obstacles
	if obs: 
		plt.scatter(agent.obs_pos[0], agent.obs_pos[1], c="red")

	#	Trajectory
	if trajectory:
		plt.plot(trajectory['x'], trajectory['y'], 'b--')
	plt.pause(0.02)