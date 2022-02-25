import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import time
from Buffer import Buffer
from bot import Botenv
from plot_trajectory import plot_trajectory
from networks import Actor, Critic

ENV_SIZE = 10
VELOCITY = 5
NUM_EPS = 1000
SHOW = False
# Discount Factor
GAMMA = 0.99
# Used to update Target Network
TAU = 0.005

EPS = 0.7
EPS_DECAY = 0.99

bot = Botenv(10, 0.01, ENV_SIZE, VELOCITY)

state_size = bot.state.shape[0]
action_size = 1

actor = Actor(state_size, action_size)
critic = Critic(state_size, action_size)
actor_target = Actor(state_size, action_size)
critic_target = Critic(state_size, action_size)

actor_target.set_weights(actor.get_weights())
critic_target.set_weights(critic.get_weights())

# Actor/Critic Learning Rates
actor_lr = 1e-4
critic_lr = 1e-3

actor_optimizer = Adam(actor_lr)
critic_optimizer = Adam(critic_lr)

buffer_size = 64
buffer_capacity = 1000000
buffer_file = None

buffer = Buffer(buffer_file, buffer_size, buffer_capacity, state_size, action_size)

def update(state_batch, next_state_batch, 
           action_batch, reward_batch):
    # Critic Loss 
    with tf.GradientTape() as tape:
        target_actions = actor_target(next_state_batch, training=True)
        target_q = critic_target([next_state_batch, target_actions], training=True)
        y = reward_batch + GAMMA * target_q
        cur_q = critic([state_batch, action_batch], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - cur_q))
    critic_grad = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grad, critic.trainable_variables))
    # Actor Loss
    with tf.GradientTape() as tape:
        actions = actor(state_batch, training=True)
        q_value = critic([state_batch, actions], training=True)
        actor_loss = -tf.math.reduce_mean(q_value)
    actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grad, actor.trainable_variables))
    
def learn():
    r = min(buffer.buffer_counter, buffer_size)
    batch_indices = np.random.choice(r, buffer_size)
    
    state_batch = tf.convert_to_tensor(buffer.state_buffer[batch_indices])
    next_state_batch = tf.convert_to_tensor(buffer.next_state_buffer[batch_indices])
    action_batch = tf.convert_to_tensor(buffer.action_buffer[batch_indices])
    reward_batch = tf.convert_to_tensor(buffer.reward_buffer[batch_indices])
    reward_batch = tf.cast(reward_batch, dtype=tf.float32)
    
    update(state_batch, next_state_batch,
          action_batch, reward_batch)

def policy(state):
    sampled_actions = tf.squeeze(actor(state))
    sampled_actions = sampled_actions.numpy()
    legal_action = np.clip(sampled_actions, -1, 1)
    return [np.squeeze(legal_action)]

# Update Target Network slowly 

def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

episode_rewards = []
action = 0
TAU = 0.005

for ep in range(NUM_EPS):
    if ep%50 == 0:
        trajectory = {'x': [], 'y': []}
        show = True
    else:
        show = False
    
    bot.reset()
    score = 0
    while(bot.T > 0):
        cur_state = bot.state
        tf_cur_state = tf.expand_dims(tf.convert_to_tensor(cur_state), 0)
        
        if np.random.random() > EPS:
            action = policy(tf_cur_state)
        else:
            action = [np.random.uniform(-1, 1)]
            action = tf.convert_to_tensor(action)

        next_state, reward, done = bot.action(action[0])
        obs = (cur_state, action, next_state, reward)
        buffer.record(obs)
        score += reward
        
        learn()
        update_target(actor_target.variables, actor.variables, TAU)
        update_target(critic_target.variables, critic.variables, TAU)
        
        if show:
            trajectory['x'].append(bot.x)
            trajectory['y'].append(bot.y)
            plot_trajectory(SIZE=ENV_SIZE, agent=bot,
                           trajectory=trajectory)

        if done:
            break
            
    episode_rewards.append(score)
    print(f"Score for Episode #{ep} = {score}")
    plt.close()

filename = f"replay_{int(time.time())}"
buffer.save(filename)
actor.save("actor.h5")
critic.save("critic.h5")