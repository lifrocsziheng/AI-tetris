import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import tf_agents
import gymnasium
import random
from tf_agents.environments import suite_gym


# trial run

env = gymnasium.make("ALE/Tetris-v5", render_mode="human") # tetris game from openai Gym

states = env.observation_space
actions = env.action_space
print(env.reset())


score = 0 
action = random.randint(0,4)
observation, reward, terminated, truncated, info = env.step(action)
while not terminated:
    action = random.randint(0,4)

    observation, reward, terminated, truncated, info = env.step(action)
score += reward
observation, info = env.reset()

env.close()


states = env.observation_space.shape[0]
actions = env.action_space.n


from tf_agents.agents import DqnAgent
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.networks import sequential
from tf_agents.environments import tf_py_environment
import tf_agents
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

train_py_env = suite_gym.load("ALE/Pacman-v5")
eval_py_env = suite_gym.load("ALE/Pacman-v5")
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# implement NN

model = sequential.Sequential([
    Flatten(input_shape=(1,states)),
    Dense(24, activation='relu'),
    Dense(24, activation='relu'),
    Dense(actions, activation='linear')
])

optimizer = Adam(learning_rate = 0.001)

train_step_counter = tf.Variable(0)

agent = DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=model,
    optimizer = optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()


eval_policy = agent.policy
collect_policy = agent.collect_policy

policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                         train_env.action_spec())


example_environment = tf_py_environment.TFPyEnvironment(
    suite_gym.load("ALE/Pacman-v5"))

time_step = example_environment.reset()
policy.action(time_step)

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.time_step_spec()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


num_eval_episodes = 10 

compute_avg_return(eval_env, policy)

