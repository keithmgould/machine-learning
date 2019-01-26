import tensorflow as tf
import numpy as np

import os
import time

from collections import deque
import functools
import gc
import sys

###

CODE_PATH = os.path.abspath(os.path.dirname(__file__)) + "/"
MODEL_PATH = CODE_PATH + "model/"

class Config(object):

  def __init__(self):
    self.action_map = lambda x: x
    self.observation_preprocess = lambda x: x
    self.rewards_discount = rewards_discount
    self.normalize_rewards = False
    self.rollout_size = 1

class FrameDiff(object):
  def __init__(self, env):
    self.wrapped = env

  def render(self):
    return self.wrapped.render()

  def reset(self):
    observation = self.wrapped.reset()
    self.observation_old = observation
    return (observation, observation)

  def step(self, action):
    observation_new, *stuff = self.wrapped.step(action)
    observation = (self.observation_old, observation_new)
    self.observation_old = observation_new
    return (observation, *stuff)

###

def map_graph():
  graph = tf.get_default_graph()

  tensors = [
    "rollout/observations",
    "rollout/actions",
    "train/loss",
  ]
  operations = [
    "train/train_op",
  ]

  graph_map = {}

  graph_map.update({x: graph.get_tensor_by_name(x + ":0") for x in tensors})
  graph_map.update({x: graph.get_operation_by_name(x) for x in operations})

  return graph_map

def count_variables():
  total_parameters = 0

  for variable in tf.trainable_variables():
    shape = variable.get_shape()
    print(shape)
    variable_parameters = 1

    for dim in shape:
      variable_parameters *= dim.value

    print(variable_parameters)
    total_parameters += variable_parameters

  print(total_parameters)

###

def run_inference(sess, graph_map, observations):
  feed_dict={
    graph_map["rollout/observations"]: observations,
  }
  actions = sess.run(graph_map["rollout/actions"], feed_dict=feed_dict)
  return actions

def run_training(sess, graph_map):
  loss, _ = sess.run([graph_map["train/loss"], graph_map["train/train_op"]])
  return loss

###

def observation_preprocess(config, observation):
  if type(observation) == tuple:
    old, new = observation
    old = config.observation_preprocess(old)
    new = config.observation_preprocess(new)
    return new - old
  else:
    observation = config.observation_preprocess(observation)
    return observation

def rewards_discount(rewards, decay=.9, reset=False):
  rewards2 = np.empty_like(rewards, np.float32)
  running_add = 0

  for index in reversed(range(len(rewards))):
    reward = rewards[index]
    if reset and reward != 0:
      running_add = reward
    else:
      running_add *= decay
      running_add += reward
    rewards2[index] = running_add

  return rewards2

def rewards_normalize(rewards):
  mean = rewards.mean()
  std = rewards.std()

  rewards -= mean
  rewards /= std

  return rewards

###

def main_testing(sess, graph_map, config, env, delta=0.02):
  observation = env.reset()
  observation = observation_preprocess(config, observation)

  done = False

  while not done:
    env.render()

    action = run_inference(sess, graph_map, [observation])[0]

    observation, reward, done, *info = env.step(config.action_map(action))
    observation = observation_preprocess(config, observation)

    time.sleep(delta)

def main_training(sess, graph_map, config, env, memory):
  epoch_memory = []
  episode_memory = []

  scores = []

  while len(epoch_memory) < config.rollout_size:
    observation = env.reset()
    observation = observation_preprocess(config, observation)

    done = False

    while not done:
      action = run_inference(sess, graph_map, [observation])[0]
      old_observation = observation

      observation, reward, done, *info = env.step(config.action_map(action))
      observation = observation_preprocess(config, observation)

      episode_memory.append((old_observation, action, reward))

    q_observations, q_actions, q_rewards = zip(*episode_memory)
    scores.append(sum(q_rewards))

    q_rewards = config.rewards_discount(q_rewards)

    if config.normalize_rewards:
      q_rewards = rewards_normalize(q_rewards)

    epoch_memory.extend(zip(q_observations, q_actions, q_rewards))
    episode_memory = []

  memory.extend(epoch_memory)
  score = sum(scores) / len(scores)
  loss = run_training(sess, graph_map)
  return score, loss

###

def main(env, test=False, load=None, save=None, log=None, **kwargs):
  tf.reset_default_graph()

  tf_config = tf.ConfigProto()
  tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

  build_graph()
  # count_variables()
  graph_map = map_graph()

  init_op = tf.global_variables_initializer()
  saver = tf.train.Saver()

  with tf.Session(config=tf_config) as sess:
    if load:
      saver.restore(sess, MODEL_PATH + load + "/model")
    else:
      sess.run(init_op)

    if save:
      os.makedirs(MODEL_PATH + save['path'], exist_ok=True)

    try:
      if test:
        while True:
          main_testing(sess, graph_map, config, env, **kwargs)
      else:
        train_step = 0
        smooth_score = None

        while True:
          score, loss = main_training(sess, graph_map, config, env, MEMORY, **kwargs)

          if smooth_score is None:
            smooth_score = score
          else:
            smooth_score *= .9
            smooth_score += .1 * score

          train_step += 1

          if log and log(train_step=train_step, smooth_score=smooth_score, loss=loss):
            env.render()

          if save and (train_step % save['interval'] == 0):
            gc.collect()

            saver.save(sess, MODEL_PATH + save['path'] + "/model")
    except KeyboardInterrupt:
      pass
