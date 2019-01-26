#!/usr/bin/python3
from common import *

import gym
import gym.spaces

###

BATCH_SIZE = 10000
MEMORY_CAPACITY = 100000
MEMORY = deque(maxlen=MEMORY_CAPACITY)

def gen():
  for m in list(MEMORY):
    yield m

def observation_preprocess(observation):
  observation = observation[35:195]
  observation = observation[::2,::2,0]
  observation[observation == 144] = 0
  observation[observation == 109] = 0
  observation[observation != 0] = 1
  observation = observation.astype(np.int8)
  return observation

config = Config()
config.action_map = lambda x: [0, 2, 3][x]
config.rewards_discount = functools.partial(rewards_discount, decay=.99, reset=True)
config.normalize_rewards = True
config.observation_preprocess = observation_preprocess

###

def build_model(observations):
  observations = tf.cast(observations, tf.float32)
  flat = tf.layers.flatten(observations)

  with tf.variable_scope("iso", reuse=tf.AUTO_REUSE):
    features = tf.layers.dense(flat, 200, activation=tf.nn.relu, use_bias=False, name="features")
    logits = tf.layers.dense(features, 3, use_bias=False, kernel_initializer=tf.zeros_initializer(), name="logits")

  return logits

def build_graph():
  with tf.name_scope("rollout"):
    observations = tf.placeholder(tf.int8, [None, 80, 80], name="observations")
    logits = build_model(observations)

    actions = tf.multinomial(logits, 1)
    actions = tf.squeeze(actions, 1, name="actions")

  with tf.name_scope("dataset"):
    ds = tf.data.Dataset.from_generator(gen, output_types=(tf.int8, tf.int32, tf.float32))
    ds = ds.shuffle(MEMORY_CAPACITY).repeat().batch(BATCH_SIZE)
    iterator = ds.make_one_shot_iterator()

  with tf.name_scope("train"):
    q_observations, q_actions, q_rewards = iterator.get_next()
    q_observations.set_shape((BATCH_SIZE, 80, 80))
    q_logits = build_model(q_observations)

    xent = tf.losses.sparse_softmax_cross_entropy(q_actions, q_logits, q_rewards, scope="xent")
    effort = 3e-7 * tf.reduce_mean(tf.nn.softmax(logits) * [0., 1., 1.])
    loss = tf.identity(xent + effort, "loss")

    optimizer = tf.train.RMSPropOptimizer(1e-3, .99)
    train_op = optimizer.minimize(loss, name="train_op")

###

def log(train_step, smooth_score, loss):
  print("training step:", train_step, "| score:", "{:.2f}".format(smooth_score), "| loss:", "{:.10f}".format(loss))
  return False

args = sys.argv[1:]
test = "test" in args
load = "load" in args
save = "save" in args

load = "atari-pong" if load else None
save = {
  'path': "atari-pong",
  'interval': 16,
} if save else None

env = gym.make("Pong-v4")
env = FrameDiff(env)

main(env, test=test, load=load, save=save, log=log)
