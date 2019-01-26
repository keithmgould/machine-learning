#!/usr/bin/python3
from common import *

import cv2
import random

###

SIZE = 6
MAX = 12

BATCH_SIZE = 10000
MEMORY_CAPACITY = 100000
MEMORY = deque(maxlen=MEMORY_CAPACITY)

def gen():
  for m in list(MEMORY):
    yield m

class ThreesEnv():

  def __init__(self, test=False):
    self.test = test

  def render(self):
    tiles = self.tiles.astype(np.int32)

    h = tiles * 17
    s = np.full([SIZE, SIZE], 255)
    v = np.full([SIZE, SIZE], 255)

    s[tiles == 0] = 0

    image = np.stack([h, s, v], 2)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    cv2.imshow("threes", image)
    cv2.waitKey(1)

  def reset(self):
    self.tiles = np.zeros([SIZE, SIZE], np.uint8)
    self.count = 0

    for _ in range(2):
      self.spawn()

    return self.tiles

  def spawn(self):
    i, j = np.where(self.tiles == 0)
    r = np.random.choice(len(i), 1)
    self.tiles[i[r], j[r]] = random.choice([1, 2])
    self.count += 1

  def step(self, action):
    self.tiles = np.rot90(self.tiles, action)
    reward = 0.
    done = False

    for y in range(SIZE):
      for i in reversed(range(1, SIZE)):
        for x in range(i):
          a = self.tiles[y][x]
          b = self.tiles[y][x + 1]

          if a == 0:
            self.tiles[y][x] = b
            self.tiles[y][x + 1] = 0
          elif a == b and a < MAX:
            self.tiles[y][x] = a + 1
            self.tiles[y][x + 1] = 0
            self.count -= 1
            reward += a ** 2

            if a == MAX - 1:
              done = True
              print("heureka!")

    if self.test:
      self.tiles = np.rot90(self.tiles, -action)

    reward /= self.count ** .5

    if self.count == SIZE * SIZE:
      reward -= (MAX * SIZE) ** 2
      done = True

    if not done:
      self.spawn()

    return self.tiles, reward, done

config = Config()
config.rewards_discount = functools.partial(rewards_discount, decay=.9)

###

def build_model(observations):
  observations = tf.expand_dims(observations, 3)
  observations = tf.cast(observations, tf.float32)

  logits = []

  for k in range(4):
    feed = observations

    with tf.variable_scope("iso", reuse=tf.AUTO_REUSE):
      feed = tf.layers.conv2d(feed, 32, 2, activation=tf.nn.relu)
      feed = tf.layers.conv2d(feed, 64, 3, activation=tf.nn.relu)

      rotated = tf.image.rot90(feed, k)
      flat = tf.layers.flatten(rotated)

      features = tf.layers.dense(flat, 256, activation=tf.nn.relu)
      logit = tf.layers.dense(features, 1, use_bias=False)

    logits.append(logit)

  logits = tf.concat(logits, 1)
  return logits

def build_graph():
  with tf.name_scope("rollout"):
    observations = tf.placeholder(tf.uint8, [None, SIZE, SIZE], name="observations")
    logits = build_model(observations)

    actions = tf.multinomial(logits, 1)
    actions = tf.squeeze(actions, 1, name="actions")

  with tf.name_scope("dataset"):
    ds = tf.data.Dataset.from_generator(gen, output_types=(tf.uint8, tf.int32, tf.float32))
    ds = ds.shuffle(MEMORY_CAPACITY).repeat().batch(BATCH_SIZE)
    iterator = ds.make_one_shot_iterator()

  with tf.name_scope("train"):
    q_observations, q_actions, q_rewards = iterator.get_next()
    q_observations.set_shape((BATCH_SIZE, SIZE, SIZE))
    q_logits = build_model(q_observations)

    xent = tf.losses.sparse_softmax_cross_entropy(q_actions, q_logits, q_rewards, scope="xent")
    loss = tf.identity(xent, "loss")

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, name="train_op")

###

def log(train_step, smooth_score, loss):
  print("training step:", train_step, "| score:", "{:.0f}".format(smooth_score), "| loss:", "{:.2f}".format(loss))
  return True

args = sys.argv[1:]
test = "test" in args
load = "load" in args
save = "save" in args

load = "threes" if load else None
save = {
  'path': "threes",
  'interval': 32,
} if save else None

env = ThreesEnv(test=test)

main(env, test=test, load=load, save=save, log=log, delta=1.)
