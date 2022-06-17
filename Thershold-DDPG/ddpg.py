# 自我測試版
import random
import math
import matplotlib.pyplot as plt
import time
from ENV import Environment
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#####################  hyper parameters  ####################
# MAX_EPISODES = 50000

LR_A = 0.00001  # learning rate for actor
LR_C = 0.0002  # learning rate for critic
# LR_A = 0.1  # learning rate for actor
# LR_C = 0.2  # learning rate for critic
GAMMA = 0.9  # optimal reward discount 0.001
# GAMMA = 0.999  # reward discount
TAU = 0.01  # soft replacement
VAR_MIN = 0.01
# MEMORY_CAPACITY = 5000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
OUTPUT_GRAPH = False


###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        # memory里存放当前和下一个state，动作和奖励
        self.memory = np.zeros(
            (MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')  # 输入
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval',
                              trainable=True)  # 　trainable　是否參與訓練
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        self.ae_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(
            td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(
            LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        temp = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        # return np.clip(temp, -2, 2)
        return temp

    def learn(self):
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs,
                      self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # transition = np.hstack((s, [a], [r], s_))
        # replace the old memory with new memory
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def store_transition2(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # transition = np.hstack((s, [a], [r], s_))
        # replace the old memory with new memory
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(
                s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(
                net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(
                net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(
                net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound[1], name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.get_variable(
                'w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable(
                'w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(
                net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(
                net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################
np.random.seed(1)
tf.set_random_seed(1)

env = Environment()
s_dim = 7
a_dim = 3
a_bound = [-1, 1]  # [-1,1]
RL = DDPG(a_dim, s_dim, a_bound)
MAX_EPISODES = 3000
T = 100
# var = 1  # control exploration
var = 1  # control exploration
t1 = time.time()
episode_list = []
DG_list = []
offload = 0
local = 0
penalty = 5
delay = 0

for episode in range(MAX_EPISODES):
    # s = env.reset()
    i = 0
    for i in range(T):
        obs = env.observed()
        # print("obs", obs[0], obs[1])
        # Add exploration noise

        actions = RL.choose_action(obs)
        # [-0.99865615  1.          0.5830561 ]
        #print('choose action', actions)

        lower_bound = 0.01    # 0.1
        actions[1] = np.clip(np.random.normal(
            actions[1], var), lower_bound, 0.99)
        actions[2] = np.clip(np.random.normal(
            actions[2], var), lower_bound, 0.99)

        #print("actions noise", actions)
        w_n = actions[1]  # 0.1 or 0.99
        f_n = actions[2]  # 0.1 or 0.99

        # if w_n != 0.0 and f_n != 0.0 and obs[0] != 0 and obs[1] != 0:
        R_n = w_n*obs[0]*env.Wmax * \
            math.log(1+(obs[5]/(w_n*obs[0]*env.Wmax*env.n0)), 2)
        T_trans = obs[2] / R_n
        T_MEC = obs[3] / (f_n*obs[1]*env.Fmax)
        T_offload = round((T_trans + T_MEC), 5)
        #print('Tolerable delay', obs[4])
        #print("T_offload", T_offload)
        if T_offload < obs[4]:
            actions[0] = 1
            # offload
            # r1 = env.step(actions, round(T_trans, 5), round(T_MEC, 5))
            # r2 = env.step(actions, round(T_trans, 5), round(T_MEC, 5))
            reward = env.step(actions, round(T_trans, 5), round(T_MEC, 5))
            # s_ = obs
            s_ = obs.copy()
            # print('S_', s_)
            s_[0] -= (s_[0] * w_n)
            s_[1] -= (s_[1] * f_n)
            # print(obs, s_)

            offload += 1

        else:
            # loacl
            actions[0] = actions[1] = actions[2] = 0
            T_local = obs[3] / obs[6]
            E_local = env.kn * pow(obs[6], 2) * obs[3]
            reward = env.alpha * T_local + env.beta * E_local
            # reward = 1
            # s_ = obs
            s_ = obs.copy()
            # s_[6] = 0
            # print(obs, s_)

            local += 1
        #reward -= 0.004
        RL.store_transition(obs, actions, -reward, s_)
        env.reward_list.append(reward)

        if RL.pointer > MEMORY_CAPACITY:
            # var = max([var * 0.9997, VAR_MIN])  # decay the action randomness
            var *= .9997
            RL.learn()
        # obs = s_
        time.sleep(0.01)

    # episode_list = np.append(episode_list, ep_reward)
    episode_list.append(env.show_reward(100))
    file = open('DG_comp.txt', 'w')
    for line in episode_list:
        file.write(str(line))
        file.write("\n")
    file.close
    # print('reward', episode_list)
    # print("actions", actions)
    print('Episode:', episode, ' Steps: %2d' % i, 'Reward',
          episode_list[-1], "local", local, "offload", offload)
    # print('OBS', obs)

    # # Evaluate episode
    # if (i + 1) % 50 == 0:
    #     eval_policy(ddpg, env)

print('Running time: ', time.time() - t1)
plt.plot(np.arange(len(episode_list)), episode_list)
plt.xlabel("Episode")
plt.ylabel("cost")
plt.savefig("ddpg_comp.png")
plt.show()
