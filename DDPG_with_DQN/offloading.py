# 自我測試版
#from Qlearning import QLearningTable
import random
import math
from tkinter import W
import matplotlib.pyplot as plt
import time
from ENV import Environment
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#####################  hyper parameters  ####################
# MAX_EPISODES = 50000

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
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
        self.sess.run(
            self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s,  a, r, s_):
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


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.001,  # 0.001
            e_greedy=0.99,
            replace_target_iter=200,
            memory_size=MEMORY_CAPACITY,
            batch_size=BATCH_SIZE,
            # e_greedy_increment=8.684615e-05,
            # e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = 0.8
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        # memory里存放当前和下一个state，动作和奖励
        self.memory = np.zeros(
            (MEMORY_CAPACITY, s_dim * 2 + 2), dtype=np.float32)

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [
                tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(
            tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(
            0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            # e2 = tf.layers.dense(e1, 48, tf.nn.relu6, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='e2')
            e3 = tf.layers.dense(e1, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e3')
            self.q_eval = tf.layers.dense(e3, self.n_actions, tf.nn.softmax, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 100, tf.nn.relu6, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            # t2 = tf.layers.dense(t1, 48, tf.nn.relu6, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='t2')
            t3 = tf.layers.dense(t1, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t3')
            self.q_next = tf.layers.dense(t3, self.n_actions, tf.nn.softmax, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t4')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * \
                tf.reduce_max(self.q_next, axis=1,
                              name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack(
                [tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(
                params=self.q_eval, indices=a_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(
                self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            # self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            self._train_op = tf.train.AdamOptimizer(
                self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, [r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(
                self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        #self.epsilon += (0.01/10000)


###############################  training  ####################################
np.random.seed(1)
tf.set_random_seed(1)

env = Environment()
s_dim = 7
a_dim = 2
action_space = ['0', '1']
n_actions = len(action_space)
n_features = 7
a_bound = [-1, 1]  # [-1,1]
RL = DDPG(a_dim, s_dim, a_bound)
DQN = DeepQNetwork(n_actions, n_features)
#QL = QLearningTable(actions=list(range(env.n_actions)))
MAX_EPISODES = 3000
T = 100
N = 100
# var = 1  # control exploration

var = 0.1  # control exploration
t1 = time.time()
episode_list = []
delay_list = []
energy_list = []
offload = 0
local = 0
delay = 0
r = 0

for episode in range(MAX_EPISODES):

    i = 0
    for i in range(T):
        obs = env.observed()
        # print("obs", obs[0], obs[1])
        # Add exploration noise
        #a0 = RL.choose_action_d(obs)
        # print('DQN_a0', a0
        a = DQN.choose_action(obs)
        #print("DQN action", a)
        actions = RL.choose_action(obs)
        lower_bound = 0.01    # 0.1
        actions[0] = np.clip(np.random.normal(
            actions[0], var), lower_bound, 0.8)
        actions[1] = np.clip(np.random.normal(
            actions[1], var), lower_bound, 0.8)

        #print("actions noise", actions)
        w_n = actions[0]  # 0.01 or 0.99
        f_n = actions[1]  # 0.01 or 0.99

        if a == 1:

            # offload
            R_n = w_n*obs[0]*env.Wmax * \
                math.log(1+(obs[5]/(w_n*obs[0]*env.Wmax*env.n0)), 2)
            T_trans = obs[2] / R_n
            T_MEC = obs[3] / (f_n*obs[1]*env.Fmax)
            T_offload = round((T_trans + T_MEC), 5)
            reward = env.step(actions, round(T_trans, 5), round(T_MEC, 5))
            s_ = obs.copy()
            # print('S_', s_)
            s_[0] -= (s_[0] * w_n)
            s_[1] -= (s_[1] * f_n)
            # print(obs, s_)
            offload += 1

            if T_offload > obs[4]:
                # print("NO")
                reward += 0.005

            RL.store_transition(obs, actions, -reward, s_)

        else:
            # loacl
            actions[0] = actions[1] = 0
            T_local = obs[3] / obs[6]
            E_local = env.kn * pow(obs[6], 2) * obs[3]
            reward = env.alpha * T_local + env.beta * E_local

            #reward = 10
            # s_ = obs
            s_ = obs.copy()
            # s_[6] = 0
            # print(obs, s_)
            local += 1

        DQN.store_transition(obs, a, -reward, s_)
        #print("reward", reward)

        env.reward_list.append(reward)

        if RL.pointer > 10000:
            # var = max([var * 0.9997, VAR_MIN])  # decay the action randomness
            #var *= .9997
            RL.learn()
            DQN.learn()
            time.sleep(0.01)
            # time.sleep(0.02)
        #x = random.uniform(0.04, 0.01)
        # time.sleep(0.01)
    if env.show_reward(N) < 0.03:
        episode_list.append(env.show_reward(N))
    file = open('RL.txt', 'w')
    for line in episode_list:
        file.write(str(line))
        file.write("\n")
    file.close
    print('Episode:', episode, ' Steps: %2d' % i, 'Reward',
          episode_list[-1], "local", local, "offload", offload)


print('Running time: ', time.time() - t1)
#plt.axis([0, MAX_EPISODES, 0.004, 0.03])
plt.plot(episode_list)
plt.xlabel("Episode")
plt.ylabel("cost")
plt.savefig("ddpg.png")
plt.show()
