import random
import math
import matplotlib.pyplot as plt
import time
from ENV import Environment
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
MAX_EPISODES = 200
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32


# Deep Q Network off-policy
class DQN(object):
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.1,
            reward_decay=0.001,
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
        # self.epsilon_increment = e_greedy_increment
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.epsilon = 0.9
        # self.epsilon = 0.9

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        # memory里存放当前和下一个state，动作和奖励
        self.memory = np.zeros(
            (MEMORY_CAPACITY, n_features * 2 + 2), dtype=np.float32)

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
        index = self.memory_counter % MEMORY_CAPACITY
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
            action = np.random.randint(0, 2)
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
        self.learn_step_counter += 1


env = Environment()
action_space = ['0', '1']
n_actions = len(action_space)
n_features = 7
RL = DQN(n_actions, n_features, output_graph=False)
#QL = QLearningTable(actions=list(range(env.n_actions)))
MAX_EPISODES = 3000
T = 100
# var = 1  # control exploration
var = 0.1  # control exploration
t1 = time.time()
episode_list = []
delay_list = []
actions = []
offload = 0
local = 0
penalty = 5
delay = 0
r = 0

for episode in range(MAX_EPISODES):

    if episode % 10 == 0:
        time.sleep(0.1)
    i = 0
    for i in range(T):
        obs = env.observed()
        # print("obs", obs[0], obs[1])
        # Add exploration noise
        #a0 = RL.choose_action_d(obs)
        #print('DQN_a0', a0)
        a = RL.choose_action(obs)
        #print("DQN action", a)

        w_n = 0.2  # 0.1 or 0.99
        f_n = 0.3  # 0.1 or 0.99
        actions = [0.3, 0.4]

        if a == 1:
            # offload
            R_n = w_n*obs[0]*env.Wmax * \
                math.log(1+(obs[5]/(w_n*obs[0]*env.Wmax*env.n0)), 2)
            T_trans = obs[2] / R_n
            T_MEC = obs[3] / (f_n*obs[1]*env.Fmax)
            T_offload = round((T_trans + T_MEC), 5)
            reward = env.step(actions, round(T_trans, 5), round(T_MEC, 5))

            #reward = 10
            s_ = obs.copy()
            # print('S_', s_)
            s_[0] -= (s_[0] * w_n)
            s_[1] -= (s_[1] * f_n)
            # print(obs, s_)
            offload += 1

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

        #reward = reward + 0.5
        RL.store_transition(obs, a, -reward, s_)

        env.reward_list.append(reward)

        if RL.memory_counter > MEMORY_CAPACITY:
            # var = max([var * 0.9997, VAR_MIN])  # decay the action randomness
            RL.learn()

    # episode_list = np.append(episode_list, ep_reward)
    episode_list.append(env.show_reward(T))
    # print('reward', episode_list)
    # print("actions", actions)
    print('Episode:', episode, ' Steps: %2d' % i, 'Reward',
          episode_list[-1], "local", local, "offload", offload)
    # print('OBS', obs)

    # # Evaluate episode
    # if (i + 1) % 50 == 0:
    #     eval_policy(ddpg, env)

print('Running time: ', time.time() - t1)
plt.plot(episode_list)
plt.xlabel("Episode")
plt.ylabel("reward")
plt.savefig("dqn.png")
plt.show()
