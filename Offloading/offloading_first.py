from ENV import Environment
from DQN_of_model_DDPG import DDPG
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import random


class Offload(object):
    env = Environment()
    M = 3000
    T = 100

    off_list = []
    Offload_list = []
    episode_list = []
    actions = [0]*2

    for episode in range(M):

        for i in range(T):
            obs = env.observed()
            w_n = 1
            f_n = 1
            actions[0] = 1
            actions[1] = 1

            R_n = w_n*obs[0]*env.Wmax * \
                math.log(1+(obs[5]/(w_n*obs[0]*env.Wmax*env.n0)), 2)
            T_trans = obs[2] / R_n
            T_MEC = obs[3] / (f_n*obs[1]*env.Fmax)
            T_offload = T_trans + T_MEC
            E_offload = env.p * T_offload
            reward = env.step(actions, round(T_trans, 5), round(T_MEC, 5))
            env.reward_list.append(reward)

        time.sleep(0.01)
        off_list.append(env.show_reward(100))

        print(episode, off_list[-1])

    # x = np.arange(len(episode_list))
    # plt.plot(x, episode_list, c="b", label='offload')  # b
    # plt.plot(x, local_list, c="r", label='local')  # r


if __name__ == '__main__':
    offload = Offload()
    #local = Local()

    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.plot(np.arange(len(offload.off_list)),
             offload.off_list, c="b", label='offload')  # b
    plt.savefig('result_eng.png')
    plt.show()
