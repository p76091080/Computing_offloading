from ENV import Environment
import time
import numpy as np
import random
import math


class random(object):
    env = Environment()

    M = 3000
    T = 100
    alpha = 0.5
    beta = 0.5
    Random_list = []
    actions = [0]*2
    Random = []
    offload = 0
    local = 0
    for episode in range(M):

        for i in range(T):
            obs = env.observed()
            # print(obs)
            # print(random.randint(0, 1))
            if random.randint(0, 1) == 1:
                # offload
                w_n = random.uniform(0.1, 0.99)
                f_n = random.uniform(0.1, 0.99)
                # print(w_n, f_n)
                actions[0] = w_n
                actions[1] = f_n

                R_n = w_n*obs[0]*env.Wmax * \
                    math.log(1+(obs[5]/(w_n*obs[0]*env.Wmax*env.n0)), 2)
                T_trans = obs[2] / R_n
                T_MEC = obs[3] / (f_n*obs[1]*env.Fmax)
                T_offload = T_trans + T_MEC
                E_offload = env.p * T_offload

                reward = env.step(actions, round(T_trans, 7), round(T_MEC, 7))
                env.reward_list.append(reward)

                offload += 1
            else:
                # loacl
                T_local = obs[4]
                E_local = env.kn * pow(obs[6], 2) * obs[3]
                cost = alpha * T_local + beta * E_local
                env.reward_list.append(cost)
                local += 1

            time.sleep(0.01)
        Random_list.append(env.show_reward(100))
        print(episode, Random_list[-1])

        file = open('Random.txt', 'w')
        for line in Random_list:
            file.write(str(line))
            file.write("\n")
        file.close

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(Random_list)), Random_list)
    plt.title("Random")
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.savefig('Random.png')
    plt.show()
