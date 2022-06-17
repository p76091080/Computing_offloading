from ENV import Environment
import time
import numpy as np


class local(object):
    env = Environment()

    M = 3000
    T = 100
    alpha = 0.5
    beta = 0.5
    local_list = []
    reward = []
    offload = 0
    local = 0
    for episode in range(M):
        if episode % 10 == 0:
            time.sleep(0.1)
        for i in range(T):
            obs = env.observed()
            # print(obs)

            # loacl
            T_local = obs[4]
            E_local = env.kn * pow(obs[6], 2) * obs[3]
            cost = alpha * T_local + beta * E_local
            env.reward_list.append(cost)
            local += 1

        local_list.append(env.show_reward(100))
        print(episode, local_list[-1])

        file = open('Local.txt', 'w')
        for line in local_list:
            file.write(str(line))
            file.write("\n")
        file.close
    # print('Episode:', episode, ' Steps: %2d' %i, 'Reward', env.reward_list[-1])
    # print(local_list)
    '''import matplotlib.pyplot as plt
    plt.plot(np.arange(len(local_list)), local_list)
    plt.title("Local computing")
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.savefig('Local_result.png')
    plt.show()
    '''
