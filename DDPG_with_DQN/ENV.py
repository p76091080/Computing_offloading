# from socket import socket, AF_INET, SOCK_DGRAM, timeout
import threading
import time
import random
import numpy as np
import os


# lock = threading.Lock()
pi = 3.141592653589793


class Environment:
    def __init__(self,
                 n_actions=4,
                 n_features=5*1e9,
                 transmit_power=2,
                 bandwidth=10*1e6,
                 M_computing_capacity=5*1e9,
                 noise_PSD=1e-13,
                 kn=1e-26,
                 Ad=4.11,
                 carrier_frequency=900*1e6
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.p = 2
        self.Ad = Ad
        self.fc = carrier_frequency
        self.n0 = noise_PSD*2
        self.kn = kn

        self.memory_lock = threading.Lock()
        self.log_dir = "logs/"
        # check path exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_filename = time.strftime("%y%m%d%H%M%S.txt", time.localtime())

        self.residual_radio = 1
        self.residual_computing = 1
        self.Fmax = M_computing_capacity
        self.Wmax = bandwidth

        self.alpha = 0.5
        self.beta = 0.5

        self.reward_list = []

    def reset(self):
        self.reset_env()
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.state = np.append(self.e_battery_uav, self.loc_uav)
        self.state = np.append(self.state, self.sum_task_size)
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.block_flag_list)
        return self._get_obs()

    def thread_radio(self, T_tras, cost_tras):
        self.memory_lock.acquire()
        self.residual_radio -= cost_tras
        self.memory_lock.release()
        # wait for transmission
        # print("sleep", T_tras)
        time.sleep(T_tras)
        # end waiting
        self.memory_lock.acquire()
        self.residual_radio += cost_tras
        # print("end tras", self.residual_radio)
        self.memory_lock.release()

    def thread_computing(self, T_computing, cost_computing):
        self.memory_lock.acquire()
        self.residual_computing -= cost_computing
        self.memory_lock.release()
        # wait for transmission
        # print("sleep", T_computing)
        time.sleep(T_computing)
        # end waiting
        self.memory_lock.acquire()
        self.residual_computing += cost_computing
        # print("end computing", self.residual_computing)
        self.memory_lock.release()

    def step(self, actions, T_offload, T_computing):
        if T_offload != 0:
            # offloading
            # print(T_offload, T_computing)
            # print("use", actions[1], self.residual_radio, actions[2], self.residual_computing)

            threading.Thread(target=self.thread_radio, args=(
                T_offload, actions[0] * self.residual_radio)).start()  # target: 被執行的物件，由run()方法執行 args: target物件使用的引數
            threading.Thread(target=self.thread_computing, args=(
                T_computing, actions[1] * self.residual_computing)).start()

            return (self.alpha * (T_offload + T_computing) + (self.beta * self.p * T_offload))
        else:
            pass

    def observed(self):
        x = self.residual_radio
        y = self.residual_computing
        if random.randint(0, 1) == 0:
            # Type-1 task
            d = random.uniform(150*1e3, 170*1e3)
            c = random.uniform(4*1e6, 6*1e6)
        else:
            d = random.uniform(70*1e3, 90*1e3)
            c = random.uniform(9*1e6, 11*1e6)

        dis_n = random.uniform(10, 20)
        h = self.Ad*pow(3*1e8/(4*pi*self.fc*dis_n), 2)
        fln = random.uniform(0.4*1e9, 0.6*1e9)  # 　local computing capacity
        tau = c/fln                             # tolerable delay
        return np.array([x, y, d, c, tau, self.p*h, fln])

    def show_reward(self, count):
        # for i in range(len(self.reward_list)):
        return sum(self.reward_list[-count:]) / count  # 最後5個相加
