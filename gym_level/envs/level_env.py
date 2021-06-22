import math

import numpy as np
from scipy.integrate import solve_ivp

import gym
from gym import spaces
from gym.utils import seeding
from gym_level.envs.render import StatePlotter


class LevelEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    #     'render.modes': ['human', 'rgb_array'],
    #     'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity=0):
        self.action_limit = [-10.0, 10.0]
        self.sensor_limit = [0.0, 100.0]
        self.valve_limit = [0, 100]
        self.tank1_limit = [0, 10]
        self.ro = 1000
        self.spg = 1
        self.A1 = 3
        self.cv_1 = 0.025
        self.cv_i = 0.0424

        self.target = 50

        self.vp_1 = 1
        self.m1 = 50

        self.visualization = None

        self.action_space = spaces.Box(
            low=np.array([self.action_limit[0]], dtype=np.float32),
            high=np.array([self.action_limit[1]], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array(
                [-1, -1, -1, -1, -1],
                dtype=np.float32
            ),
            high=np.array(
                [1, 1, 1, 1, 1],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_step = 0
        self.vp_1 = 1  # self.np_random.uniform(low=0.8, high=1)
        self.target = self.np_random.uniform(low=20, high=80)
        self.m1 = self.np_random.uniform(low=40, high=60)
        fi = self.cv_i * self.m1 / 100 * np.sqrt(50 / (self.spg)) * 60
        h1 = math.pow(fi / (self.cv_1 * self.vp_1 * 60), 2) * self.spg * 1000 / (self.ro * 10)
        self.state = np.array([h1])

        h1m = (h1 - self.tank1_limit[0]) / (self.tank1_limit[1] - self.tank1_limit[0]) * 100
        self.act_before = 0
        return np.array([(h1m - 50) / 50, (self.m1 - 50) / 50, (self.target - 50) / 50, 0, 0])

    def step(self, action):

        self.current_step += 1

        self.m1 += self.act_before
        self.act_before = action[0]
        self.m1 = min(max(self.m1, self.valve_limit[0]), self.valve_limit[1])

        res = solve_ivp(self._derivative, [0, 1], self.state)

        h1 = res.y[0, -1]
        h1 = min(max(h1, self.tank1_limit[0]), self.tank1_limit[1])
        diff = h1 - self.state[0]

        h1m = (h1 - self.tank1_limit[0]) / (self.tank1_limit[1] - self.tank1_limit[0]) * 100

        reward = -np.abs(self.target - h1m) * 0.5

        self.state = np.array([h1])
        return np.array([(h1m - 50) / 50, (self.m1 - 50) / 50, (self.target - 50) / 50, (diff) / 10, (self.act_before) / 10]), reward, False, {}

    def _derivative(self, t, y):
        h1 = y[0] if y[0] >= 0 else 0

        fi = self.cv_i * self.m1 / 100 * np.sqrt(50 / (self.spg)) * 60
        f1 = self.cv_1 * self.vp_1 * np.sqrt(self.ro*10*h1 / (self.spg * 1000)) * 60

        dh1_dt = (fi - f1) / self.A1

        return np.array([dh1_dt])

    def render(self, mode='human'):
        if self.visualization == None:
            self.visualization = StatePlotter('Level')

        self.visualization.render(self.current_step, self.state[0], self.m1)

    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None
