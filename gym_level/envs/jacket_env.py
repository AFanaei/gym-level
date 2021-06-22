import math

import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

import gym
from gym import spaces
from gym.utils import seeding
from gym_level.envs.render import StatePlotter


class JacketEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    #     'render.modes': ['human', 'rgb_array'],
    #     'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity=0):
        self.action_limit = [-10.0, 10.0] # percent
        self.sensor_limit = [0.0, 100.0] # percent
        self.valve_limit = [0, 100] # percent
        self.temp_limit = [550, 750] # rankin
        self.temp_tav = 2 # min
        self.c_limit = [0, 0.5] # lbmol/ft3
        self.c_dead = 3 # min
        # reaction
        self.k_0 = 8.33e8 # ft3/lbmol.min
        self.E = 27820 # btu/lbmol
        self.R = 1.987 # btu/lbmol.R
        self.DH_r = -12000 # btu/lbmol
        # heat
        self.A = 36 # ft2
        self.U = 1.25 # but/min.ft2.R
        # inside
        self.V = 13.26 # ft3
        self.ro = 55 # lbm/ft3
        self.Cp = 0.88 # btu/lbm.R
        # jacket
        self.V_c = 1.56 # ft3
        self.ro_c = 62.4 #lbm/ft3
        self.Cp_c = 1 # btu/lbm.R
        # valve
        self.Cv_max = 11
        self.DP_v = 10 # psi

        self.visualization = None

        self.action_space = spaces.Box(
            low=np.array([self.action_limit[0], self.action_limit[0]], dtype=np.float32),
            high=np.array([self.action_limit[1], self.action_limit[1]], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array(
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                dtype=np.float32
            ),
            high=np.array(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
        self.input = dict(
          T_i=635, # R
          Ca_i=0.6, # lbmol/ft3
          Tc_i=540, # R
          m1=25, # %
          m2=75, # %
        )

        self.state = fsolve(lambda x, *args: self._derivative(0, x), [0.23091, 667.451, 588.792, 58.73])
        self.c_s0 = self.c_s1 = self.c_s2 = self.c_s3 = (self.state[0] - self.c_limit[0]) / (self.c_limit[1] - self.c_limit[0]) * 100
        self.target = np.array([self.c_s0 + 1, self.state[3] + 1])
        self.history = [
            (np.array([self.state[3], self.c_s0, 25, 75]) -50) / 100,
            (np.array([self.state[3], self.c_s0, 25, 75]) -50) / 100,
            (np.array([self.state[3], self.c_s0, 25, 75]) -50) / 100
        ]
        indexes = [0, 1, 2]
        return np.concatenate([self.history[indexes[0]], self.history[indexes[1]], self.history[indexes[2]], (self.target - 50) / 100])

    def step(self, action):
        self.current_step += 1
        m1 = self.input['m1']
        m1 += action[0]
        m1 = min(max(m1, self.valve_limit[0]), self.valve_limit[1])
        self.input['m1'] = m1

        m2 = self.input['m2']
        m2 += action[1]
        m2 = min(max(m2, self.valve_limit[0]), self.valve_limit[1])
        self.input['m2'] = m2

        res = solve_ivp(self._derivative, [0, 1], self.state)

        self.state = res.y[:, -1]
        self.c_s3, self.c_s2, self.c_s1 = self.c_s2, self.c_s1, self.c_s0
        self.c_s0 = (self.state[0] - self.c_limit[0]) / (self.c_limit[1] - self.c_limit[0]) * 100

        reward = -norm(self.target - np.array([self.c_s0, self.state[3]])) * 0.1

        indexes = [(1 + self.current_step - i) % 3 for i in range(2, -1, -1)]
        self.history[indexes[-1]] = (np.array([self.state[3], self.c_s0, self.input['m1'], self.input['m2']]) - 50) / 100
        return np.concatenate([self.history[indexes[0]], self.history[indexes[1]], self.history[indexes[2]], (self.target - 50) / 100]), reward, False, {}

    def _derivative(self, t, y):
        Ca, T, T_c, T_m = y

        f = self.Cv_max * self.input['m1'] / 100 * np.sqrt(self.DP_v / (self.ro / self.ro_c)) * 1 / 7.48
        f_c = self.Cv_max * (1 - self.input['m2'] / 100) * np.sqrt(self.DP_v) * 1 / 7.48

        Ra = self.k_0 * np.exp(-self.E / (self.R * T)) * Ca ** 2
        dCa_dt = f * (self.input['Ca_i'] - Ca) / self.V - Ra
        dT_dt = f * (self.input['T_i'] - T) / self.V - self.DH_r * Ra / (self.ro * self.Cp) - self.U * self.A * (T - T_c) / (self.ro * self.Cp * self.V)
        dTc_dt = f_c * (self.input['Tc_i'] - T_c) / self.V_c + self.U * self.A * (T - T_c) / (self.ro_c * self.Cp_c * self.V_c)
        T_s = (T - self.temp_limit[0]) / (self.temp_limit[1] - self.temp_limit[0]) * 100
        dTm_dt = (T_s - T_m) / self.temp_tav
        return np.array([dCa_dt, dT_dt, dTc_dt, dTm_dt])

    def render(self, mode='human'):
        if self.visualization == None:
            self.visualization = StatePlotter('Jacket', 2, 2)

        self.visualization.render(
            self.current_step,
            np.array([self.state[3], self.c_s0]),
            np.array([self.input['m1'], self.input['m2']])
        )

    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None
