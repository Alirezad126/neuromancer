
import numpy as np
import gym
import torch
from gym import spaces
from neuromancer import psl
from neuromancer.psl.building_envelope import LinearBuildingEnvelope



class SimpleBuildingEnv(gym.Env):
    """Custom Environment for controlling the building indoor temperature using RL"""

    metadata = {'render.modes': ['human']}

    def __init__(self, simulator=None, min_temp = None, max_temp = None ,random_seed=0):
        super().__init__()

        self.simulator = simulator if isinstance(simulator, LinearBuildingEnvelope) else psl.systems['LinearSimpleSingleZone'](seed=random_seed)

        self.action_space = spaces.Box(-np.inf, np.inf, shape=self.simulator.get_U(1).shape[-1], dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.simulator.x0.shape, dtype=np.float32)

        self.min_temp = min_temp
        self.max_temp = max_temp
        self.t = 0
        self.A = torch.tensor(self.simulator.params[2]['A'])
        self.B = torch.tensor(self.simulator.params[2]['Beta'])
        self.C = torch.tensor(self.simulator.params[2]['C'])
        self.E = torch.tensor(self.simulator.params[2]['E'])

        self.U_min = torch.tensor(self.simulator.umin)
        self.U_max = torch.tensor(self.simulator.umax)


        obs = self.reset()

    def build_state(self):
        return [self.x, self.min_temp, self.max_temp, self.d]

    def step(self, action, D):
        self.t += 1
        sim = self.simulator(nsim=1, x0=self.x, U=action, D=self.d)
        self.x = sim["X"].squeeze()
        self.y = sim["Y"].squeeze()
        self.d = D if D is not None else self.simulator.get_D(1)
        reward, cost = self.get_reward_cost(action, self.y)
        next_obs = self.build_state()
        return

    def reset(self):
        self.simulator = psl.systems['LinearBuildingEnvelope'](seed=np.random.randint(1))
        self.t = 0
        self.x = self.simulator.get_x0()
        self.d = self.simulator.get_d(1)
        obs = self.build_state()
        return obs

    def get_reward_cost(self, action, output):

        return



if __name__ == "__main__":
    env = SimpleBuildingEnv()
