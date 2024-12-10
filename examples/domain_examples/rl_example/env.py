
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

        self.action_space = spaces.Box(low=-np.inf,
                                       high=np.inf,
                                       dtype=np.float32,
                                       shape=(self.simulator.get_U(2).shape[-1],))
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            dtype=np.float32,
                                            shape=(self.simulator.get_x0().shape[0] + self.simulator.get_D(1).shape[-1] + 2,))

        self.min_temp = min_temp
        self.max_temp = max_temp
        self.t = 0
        self.U_min = torch.tensor(self.simulator.umin)
        self.U_max = torch.tensor(self.simulator.umax)
        self.prev_action = None

    def build_state(self):
        observations =  [self.x, self.min_temp, self.max_temp, self.d]

        combined_obs = []
        for o in observations:
            if isinstance(o, np.ndarray):  # Flatten if it's a NumPy array
                combined_obs.extend(o.ravel())
            else:  # Keep `None` as-is
                combined_obs.append(o)

        return combined_obs

    def step(self, action, D):
        self.t += 1
        sim = self.simulator.simulate(nsim=1, x0=self.x, U=action, D=self.d)
        self.x = sim["X"].reshape(self.simulator.get_x0().shape[0],)
        self.y = sim["Y"]
        self.d = D if D is not None else self.simulator.get_D(1)
        reward = self.get_reward(action)
        cost = self.get_cost(self.y)
        self.prev_action = action
        next_obs = self.build_state()
        return next_obs, reward, cost

    def reset(self):
        self.simulator = psl.systems['LinearSimpleSingleZone'](seed=np.random.randint(1000))
        self.t = 0
        self.x = self.simulator.get_x0()
        self.d = self.simulator.get_D(1)
        self.min_temp = self.min_temp+(self.max_temp-self.min_temp)*np.random.rand()
        self.max_temp = self.min_temp + 2.0
        obs = self.build_state()

        return obs
        # Combine all elements into a single list


    def get_reward(self, action):
        if self.prev_action is None:
            reward = - (action * 0.01)
        else:
            reward = - (action * 0.01) - abs(action - self.prev_action) * 0.1
        return reward

    def get_cost(self, output):
        cost = np.maximum(self.min_temp - output, 0) + np.maximum(output - self.max_temp, 0)
        return cost



if __name__ == "__main__":
    env = SimpleBuildingEnv(min_temp=20, max_temp=30)
    obs = env.reset()
    print(obs)
