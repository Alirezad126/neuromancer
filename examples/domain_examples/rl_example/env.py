import numpy as np
import gym
import torch
from gym import spaces
from neuromancer import psl
from neuromancer.psl.building_envelope import LinearBuildingEnvelope

# Set the device for computations (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleBuildingEnv(gym.Env):
    """Custom Environment for controlling the building indoor temperature using reinforcement learning."""

    def __init__(self, simulator=None, min_temp=None, max_temp=None, random_seed=0, timesteps=5000, samples=1000):
        super().__init__()

        # Use the provided simulator or default to LinearSimpleSingleZone
        self.simulator = simulator if isinstance(simulator, LinearBuildingEnvelope) else psl.systems[
            'LinearSimpleSingleZone'](seed=random_seed)

        self.def_min_temp = min_temp
        self.def_max_temp = max_temp
        self.t = 0  # Current timestep
        self.timesteps = timesteps  # Number of timesteps per trajectory
        self.samples = samples  # Number of samples for normalization computation

        # Action constraints
        self.U_min = self.simulator.umin
        self.U_max = self.simulator.umax
        self.prev_action = 0.0  # Initialize previous action

        # Define action space: a single scalar action in [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.]),
            high=np.array([1.]),
            dtype=np.float32,
            shape=(self.simulator.get_U(2).shape[-1],)
        )

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
            shape=(
                self.simulator.get_x0().shape[0] + self.simulator.get_D(1).shape[-1] + 2,
            )
        )

        # Generate trajectory for disturbances and initialize normalization statistics
        self.trajectory = self.generate_trajectory()
        self.state_means, self.state_stds = self.compute_state_normalization()

        # Initialize the state
        self.reset()

    def build_state(self):
        """Construct the current state with normalized features."""
        # Combine current state, disturbance, and temperature bounds
        observations = [
            torch.tensor(self.x, dtype=torch.float32) if not isinstance(self.x, torch.Tensor) else self.x,
            torch.tensor(self.d, dtype=torch.float32) if not isinstance(self.d, torch.Tensor) else self.d,
            torch.tensor([self.y_min, self.y_max], dtype=torch.float32)
        ]

        # Flatten and concatenate observations
        combined_obs = torch.cat([o.flatten() for o in observations], dim=0)

        # Normalize the state
        normalized_obs = (combined_obs - self.state_means) / self.state_stds

        return normalized_obs.tolist()

    def step(self, action):
        """Take a step in the environment by the given action. Returns the next state, reward, cost, and done flag. """
        self.t += 1

        # Scale the action
        action_to_do = action * self.U_max

        # Convert disturbances and state to numpy arrays if necessary
        if isinstance(self.d, torch.Tensor):
            self.d = self.d.detach().cpu().numpy()
        if self.d.ndim == 1:
            self.d = self.d[np.newaxis, :]

        if isinstance(self.x, torch.Tensor):
            self.x = self.x.detach().cpu().numpy()

        # Simulate the next state
        sim = self.simulator.simulate(nsim=1, x0=self.x, U=action_to_do, D=self.d)
        self.x = torch.tensor(sim["X"].reshape(self.simulator.get_x0().shape[0]), dtype=torch.float32)
        self.y = sim["Y"]

        # Calculate reward and cost
        reward = self.get_reward(action)
        cost = self.get_cost(self.y)

        self.prev_action = action

        # Update disturbance for the next timestep or reset trajectory
        if self.t < self.timesteps:
            self.d = self.trajectory["D"][self.t].unsqueeze(0)
        else:
            self.t = 0
            self.trajectory = self.generate_trajectory()
            self.d = self.trajectory["D"][self.t].unsqueeze(0)

        next_obs = self.build_state()
        return next_obs, reward, cost, False  # The episode does not terminate

    def reset(self):
        """Reset the environment to the initial state. Returns the initial state of the environment."""
        self.trajectory = self.generate_trajectory()
        self.x = self.trajectory["x0"]
        self.y = self.x[3].reshape(1,1)
        self.d = self.trajectory["D"][0]  # Start with the first disturbance
        self.t = 0
        self.prev_action = 0.0
        obs = self.build_state()
        return obs

    def change_refs(self):
        """Update temperature reference bounds."""
        self.y_min = self.def_min_temp + (self.def_max_temp - self.def_min_temp) * np.random.rand()
        self.y_max = self.y_min + 2.0
        return True

    def get_reward(self, action):
        """Calculate the reward based on the action."""
        reward = - (action * 0.01) * self.simulator.umax
        return reward

    def get_cost(self, output):
        """Calculate the cost based on the output."""
        cost = (np.maximum(self.y_min - output, 0) + np.maximum(output - self.y_max, 0)) * 50.0
        return cost


    def generate_trajectory(self):
        """Generate the trajectory for disturbances and initial state."""
        x0 = torch.tensor(self.simulator.get_x0(), dtype=torch.float32)
        D = torch.tensor(self.simulator.get_D(self.timesteps), dtype=torch.float32)

        # Define temperature range randomly within limits
        self.y_min = self.def_min_temp + (self.def_max_temp - self.def_min_temp) * np.random.rand()
        self.y_max = self.y_min + 2.0

        return {"x0": x0, "D": D}

    def compute_state_normalization(self):
        """Compute mean and standard deviation for normalizing state elements based on multiple samples."""
        all_states = []

        for _ in range(self.samples):
            # Generate random initial state and disturbances
            x0 = torch.tensor(self.simulator.get_x0(), dtype=torch.float32)
            D = torch.tensor(self.simulator.get_D(2000), dtype=torch.float32)

            # Define random temperature range
            min_temp = self.def_min_temp + (self.def_max_temp - self.def_min_temp) * np.random.rand()
            max_temp = min_temp + 2.0

            for t in range(2000):
                state = torch.cat([x0, D[t], torch.tensor([min_temp, max_temp], dtype=torch.float32)])
                all_states.append(state)

        all_states = torch.stack(all_states)
        state_means = all_states.mean(dim=0)
        state_stds = all_states.std(dim=0)

        # Avoid zero standard deviations
        state_stds = torch.where(state_stds == 0, torch.tensor(1e-8, dtype=state_stds.dtype), state_stds)

        return state_means, state_stds
