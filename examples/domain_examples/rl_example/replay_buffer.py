import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        """
        Initialize the replay buffer.

        Args:
            max_size (int): Maximum number of transitions to store in the buffer.
        """
        self.storage = deque(maxlen=int(max_size))

    def add(self, transition):
        """
        Add a transition to the replay buffer.

        Args:
            transition (tuple): A tuple containing (state, next_state, action, reward, cost).
        """
        self.storage.append(transition)

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: Batch of (states, next_states, actions, rewards, costs).
        """
        indices = torch.randint(0, len(self.storage), (batch_size,))
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_costs, batch_dones = [], [], [], [], [], []

        for i in indices:
            state, next_state, action, reward, cost, done= self.storage[i]
            batch_states.append(state)
            batch_next_states.append(next_state)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_costs.append(cost)
            batch_dones.append(done)

        return (
            torch.stack(batch_states),
            torch.stack(batch_next_states),
            torch.stack(batch_actions),
            torch.tensor(batch_rewards, dtype=torch.float32).view(-1, 1),
            torch.tensor(batch_costs, dtype=torch.float32).view(-1, 1),
            torch.tensor(batch_dones, dtype=torch.float32).view(-1, 1),

        )
