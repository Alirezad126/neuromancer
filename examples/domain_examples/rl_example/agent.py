import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Using Cuda if Available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Defining Actor, Critic, and Cost Networks:
# Actor Network:
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, activation=nn.ReLU, output_activation=nn.Sigmoid):

        super(Actor, self).__init__()

        layers = []
        input_dim = state_dim

        # Create hidden layers with activation
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation is not None:
                layers.append(activation())
            input_dim = hidden_dim

        # Output layer with optional activation
        layers.append(nn.Linear(input_dim, action_dim))
        if output_activation is not None:
            layers.append(output_activation())

        # Combine layers into a Sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, state):

        return self.model(state)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, activation=nn.ReLU):

        super(Critic, self).__init__()
        layers = []
        input_dim = state_dim + action_dim

        # Create hidden layers with activation functions
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())  # Add activation after each layer
            input_dim = hidden_dim

        # Output layer (no activation here)
        layers.append(nn.Linear(input_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=-1)  # Concatenate state and action
        return self.layers(x)



class PD_DDPG:
    def __init__(self, state_dim, action_dim, max_action, lambda_, lambda_step, constraint_limit, lr_critic, lr_actor, hidden_layers):
        """
        Initialize the PD-DDPG agent.

        Args:
            state_dim (int): Dimension of state space.
            action_dim (int): Dimension of state space.
            max_action (float): Maximum action value.
            lambda_ (float): Initial dual variable for the constraint.
            lambda_step (float): Step size for updating lambda.
            constraint_limit (float): Constraint tolerance limit.
            lr_critic (float): Learning rate for the critic.
            lr_actor (float): Learning rate for the actor.
            hidden_layers (list): List of integers specifying the number of neurons in each hidden layer.
        """

        # Actor network and optimizer
        self.actor = Actor(state_dim, action_dim, hidden_layers).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_layers).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic network and optimizer
        self.critic = Critic(state_dim, action_dim, hidden_layers).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_layers).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Cost network and optimizer
        self.cost = Critic(state_dim, action_dim, hidden_layers).to(device)
        self.cost_target = Critic(state_dim, action_dim, hidden_layers).to(device)
        self.cost_target.load_state_dict(self.cost.state_dict())
        self.cost_optimizer = torch.optim.Adam(self.cost.parameters(), lr=lr_critic)

        # Lambda and constraint
        self.lambda_ = torch.tensor([lambda_], requires_grad=False).to(device)
        self.lambda_step = lambda_step
        self.constraint_limit = constraint_limit

        self.max_action = max_action


    def store_transition(self, replay_buffer, state, next_state, action, reward, cost, done):
        """
        Normalizing the transition and storing it in the replay buffer.

        """
        # Update the normalizer
        state = torch.tensor(state, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        cost = torch.tensor(cost, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device)


        # Store in the replay buffer
        replay_buffer.add((state, next_state, action, reward, cost, done))


    def select_action(self, state):
        """
        Select an action for a given state.

        Args:
            state (np.ndarray or list): Current state.

        Returns:
            np.ndarray: Action.
        """
        # Ensure the state is a tensor
        if isinstance(state, list):
            state = torch.tensor(state, dtype=torch.float32)
        elif isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        state = state.to(device).unsqueeze(0)  # Add batch dimension
        action = self.actor(state).cpu().data.numpy().reshape(1, 1) * self.max_action
        return np.clip(action, 0, self.max_action)

    def train(self, replay_buffer, iterations, batch_size=100, discount_Q=0.99, discount_C=0.0, tau=0.005, policy_freq=2):
        """
        Train the PD-DDPG agent.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer.
            iterations (int): Number of training iterations.
            batch_size (int): Batch size.
            discount (float): Discount factor.
            tau (float): Soft update coefficient.
            policy_freq (int): Frequency of policy updates.
        """
        for it in range(iterations):
            # Sample batch of transitions
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_costs, batch_dones = replay_buffer.sample(batch_size)

            # Convert to tensors
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            cost = torch.Tensor(batch_costs).to(device)
            done = torch.Tensor(batch_dones).to(device)
            # Compute target values
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_C = self.cost_target(next_state, next_action)
            target_Q = reward + ((1 - done) * discount_Q * target_Q).detach()
            target_C = cost + ((1 - done) * discount_C * target_C).detach()

            # Update critic
            current_Q = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            #Update cost network
            current_C = self.cost(state, action)
            cost_loss = F.mse_loss(current_C, target_C)
            self.cost_optimizer.zero_grad()
            cost_loss.backward()
            self.cost_optimizer.step()





            # # Delayed policy updates
            if it % policy_freq == 0:

                # Update actor network
                Q_value = self.critic(state, self.actor(state))
                C_value = self.cost(state, self.actor(state))
                actor_loss = -(Q_value - self.lambda_.detach() * C_value).mean() + nn.MSELoss()(self.actor(state), self.actor_target(next_state))
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.cost.parameters(), self.cost_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                #Update lambda
                lambda_gradient = (self.cost(state, self.actor(state)) - self.constraint_limit).mean()
                self.lambda_ = torch.clamp(self.lambda_ + self.lambda_step * lambda_gradient, min=0).detach()

    def save(self, filename, directory):
        """
        Save model weights.
        """
        torch.save(self.actor.state_dict(), f'{directory}/{filename}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{filename}_critic.pth')
        torch.save(self.cost.state_dict(), f'{directory}/{filename}_cost.pth')

    def load(self, filename, directory):
        """
        Load model weights.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        load_path = lambda model_type: torch.load(f'{directory}/{filename}_{model_type}.pth', map_location=device)
    
        self.actor.load_state_dict(load_path('actor'))
        self.critic.load_state_dict(load_path('critic'))
        self.cost.load_state_dict(load_path('cost'))
