import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from neuromancer import psl
# Using Cuda if Available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_policy(policy, env):
    """Evaluate the policy on the given environment."""
    obs = env.reset()
    episode_reward = 0
    episode_cost = 0

    while env.t != env.timesteps - 1:
        action = policy.select_action(obs)
        next_obs, reward, cost, done = env.step(action)
        obs = next_obs
        episode_reward += reward
        episode_cost += cost

        if env.t % 500 == 0:
            env.change_refs()

    return episode_reward, episode_cost

def train_agent(agent, replay_buffer, batch_size, iterations, discount, policy_freq):
    """Train the agent using the replay buffer."""
    if len(replay_buffer.storage) > batch_size:
        print("Agent is training...")
        agent.train(replay_buffer, iterations=iterations, batch_size=batch_size, discount=discount, policy_freq=policy_freq)

def training_loop(env, agent, replay_buffer, exploration, expl_noise, batch_size, num_steps, iterations, discount, policy_freq):
    """Main training loop for RL agent."""

    # Initialize variables
    obs = env.reset()
    total_ts = 0  # Total timesteps
    eval_reward_hist = []
    eval_cost_hist = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    while total_ts < num_steps:
        # Select action based on exploration phase
        action = (
            env.action_space.sample().reshape(1, 1)
            if total_ts < exploration
            else agent.select_action(obs) +
            (np.random.normal(0, expl_noise, size=env.action_space.shape[0]) if expl_noise != 0 else 0)
        ).clip(env.action_space.low, env.action_space.high)

        # Take a step in the environment
        next_obs, reward, cost, done = env.step(action)

        # Log agent’s lambda parameter
        current_lambda = agent.lambda_.cpu().detach().numpy()[0]

        # Store the transition in the replay buffer
        agent.store_transition(replay_buffer, obs, next_obs, action.reshape(1,), reward, cost, done)

        # Update the observation
        obs = next_obs

        # Train the agent periodically
        if total_ts >= 3000 and total_ts % 200 == 0:
            train_agent(agent, replay_buffer, batch_size, iterations, discount, policy_freq)

        # Change environment references periodically
        if (total_ts + 1) % 200 == 0:
            env.change_refs()
            print(f"References reset at step {total_ts}.", "Agent's Lambda: ", current_lambda)

        # Evaluate policy and reset the environment periodically
        if (total_ts + 1) % env.timesteps == 0:
            print("Resetting the environment.")
            expl_noise = max(0.1, expl_noise * (1 - total_ts / num_steps))
            eval_reward, eval_cost = evaluate_policy(agent, env)
            print("Episodic reward in evaluation: ", eval_reward, "Episodic cost in evaluation: ", eval_cost)
            eval_cost_hist.append(eval_cost)
            eval_reward_hist.append(eval_reward)

            obs = env.reset()

            # Save the agent with a unique filename
            filename = f"learned_{timestamp}"
            agent.save(filename=filename, directory="./model")

        # Increment timestep counter
        total_ts += 1

    print("Training completed.")
    return eval_reward_hist, eval_cost_hist


def evaluate_consecutive_episodes(env, agent, horizon, min_temp, max_temp, random_seed=10):
    """Evaluate the agent in consecutive episodes and return evaluation history and plots."""

    # Initialize environment and evaluation histories
    eval_hist_rewards = []
    eval_hist_cost = []
    eval_hist_actions = []
    eval_hist_outputs = []
    eval_hist_ymin = []
    eval_hist_ymax = []
    eval_hist_dist = []
    obs_hist = []

    env.reset()
    np_refs = psl.signals.step(horizon + 2, 1, min=min_temp, max=max_temp, randsteps=5)
    env.y_min = np_refs[0]
    env.y_max = env.y_min + 2
    eval_hist_ymin.append(env.y_min)
    eval_hist_ymax.append(env.y_max)
    obs = env.build_state()
    eval_hist_outputs.append(env.y)

    for _ in range(horizon):
        action = agent.select_action(obs)
        next_obs, reward, cost, done = env.step(action)

        # Log step details
        eval_hist_rewards.append(reward)
        eval_hist_cost.append(cost)
        eval_hist_actions.append(action)
        eval_hist_outputs.append(env.y)
        eval_hist_ymin.append(env.y_min)
        eval_hist_ymax.append(env.y_max)
        denormalized_obs = torch.tensor(obs) * env.state_stds + env.state_means
        obs_hist.append(denormalized_obs)

        env.y_min = np_refs[_ + 1]
        env.y_max = env.y_min + 2

        obs = next_obs

    outputs = {"x0": env.trajectory["x0"], "D": env.trajectory["D"][0:horizon], "ref": np_refs, "y": eval_hist_outputs,
               "rewards": eval_hist_rewards, "costs": eval_hist_cost, "actions": eval_hist_actions,
               "obs_hist": obs_hist}
    return outputs


import matplotlib.pyplot as plt
import numpy as np
def plot(eval_hist_outputs, eval_hist_ymin, eval_hist_ymax, eval_hist_actions, eval_hist_dist, trajectories):
    # Generate plots
    plt.figure(figsize=(20, 5))
    plt.plot([i[0][0] for i in eval_hist_outputs], label="Controlled Zone Temperature (RL)", color='blue', linestyle='-',
             linewidth=6, alpha=0.5)

    plt.plot([i[0].cpu().detach() for i in trajectories['y'][0]], label="Controlled Zone Temperature (DPC)", color='red', linestyle='-',
             linewidth=6 , alpha=0.5)


    x = np.arange(len(eval_hist_ymin) - 1)
    plt.step(x, np.ravel(eval_hist_ymin[:-1]), where='post', label="Y Min", color='green', linestyle='--')
    plt.step(x, np.ravel(eval_hist_ymax[:-1]), where='post', label="Y Max", color='red', linestyle='--')
    plt.fill_between(x, np.ravel(eval_hist_ymin[:-1]), np.ravel(eval_hist_ymax[:-1]),
                     where=(np.ravel(eval_hist_ymax[:-1])) > np.ravel(eval_hist_ymin[:-1]),
                     interpolate=True, color='yellow', alpha=0.3, label='Temperature Valid Region')
    plt.xlabel("Index", fontsize=10)
    plt.ylabel("Values", fontsize=10)
    plt.title("Evaluation of DPC and RL models", fontsize=20)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()

    plt.figure(figsize=(20, 5))
    plt.plot([(i[0][0]) * 5000.0 for i in eval_hist_actions],
         label="RL Actions", color='blue',
         linestyle='-', linewidth=6, alpha=0.5)

    plt.plot([i.cpu().detach() for i in trajectories['u'][0]],
             label="DPC Actions", color='red',
             linestyle='-', linewidth=6, alpha=0.5)



    plt.ylim(0, 5000)
    plt.xlabel("Index", fontsize=14)
    plt.ylabel("Actions", fontsize=14)
    plt.title("Actions", fontsize=20)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()

    plt.figure(figsize=(20, 5))
    plt.plot([i[0] for i in eval_hist_dist], label="Outdoor Air Temperature", color='red', linestyle='-',linewidth=6)
    plt.plot([i[1] for i in eval_hist_dist], label="Occupant Heat Load", color='green', linestyle='-',linewidth=6)
    plt.plot([i[2] for i in eval_hist_dist], label="Solar Radiation", color='blue', linestyle='-',linewidth=6)
    plt.xlabel("Index", fontsize=14)
    plt.ylabel("Disturbances", fontsize=14)
    plt.title("Evaluation History Disturbances", fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()

    plt.show()
