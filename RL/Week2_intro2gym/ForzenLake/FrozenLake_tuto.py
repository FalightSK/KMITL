# Import all neccessary Enviroment into the code
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Set the seaborn theme for creating Graph
sns.set_theme()

# Define Parameters object to training
class Params(NamedTuple):
    total_episodes: int  # The total interation of training (epoch)
    learning_rate: float  # Scale of value to update in Q table
    gamma: float  # Discounting rate | distribute action probability
    epsilon: float  # Probability to choosing many actions
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Determine random behavior (Make it the same everytime when runing code)
    is_slippery: bool  # If True, there will be noise in the system with prob of 1/3 the slip in the side way
    n_runs: int  # Like a batch size, determining how many tries in one iteration
    action_size: int  # Number of possible actions
    state_size: int # The number of states which model can encounter at a time
    proba_frozen: float # Prob of tile being forzen
    savefig_folder: Path # Saving Path

# This code using Q-learning algorithm to handle with action and policy

params = Params(
    total_episodes=100,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=4,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
    savefig_folder=Path("C:/Users/AIxRAI_08/Desktop/GYM/Week2/static"),
)


rng = np.random.default_rng(params.seed)

params.savefig_folder.mkdir(parents=True, exist_ok=True)


env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    render_mode="rgb_array",
    desc=generate_random_map(
        size=params.map_size, p=params.proba_frozen, seed=params.seed
    ),
)

params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)


class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, action_space, state, qtable):
        explor_exploit_tradeoff = rng.uniform(0, 1)

        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        else:
            if np.all(qtable[state, :]) == qtable[state, 0]:
                action = action_space.sample()
            else:
                action = np.argmax(qtable[state, :])
        return action


def run_env():
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.total_episodes, params.state_size, params.action_size))
    all_states = []
    all_actions = []
    avg_q_values = []

    for run in range(params.n_runs):
        learner.reset_qtable()

        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):
            state = env.reset(seed=params.seed)[0]
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space, state=state, qtable=learner.qtable
                )

                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                new_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state
                )

                total_rewards += reward
                step += 1

                state = new_state

            rewards[episode, run] = total_rewards
            steps[episode, run] = step

            # Store Q-table for this episode
            qtables[episode] = learner.qtable.copy()

            # Calculate and store average Q-value after each episode
            avg_q_value = np.mean(learner.qtable)
            avg_q_values.append(avg_q_value)

            # Print average Q-value after each episode
            print(f"Episode {episode}: Average Q-value = {avg_q_value:.3f}")

            # Print Q-table after each episode for 4x4 environment
            if params.map_size == 4 and episode == params.total_episodes - 1:
                print("\nQ-table after episode:")
                qtable_df = pd.DataFrame(learner.qtable)
                print(qtable_df)

                # Save Q-table as CSV file
                qtable_filename = params.savefig_folder / f"qtable_all_episodes.csv"
                qtable_df.to_csv(qtable_filename, index=False)

    return rewards, steps, episodes, qtables, all_states, all_actions, avg_q_values


def postprocess(episodes, params, rewards, steps, map_size):
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten(),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st


def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty((map_size, map_size), dtype=str)
    
    for i in range(map_size):
        for j in range(map_size):
            action = qtable_best_action[i, j]
            qvalue = qtable_val_max[i, j]
            if qvalue > 0:
                qtable_directions[i, j] = directions[action]
            else:
                qtable_directions[i, j] = ''
    
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, env, map_size):
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title=f"Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"frozenlake_q_values_last_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    
    # Return the figure object
    return fig


def plot_states_actions_distribution(states, actions, map_size):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    
    # Display the image and stop execution
    plt.show()


map_sizes = [4]
res_all = pd.DataFrame()
st_all = pd.DataFrame()

for map_size in map_sizes:
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=params.is_slippery,
        render_mode="rgb_array",
        desc=generate_random_map(
            size=map_size, p=params.proba_frozen, seed=params.seed
        ),
    )

    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)
    env.action_space.seed(
        params.seed
    )
    learner = Qlearning(
        learning_rate=params.learning_rate,
        gamma=params.gamma,
        state_size=params.state_size,
        action_size=params.action_size,
    )
    explorer = EpsilonGreedy(
        epsilon=params.epsilon,
    )

    print(f"Map size: {map_size}x{map_size}")
    rewards, steps, episodes, qtables, all_states, all_actions, avg_q_values = run_env()

    # Save the results in dataframes
    res, st = postprocess(episodes, params, rewards, steps, map_size)
    res_all = pd.concat([res_all, res], ignore_index=True)
    st_all = pd.concat([st_all, st], ignore_index=True)

    # Save Q-table data for all episodes in a single CSV file
    qtable_all_episodes_df = pd.DataFrame(np.reshape(qtables, (params.total_episodes, -1)))
    qtable_all_episodes_df.to_csv(params.savefig_folder / f"qtable_all_episodes_{map_size}x{map_size}.csv", index=False)

    # Plot Q-values after the last episode
    plot_q_values_map(qtables[-1], env, map_size)

    plot_states_actions_distribution(all_states, all_actions, map_size)


# Display cumulative results and stop execution
print(res_all)
print(st_all)
