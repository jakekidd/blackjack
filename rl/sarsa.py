import random
import math
from collections import defaultdict
from typing import Dict, Tuple
from sim.environment import Environment
from utils.logger import Logger, LogLevel
from sim.renderer import Renderer

class SarsaAgent:
    def __init__(self, logger: Logger, gamma=0.9, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1, use_softmax=True):
        """
        Initialize the SARSA agent.

        Args:
            logger (Logger): Logger instance for logging events.
            gamma (float): Discount factor for future rewards.
            alpha (float): Learning rate for Q-value updates.
            epsilon (float): Exploration rate for epsilon-greedy policy.
            epsilon_decay (float): Decay rate for epsilon after each episode.
            min_epsilon (float): Minimum value for epsilon.
            use_softmax (bool): Whether to use softmax for action selection.
        """
        self.logger = logger
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.use_softmax = use_softmax

        # Initialize the Q-table as a dictionary mapping state-action pairs to Q-values.
        self.q_table = defaultdict(lambda: {0: 0.0, 1: 0.0})
        # Track state-action visit counts
        self.state_action_counts = defaultdict(lambda: {0: 0, 1: 0})

    def choose_action(self, state: Dict[str, any]) -> int:
        """
        Choose an action using an epsilon-greedy or softmax policy.

        Args:
            state (dict): The current state of the environment.

        Returns:
            int: The chosen action (0 for hit, 1 for stand).
        """
        # Convert the state to a hashable tuple key
        state_key = (
            max(state["player_total"], 12),  # Start player sums at 12
            state["usable_ace"],
            state["dealer_card"],
        )

        if self.use_softmax:
            # Softmax action selection
            q_values = self.q_table[state_key]
            exp_values = {a: math.exp(q) for a, q in q_values.items()}
            total = sum(exp_values.values())
            probabilities = {a: exp / total for a, exp in exp_values.items()}
            action = random.choices(list(probabilities.keys()), weights=list(probabilities.values()))[0]
            self.logger.log(LogLevel.DEBUG, f"Softmax: Chose action {action} for state {state_key}.")
        else:
            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                action = random.choice([0, 1])  # Explore
                self.logger.log(LogLevel.DEBUG, f"Exploration: Chose random action {action} for state {state_key}.")
            else:
                action = max(self.q_table[state_key], key=self.q_table[state_key].get)  # Exploit
                self.logger.log(LogLevel.DEBUG, f"Exploitation: Chose action {action} for state {state_key}.")

        return action

    def update_q_table(self, state: Tuple, action: int, reward: float, next_state: Tuple, next_action: int):
        """
        Update the Q-value using the SARSA update rule.

        Args:
            state (tuple): The current state.
            action (int): The action taken in the current state.
            reward (float): The reward received after taking the action.
            next_state (tuple): The next state reached after taking the action.
            next_action (int): The next action to be taken (on-policy).
        """
        self.state_action_counts[state][action] += 1  # Increment visit count
        dynamic_alpha = 1 / (1 + self.state_action_counts[state][action])  # Dynamic learning rate

        # SARSA update rule
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        self.q_table[state][action] += dynamic_alpha * (reward + self.gamma * next_q - current_q)

        self.logger.log(
            LogLevel.DEBUG,
            f"Updated Q-value for state {state}, action {action}: {self.q_table[state][action]:.4f}"
        )

    def train(self, env: Environment, episodes=1000, renderer: Renderer = None):
        """
        Train the agent using the SARSA algorithm.

        Args:
            env (Environment): The Blackjack environment.
            episodes (int): Number of training episodes.
            renderer (Renderer): Renderer instance for displaying stats.

        Returns:
            Tuple[List[int], int, int, int]: Rewards per episode, wins, losses, draws.
        """
        rewards = []  # Track rewards per episode.
        wins, losses, draws = 0, 0, 0  # Outcome counters.

        stats = {
            "Wins": 0,
            "Losses": 0,
            "Draws": 0,
            "Win Rate": 0.0,
            "Epsilon": self.epsilon,
            "Total Rewards": 0,
        }
        snapshots = []

        for episode in range(episodes):
            state = env.reset()
            state_key = (
                max(state["player_total"], 12),  # Start player sums at 12
                state["usable_ace"],
                state["dealer_card"],
            )
            action = self.choose_action(state)  # Choose the first action.
            total_reward = 0
            done = False

            # Add reward shaping logic
            # if not done:
            #     if state["player_total"] >= 17:
            #         reward -= 0.5  # Penalize risky hits
            #     if state["player_total"] < 21:
            #         reward += 0.1 * (21 - state["player_total"])  # Reward getting closer to 21

            while not done:
                next_state, reward, done, _ = env.step(action)  # Take the action.
                total_reward += reward

                next_state_key = (
                    max(next_state["player_total"], 12),  # Start player sums at 12
                    next_state["usable_ace"],
                    next_state["dealer_card"],
                )

                if not done:
                    next_action = self.choose_action(next_state)  # Choose the next action (on-policy).
                else:
                    next_action = 0

                # Update Q-values using SARSA rule.
                self.update_q_table(state_key, action, reward, next_state_key, next_action)

                # Transition to the next state and action.
                state_key = next_state_key
                action = next_action

            # Track game outcome for wins/losses/draws.
            if env.player_hand and env.dealer_hand:
                player_total, _ = env._hand_value(env.player_hand)
                dealer_total, _ = env._hand_value(env.dealer_hand)

                if player_total > 21:  # Player busts
                    losses += 1
                elif dealer_total > 21 or player_total > dealer_total:  # Dealer busts or player wins
                    wins += 1
                elif player_total < dealer_total:  # Dealer wins
                    losses += 1
                else:  # Draw
                    draws += 1

            stats["Wins"] = wins
            stats["Losses"] = losses
            stats["Draws"] = draws
            stats["Win Rate"] = wins / (episode + 1)
            stats["Epsilon"] = self.epsilon
            rewards.append(total_reward)
            stats["Total Rewards"] += total_reward

            # Decay epsilon dynamically.
            # self.epsilon = max(self.min_epsilon, self.epsilon * (1 - (episode / episodes) ** 2))
            # Decay epsilon.
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            # self.epsilon = min(1.0 / episode + 1, self.min_epsilon)
            # Slower epsilon decay early on, with a controllable factor `decay_factor`
            # decay_factor = 0.1  # Adjust this value to control how slowly epsilon decays early on
            # progress = episode / episodes  # Calculate progress as a fraction of total episodes
            # self.epsilon = max(self.min_epsilon, self.epsilon * (self.epsilon_decay ** (1 - decay_factor * progress)))
            # Non-linear exploration decay
            # progress = episode / episodes
            # self.epsilon = max(self.min_epsilon, self.epsilon * (1 - progress ** 2))

            # Decay learning rate dynamically.
            self.alpha = max(0.01, self.alpha * 0.99)

            if episode % 100 == 0 or episode == episodes - 1:
                snapshots.append(stats.copy())
                self.logger.log(
                    LogLevel.INFO,
                    f"Episode {episode + 1}/{episodes} complete. Reward: {total_reward}, Epsilon: {self.epsilon:.4f}, Alpha: {self.alpha:.4f}"
                )
                # Render statistics if renderer is provided.
                if renderer:
                    renderer.render(episode + 1, episodes, stats, [])

        return rewards, wins, losses, draws, {"snapshots": snapshots}
