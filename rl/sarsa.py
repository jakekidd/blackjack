import random
from typing import Dict, Tuple
from sim.environment import Environment
from utils.logger import Logger, LogLevel
from sim.renderer import Renderer

class SarsaAgent:
    def __init__(self, logger: Logger, gamma=0.9, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1):
        """
        Initialize the SARSA agent.

        Args:
            logger (Logger): Logger instance for logging events.
            gamma (float): Discount factor for future rewards.
            alpha (float): Learning rate for Q-value updates.
            epsilon (float): Exploration rate for epsilon-greedy policy.
            epsilon_decay (float): Decay rate for epsilon after each episode.
            min_epsilon (float): Minimum value for epsilon.
        """
        self.logger = logger
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Initialize the Q-table as a dictionary mapping state-action pairs to Q-values.
        # State is represented as (player_total, usable_ace, dealer_card, hand_count, deck_composition).
        self.q_table = {}

    def choose_action(self, state: Dict[str, any]) -> int:
        """
        Choose an action using an epsilon-greedy policy.

        Args:
            state (dict): The current state of the environment.

        Returns:
            int: The chosen action (0 for hit, 1 for stand).
        """
        # Convert the state to a hashable tuple key
        state_key = (
            state["player_total"],
            state["usable_ace"],
            state["dealer_card"],
            state["hand_count"],
            tuple(state.get("deck_composition", []))
        )

        # Initialize Q-values for this state if it doesn't exist.
        if state_key not in self.q_table:
            self.q_table[state_key] = {0: 0.0, 1: 0.0}

        # Epsilon-greedy action selection.
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
        # Initialize Q-values for the next state if not present.
        if next_state not in self.q_table:
            self.q_table[next_state] = {0: 0.0, 1: 0.0}

        # SARSA update rule.
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        self.q_table[state][action] += self.alpha * (reward + self.gamma * next_q - current_q)

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
            "Epsilon": self.epsilon,
        }

        for episode in range(episodes):
            state = env.reset()  # Reset environment.
            state_key = (
                state["player_total"],
                state["usable_ace"],
                state["dealer_card"],
                state["hand_count"],
                tuple(state.get("deck_composition", []))
            )

            action = self.choose_action(state)  # Choose the first action.
            total_reward = 0
            done = False

            while not done:
                next_state, reward, done, _ = env.step(action)  # Take the action.
                total_reward += reward

                next_state_key = (
                    next_state["player_total"],
                    next_state["usable_ace"],
                    next_state["dealer_card"],
                    next_state["hand_count"],
                    tuple(next_state.get("deck_composition", []))
                )

                if not done:
                    next_action = self.choose_action(next_state)  # Choose the next action (on-policy).
                else:
                    next_action = None  # No next action if the episode is over.

                # Update Q-values using SARSA rule.
                self.update_q_table(state_key, action, reward, next_state_key, next_action if next_action is not None else 0)

                # Transition to the next state and action.
                state_key = next_state_key
                action = next_action

            # Track statistics.
            if total_reward > 0:
                wins += 1
                stats["Wins"] = wins
            elif total_reward < 0:
                losses += 1
                stats["Losses"] = losses
            else:
                draws += 1
                stats["Draws"] = draws

            stats["Epsilon"] = self.epsilon
            rewards.append(total_reward)

            # Decay epsilon.
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            self.logger.log(
                LogLevel.INFO,
                f"Episode {episode + 1}/{episodes} complete. Reward: {total_reward}, Epsilon: {self.epsilon:.4f}"
            )

            # Render statistics if renderer is provided.
            if renderer:
                renderer.render(episode + 1, episodes, stats, [])

        return rewards, wins, losses, draws
