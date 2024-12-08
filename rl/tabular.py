import random
from sim.environment import Environment
from sim.renderer import Renderer
from utils.logger import Logger, LogLevel

class MonteCarloAgent:
    def __init__(self, logger: Logger, gamma=0.9, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1):
        """
        Initialize the Monte Carlo agent.

        Args:
            logger (Logger): Logger instance for logging events.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Exploration rate for epsilon-greedy policy.
            epsilon_decay (float): Decay rate for epsilon after each episode.
            min_epsilon (float): Minimum value for epsilon.
        """
        self.logger = logger
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Initialize the Q-table as a dictionary mapping state-action pairs to Q-values.
        # The state is a tuple (player_total, usable_ace, dealer_card, hand_count, deck_composition).
        self.q_table = {}

        # Store experiences for Monte Carlo updates.
        self.episode = []

    def choose_action(self, state):
        """
        Choose an action using an epsilon-greedy policy.

        Args:
            state (dict): The current state of the environment.

        Returns:
            int: The chosen action (0 for hit, 1 for stand).
        """
        # Unpack state values.
        player_total = state["player_total"]
        usable_ace = state["usable_ace"]
        dealer_card = state["dealer_card"]
        hand_count = state["hand_count"]
        # Include deck composition only if it exists.
        deck_composition = tuple(state.get("deck_composition", []))

        # Update state key to include new inputs.
        state_key = (player_total, usable_ace, dealer_card, hand_count, deck_composition)

        # Ensure the state exists in the Q-table.
        if state_key not in self.q_table:
            self.q_table[state_key] = {0: 0.0, 1: 0.0}

        # Epsilon-greedy action selection.
        if random.random() < self.epsilon:
            # Explore: choose a random action.
            action = random.choice([0, 1])
            self.logger.log(LogLevel.DEBUG, f"Exploration: Chose random action {action} for state {state_key}.")
        else:
            # Exploit: choose the action with the highest Q-value.
            action = max(self.q_table[state_key], key=self.q_table[state_key].get)
            self.logger.log(LogLevel.DEBUG, f"Exploitation: Chose action {action} for state {state_key}.")

        return action

    def update_q_table(self):
        """
        Update the Q-table using Monte Carlo policy evaluation.
        """
        # First, calculate returns (cumulative rewards) for each state-action pair in the episode.
        g = 0  # Initialize return (cumulative reward).
        visited_state_actions = set()  # To ensure each state-action pair is only updated once per episode.

        # Process the episode in reverse (backward updates).
        for state, action, reward in reversed(self.episode):
            g = reward + self.gamma * g  # Update return with the discounted reward.
            state_action = (state, action)

            if state_action not in visited_state_actions:
                visited_state_actions.add(state_action)

                # Ensure the state exists in the Q-table.
                if state not in self.q_table:
                    self.q_table[state] = {0: 0.0, 1: 0.0}

                # Update Q-value with incremental mean.
                old_value = self.q_table[state][action]
                self.q_table[state][action] += (g - old_value) / 1  # Update with constant learning (can track counts too).

                self.logger.log(
                    LogLevel.DEBUG,
                    f"Updated Q-value for state {state}, action {action}: old={old_value}, new={self.q_table[state][action]}.")

    def train(self, env: Environment, episodes=1000, renderer: Renderer=None):
        """
        Train the agent using Monte Carlo methods.

        Args:
            env (Environment): The Blackjack environment.
            renderer (Renderer): The curses-based renderer for statistics.
            episodes (int): Number of training episodes.

        Returns:
            Tuple[List[int], int, int, int]: Rewards per episode, wins, losses, draws.
        """
        rewards = []  # List to track rewards per episode.
        wins, losses, draws = 0, 0, 0  # Outcome counters.

        # Initialize stats for the renderer.
        stats = {
            "Wins": 0,
            "Losses": 0,
            "Draws": 0,
            "Hits": 0,
            "Stands": 0,
            "Total Rewards": 0,
            "Epsilon": self.epsilon,
        }

        for episode in range(episodes):
            # Reset environment and start a new episode.
            state = env.reset()
            self.episode = []
            total_reward = 0

            done = False

            while not done:
                # Choose an action using the policy.
                action = self.choose_action(state)

                # Update stats for hits and stands.
                if action == 0:
                    stats["Hits"] += 1
                elif action == 1:
                    stats["Stands"] += 1

                # Take the action in the environment.
                next_state, reward, done, _ = env.step(action)

                # Store the experience in the episode buffer.
                self.episode.append((
                    (state["player_total"], state["usable_ace"], state["dealer_card"],
                    state["hand_count"], tuple(state.get("deck_composition", []))),  # State.
                    action,  # Action.
                    reward  # Reward.
                ))

                # Update the total reward.
                total_reward += reward

                # Move to the next state.
                state = next_state

            # Update statistics for wins, losses, and draws.
            if total_reward > 0:
                wins += 1
                stats["Wins"] = wins
            elif total_reward < 0:
                losses += 1
                stats["Losses"] = losses
            else:
                draws += 1
                stats["Draws"] = draws

            # Update total rewards and epsilon.
            stats["Total Rewards"] = sum(rewards)
            stats["Epsilon"] = self.epsilon

            # Append the total reward for this episode.
            rewards.append(total_reward)

            # Update the Q-table after the episode.
            self.update_q_table()

            # Decay epsilon (reduce exploration over time).
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Render updated stats and logs.
            if renderer:
                renderer.render(episode + 1, episodes, stats, [])

        return rewards, wins, losses, draws
