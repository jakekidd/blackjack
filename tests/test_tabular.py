import unittest
from unittest.mock import MagicMock
from rl.tabular import MonteCarloAgent
from sim.environment import Environment
from utils.logger import Logger, LogLevel

class TestMonteCarloAgent(unittest.TestCase):

    def setUp(self):
        """
        Set up a mock logger and initialize the Monte Carlo agent and environment for testing.
        """
        self.logger = Logger(session_id="test_session", log_to_console=False)
        self.logger.log = MagicMock()  # Mock the logger to suppress output during tests
        self.env = Environment(logger=self.logger)
        self.agent = MonteCarloAgent(logger=self.logger, gamma=0.9, epsilon=0.5)

    def test_choose_action_exploration(self):
        """
        Test that the agent explores correctly based on epsilon.
        """
        state = {"player_total": 15, "usable_ace": True, "dealer_card": 10}
        self.agent.epsilon = 1.0  # Force exploration

        actions = [self.agent.choose_action(state) for _ in range(100)]
        self.assertIn(0, actions)  # Ensure "hit" is chosen
        self.assertIn(1, actions)  # Ensure "stand" is chosen
        self.logger.log.assert_called()  # Check that logging happens

    def test_choose_action_exploitation(self):
        """
        Test that the agent exploits the highest Q-value when epsilon is low.
        """
        state = {"player_total": 15, "usable_ace": True, "dealer_card": 10}
        state_key = (15, True, 10)

        # Initialize Q-values for testing
        self.agent.q_table[state_key] = {0: 0.5, 1: 0.8}  # Q(hit) = 0.5, Q(stand) = 0.8
        self.agent.epsilon = 0.0  # Force exploitation

        action = self.agent.choose_action(state)
        self.assertEqual(action, 1)  # Agent should choose "stand" (action 1)
        self.logger.log.assert_called()  # Check that logging happens

    def test_update_q_table(self):
        """
        Test that the Q-table updates correctly using Monte Carlo returns.
        """
        # Set up a fake episode
        self.agent.episode = [
            ((15, True, 10), 0, -1),  # State, action, reward
            ((18, False, 10), 1, 1)
        ]

        # Initialize Q-values for testing
        self.agent.q_table = {
            (15, True, 10): {0: 0.0, 1: 0.0},
            (18, False, 10): {0: 0.0, 1: 0.0}
        }

        self.agent.update_q_table()

        # Check updated Q-values
        self.assertNotEqual(self.agent.q_table[(15, True, 10)][0], 0.0)
        self.assertNotEqual(self.agent.q_table[(18, False, 10)][1], 0.0)
        self.logger.log.assert_called()  # Check that logging happens

    def test_train(self):
        """
        Test the train method with the environment.
        """
        self.agent.train(self.env, episodes=10)  # Train for 10 episodes

        # Ensure Q-table is populated
        self.assertTrue(len(self.agent.q_table) > 0)
        self.logger.log.assert_called_with(LogLevel.INFO, "Training complete.")  # Check training completion log

if __name__ == "__main__":
    unittest.main()
