import unittest
from unittest.mock import MagicMock
from utils.logger import Logger, LogLevel
from sim.environment import Environment

class TestEnvironment(unittest.TestCase):

    def setUp(self):
        """Set up a mock logger and initialize the environment for testing."""
        self.logger = Logger(session_id="test_session", log_to_console=False)
        self.logger.log = MagicMock()  # Mock the log method to avoid actual logging
        self.env = Environment(logger=self.logger)

    def test_environment_initialization(self):
        """Test that the environment initializes correctly."""
        state = self.env.reset()
        self.assertIn("player_total", state)
        self.assertIn("usable_ace", state)
        self.assertIn("dealer_card", state)
        self.assertFalse(self.env.done)
        self.assertEqual(self.env.reward, 0)

    def test_reset_functionality(self):
        """Test that reset initializes a new game state."""
        initial_state = self.env.reset()
        self.assertIsNotNone(initial_state)
        self.assertEqual(len(self.env.player_hand), 2)
        self.assertEqual(len(self.env.dealer_hand), 2)
        self.assertFalse(self.env.done)
        self.assertEqual(self.env.reward, 0)

    def test_hit_action(self):
        """Test the hit action in the environment."""
        self.env.reset()
        initial_hand_size = len(self.env.player_hand)
        state, reward, done, _ = self.env.step(0)  # Hit action
        self.assertEqual(len(self.env.player_hand), initial_hand_size + 1)
        self.assertFalse(done)
        self.assertEqual(reward, 0)

    def test_bust_on_hit(self):
        """Test that the player loses if they bust after hitting."""
        self.env.reset()
        # Force the player's hand to a value that will bust on a hit
        self.env.player_hand = [10, 10]
        state, reward, done, _ = self.env.step(0)  # Hit action
        self.assertTrue(done)
        self.assertEqual(reward, -1)

    def test_stand_action(self):
        """Test the stand action in the environment."""
        self.env.reset()
        state, reward, done, _ = self.env.step(1)  # Stand action
        self.assertTrue(done)
        self.assertIn(reward, [-1, 0, 1])  # Reward could be win, loss, or draw

    def test_dealer_behavior(self):
        """Test that the dealer hits until reaching 17 or higher."""
        self.env.reset()
        self.env.player_hand = [10, 10]  # Ensure the player stands immediately
        self.env.dealer_hand = [5, 2]  # Force the dealer to draw
        _, _, _, _ = self.env.step(1)  # Stand action
        dealer_total, _ = self.env._hand_value(self.env.dealer_hand)
        self.assertGreaterEqual(dealer_total, 17)

    def test_logging_calls(self):
        """Test that logging is invoked correctly during environment operations."""
        self.env.reset()
        self.logger.log.assert_called_with(LogLevel.INFO, "Resetting the environment for a new game.")
        self.env.step(0)  # Hit action
        self.logger.log.assert_any_call(LogLevel.INFO, "Action taken: Hit")
        self.env.step(1)  # Stand action
        self.logger.log.assert_any_call(LogLevel.INFO, "Action taken: Stand")

    def test_logging_calls(self):
        """Test that logging is invoked correctly during environment operations."""
        self.env.reset()
        self.logger.log.assert_any_call(LogLevel.INFO, "Resetting the environment for a new game.")
        self.env.step(0)  # Hit action
        self.logger.log.assert_any_call(LogLevel.INFO, "Action taken: Hit")
        self.env.step(1)  # Stand action
        self.logger.log.assert_any_call(LogLevel.INFO, "Action taken: Stand")


if __name__ == "__main__":
    unittest.main()
