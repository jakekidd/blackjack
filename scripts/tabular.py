import json
from sim.environment import Environment
from sim.renderer import Renderer
from rl.tabular import MonteCarloAgent
from utils.logger import Logger, LogLevel
from utils.visualization import plot
from utils.misc import gen_session_id

# Generate a session ID
session_id = gen_session_id()

# Initialize the logger
logger = Logger(session_id=session_id)

# Initialize the environment and the Monte Carlo agent
env = Environment(logger=logger, multi_round_mode=True)
# TODO: command line args ..?
agent = MonteCarloAgent(
    logger=logger,
    gamma=0.95,
    epsilon_decay=0.995,
    min_epsilon=0.2
)

# Configuration
EPISODES = 100_000      # Number of training episodes
LOG_INTERVAL = 1_000    # How often to log progress

# Initialize Renderer
renderer = Renderer()
renderer.initialize()

try:
    # Training
    logger.log(LogLevel.INFO, f"Starting Monte Carlo training session: {session_id}")
    rewards, wins, losses, draws = agent.train(env, episodes=EPISODES, renderer=renderer)
except Exception as e:
    raise e
finally:
    renderer.cleanup()

# Save the Q-table
output_file = f"data/models/q_table_{session_id}.json"
logger.log(LogLevel.INFO, f"Saving Q-table to {output_file}")

# Helper function to save the Q-table
def save_q_table(q_table, output_file):
    """
    Save the Q-table to a JSON file with tuple keys converted to strings.

    Args:
        q_table (dict): The Q-table to save.
        output_file (str): File path to save the Q-table.
    """
    q_table_serializable = {str(k): v for k, v in q_table.items()}
    with open(output_file, 'w') as f:
        json.dump(q_table_serializable, f)
    logger.log(LogLevel.INFO, f"Q-table saved to {output_file}")

# Helper function to load the Q-table
def load_q_table(input_file):
    """
    Load the Q-table from a JSON file, converting string keys back to tuples.

    Args:
        input_file (str): File path to load the Q-table from.

    Returns:
        dict: The reconstructed Q-table.
    """
    with open(input_file, 'r') as f:
        q_table_serializable = json.load(f)
    return {eval(k): v for k, v in q_table_serializable.items()}  # Use eval to convert string back to tuple

# Save the Q-table using the helper function
save_q_table(agent.q_table, output_file)

# Final log message
logger.log(LogLevel.INFO, "Training session complete.")
logger.log(LogLevel.INFO, f"Session ID: {session_id}")

# Visualizations
logger.log(LogLevel.INFO, "Generating visualizations...")
plot(rewards, wins, losses, draws, rolling_window=100, max_points=1000)
logger.log(LogLevel.INFO, "Visualizations complete.")
