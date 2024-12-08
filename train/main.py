import argparse
import json
from sim.environment import Environment
from sim.renderer import Renderer
from rl.tabular import MonteCarloAgent
from rl.sarsa import SarsaAgent  # Assuming SarsaAgent is defined here
from utils.logger import Logger, LogLevel
from utils.visualization import plot
from utils.misc import gen_session_id


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Blackjack agent using RL.")
    parser.add_argument(
        "--agent", type=str, choices=["tab", "sarsa"], default="tab",
        help="Specify the agent type (tab, sarsa, etc). Default is tab (monte carlo)."
    )
    parser.add_argument("--ep", type=int, default=100_000, help="Number of training episodes.")
    parser.add_argument("--li", type=int, default=1_000, help="How often to log progress.")
    return parser.parse_args()


def initialize_agent(agent_type, logger, **kwargs):
    """Initialize the specified agent."""
    if agent_type == "tab":
        return MonteCarloAgent(
            logger=logger,
            gamma=kwargs.get("gamma", 0.95),
            epsilon_decay=kwargs.get("epsilon_decay", 0.995),
            min_epsilon=kwargs.get("min_epsilon", 0.2)
        )
    elif agent_type == "sarsa":
        return SarsaAgent(
            logger=logger,
            gamma=kwargs.get("gamma", 0.95),
            alpha=kwargs.get("alpha", 0.1),
            epsilon=kwargs.get("epsilon", 1.0),
            epsilon_decay=kwargs.get("epsilon_decay", 0.995),
            min_epsilon=kwargs.get("min_epsilon", 0.2)
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def save_q_table(q_table, output_file):
    """Save the Q-table to a JSON file."""
    q_table_serializable = {str(k): v for k, v in q_table.items()}
    with open(output_file, 'w') as f:
        json.dump(q_table_serializable, f)


def load_q_table(input_file):
    """Load the Q-table from a JSON file."""
    with open(input_file, 'r') as f:
        q_table_serializable = json.load(f)
    return {eval(k): v for k, v in q_table_serializable.items()}  # Use eval to convert string back to tuple


def main():
    args = parse_arguments()

    # Configuration
    EPISODES = getattr(args, 'ep', 100_000)  # Default to 100_000 if not provided
    # LOG_INTERVAL = getattr(args, 'li', 1_000)  # Default to 1_000 if not provided

    # Generate a session ID
    session_id = gen_session_id()

    # Initialize Renderer.
    titles = {
        "tab": "Monte Carlo",
        "sarsa": "Sarsa"
    }
    renderer = Renderer(title=f"{titles[args.agent]} Training")
    renderer.initialize()

    try:
        # Initialize the logger
        logger = Logger(session_id=session_id, log_to_console=False, renderer=renderer)

        # Initialize the environment
        env = Environment(logger=logger, multi_round_mode=True)

        # Initialize the agent
        agent = initialize_agent(
            args.agent,
            logger=logger,
            gamma=0.95,
            epsilon_decay=0.995,
            min_epsilon=0.2,
            alpha=0.1,
            epsilon=1.0
        )

        # Training
        logger.log(LogLevel.INFO, f"Starting {agent.__class__.__name__} training session: {session_id}")
        rewards, wins, losses, draws = agent.train(env, episodes=EPISODES, renderer=renderer)

        # Save the Q-table
        output_file = f"data/models/q_table_{session_id}.json"
        logger.log(LogLevel.INFO, f"Saving Q-table to {output_file}")
        save_q_table(agent.q_table, output_file)

        # Visualizations
        logger.log(LogLevel.INFO, "Generating visualizations...")
        plot(rewards, wins, losses, draws, rolling_window=100, max_points=1000)
        logger.log(LogLevel.INFO, "Visualizations complete.")

    except Exception as e:
        raise e
    finally:
        renderer.cleanup()


if __name__ == "__main__":
    main()
