import curses
from typing import List

class Renderer:
    """
    A simple curses-based renderer to display training statistics.
    """
    def __init__(self):
        self.screen = None

    def initialize(self):
        """Initializes the curses screen."""
        self.screen = curses.initscr()
        curses.start_color()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)  # Hide the cursor.
        self.screen.keypad(True)

        # Initialize color pairs.
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)

    def cleanup(self):
        """Cleans up the curses environment."""
        curses.nocbreak()
        self.screen.keypad(False)
        curses.echo()
        curses.endwin()

    def render(self, episode: int, total_episodes: int, stats: dict, log_tail: List[str]):
        """
        Renders the training statistics and logs.

        Args:
            episode (int): The current episode number.
            total_episodes (int): The total number of episodes.
            stats (dict): A dictionary of statistics to display.
            log_tail (List[str]): The tail of the logs to display.
        """
        self.screen.clear()

        # Title
        self.screen.addstr(0, 2, "Blackjack Training Renderer", curses.A_BOLD | curses.color_pair(4))

        # Progress Bar
        progress = int((episode / total_episodes) * 50)
        progress_bar = "[" + "#" * progress + "-" * (50 - progress) + "]"
        self.screen.addstr(2, 2, f"Progress: {progress_bar} {episode}/{total_episodes}", curses.color_pair(3))

        # Statistics
        self.screen.addstr(4, 2, "Statistics:", curses.A_BOLD | curses.color_pair(2))
        row = 5
        for key, value in stats.items():
            self.screen.addstr(row, 4, f"{key}: {value}", curses.color_pair(1 if key == "Losses" else 2))
            row += 1

        # Log Tail
        self.screen.addstr(row + 1, 2, "Log Tail:", curses.A_BOLD | curses.color_pair(4))
        for i, log in enumerate(log_tail[-5:]):  # Display the last 5 logs.
            self.screen.addstr(row + 2 + i, 4, log, curses.color_pair(3))

        self.screen.refresh()

# Example usage
if __name__ == "__main__":
    import time
    import random

    renderer = Renderer()
    try:
        renderer.initialize()

        total_episodes = 100
        stats = {
            "Wins": 0,
            "Losses": 0,
            "Draws": 0,
            "Hits": 0,
            "Stands": 0,
        }

        log_tail = []

        for episode in range(1, total_episodes + 1):
            # Simulate statistics updates
            stats["Wins"] += random.randint(0, 1)
            stats["Losses"] += random.randint(0, 1)
            stats["Draws"] += random.randint(0, 1)
            stats["Hits"] += random.randint(0, 5)
            stats["Stands"] += random.randint(0, 5)

            # Simulate log updates
            log_tail.append(f"Episode {episode}: Random event log.")

            # Render the screen
            renderer.render(episode, total_episodes, stats, log_tail)

            time.sleep(0.1)

    finally:
        renderer.cleanup()