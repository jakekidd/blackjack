import curses
from typing import List
import time  # Added for tracking timestamps

class Renderer:
    """
    A simple curses-based renderer to display training statistics.
    """
    def __init__(self, title: str = None):
        self.screen = None
        self.title = title
        self.log_feed = []
        self.start_time = None  # Tracks the start time of the training session
        self.last_episode_time = None  # Tracks the last episode's timestamp
        self.last_episode = 0  # Tracks the last rendered episode

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

        # Initialize start time when the screen is initialized
        self.start_time = time.time()
        self.last_episode_time = self.start_time

    def cleanup(self):
        """Cleans up the curses environment."""
        curses.nocbreak()
        self.screen.keypad(False)
        curses.echo()
        curses.endwin()

    def log(self, line: str):
        """Update the log feed to display tail."""
        if type(line) != type(""):
            raise TypeError("log line not a str", line)
        self.log_feed.append(line)
        if len(self.log_feed) > 5:
            self.log_feed = self.log_feed[1:]

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
        self.screen.addstr(0, 2, "B L A C K   J A C K" if self.title is None else self.title, curses.A_BOLD | curses.color_pair(4))

        # Progress Bar
        progress = int((episode / total_episodes) * 50)
        progress_bar = "[" + "#" * progress + "-" * (50 - progress) + "]"
        self.screen.addstr(2, 2, f"Progress: {progress_bar} {episode}/{total_episodes}", curses.color_pair(3))

        # Estimated Time Remaining
        if episode > self.last_episode:
            current_time = time.time()
            elapsed_since_last = current_time - self.last_episode_time
            episodes_since_last = episode - self.last_episode
            time_per_episode = elapsed_since_last / episodes_since_last
            remaining_time = (total_episodes - episode) * time_per_episode

            # Update timestamps and episode count
            self.last_episode_time = current_time
            self.last_episode = episode

            minutes, seconds = divmod(int(remaining_time), 60)
            self.screen.addstr(3, 2, f"Estimated Time Remaining: {minutes}m {seconds}s", curses.color_pair(2))

        # Statistics
        self.screen.addstr(5, 2, "Statistics:", curses.A_BOLD | curses.color_pair(2))
        row = 6
        for key, value in stats.items():
            self.screen.addstr(row, 4, f"{key}: {value}", curses.color_pair(1 if key == "Losses" else 2))
            row += 1

        # Log Tail
        self.screen.addstr(row + 1, 2, "Log Tail:", curses.A_BOLD | curses.color_pair(4))
        # TODO: Logs come from outer consumer.
        log_tail = self.log_feed
        for i, log in enumerate(log_tail[-5:]):  # Display the last 5 logs.
            self.screen.addstr(row + 2 + i, 4, log, curses.color_pair(3))

        self.screen.refresh()
