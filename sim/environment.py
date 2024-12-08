import random
from typing import Tuple, Dict, Any, List, Optional
from utils.logger import Logger, LogLevel

class Environment:
    def __init__(self, logger: Logger, multi_round_mode=False, stand_penalty_decay=0.001):
        """
        Initialize the Environment.

        Args:
            logger (Logger): Logger instance for logging events.
            multi_round_mode (bool): Whether to enable multi-round games with deck tracking.
            stand_penalty_decay (float): The decay factor \(\phi\) for the stand penalty over episodes.
        """
        self.logger = logger
        self.logger.log(LogLevel.DEBUG, "Initializing the environment.")

        self.multi_round_mode = multi_round_mode
        self.stand_penalty_decay = stand_penalty_decay
        self.stand_penalty = 0.1  # Initial penalty for standing below total 13.

        self.deck = self._create_deck()
        self.player_hand = []
        self.dealer_hand = []
        self.done = False
        self.reward = 0

    def _create_deck(self):
        """Create and shuffle a standard deck of cards."""
        self.logger.log(LogLevel.DEBUG, "Creating and shuffling a new deck.")
        deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        random.shuffle(deck)
        return deck

    def _draw_card(self):
        """Draw a card from the deck."""
        if not self.deck:
            if self.multi_round_mode:
                self.logger.log(LogLevel.INFO, "Deck depleted. Resetting for new rounds.")
                self.deck = self._create_deck()
            else:
                self.logger.log(LogLevel.WARN, "Deck is empty. Reshuffling.")
                self.deck = self._create_deck()
        card = self.deck.pop()
        self.logger.log(LogLevel.DEBUG, f"Drew card: {card}")
        return card

    def _hand_value(self, hand: list) -> Tuple[int, bool]:
        """
        Calculate the total value of a hand.

        Args:
            hand (list): List of card values in the hand.

        Returns:
            Tuple[int, bool]: Total value of the hand and whether it has a usable ace.
        """
        total = sum(hand)
        usable_ace = 1 in hand and total + 10 <= 21
        if usable_ace:
            total += 10
        self.logger.log(LogLevel.DEBUG, f"Calculated hand value: {total}, usable ace: {usable_ace}")
        return total, usable_ace

    def _deck_composition(self) -> Dict[int, int]:
        """
        Calculate the current deck composition.

        Returns:
            Dict[int, int]: A dictionary where keys are card values and values are remaining counts.
        """
        composition = {value: self.deck.count(value) for value in set(self.deck)}
        self.logger.log(LogLevel.DEBUG, f"Deck composition: {composition}")
        return composition

    def reset(self) -> Dict[str, Any]:
        """Reset the environment for a new episode."""
        self.logger.log(LogLevel.DEBUG, "Resetting the environment for a new game.")
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card()]
        self.done = False
        self.reward = 0

        player_total, usable_ace = self._hand_value(self.player_hand)
        dealer_card = self.dealer_hand[0]
        hand_count = len(self.player_hand)

        state = {
            "player_total": player_total,
            "usable_ace": usable_ace,
            "dealer_card": dealer_card,
            "hand_count": hand_count
        }

        if self.multi_round_mode:
            state["deck_composition"] = self._deck_composition()

        self.logger.log(LogLevel.DEBUG, f"Initial state: {state}")
        return state

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, None]:
        """
        Perform an action: 0 = Hit, 1 = Stand.

        Args:
            action (int): Action to take (0 for hit, 1 for stand).

        Returns:
            Tuple[Dict[str, Any], float, bool, None]:
                - Updated state
                - Reward
                - Whether the episode is done
                - Additional info (None for now)
        """
        self.logger.log(LogLevel.DEBUG, f"Action taken: {'Hit' if action == 0 else 'Stand'}")

        if self.done:
            self.logger.log(LogLevel.ERROR, "Attempted to step in a completed episode. Reset the environment.")
            raise ValueError("Episode has ended. Please reset the environment.")

        if action == 0:  # Hit
            self.player_hand.append(self._draw_card())
            player_total, usable_ace = self._hand_value(self.player_hand)
            if player_total > 21:
                self.logger.log(LogLevel.DEBUG, "Player busts. Game over.")
                self.done = True
                self.reward = -1
            else:
                self.reward = 0

        elif action == 1:  # Stand
            player_total, _ = self._hand_value(self.player_hand)
            if player_total < 13:
                penalty = self.stand_penalty * (13 - player_total)
                self.reward -= penalty
                self.logger.log(LogLevel.INFO, f"Applied stand penalty: {penalty:.2f}.")

            self.done = True
            dealer_total, _ = self._hand_value(self.dealer_hand)

            self.logger.log(LogLevel.DEBUG, "Dealer begins drawing cards.")
            while dealer_total < 17:
                self.dealer_hand.append(self._draw_card())
                dealer_total, _ = self._hand_value(self.dealer_hand)

            if dealer_total > 21 or player_total > dealer_total:
                self.logger.log(LogLevel.INFO, "Player wins.")
                if player_total == 21:
                    if len(self.player_hand) == 2:
                        self.reward += 2
                    else:
                        self.reward += 1.5
                else:
                    self.reward += 1
            elif player_total < dealer_total:
                self.logger.log(LogLevel.INFO, "Dealer wins.")
                self.reward -= 1
            else:
                self.logger.log(LogLevel.INFO, "It's a draw.")

        # Update state and decay penalty.
        self.stand_penalty *= (1 - self.stand_penalty_decay)
        player_total, usable_ace = self._hand_value(self.player_hand)
        dealer_card = self.dealer_hand[0]
        hand_count = len(self.player_hand)

        state = {
            "player_total": player_total,
            "usable_ace": usable_ace,
            "dealer_card": dealer_card,
            "hand_count": hand_count
        }

        if self.multi_round_mode:
            state["deck_composition"] = self._deck_composition()

        self.logger.log(LogLevel.DEBUG, f"Step result - Player: {self.player_hand}, Dealer: {self.dealer_hand}")
        return state, self.reward, self.done, None

    def render(self):
        """Render the current state of the game."""
        self.logger.log(LogLevel.INFO, "Rendering current game state.")
        print(f"Player hand: {self.player_hand}, total: {self._hand_value(self.player_hand)[0]}")
        print(f"Dealer hand: [{self.dealer_hand[0]}, ?]")

    def probability_of_bust(self, player_total: int, deck_composition: Optional[Dict[int, int]] = None) -> float:
        """
        Calculate the probability of bust if the player draws a card.

        Args:
            player_total (int): The current total of the player's hand.
            deck_composition (Optional[Dict[int, int]]): The current deck composition. If None, assumes a full deck.

        Returns:
            float: The probability of busting.
        """
        if deck_composition is None:
            deck_composition = {value: 4 for value in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        total_cards = sum(deck_composition.values())
        if total_cards == 0:
            return 0.0

        busting_cards = [card for card in deck_composition if player_total + card > 21]
        bust_count = sum(deck_composition[card] for card in busting_cards)

        probability = bust_count / total_cards
        self.logger.log(LogLevel.INFO, f"Probability of bust: {probability:.2f}")
        return probability

# Example of usage
if __name__ == "__main__":
    logger = Logger(session_id="test_session")
    env = Environment(logger, multi_round_mode=True, stand_penalty_decay=0.001)
    state = env.reset()
    print("Initial state:", state)

    while not env.done:
        action = random.choice([0, 1])
        state, reward, done, _ = env.step(action)
        env.render()
        if done:
            print("Game over! Reward:", reward)
