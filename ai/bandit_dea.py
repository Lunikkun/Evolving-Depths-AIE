import random
from typing import Dict, Optional

from config import MAX_HP


class BanditDirector:
    """Epsilon-greedy multi-armed bandit for room difficulty pacing."""

    _EMA_ALPHA = 0.2  # smoothing factor shared by all EMA signals

    def __init__(self, epsilon: float = 0.2) -> None:
        self.epsilon = epsilon
        self.arms = [0, 1, 2]
        self.counts: Dict[int, int] = {arm: 0 for arm in self.arms}
        # Optimistic initialisation: all arms start at 0.5 to force early exploration
        # of every branch before the bandit can converge on the first lucky arm.
        self.values: Dict[int, float] = {arm: 0.5 for arm in self.arms}
        self._target_time: float = 0.0  # EMA of clear time; bootstrapped on first room
        self._target_kpr: float = 0.0   # EMA of key_press_rate; bootstrapped on first room
        self._rooms_seen: int = 0

    def _update_emas(self, time_taken: float, key_press_rate: float) -> None:
        """Bootstrap on first room, then EMA-update both signals."""
        if self._rooms_seen == 0:
            self._target_time = time_taken
            self._target_kpr = key_press_rate
        else:
            a = self._EMA_ALPHA
            self._target_time = a * time_taken + (1.0 - a) * self._target_time
            self._target_kpr = a * key_press_rate + (1.0 - a) * self._target_kpr
        self._rooms_seen += 1

    def reward_from_metrics(self, time_taken: float, hp_lost: int, key_press_rate: float = 0.0) -> float:
        """Flow score — weights T=4, H=1, K=4 (sum=9), normalised on the player's own EMA baseline."""
        target_t = max(1.0, self._target_time)
        time_component = max(0.0, 1.0 - (abs(time_taken - target_t) / target_t))

        hp_component = max(0.0, 1.0 - (hp_lost / MAX_HP))

        # KPR: deviation from player's own rhythm lowers the score
        # (rate drop = boredom/stuck, rate spike = panic).
        target_k = max(0.1, self._target_kpr)
        kpr_component = max(0.0, 1.0 - (abs(key_press_rate - target_k) / target_k))

        return (4 * time_component + 2 * hp_component + 4 * kpr_component) / 10.0

    def update_reward(self, arm: Optional[int], time_taken: float, hp_lost: int, key_press_rate: float = 0.0) -> None:
        if arm is None:
            return

        self._update_emas(time_taken, key_press_rate)
        reward = self.reward_from_metrics(time_taken, hp_lost, key_press_rate)
        self.counts[arm] += 1
        n = self.counts[arm]
        old_value = self.values[arm]
        self.values[arm] = old_value + (reward - old_value) / n

    def choose_difficulty(self) -> int:
        if random.random() < self.epsilon:
            return random.choice(self.arms)

        best_value = max(self.values.values())
        best_arms = [arm for arm, value in self.values.items() if value == best_value]
        return random.choice(best_arms)
