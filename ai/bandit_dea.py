import random
from typing import Dict, Optional


class BanditDirector:
    """Epsilon-greedy multi-armed bandit for room difficulty pacing."""

    def __init__(self, epsilon: float = 0.25) -> None:
        self.epsilon = epsilon
        self.arms = [0, 1, 2]
        self.counts: Dict[int, int] = {arm: 0 for arm in self.arms}
        # Optimistic initialisation: arm 2 starts favoured so the bandit
        # explores harder rooms before falling back to safe/medium.
        self.values: Dict[int, float] = {0: 0.30, 1: 0.45, 2: 0.55}

    @staticmethod
    def _closeness(value: float, target: float, tolerance: float) -> float:
        """Score in [0,1]: 1.0 at target, 0.0 when |diff| >= tolerance."""
        return max(0.0, 1.0 - abs(value - target) / max(1e-6, tolerance))

    def reward_from_metrics(
        self,
        time_taken: float,
        hp_lost: int,
        avg_enemy_distance: float = 0.0,
        kpr: float = 0.0,
    ) -> float:
        """
        Reward the challenge sweet-spot.
        Rooms cleared too fast with no damage score LOW (too easy = arm 0 penalised).
        Rooms with moderate time, some pressure and active input score HIGH.

        target_time = 25s,  tolerance = 20  -> band 5-45s; <10s scores below 0.25
        target_hp   = 15,   tolerance = 25  -> 0 hp scores ~0.40, not 1.0
        target_kpr  = 1.5,  tolerance = 1.8
        """
        time_component = self._closeness(max(0.1, float(time_taken)), 25.0, 20.0)
        hp_component   = self._closeness(max(0.0, float(hp_lost)),    15.0, 25.0)
        kpr_component  = self._closeness(max(0.0, float(kpr)),         1.5,  1.8)

        return (
            0.40 * time_component
            + 0.35 * kpr_component
            + 0.25 * hp_component
        )

    def update_reward(
        self,
        arm: Optional[int],
        time_taken: float,
        hp_lost: int,
        avg_enemy_distance: float = 0.0,
        kpr: float = 0.0,
    ) -> None:
        if arm is None:
            return
        reward = self.reward_from_metrics(time_taken, hp_lost, avg_enemy_distance, kpr)
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
