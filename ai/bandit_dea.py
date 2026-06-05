import random
from typing import Dict, Optional
import config
from config import MAX_HP

class BanditDirector:
    """Epsilon-greedy multi-armed bandit for room difficulty pacing."""

    _EMA_ALPHA = 0.2  # smoothing factor shared by all EMA signals

    def __init__(self, epsilon: float = 0.2) -> None:
        self.epsilon = epsilon
        self.arms = [0, 1, 2]
        self.counts: Dict[int, int] = {arm: 0 for arm in self.arms}
        self.values: Dict[int, float] = {arm: 0.5 for arm in self.arms}
        self._target_time: float = 0.0  
        self._target_kpr: float = 0.0   
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

    def choose_difficulty(self) -> int:
        """Selects the best difficulty level (arm) using Epsilon-Greedy."""
        if random.random() < self.epsilon:
            return random.choice(self.arms)
        
        # Exploit: return index of the highest value
        best_value = max(self.values.values())
        best_arms = [arm for arm, value in self.values.items() if value == best_value]
        return random.choice(best_arms)

    def reward_from_metrics(self, time_taken: float, hp_lost: int, kpr: float = 0.0) -> float:
        """Calculates reward with Flow-Aware metrics (Intensity + Stability)."""
        # Normalization
        norm_time = max(0, min(1, 1 - (time_taken / 20.0))) 
        norm_hp = max(0, min(1, 1 - (hp_lost / 20.0)))
        
        # Stability Calculation
        if self._target_kpr > 0.1:
            diff = abs(kpr - self._target_kpr)
            stability_factor = max(0.0, 1.0 - (diff / (self._target_kpr + 0.5)))
        else:
            stability_factor = 0.5

        norm_kpr = max(0, min(1, kpr / 5.0))
        engagement_score = norm_kpr * stability_factor
        
        # Weights
        w_time = getattr(config, 'WEIGHT_TIME', 0.2)
        w_hp = getattr(config, 'WEIGHT_HP', 0.1)
        w_eng = getattr(config, 'WEIGHT_ENGAGEMENT', 0.7)
        
        return (norm_time * w_time) + (norm_hp * w_hp) + (engagement_score * w_eng)

    def update_reward(self, arm: Optional[int], time_taken: float, hp_lost: int, key_press_rate: float = 0.0) -> None:
        """Updates internal model based on performance."""
        if arm is None:
            return

        self._update_emas(time_taken, key_press_rate)
        reward = self.reward_from_metrics(time_taken, hp_lost, key_press_rate)
        
        self.counts[arm] += 1
        n = self.counts[arm]
        old_value = self.values[arm]
        self.values[arm] = old_value + (reward - old_value) / n