import random
from typing import Dict, Optional
import config

class BanditDirector:
    _EMA_ALPHA = 0.2

    def __init__(self, epsilon: float = 0.2) -> None:
        self.epsilon = epsilon
        self.arms = [0, 1, 2]
        self.counts: Dict[int, int] = {arm: 0 for arm in self.arms}
        self.values: Dict[int, float] = {arm: 0.5 for arm in self.arms}
        self._target_time: float = 0.0  
        self._target_kpr: float = 0.0   
        self._rooms_seen: int = 0

    def _update_emas(self, time_taken: float, key_press_rate: float) -> None:
        if self._rooms_seen == 0:
            self._target_time = time_taken
            self._target_kpr = key_press_rate
        else:
            a = self._EMA_ALPHA
            self._target_time = a * time_taken + (1.0 - a) * self._target_time
            self._target_kpr = a * key_press_rate + (1.0 - a) * self._target_kpr
        self._rooms_seen += 1

    def choose_difficulty(self) -> int:
        if random.random() < self.epsilon:
            return random.choice(self.arms)

        best_value = max(self.values.values())
        best_arms = [arm for arm, value in self.values.items() if value == best_value]
        return random.choice(best_arms)

    def reward_from_metrics(
        self,
        time_taken: float,
        enemy_hits_taken: int,
        avg_enemy_distance: float,
        kpr: float = 0.0,
    ) -> float:
        baseline_time = self._target_time if self._rooms_seen > 0 else 15.0
        norm_time = max(0.0, min(1.0, 1.0 - (time_taken / (baseline_time * 2.0))))
        norm_hits = max(0.0, min(1.0, 1.0 - (enemy_hits_taken / 4.0)))
        norm_distance = max(0.0, min(1.0, avg_enemy_distance / 8.0))
        pressure_score = (0.65 * norm_distance) + (0.35 * norm_hits)

        if self._target_kpr > 0.1:
            diff = abs(kpr - self._target_kpr)
            stability_factor = max(0.0, 1.0 - (diff / (self._target_kpr + 0.5)))
        else:
            stability_factor = 0.5

        norm_kpr = max(0.0, min(1.0, kpr / 5.0))
        engagement_score = norm_kpr * stability_factor

        w_time = getattr(config, 'WEIGHT_TIME', 0.45)
        w_pressure = getattr(config, 'WEIGHT_PRESSURE', 0.20)
        w_eng = getattr(config, 'WEIGHT_ENGAGEMENT', 0.35)

        return (norm_time * w_time) + (pressure_score * w_pressure) + (engagement_score * w_eng)

    def update_reward(
        self,
        arm: Optional[int],
        time_taken: float,
        enemy_hits_taken: int,
        avg_enemy_distance: float,
        key_press_rate: float = 0.0,
    ) -> None:
        if arm is None:
            return

        self._update_emas(time_taken, key_press_rate)
        reward = self.reward_from_metrics(time_taken, enemy_hits_taken, avg_enemy_distance, key_press_rate)
        
        self.counts[arm] += 1
        n = self.counts[arm]
        old_value = self.values[arm]
        self.values[arm] = old_value + (reward - old_value) / n