from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np

from ai.bandit_dea import BanditDirector
from ai.generator import RoomGenerator
from ai.verifier import AStarVerifier


class Director:
    """Orchestrates Observe -> Decide -> Apply -> Verify iteration."""

    def __init__(self, use_bandit: bool = True) -> None:
        self.use_bandit = use_bandit
        self.bandit = BanditDirector(epsilon=0.2)
        self.generator = RoomGenerator()
        self.verifier = AStarVerifier(start=(0, 0))
        self.last_difficulty: Optional[int] = None
        self.last_engagement: float = 0.0 

    def _choose_difficulty(self) -> int:
        if self.use_bandit:
            return self.bandit.choose_difficulty()
        return random.choice([0, 1, 2])

    def generate_next_room(
        self,
        player_stats: Optional[Tuple[float, int]] = None,
        key_press_rate: float = 0.0,
        exits: Optional[list] = None  # <--- AGGIUNGI QUESTO
    ) -> Tuple[np.ndarray, int, int]:
        # Observe + Infer: update reward from previous room metrics.
        if player_stats is not None and self.last_difficulty is not None:
            if len(player_stats) == 4:
                time_taken, hp_lost, avg_enemy_distance, kpr = player_stats
            else:
                time_taken, hp_lost = player_stats
                avg_enemy_distance, kpr = 0.0, 0.0
            self.bandit.update_reward(
                self.last_difficulty, time_taken, hp_lost,
                avg_enemy_distance, kpr,
            )

        # Decide — save the originally chosen arm before any fallback
        chosen_arm = self._choose_difficulty()
        target_difficulty = chosen_arm

        # Apply + Verify — hard cap per tier, then drop one level
        attempts = 0
        candidate_map = None
        current_tier = target_difficulty
        while attempts < 90:
            attempts += 1
            candidate_map = self.generator.create_map(current_tier)
            if self.verifier.check_playability(candidate_map):
                self.last_difficulty = chosen_arm  # always credit the chosen arm
                return candidate_map, chosen_arm, attempts
            if attempts % 30 == 0 and current_tier > 0:
                current_tier -= 1

        # Guaranteed fallback — still credit chosen arm so bandit gets signal
        self.last_difficulty = chosen_arm
        return candidate_map, chosen_arm, attempts