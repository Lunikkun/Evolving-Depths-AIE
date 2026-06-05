from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np

from ai.bandit_dea import BanditDirector
from ai.generator import RoomGenerator
from ai.verifier import AStarVerifier
from game_logic.runtime_logger import log_event


class Director:
    """Orchestrates Observe -> Decide -> Apply -> Verify iteration."""

    def __init__(self, use_bandit: bool = True) -> None:
        self.use_bandit = use_bandit
        self.bandit = BanditDirector(epsilon=0.2)
        self.generator = RoomGenerator()
        self.verifier = AStarVerifier(start=(0, 0))
        self.last_difficulty: Optional[int] = None
        self.last_engagement: float = 0.5

    def _choose_difficulty(self) -> int:
        if self.use_bandit:
            difficulty = self.bandit.choose_difficulty()
            log_event(f"DIRECTOR choose_difficulty mode=bandit value={difficulty}")
            return difficulty
        difficulty = random.choice([0, 1, 2])
        log_event(f"DIRECTOR choose_difficulty mode=random value={difficulty}")
        return difficulty

    def generate_next_room(
        self,
        player_stats: Optional[Tuple[float, int]] = None,
        key_press_rate: float = 0.0,
        exits: Optional[dict] = None
    ) -> Tuple[np.ndarray, int, int]:
        # Observe + Infer: update reward from previous room metrics.
        if player_stats is not None and self.last_difficulty is not None:
            time_taken, hp_lost = player_stats
            self.last_engagement = self.bandit.reward_from_metrics(time_taken, hp_lost, key_press_rate)
            self.bandit.update_reward(self.last_difficulty, time_taken, hp_lost, key_press_rate)
            log_event(
                f"DIRECTOR update_reward arm={self.last_difficulty} time_taken={time_taken:.3f} hp_lost={hp_lost} kpr={key_press_rate:.3f} engagement={self.last_engagement:.3f}"
            )

        # Decide
        target_difficulty = self._choose_difficulty()

        # Apply + Verify
        attempts = 0
        while True:
            attempts += 1
            candidate_map = self.generator.create_map(target_difficulty, engagement=self.last_engagement)
            
            # Passiamo le porte calcolate al verifier
            if self.verifier.check_playability(candidate_map, exits):
                self.last_difficulty = target_difficulty
                log_event(
                    f"DIRECTOR room_validated difficulty={target_difficulty} attempts={attempts} wall_prob={self.generator.last_wall_prob:.3f} enemies={self.generator.last_enemy_count}"
                )
                return candidate_map, target_difficulty, attempts