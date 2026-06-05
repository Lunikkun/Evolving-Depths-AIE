from __future__ import annotations

import csv
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

# Assicurati di importare get_session_dir
from game_logic.runtime_logger import log_event, get_session_dir

@dataclass
class RoomMetrics:
    time_taken: float
    hp_lost: int

class Observer:
    """Collects room metrics and persists logs for evaluation."""

    def __init__(self) -> None:
        self.session_id = str(uuid.uuid4())
        
        # Recupera la cartella creata dal logger per questa sessione
        session_dir = get_session_dir()
        
        # Imposta i percorsi dei file DENTRO la cartella della sessione
        self.log_path = os.path.join(session_dir, "session_metrics.csv")
        self.engagement_log_path = os.path.join(session_dir, "engagement_metrics.csv")

        self._room_start_time: Optional[float] = None
        self._room_start_hp: Optional[int] = None
        self._room_number: Optional[int] = None
        self._difficulty: Optional[int] = None
        self._use_bandit: Optional[bool] = None

        self._base_header = [
            "id_sessione", "num_stanza", "use_bandit", 
            "difficulty_chosen", "time_taken", "hp_lost"
        ]

        self._engagement_header = [
            "id_sessione", "num_stanza", "dungeon_room_id", "difficulty_chosen",
            "time_taken", "hp_lost", "flow_score", "enemy_delay",
            "enemy_spawned", "enemy_hits_taken", "damage_blocked",
            "powerups_spawned", "powerups_collected", "heal_collected",
            "speed_collected", "teleport_present", "teleport_uses",
            "moves_made", "key_press_rate", "avg_enemy_distance",
            "room_profile", "objective_kind", "elite_modifier",
            "dungeon_event", "relics_collected", "relic_goal", "final_door_open"
        ]

        self._ensure_csv_header(self.log_path, self._base_header)
        self._ensure_csv_header(self.engagement_log_path, self._engagement_header)
        

    def _ensure_csv_header(self, path: str, expected_header: list[str]) -> None:
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(expected_header)
            return

        first_line = ""
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        current_header = first_line.split(",") if first_line else []
        if current_header == expected_header:
            return

        backup = f"{path}.bak-{int(time.time())}"
        os.replace(path, backup)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(expected_header)
        log_event(f"OBSERVER migrated_csv_header path={path} backup={backup}")

    def start_room(self, room_number: int, difficulty: int, use_bandit: bool, current_hp: int) -> None:
        self._room_number = room_number
        self._difficulty = difficulty
        self._use_bandit = use_bandit
        self._room_start_time = time.time()
        self._room_start_hp = current_hp
        log_event(
            f"OBSERVER start_room room={room_number} difficulty={difficulty} use_bandit={use_bandit} hp={current_hp}"
        )

    def complete_room(self, current_hp: int, extra_metrics: Optional[dict] = None) -> RoomMetrics:
        if self._room_start_time is None or self._room_start_hp is None:
            return RoomMetrics(time_taken=0.0, hp_lost=0)

        time_taken = max(0.0, time.time() - self._room_start_time)
        hp_lost = max(0, self._room_start_hp - current_hp)

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.session_id,
                    self._room_number,
                    self._use_bandit,
                    self._difficulty,
                    round(time_taken, 2),
                    hp_lost,
                ]
            )

        extra = extra_metrics or {}
        with open(self.engagement_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.session_id,
                    self._room_number,
                    int(extra.get("dungeon_room_id", -1)),
                    self._difficulty,
                    round(time_taken, 2),
                    hp_lost,
                    round(float(extra.get("flow_score", 0.0)), 3),
                    round(float(extra.get("enemy_delay", 0.0)), 3),
                    int(extra.get("enemy_spawned", 0)),
                    int(extra.get("enemy_hits_taken", 0)),
                    int(extra.get("damage_blocked", 0)),
                    int(extra.get("powerups_spawned", 0)),
                    int(extra.get("powerups_collected", 0)),
                    int(extra.get("heal_collected", 0)),
                    int(extra.get("speed_collected", 0)),
                    bool(extra.get("teleport_present", False)),
                    int(extra.get("teleport_uses", 0)),
                    int(extra.get("moves_made", 0)),
                    round(float(extra.get("key_press_rate", 0.0)), 3),
                    round(float(extra.get("avg_enemy_distance", 0.0)), 3),
                    str(extra.get("room_profile", "unknown")),
                    str(extra.get("objective_kind", "unknown")),
                    str(extra.get("elite_modifier", "")),
                    str(extra.get("dungeon_event", "")),
                    int(extra.get("relics_collected", 0)),
                    int(extra.get("relic_goal", 0)),
                    bool(extra.get("final_door_open", False)),
                ]
            )

        log_event(
            f"OBSERVER complete_room room={self._room_number} time_taken={time_taken:.3f} hp_lost={hp_lost}"
        )

        return RoomMetrics(time_taken=time_taken, hp_lost=hp_lost)