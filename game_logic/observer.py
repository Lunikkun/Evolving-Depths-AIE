from __future__ import annotations

import csv
import json
import os
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from game_logic.runtime_logger import get_session_dir, log_event, log_structured_event

@dataclass
class RoomMetrics:
    time_taken: float
    enemy_hits_taken: int
    avg_enemy_distance: float


class Observer:
    def __init__(self) -> None:
        self.session_id = str(uuid.uuid4())
        session_dir = get_session_dir()
        self.log_path = os.path.join(session_dir, "session_metrics.csv")
        self.engagement_log_path = os.path.join(session_dir, "engagement_metrics.csv")
        self.room_summary_jsonl_path = os.path.join(session_dir, "room_summary.jsonl")

        self._room_start_time: Optional[float] = None
        self._room_start_hp: Optional[int] = None
        self._room_number: Optional[int] = None
        self._difficulty: Optional[int] = None
        self._use_bandit: Optional[bool] = None

        self._base_header = [
            "session_id", "room_number", "use_bandit",
            "difficulty_chosen", "room_started_at", "room_completed_at",
            "time_taken", "room_start_hp", "room_end_hp", "hp_lost"
        ]

        self._engagement_header = [
            "session_id", "room_number", "dungeon_room_id", "difficulty_chosen",
            "room_started_at", "room_completed_at", "time_taken", "room_start_hp",
            "room_end_hp", "hp_lost", "flow_score", "enemy_delay",
            "enemy_spawned", "enemy_hits_taken", "damage_blocked",
            "powerups_spawned", "powerups_collected", "heal_collected",
            "speed_collected", "shield_collected", "teleport_present", "teleport_uses",
            "moves_made", "key_press_rate", "avg_enemy_distance",
            "enemy_kills", "enemies_remaining", "powerups_remaining",
            "room_entries", "combo_streak", "speed_boost_steps", "shield_hits",
            "room_profile", "room_special_type", "objective_kind", "elite_modifier",
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
        started_at_iso = datetime.fromtimestamp(self._room_start_time, timezone.utc).isoformat()
        log_event(
            f"OBSERVER start_room room={room_number} difficulty={difficulty} use_bandit={use_bandit} hp={current_hp}"
        )
        log_structured_event(
            "room_start",
            session_id=self.session_id,
            room_number=room_number,
            difficulty=difficulty,
            use_bandit=use_bandit,
            room_start_hp=current_hp,
            room_started_at=started_at_iso,
        )

    def complete_room(self, current_hp: int, extra_metrics: Optional[dict] = None) -> RoomMetrics:
        if self._room_start_time is None or self._room_start_hp is None:
            return RoomMetrics(time_taken=0.0, enemy_hits_taken=0, avg_enemy_distance=0.0)

        extra = extra_metrics or {}
        time_taken = max(0.0, time.time() - self._room_start_time)
        hp_lost = max(0, self._room_start_hp - current_hp)
        completed_at = time.time()
        started_at_iso = datetime.fromtimestamp(self._room_start_time, timezone.utc).isoformat()
        completed_at_iso = datetime.fromtimestamp(completed_at, timezone.utc).isoformat()

        summary_record = {
            "session_id": self.session_id,
            "room_number": self._room_number,
            "dungeon_room_id": int(extra.get("dungeon_room_id", -1)),
            "difficulty_chosen": self._difficulty,
            "use_bandit": self._use_bandit,
            "room_started_at": started_at_iso,
            "room_completed_at": completed_at_iso,
            "time_taken": round(time_taken, 3),
            "room_start_hp": self._room_start_hp,
            "room_end_hp": current_hp,
            "hp_lost": hp_lost,
            "flow_score": round(float(extra.get("flow_score", 0.0)), 3),
            "enemy_delay": round(float(extra.get("enemy_delay", 0.0)), 3),
            "enemy_spawned": int(extra.get("enemy_spawned", 0)),
            "enemy_hits_taken": int(extra.get("enemy_hits_taken", 0)),
            "damage_blocked": int(extra.get("damage_blocked", 0)),
            "powerups_spawned": int(extra.get("powerups_spawned", 0)),
            "powerups_collected": int(extra.get("powerups_collected", 0)),
            "heal_collected": int(extra.get("heal_collected", 0)),
            "speed_collected": int(extra.get("speed_collected", 0)),
            "shield_collected": int(extra.get("shield_collected", 0)),
            "teleport_present": bool(extra.get("teleport_present", False)),
            "teleport_uses": int(extra.get("teleport_uses", 0)),
            "moves_made": int(extra.get("moves_made", 0)),
            "key_press_rate": round(float(extra.get("key_press_rate", 0.0)), 3),
            "avg_enemy_distance": round(float(extra.get("avg_enemy_distance", 0.0)), 3),
            "enemy_kills": int(extra.get("enemy_kills", 0)),
            "enemies_remaining": int(extra.get("enemies_remaining", 0)),
            "powerups_remaining": int(extra.get("powerups_remaining", 0)),
            "room_entries": int(extra.get("room_entries", 0)),
            "combo_streak": int(extra.get("combo_streak", 0)),
            "speed_boost_steps": int(extra.get("speed_boost_steps", 0)),
            "shield_hits": int(extra.get("shield_hits", 0)),
            "room_profile": str(extra.get("room_profile", "unknown")),
            "room_special_type": str(extra.get("room_special_type", "normal")),
            "objective_kind": str(extra.get("objective_kind", "unknown")),
            "elite_modifier": str(extra.get("elite_modifier", "")),
            "dungeon_event": str(extra.get("dungeon_event", "")),
            "relics_collected": int(extra.get("relics_collected", 0)),
            "relic_goal": int(extra.get("relic_goal", 0)),
            "final_door_open": bool(extra.get("final_door_open", False)),
        }

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    summary_record["session_id"],
                    summary_record["room_number"],
                    summary_record["use_bandit"],
                    summary_record["difficulty_chosen"],
                    summary_record["room_started_at"],
                    summary_record["room_completed_at"],
                    summary_record["time_taken"],
                    summary_record["room_start_hp"],
                    summary_record["room_end_hp"],
                    summary_record["hp_lost"],
                ]
            )

        with open(self.engagement_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    summary_record["session_id"],
                    summary_record["room_number"],
                    summary_record["dungeon_room_id"],
                    summary_record["difficulty_chosen"],
                    summary_record["room_started_at"],
                    summary_record["room_completed_at"],
                    summary_record["time_taken"],
                    summary_record["room_start_hp"],
                    summary_record["room_end_hp"],
                    summary_record["hp_lost"],
                    summary_record["flow_score"],
                    summary_record["enemy_delay"],
                    summary_record["enemy_spawned"],
                    summary_record["enemy_hits_taken"],
                    summary_record["damage_blocked"],
                    summary_record["powerups_spawned"],
                    summary_record["powerups_collected"],
                    summary_record["heal_collected"],
                    summary_record["speed_collected"],
                    summary_record["shield_collected"],
                    summary_record["teleport_present"],
                    summary_record["teleport_uses"],
                    summary_record["moves_made"],
                    summary_record["key_press_rate"],
                    summary_record["avg_enemy_distance"],
                    summary_record["enemy_kills"],
                    summary_record["enemies_remaining"],
                    summary_record["powerups_remaining"],
                    summary_record["room_entries"],
                    summary_record["combo_streak"],
                    summary_record["speed_boost_steps"],
                    summary_record["shield_hits"],
                    summary_record["room_profile"],
                    summary_record["room_special_type"],
                    summary_record["objective_kind"],
                    summary_record["elite_modifier"],
                    summary_record["dungeon_event"],
                    summary_record["relics_collected"],
                    summary_record["relic_goal"],
                    summary_record["final_door_open"],
                ]
            )

        with open(self.room_summary_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_record, ensure_ascii=True, sort_keys=True) + "\n")

        log_event(
            "OBSERVER complete_room "
            f"room={self._room_number} time_taken={time_taken:.3f} "
            f"hits={summary_record['enemy_hits_taken']} avg_enemy_distance={summary_record['avg_enemy_distance']:.3f}"
        )
        log_structured_event("room_complete", **summary_record)

        return RoomMetrics(
            time_taken=time_taken,
            enemy_hits_taken=summary_record["enemy_hits_taken"],
            avg_enemy_distance=summary_record["avg_enemy_distance"],
        )