import csv
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

@dataclass
class RoomMetrics:
    time_taken: float
    hp_lost: int
    enemy_hits_taken: int = 0
    avg_enemy_distance: float = 0.0

class Observer:
    """Collects room metrics and persists logs for evaluation."""

    def __init__(self, log_path: str = "logs/session_metrics.csv", engagement_path: str = "logs/engagement_metrics.csv") -> None:
        self.session_id = str(uuid.uuid4())
        self.log_path = log_path
        self.engagement_path = engagement_path  # Nuovo percorso per engagement

        self._room_start_time: Optional[float] = None
        self._room_start_hp: Optional[int] = None
        self._room_number: Optional[int] = None
        self._difficulty: Optional[int] = None
        self._use_bandit: Optional[bool] = None

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Inizializza session_metrics.csv
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id_sessione", "num_stanza", "use_bandit", "difficulty_chosen", "time_taken", "hp_lost"])

        # Inizializza engagement_metrics.csv
        if not os.path.exists(self.engagement_path):
            with open(self.engagement_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Assicurati che questi campi corrispondano alle chiavi presenti nel tuo dict 'snap'
                writer.writerow(["session_id", "num_stanza", "difficulty_chosen", "kpr", "enemy_dist", "profile"])

    def start_room(self, room_number: int, difficulty: int, use_bandit: bool, current_hp: int) -> None:
        self._room_number = room_number
        self._difficulty = difficulty
        self._use_bandit = use_bandit
        self._room_start_time = time.time()
        self._room_start_hp = current_hp

    # AGGIORNA LA FIRMA QUI
    def complete_room(self, current_hp: int, snap: dict) -> RoomMetrics:
        if self._room_start_time is None or self._room_start_hp is None:
            return RoomMetrics(time_taken=0.0, hp_lost=0)

        time_taken = max(0.0, time.time() - self._room_start_time)
        hp_lost = max(0, self._room_start_hp - current_hp)

        # Scrittura session_metrics
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([self.session_id, self._room_number, self._use_bandit, self._difficulty, round(time_taken, 2), hp_lost])

        # Scrittura engagement_metrics
        with open(self.engagement_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.session_id, 
                self._room_number, 
                self._difficulty, 
                snap.get('key_press_rate', 0), 
                snap.get('avg_enemy_distance', 0), 
                snap.get('profile', 'none')
            ])

        return RoomMetrics(
            time_taken=time_taken,
            hp_lost=hp_lost,
            enemy_hits_taken=int(snap.get('enemy_hits_taken', 0)),
            avg_enemy_distance=float(snap.get('avg_enemy_distance', 0.0)),
        )