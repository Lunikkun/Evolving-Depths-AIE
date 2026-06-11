from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import main
from game_logic.director import Director

Vec2 = Tuple[int, int]
Move = Tuple[int, int]

DIRECTIONS: List[Move] = [(0, -1), (0, 1), (-1, 0), (1, 0)]

SESSION_METRICS_HEADER = [
    "session_id", "room_number", "use_bandit",
    "difficulty_chosen", "room_started_at", "room_completed_at",
    "time_taken", "room_start_hp", "room_end_hp", "hp_lost",
]

ENGAGEMENT_METRICS_HEADER = [
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
    "dungeon_event", "goals_current", "relics_collected", "relic_goal", "final_door_open",
]


@dataclass
class EpisodeResult:
    scenario: str
    person_id: int
    agent: str
    bandit: bool
    seed: int
    won: bool
    game_over: bool
    steps: int
    rooms_entered: int
    rooms_cleared: int
    hp_end: int
    relics_collected: int
    relic_goal: int
    flow_end: float
    total_time_sim: float


def _neighbors(pos: Vec2) -> List[Vec2]:
    x, y = pos
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _shortest_next_step(start: Vec2, goal: Vec2, room: main.DungeonRoom) -> Optional[Vec2]:
    if start == goal:
        return start
    queue: deque[Vec2] = deque([start])
    parent: Dict[Vec2, Optional[Vec2]] = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for nxt in _neighbors(cur):
            if nxt in parent:
                continue
            if not main.is_walkable(room.grid, nxt):
                continue
            parent[nxt] = cur
            queue.append(nxt)
    if goal not in parent:
        return None
    step = goal
    while parent[step] is not None and parent[step] != start:
        step = parent[step]
    return step


def _move_from_step(src: Vec2, dst: Optional[Vec2]) -> Move:
    if dst is None:
        return (0, 0)
    return (dst[0] - src[0], dst[1] - src[1])


def _fast_greedy_step(start: Vec2, goal: Vec2, room: main.DungeonRoom) -> Optional[Vec2]:
    best: Optional[Vec2] = None
    best_dist = 10**9
    base_dist = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
    for nx, ny in _neighbors(start):
        if not main.is_walkable(room.grid, (nx, ny)):
            continue
        dist = abs(goal[0] - nx) + abs(goal[1] - ny)
        if dist < best_dist:
            best_dist = dist
            best = (nx, ny)
    if best is not None and best_dist < base_dist:
        return best
    return None


def _random_policy(state: main.GameState, room: main.DungeonRoom, dungeon: main.DungeonState) -> Move:
    choices: List[Move] = []
    for dx, dy in DIRECTIONS:
        nx, ny = state.player_pos[0] + dx, state.player_pos[1] + dy
        if main.is_walkable(room.grid, (nx, ny)):
            choices.append((dx, dy))
    return random.choice(choices) if choices else (0, 0)


def _target_exit(room: main.DungeonRoom) -> Optional[Vec2]:
    if not room.exits:
        return None
    return random.choice(list(room.exits.keys()))


def _exit_rush_policy(state: main.GameState, room: main.DungeonRoom, dungeon: main.DungeonState) -> Move:
    target = _target_exit(room)
    if target is None:
        return _random_policy(state, room, dungeon)
    step = _fast_greedy_step(state.player_pos, target, room)
    if step is None:
        step = _shortest_next_step(state.player_pos, target, room)
    if step is None:
        return _random_policy(state, room, dungeon)
    return _move_from_step(state.player_pos, step)


def _safe_policy(state: main.GameState, room: main.DungeonRoom, dungeon: main.DungeonState) -> Move:
    targets: List[Vec2] = []
    if room.relic_positions:
        targets.extend(room.relic_positions)
    if room.powerups:
        targets.extend(room.powerups.keys())
    if not targets:
        t_exit = _target_exit(room)
        if t_exit is not None:
            targets.append(t_exit)
    if not targets:
        return _random_policy(state, room, dungeon)
    best = min(targets, key=lambda p: abs(p[0] - state.player_pos[0]) + abs(p[1] - state.player_pos[1]))
    step = _fast_greedy_step(state.player_pos, best, room)
    if step is None:
        step = _shortest_next_step(state.player_pos, best, room)
    if step is None:
        return _random_policy(state, room, dungeon)
    return _move_from_step(state.player_pos, step)


def _fighter_policy(state: main.GameState, room: main.DungeonRoom, dungeon: main.DungeonState) -> Move:
    if room.enemies:
        target = min(room.enemies.keys(), key=lambda p: abs(p[0] - state.player_pos[0]) + abs(p[1] - state.player_pos[1]))
        step = _fast_greedy_step(state.player_pos, target, room)
        if step is None:
            step = _shortest_next_step(state.player_pos, target, room)
        if step is not None:
            return _move_from_step(state.player_pos, step)
    return _exit_rush_policy(state, room, dungeon)


def _goal_exit_policy(state: main.GameState, room: main.DungeonRoom, dungeon: main.DungeonState) -> Move:
    if dungeon.collected_relics < dungeon.relic_goal and room.relic_positions:
        target = min(room.relic_positions, key=lambda p: abs(p[0] - state.player_pos[0]) + abs(p[1] - state.player_pos[1]))
        step = _fast_greedy_step(state.player_pos, target, room)
        if step is None:
            step = _shortest_next_step(state.player_pos, target, room)
        if step is not None:
            return _move_from_step(state.player_pos, step)

    if dungeon.collected_relics >= dungeon.relic_goal and room.final_door_pos is not None:
        step = _fast_greedy_step(state.player_pos, room.final_door_pos, room)
        if step is None:
            step = _shortest_next_step(state.player_pos, room.final_door_pos, room)
        if step is not None:
            return _move_from_step(state.player_pos, step)

    return _exit_rush_policy(state, room, dungeon)


def _goal_target(state: main.GameState, room: main.DungeonRoom, dungeon: main.DungeonState) -> Optional[Vec2]:
    if dungeon.collected_relics < dungeon.relic_goal and room.relic_positions:
        return min(
            room.relic_positions,
            key=lambda p: abs(p[0] - state.player_pos[0]) + abs(p[1] - state.player_pos[1]),
        )
    if dungeon.collected_relics >= dungeon.relic_goal and room.final_door_pos is not None:
        return room.final_door_pos
    return _target_exit(room)


def _nearest_enemy_distance(pos: Vec2, room: main.DungeonRoom) -> int:
    if not room.enemies:
        return 10**6
    return min(abs(ex - pos[0]) + abs(ey - pos[1]) for ex, ey in room.enemies.keys())


def _avoid_enemies_policy(state: main.GameState, room: main.DungeonRoom, dungeon: main.DungeonState) -> Move:
    target = _goal_target(state, room, dungeon)
    current = state.player_pos
    current_enemy_dist = _nearest_enemy_distance(current, room)

    candidates: List[Tuple[int, int, Vec2]] = []
    for dx, dy in DIRECTIONS:
        nxt = (current[0] + dx, current[1] + dy)
        if not main.is_walkable(room.grid, nxt):
            continue
        enemy_dist = _nearest_enemy_distance(nxt, room)
        goal_dist = 10**6 if target is None else abs(target[0] - nxt[0]) + abs(target[1] - nxt[1])
        candidates.append((enemy_dist, -goal_dist, nxt))

    if not candidates:
        return (0, 0)

    safer = [c for c in candidates if c[0] > current_enemy_dist]
    if safer:
        best = max(safer, key=lambda c: (c[0], c[1]))
        return _move_from_step(current, best[2])

    best = max(candidates, key=lambda c: (c[0], c[1]))
    return _move_from_step(current, best[2])


def _simulate_episode(
    scenario_name: str,
    person_id: int,
    agent_name: str,
    policy: Callable[[main.GameState, main.DungeonRoom, main.DungeonState], Move],
    use_bandit: bool,
    seed: int,
    max_steps: int,
    rooms: int,
    relic_goal: int,
) -> Tuple[EpisodeResult, List[List[object]], List[List[object]]]:
    random.seed(seed)
    now = time.time()

    old_max_rooms = main.MAX_ROOMS
    old_relic_goal = main.RELIC_GOAL
    main.MAX_ROOMS = max(3, rooms)
    main.RELIC_GOAL = max(1, relic_goal)

    director = Director(use_bandit=use_bandit)
    dungeon = main._build_dungeon()
    state = main._new_game_state(dungeon.start_room)

    room = main._create_room(state.current_room_id, dungeon, director, None)
    main._enter_room(state, dungeon, room)

    sim_elapsed_in_room = 0.0
    dt_sec = 1.0 / max(1, main.FPS)
    steps = 0
    rooms_cleared = 0
    room_number = 1
    room_started_at_sim = 0.0
    episode_started_at = time.time()
    room_start_hp = state.hp
    session_id = f"ablation-{person_id}-{agent_name}-{'on' if use_bandit else 'off'}-{seed}-{uuid.uuid4().hex[:8]}"
    session_rows: List[List[object]] = []
    engagement_rows: List[List[object]] = []

    def _iso_from_sim(sim_sec: float) -> str:
        return datetime.fromtimestamp(episode_started_at + sim_sec, timezone.utc).isoformat()

    def _append_room_log(room_completed_at_sim: float, time_taken: float, snapshot: dict) -> None:
        hp_lost = max(0, room_start_hp - state.hp)
        started_iso = _iso_from_sim(room_started_at_sim)
        completed_iso = _iso_from_sim(room_completed_at_sim)
        session_rows.append([
            session_id,
            room_number,
            use_bandit,
            state.difficulty,
            started_iso,
            completed_iso,
            round(time_taken, 3),
            room_start_hp,
            state.hp,
            hp_lost,
        ])
        engagement_rows.append([
            session_id,
            room_number,
            int(snapshot.get("dungeon_room_id", -1)),
            state.difficulty,
            started_iso,
            completed_iso,
            round(time_taken, 3),
            room_start_hp,
            state.hp,
            hp_lost,
            round(float(snapshot.get("flow_score", 0.0)), 3),
            round(float(snapshot.get("enemy_delay", 0.0)), 3),
            int(snapshot.get("enemy_spawned", 0)),
            int(snapshot.get("enemy_hits_taken", 0)),
            int(snapshot.get("damage_blocked", 0)),
            int(snapshot.get("powerups_spawned", 0)),
            int(snapshot.get("powerups_collected", 0)),
            int(snapshot.get("heal_collected", 0)),
            int(snapshot.get("speed_collected", 0)),
            int(snapshot.get("shield_collected", 0)),
            bool(snapshot.get("teleport_present", False)),
            int(snapshot.get("teleport_uses", 0)),
            int(snapshot.get("moves_made", 0)),
            round(float(snapshot.get("key_press_rate", 0.0)), 3),
            round(float(snapshot.get("avg_enemy_distance", 0.0)), 3),
            int(snapshot.get("enemy_kills", 0)),
            int(snapshot.get("enemies_remaining", 0)),
            int(snapshot.get("powerups_remaining", 0)),
            int(snapshot.get("room_entries", 0)),
            int(snapshot.get("combo_streak", 0)),
            int(snapshot.get("speed_boost_steps", 0)),
            int(snapshot.get("shield_hits", 0)),
            str(snapshot.get("room_profile", "unknown")),
            str(snapshot.get("room_special_type", "normal")),
            str(snapshot.get("objective_kind", "unknown")),
            str(snapshot.get("elite_modifier", "")),
            str(snapshot.get("dungeon_event", "")),
            int(snapshot.get("goals_current", snapshot.get("relics_collected", 0))),
            int(snapshot.get("relics_collected", 0)),
            int(snapshot.get("relic_goal", 0)),
            bool(snapshot.get("final_door_open", False)),
        ])

    while steps < max_steps and not state.game_over and not state.is_win:
        steps += 1
        sim_elapsed_in_room += dt_sec
        state.room_start_time = now - sim_elapsed_in_room

        dx, dy = policy(state, room, dungeon)
        if dx != 0 or dy != 0:
            main._move_player(state, room, dungeon, dx, dy)

        state.enemy_timer += dt_sec
        main._update_in_room_flow(state, director)
        main._update_room_objective(state, room, dt_sec)
        if state.enemy_timer >= state.enemy_delay:
            main._move_enemies(state, room)
            state.enemy_timer = 0.0

        target_room = main._find_transition_room(state, room)
        if target_room is not None:
            rooms_cleared += 1
            snap = main._engagement_snapshot(state, dungeon)
            _append_room_log(steps * dt_sec, sim_elapsed_in_room, snap)
            kpr = float(snap.get("key_press_rate", 0.0))
            if director.bandit is not None:
                state.flow_score = director.bandit.reward_from_metrics(
                    sim_elapsed_in_room,
                    int(snap.get("enemy_hits_taken", 0)),
                    float(snap.get("avg_enemy_distance", 0.0)),
                    kpr,
                )
            else:
                state.flow_score = 0.5

            main._rotate_dungeon_event(dungeon, state.room_step + 1)
            main._advance_predator(dungeon, target_room)
            if target_room not in dungeon.rooms:
                room = main._create_room(
                    target_room,
                    dungeon,
                    director,
                    (sim_elapsed_in_room, int(snap.get("enemy_hits_taken", 0))),
                    key_press_rate=kpr,
                    force_reward_room=False,
                )
            else:
                room = dungeon.rooms[target_room]

            state.room_step += 1
            state.previous_room_id = state.current_room_id
            main._enter_room(state, dungeon, room)
            now = time.time()
            sim_elapsed_in_room = 0.0
            room_number += 1
            room_started_at_sim = steps * dt_sec
            room_start_hp = state.hp

        if room.final_door_pos is not None and state.player_pos == room.final_door_pos:
            door_cell = int(room.grid[room.final_door_pos[1], room.final_door_pos[0]])
            if door_cell == main.DOOR_OPEN and dungeon.collected_relics >= dungeon.relic_goal:
                state.is_win = True

        if state.hp <= 0:
            state.game_over = True

    total_time = steps * dt_sec
    snap_final = main._engagement_snapshot(state, dungeon)
    _append_room_log(total_time, sim_elapsed_in_room, snap_final)

    result = EpisodeResult(
        scenario=scenario_name,
        person_id=person_id,
        agent=agent_name,
        bandit=use_bandit,
        seed=seed,
        won=state.is_win,
        game_over=state.game_over,
        steps=steps,
        rooms_entered=state.room_entries,
        rooms_cleared=rooms_cleared,
        hp_end=state.hp,
        relics_collected=dungeon.collected_relics,
        relic_goal=dungeon.relic_goal,
        flow_end=state.flow_score,
        total_time_sim=total_time,
    )

    main.MAX_ROOMS = old_max_rooms
    main.RELIC_GOAL = old_relic_goal
    return result, session_rows, engagement_rows


def _write_episode_csv(path: str, rows: List[EpisodeResult]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "person_id",
            "agent",
            "bandit",
            "seed",
            "won",
            "game_over",
            "steps",
            "rooms_entered",
            "rooms_cleared",
            "hp_end",
            "relics_collected",
            "relic_goal",
            "flow_end",
            "total_time_sim",
        ])
        for r in rows:
            writer.writerow([
                r.scenario,
                r.person_id,
                r.agent,
                r.bandit,
                r.seed,
                r.won,
                r.game_over,
                r.steps,
                r.rooms_entered,
                r.rooms_cleared,
                r.hp_end,
                r.relics_collected,
                r.relic_goal,
                round(r.flow_end, 4),
                round(r.total_time_sim, 4),
            ])


def _write_summary_csv(path: str, rows: List[EpisodeResult]) -> None:
    groups: Dict[Tuple[str, int, str, bool], List[EpisodeResult]] = {}
    for row in rows:
        groups.setdefault((row.scenario, row.person_id, row.agent, row.bandit), []).append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "person_id",
            "agent",
            "bandit",
            "episodes",
            "win_rate",
            "avg_rooms_cleared",
            "avg_hp_end",
            "avg_flow_end",
            "avg_total_time_sim",
            "avg_steps",
        ])
        for (scenario, person_id, agent, bandit), vals in sorted(groups.items()):
            n = max(1, len(vals))
            win_rate = sum(1 for v in vals if v.won) / n
            avg_rooms = sum(v.rooms_cleared for v in vals) / n
            avg_hp = sum(v.hp_end for v in vals) / n
            avg_flow = sum(v.flow_end for v in vals) / n
            avg_time = sum(v.total_time_sim for v in vals) / n
            avg_steps = sum(v.steps for v in vals) / n
            writer.writerow([
                scenario,
                person_id,
                agent,
                bandit,
                n,
                round(win_rate, 4),
                round(avg_rooms, 4),
                round(avg_hp, 4),
                round(avg_flow, 4),
                round(avg_time, 4),
                round(avg_steps, 2),
            ])


def _write_global_summary_csv(path: str, rows: List[EpisodeResult]) -> None:
    groups: Dict[Tuple[str, str, bool], List[EpisodeResult]] = {}
    for row in rows:
        groups.setdefault((row.scenario, row.agent, row.bandit), []).append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "agent",
            "bandit",
            "episodes_total",
            "win_rate",
            "avg_rooms_cleared",
            "avg_hp_end",
            "avg_flow_end",
            "avg_total_time_sim",
            "avg_steps",
        ])
        for (scenario, agent, bandit), vals in sorted(groups.items()):
            n = max(1, len(vals))
            win_rate = sum(1 for v in vals if v.won) / n
            avg_rooms = sum(v.rooms_cleared for v in vals) / n
            avg_hp = sum(v.hp_end for v in vals) / n
            avg_flow = sum(v.flow_end for v in vals) / n
            avg_time = sum(v.total_time_sim for v in vals) / n
            avg_steps = sum(v.steps for v in vals) / n
            writer.writerow([
                scenario,
                agent,
                bandit,
                n,
                round(win_rate, 4),
                round(avg_rooms, 4),
                round(avg_hp, 4),
                round(avg_flow, 4),
                round(avg_time, 4),
                round(avg_steps, 2),
            ])


def _write_runtime_session_metrics_csv(path: str, rows: List[List[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SESSION_METRICS_HEADER)
        writer.writerows(rows)


def _write_runtime_engagement_metrics_csv(path: str, rows: List[List[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ENGAGEMENT_METRICS_HEADER)
        writer.writerows(rows)


def run_ablation(people: int, episodes: int, max_steps: int, seed: int, out_dir: str) -> Tuple[str, str, str]:
    scenarios: List[Tuple[str, int, int]] = [
        ("g1_r3", 1, 3),
        ("g3_r6", 3, 6),
        ("g5_r12", 5, 12),
    ]

    policies: Dict[str, Callable[[main.GameState, main.DungeonRoom, main.DungeonState], Move]] = {
        "goal_focus": _goal_exit_policy,
        "safety_focus": _avoid_enemies_policy,
    }

    results: List[EpisodeResult] = []
    session_metric_rows: List[List[object]] = []
    engagement_metric_rows: List[List[object]] = []
    s = seed
    for scenario_name, relic_goal, rooms in scenarios:
        for person_id in range(1, people + 1):
            for use_bandit in (False, True):
                for agent_name, policy in policies.items():
                    for _ in range(episodes):
                        s += 1
                        r, room_rows, engage_rows = _simulate_episode(
                            scenario_name,
                            person_id,
                            agent_name,
                            policy,
                            use_bandit,
                            s,
                            max_steps,
                            rooms,
                            relic_goal,
                        )
                        results.append(r)
                        session_metric_rows.extend(room_rows)
                        engagement_metric_rows.extend(engage_rows)

    episodes_csv = os.path.join(out_dir, "ablation_episodes.csv")
    summary_csv = os.path.join(out_dir, "ablation_summary_by_person.csv")
    global_summary_csv = os.path.join(out_dir, "ablation_summary_global.csv")
    session_metrics_csv = os.path.join(out_dir, "session_metrics.csv")
    engagement_metrics_csv = os.path.join(out_dir, "engagement_metrics.csv")
    _write_episode_csv(episodes_csv, results)
    _write_summary_csv(summary_csv, results)
    _write_global_summary_csv(global_summary_csv, results)
    _write_runtime_session_metrics_csv(session_metrics_csv, session_metric_rows)
    _write_runtime_engagement_metrics_csv(engagement_metrics_csv, engagement_metric_rows)
    print(f"session_metrics_csv={session_metrics_csv}")
    print(f"engagement_metrics_csv={engagement_metrics_csv}")
    return episodes_csv, summary_csv, global_summary_csv


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Run agent-vs-agent ablation episodes")
    parser.add_argument("--people", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", type=str, default="reports/ablation")
    args = parser.parse_args()

    episodes_csv, summary_csv, global_summary_csv = run_ablation(args.people, args.episodes, args.max_steps, args.seed, args.out_dir)
    print(f"episodes_csv={episodes_csv}")
    print(f"summary_by_person_csv={summary_csv}")
    print(f"summary_global_csv={global_summary_csv}")


if __name__ == "__main__":
    main_cli()
