from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pygame

from config import (
    DIFFICULTY_NAMES,
    DOOR_LOCKED,
    DOOR_OPEN,
    DUNGEON_MAP_H,
    DUNGEON_MAP_W,
    ENEMY,
    ENEMY_BASE_MOVE_DELAY,
    ENEMY_CONTACT_DAMAGE,
    EXIT,
    FLOOR,
    FPS,
    GRID_HEIGHT,
    GRID_WIDTH,
    HEAL_AMOUNT,
    MAX_HP,
    MAX_ROOMS,
    POWER_HEAL,
    POWER_SHIELD,
    POWER_SPEED,
    RELIC,
    RELIC_GOAL,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SHIELD_HITS,
    SPEED_BOOST_STEPS,
    TELEPORT_A,
    TELEPORT_B,
    TILE_SIZE,
    USE_BANDIT,
    WALL,
)
from game_logic.director import Director
from game_logic.observer import Observer
from game_logic.runtime_logger import log_event

Color = Tuple[int, int, int]
Vec2 = Tuple[int, int]

COLORS: dict[int, Color] = {
    FLOOR: (40, 44, 52),
    WALL: (95, 99, 109),
    ENEMY: (196, 75, 75),
    EXIT: (110, 185, 110),
    POWER_HEAL: (72, 199, 116),
    POWER_SPEED: (65, 153, 225),
    POWER_SHIELD: (141, 106, 199),
    TELEPORT_A: (71, 196, 202),
    TELEPORT_B: (232, 184, 78),
    RELIC: (245, 245, 245),
    DOOR_LOCKED: (160, 70, 70),
    DOOR_OPEN: (100, 200, 255),
}
PLAYER_COLOR: Color = (245, 210, 90)
HUD_BG: Color = (24, 26, 30)
TEXT_COLOR: Color = (235, 235, 235)
MINIMAP_BG: Color = (30, 33, 40)
SPAWN_POINT_COLOR: Color = (120, 210, 255)
PANEL_BG: Color = (36, 39, 46)

POWERUP_LABELS: dict[int, str] = {
    POWER_HEAL: "Heal",
    POWER_SPEED: "Speed",
    POWER_SHIELD: "Shield",
}

PROFILE_LABELS: dict[str, str] = {
    "arena": "Arena",
    "cave": "Cave",
    "corridor": "Corridor",
    "pillars": "Pillars",
    "crossroads": "Crossroads",
    "islands": "Islands",
}

OBJECTIVE_LABELS: dict[str, str] = {
    "none": "No objective",
    "survive": "Survive 20s",
    "eliminate": "Eliminate all",
    "altar": "Reach the altar",
    "protect_relic": "Protect the relic",
}

ELITE_LABELS: dict[str, str] = {
    "fast_enemies": "Fast Enemies",
    "fog": "Dense Fog",
    "half_heal": "Halved Heals",
    "double_relic": "Double Relic",
}

EVENT_LABELS: dict[str, str] = {
    "calm": "Unstable Calm",
    "blackout": "Blackout",
    "infestation": "Infestation",
    "predator_hunt": "Predator Hunt",
}

ENEMY_ROLE_LABELS: dict[str, str] = {
    "stalker": "Stalker",
    "splitter": "Splitter",
    "predator": "Predator",
}

ALTAR_COLOR: Color = (214, 160, 78)
RISK_EXIT_COLOR: Color = (236, 184, 72)
MERCHANT_COLOR: Color = (78, 204, 163)
SANCTUARY_COLOR: Color = (120, 190, 255)
FOG_COLOR: Color = (18, 20, 26)

ROOM_OBJECTIVE_SECONDS = 20.0
PROTECT_RELIC_SECONDS = 15.0
FLOW_COMBO_TIME_THRESHOLD = 14.0
FLOW_COMBO_HP_THRESHOLD = 6
FLOW_REWARD_STREAK = 2
FLOW_SURGE_STREAK = 3
MAX_OVERHEAL = 30
ELITE_ROOM_INTERVAL = 4
PREDATOR_DAMAGE = 12


@dataclass
class EnemyActor:
    role: str
    cooldown: int = 0
    hp: int = 1


@dataclass
class RoomObjective:
    kind: str
    timer_target: float = 0.0
    altar_pos: Optional[Vec2] = None
    progress: float = 0.0
    completed: bool = False
    failed: bool = False
    reward_granted: bool = False


@dataclass
class DungeonRoom:
    room_id: int
    grid: np.ndarray
    difficulty: int
    verify_attempts: int
    profile: str
    exits: Dict[Vec2, int]
    spawn_points: Dict[Vec2, Vec2]
    risk_exits: Set[Vec2]
    enemies: Dict[Vec2, EnemyActor]
    powerups: Dict[Vec2, int]
    teleports: Optional[Tuple[Vec2, Vec2]]
    relic_positions: Set[Vec2]
    objective: RoomObjective
    final_door_pos: Optional[Vec2]
    special_room_type: Optional[str] = None
    is_elite: bool = False
    elite_modifier: Optional[str] = None
    is_reward_room: bool = False
    visit_count: int = 0
    memory_shifted: bool = False


@dataclass
class DungeonState:
    rooms: Dict[int, DungeonRoom]
    coords_by_room: Dict[int, Vec2]
    adjacency: Dict[int, Set[int]]
    discovered: Set[int]
    start_room: int
    final_room: int
    relic_rooms: Set[int]
    relic_collected_rooms: Set[int]
    relic_goal: int
    collected_relics: int
    elite_rooms: Set[int]
    merchant_rooms: Set[int]
    sanctuary_rooms: Set[int]
    risk_reward_rooms: Set[int]
    active_event: str
    predator_room_id: Optional[int]


@dataclass
class GameState:
    room_step: int
    hp: int
    player_pos: Vec2
    current_room_id: int
    previous_room_id: Optional[int]
    difficulty: int
    verify_attempts: int
    room_profile: str
    room_special_type: str
    elite_modifier: str
    objective_kind: str
    flow_score: float
    enemy_delay: float
    enemy_timer: float
    speed_boost_steps: int
    shield_hits: int
    overheal_hp: int
    combo_streak: int
    combo_reward_pending: bool
    status_message: str
    status_timer: float
    enemies: Dict[Vec2, EnemyActor]
    enemies_spawned: int
    powerups: Dict[Vec2, int]
    powerups_spawned: int
    teleports: Optional[Tuple[Vec2, Vec2]]
    room_start_time: float
    room_start_hp: int
    moves_made: int
    enemy_kills: int
    enemy_hits_taken: int
    damage_blocked: int
    powerups_collected: int
    heal_collected: int
    speed_collected: int
    shield_collected: int
    teleport_uses: int
    enemy_distance_accum: float
    enemy_distance_samples: int
    room_entries: int
    game_over: bool
    is_win: bool
    paused: bool
    time_menu_open: float
    game_start_time: float


def _reset_room_counters(state: GameState) -> None:
    state.moves_made = 0
    state.enemy_kills = 0
    state.enemy_hits_taken = 0
    state.damage_blocked = 0
    state.powerups_collected = 0
    state.heal_collected = 0
    state.speed_collected = 0
    state.shield_collected = 0
    state.teleport_uses = 0
    state.enemy_distance_accum = 0.0
    state.enemy_distance_samples = 0


def _set_status(state: GameState, message: str, duration: float = 2.5) -> None:
    state.status_message = message
    state.status_timer = duration


def _objective_status(room: DungeonRoom) -> str:
    objective = room.objective
    label = OBJECTIVE_LABELS.get(objective.kind, objective.kind)
    if objective.kind == "none":
        return "Objective: optional"
    if objective.completed:
        return f"Optional objective: {label} [OK]"
    if objective.kind in {"survive", "protect_relic"}:
        remaining = max(0.0, objective.timer_target - objective.progress)
        return f"Optional objective: {label} ({remaining:.0f}s)"
    if objective.kind == "altar" and objective.altar_pos is not None:
        return f"Optional objective: {label} @ {objective.altar_pos[0]},{objective.altar_pos[1]}"
    return f"Optional objective: {label}"


def _objective_brief(room: DungeonRoom) -> str:
    objective = room.objective
    if objective.kind == "none":
        return "No objective"
    if objective.completed:
        return f"Obj {OBJECTIVE_LABELS.get(objective.kind, objective.kind)} OK"
    if objective.kind in {"survive", "protect_relic"}:
        remaining = max(0.0, objective.timer_target - objective.progress)
        return f"Obj {OBJECTIVE_LABELS.get(objective.kind, objective.kind)} {remaining:.0f}s"
    return f"Obj {OBJECTIVE_LABELS.get(objective.kind, objective.kind)}"


def _qualifies_for_combo(time_taken: float, hp_lost: int) -> bool:
    return False


def _advance_combo(state: GameState, time_taken: float, hp_lost: int) -> None:
    state.combo_streak = 0
    state.combo_reward_pending = False


def _role_weights(difficulty: int) -> Dict[str, float]:
    return {
        "stalker": 1.0,
    }


def _weighted_pick(weights: Dict[str, float], fallback: str) -> str:
    total = sum(max(0.01, value) for value in weights.values())
    roll = random.random() * total
    acc = 0.0
    for key, value in weights.items():
        acc += max(0.01, value)
        if roll <= acc:
            return key
    return fallback

def _show_startup_menu(screen: pygame.Surface) -> dict:
    import sys
    import config
    font_l = pygame.font.SysFont("consolas", 40, bold=True)
    font_s = pygame.font.SysFont("consolas", 22)

    cfg = {"bandit": True, "density": 1.0, "rooms": 10, "relics": 3}
    
    while True:
        screen.fill((20, 20, 25))
        title = font_l.render("GAME CONFIGURATION", True, (255, 255, 255))
        
        opts = [
            f"[1] Bandit Mode: {'ON' if cfg['bandit'] else 'OFF'}",
            f"[O/P] Enemy Density: {cfg['density']:.1f}x",
            f"[UP/DOWN] Total Rooms: {cfg['rooms']}",
            f"[LEFT/RIGHT] Relics Needed: {cfg['relics']}",
            "Press [ENTER] to START"
        ]
        
        screen.blit(title, (screen.get_width()//2 - title.get_width()//2, 100))
        for i, text in enumerate(opts):
            surf = font_s.render(text, True, (200, 200, 200))
            screen.blit(surf, (screen.get_width()//2 - surf.get_width()//2, 250 + i * 50))
            
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: cfg['bandit'] = not cfg['bandit']
                elif event.key == pygame.K_o: cfg['density'] = max(0.5, cfg['density'] - 0.1)
                elif event.key == pygame.K_p: cfg['density'] = min(2.0, cfg['density'] + 0.1)
                elif event.key == pygame.K_UP: cfg['rooms'] = min(50, cfg['rooms'] + 1)
                elif event.key == pygame.K_DOWN: cfg['rooms'] = max(5, cfg['rooms'] - 1)
                elif event.key == pygame.K_LEFT: cfg['relics'] = max(1, cfg['relics'] - 1)
                elif event.key == pygame.K_RIGHT: cfg['relics'] = min(10, cfg['relics'] + 1)
                elif event.key == pygame.K_RETURN: return cfg       

def _assign_enemy_roles(
    positions: Set[Vec2],
    difficulty: int,
    special_room_type: Optional[str],
    elite_modifier: Optional[str],
    active_event: str,
    include_predator: bool,
) -> Dict[Vec2, EnemyActor]:
    enemies: Dict[Vec2, EnemyActor] = {}
    weights = _role_weights(difficulty)
    hp_by_difficulty = {0: 2, 1: 3, 2: 4}
    enemy_hp = hp_by_difficulty.get(difficulty, 2)
    for pos in positions:
        role = _weighted_pick(weights, fallback="stalker")
        enemies[pos] = EnemyActor(role=role, cooldown=0, hp=enemy_hp)
    return enemies


def _available_special_positions(grid: np.ndarray, reserved: Set[Vec2]) -> list[Vec2]:
    pool = _available_floor_positions(grid, reserved)
    return pool


def _build_room_objective(room: DungeonRoom) -> RoomObjective:
    return RoomObjective(kind="none", completed=True)


def _engagement_snapshot(state: GameState, dungeon: DungeonState) -> dict:
    avg_enemy_distance = 0.0
    if state.enemy_distance_samples > 0:
        avg_enemy_distance = state.enemy_distance_accum / state.enemy_distance_samples
    elapsed = max(0.1, time.time() - state.room_start_time)
    key_press_rate = round(state.moves_made / elapsed, 3)
    return {
        "dungeon_room_id": state.current_room_id,
        "flow_score": state.flow_score,
        "enemy_delay": state.enemy_delay,
        "enemy_spawned": state.enemies_spawned,
        "enemy_hits_taken": state.enemy_hits_taken,
        "damage_blocked": state.damage_blocked,
        "powerups_spawned": state.powerups_spawned,
        "powerups_collected": state.powerups_collected,
        "heal_collected": state.heal_collected,
        "speed_collected": state.speed_collected,
        "shield_collected": state.shield_collected,
        "teleport_present": state.teleports is not None,
        "teleport_uses": state.teleport_uses,
        "moves_made": state.moves_made,
        "key_press_rate": key_press_rate,
        "avg_enemy_distance": avg_enemy_distance,
        "enemy_kills": state.enemy_kills,
        "enemies_remaining": len(state.enemies),
        "powerups_remaining": len(state.powerups),
        "room_entries": state.room_entries,
        "combo_streak": state.combo_streak,
        "speed_boost_steps": state.speed_boost_steps,
        "shield_hits": state.shield_hits,
        "room_profile": state.room_profile,
        "room_special_type": state.room_special_type,
        "goals_current": dungeon.collected_relics,
        "relics_collected": dungeon.collected_relics,
        "relic_goal": dungeon.relic_goal,
        "final_door_open": dungeon.collected_relics >= dungeon.relic_goal,
        "objective_kind": state.objective_kind,
        "elite_modifier": state.elite_modifier,
        "dungeon_event": dungeon.active_event,
    }


def is_walkable(grid: np.ndarray, pos: Vec2) -> bool:
    x, y = pos
    if x < 0 or y < 0 or x >= GRID_WIDTH or y >= GRID_HEIGHT:
        return False
    return int(grid[y, x]) not in {WALL, DOOR_LOCKED}


def _enemy_delay(difficulty: int, flow_score: float) -> float:
    base = ENEMY_BASE_MOVE_DELAY.get(difficulty, 0.50)
    flow = max(0.0, min(1.0, flow_score))
    scale = 1.10 - 0.50 * flow
    min_delay = {
        0: 0.32,
        1: 0.24,
        2: 0.18,
    }.get(difficulty, 0.20)
    return max(min_delay, base * scale)


def _parse_generated_room(grid: np.ndarray) -> Set[Vec2]:
    ys, xs = np.where(grid == ENEMY)
    enemies = {(int(x), int(y)) for y, x in zip(ys, xs)}
    grid[grid == ENEMY] = FLOOR
    grid[grid == EXIT] = FLOOR
    return enemies


def _available_floor_positions(grid: np.ndarray, reserved: Set[Vec2]) -> list[Vec2]:
    out: list[Vec2] = []
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if int(grid[y, x]) == FLOOR and (x, y) not in reserved and not (x <= 1 and y <= 1):
                out.append((x, y))
    random.shuffle(out)
    return out


def spawn_room_features(
    grid: np.ndarray,
    enemies: Set[Vec2],
    exits: Dict[Vec2, int],
    difficulty: int,
    reserved_points: Set[Vec2],
    bonus_powerups: int = 0,
    force_heal: bool = False,
) -> Tuple[Dict[Vec2, int], Optional[Tuple[Vec2, Vec2]]]:
    reserved = set(exits.keys()) | reserved_points | enemies | {(0, 0)}
    pool = _available_floor_positions(grid, reserved)

    powerups: Dict[Vec2, int] = {}
    powerup_count = {0: 3, 1: 4, 2: 5}.get(difficulty, 4) + bonus_powerups
    weights = [(POWER_HEAL, 0.55), (POWER_SPEED, 0.45)]

    for _ in range(min(powerup_count, len(pool))):
        pos = pool.pop()
        if force_heal and not powerups:
            chosen = POWER_HEAL
        else:
            roll = random.random()
            acc = 0.0
            chosen = POWER_HEAL
            for p_type, w in weights:
                acc += w
                if roll <= acc:
                    chosen = p_type
                    break
        powerups[pos] = chosen

    teleports: Optional[Tuple[Vec2, Vec2]] = None
    teleport_chance = {0: 0.30, 1: 0.45, 2: 0.60}.get(difficulty, 0.40)
    if len(pool) >= 2 and random.random() < teleport_chance:
        teleports = (pool.pop(), pool.pop())
    return powerups, teleports


def _build_dungeon_layout(num_rooms: int) -> Tuple[Dict[int, Vec2], Dict[int, Set[int]], int, int]:
    start = (DUNGEON_MAP_W // 2, DUNGEON_MAP_H // 2)
    occupied = {start}
    order = [start]

    attempts = 0
    while len(order) < num_rooms and attempts < 3000:
        attempts += 1
        bx, by = random.choice(order)
        neighbors = [(bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)]
        random.shuffle(neighbors)
        for nx, ny in neighbors:
            if nx < 0 or ny < 0 or nx >= DUNGEON_MAP_W or ny >= DUNGEON_MAP_H:
                continue
            if (nx, ny) in occupied:
                continue
            occupied.add((nx, ny))
            order.append((nx, ny))
            break

    if len(order) < num_rooms:
        for y in range(DUNGEON_MAP_H):
            for x in range(DUNGEON_MAP_W):
                if (x, y) not in occupied:
                    occupied.add((x, y))
                    order.append((x, y))
                if len(order) >= num_rooms:
                    break
            if len(order) >= num_rooms:
                break

    coords_by_room: Dict[int, Vec2] = {idx: coord for idx, coord in enumerate(order)}
    room_by_coord: Dict[Vec2, int] = {coord: rid for rid, coord in coords_by_room.items()}
    adjacency: Dict[int, Set[int]] = {rid: set() for rid in coords_by_room}

    for rid, (x, y) in coords_by_room.items():
        for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            other = room_by_coord.get((nx, ny))
            if other is not None:
                adjacency[rid].add(other)

    start_room = 0
    dist = {start_room: 0}
    q: deque[int] = deque([start_room])
    while q:
        cur = q.popleft()
        for nxt in adjacency[cur]:
            if nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            q.append(nxt)

    final_room = max(dist, key=lambda rid: dist[rid]) if dist else start_room
    return coords_by_room, adjacency, start_room, final_room


def _exit_layout(room_id: int, dungeon: DungeonState) -> Dict[Vec2, int]:
    x, y = dungeon.coords_by_room[room_id]
    out: Dict[Vec2, int] = {}

    left_slot = (0, random.randint(2, GRID_HEIGHT - 3))
    right_slot = (GRID_WIDTH - 1, random.randint(2, GRID_HEIGHT - 3))
    top_slot = (random.randint(2, GRID_WIDTH - 3), 0)
    bottom_slot = (random.randint(2, GRID_WIDTH - 3), GRID_HEIGHT - 1)

    for neighbor in dungeon.adjacency[room_id]:
        nx, ny = dungeon.coords_by_room[neighbor]
        if nx == x and ny == y - 1:
            out[top_slot] = neighbor
        elif nx == x and ny == y + 1:
            out[bottom_slot] = neighbor
        elif nx == x - 1 and ny == y:
            out[left_slot] = neighbor
        elif nx == x + 1 and ny == y:
            out[right_slot] = neighbor
    return out


def _spawn_points_for_exits(exits: Dict[Vec2, int]) -> Dict[Vec2, Vec2]:
    spawn_points: Dict[Vec2, Vec2] = {}
    for (x, y) in exits:
        if x == 0:
            spawn_points[(x, y)] = (1, y)
        elif x == GRID_WIDTH - 1:
            spawn_points[(x, y)] = (GRID_WIDTH - 2, y)
        elif y == 0:
            spawn_points[(x, y)] = (x, 1)
        elif y == GRID_HEIGHT - 1:
            spawn_points[(x, y)] = (x, GRID_HEIGHT - 2)
    return spawn_points


def _carve_exit_access(grid: np.ndarray, pos: Vec2) -> None:
    x, y = pos
    grid[y, x] = EXIT
    if x == 0:
        for ny in range(max(0, y - 1), min(GRID_HEIGHT, y + 2)):
            grid[ny, 0] = EXIT if ny == y else FLOOR
            grid[ny, 1] = FLOOR
            if GRID_WIDTH > 2:
                grid[ny, 2] = FLOOR
    elif x == GRID_WIDTH - 1:
        for ny in range(max(0, y - 1), min(GRID_HEIGHT, y + 2)):
            grid[ny, GRID_WIDTH - 1] = EXIT if ny == y else FLOOR
            grid[ny, GRID_WIDTH - 2] = FLOOR
            if GRID_WIDTH > 2:
                grid[ny, GRID_WIDTH - 3] = FLOOR
    if y == 0:
        for nx in range(max(0, x - 1), min(GRID_WIDTH, x + 2)):
            grid[0, nx] = EXIT if nx == x else FLOOR
            grid[1, nx] = FLOOR
            if GRID_HEIGHT > 2:
                grid[2, nx] = FLOOR
    elif y == GRID_HEIGHT - 1:
        for nx in range(max(0, x - 1), min(GRID_WIDTH, x + 2)):
            grid[GRID_HEIGHT - 1, nx] = EXIT if nx == x else FLOOR
            grid[GRID_HEIGHT - 2, nx] = FLOOR
            if GRID_HEIGHT > 2:
                grid[GRID_HEIGHT - 3, nx] = FLOOR


def _spawn_position_for_entry(room: DungeonRoom, previous_room_id: Optional[int], dungeon: DungeonState) -> Vec2:
    if previous_room_id is None:
        if room.spawn_points:
            return sorted(room.spawn_points.values())[0]
        return (1, 1)

    for door_pos, target_room_id in room.exits.items():
        if target_room_id == previous_room_id:
            return room.spawn_points.get(door_pos, (1, 1))

    if room.spawn_points:
        return sorted(room.spawn_points.values())[0]
    return (1, 1)


def _layout_distances(adjacency: Dict[int, Set[int]], start_room: int) -> Dict[int, int]:
    dist = {start_room: 0}
    queue: deque[int] = deque([start_room])
    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            q.append(nxt)
    return dist


def _pick_special_rooms(
    adjacency: Dict[int, Set[int]],
    start_room: int,
    final_room: int,
) -> Tuple[Set[int], Set[int], Set[int], Set[int], Optional[int]]:
    return set(), set(), set(), set(), None


def _choose_special_room_type(room_id: int, dungeon: DungeonState, engagement: float, force_reward_room: bool) -> Optional[str]:
    return None


def _should_be_elite(room_id: int, dungeon: DungeonState, engagement: float, special_room_type: Optional[str]) -> bool:
    return False


def _pick_elite_modifier(room_id: int) -> str:
    modifiers = ["fast_enemies", "fog", "half_heal", "double_relic"]
    return modifiers[room_id % len(modifiers)]


def _spawn_relics(
    grid: np.ndarray,
    reserved: Set[Vec2],
    count: int,
) -> Set[Vec2]:
    relic_positions: Set[Vec2] = set()
    pool = _available_floor_positions(grid, reserved)
    for _ in range(min(count, len(pool))):
        pos = pool.pop()
        relic_positions.add(pos)
        grid[pos[1], pos[0]] = RELIC
    return relic_positions


def _sync_final_door(dungeon: DungeonState) -> None:
    room = dungeon.rooms.get(dungeon.final_room)
    if room is None or room.final_door_pos is None:
        return
    x, y = room.final_door_pos
    door_open = dungeon.collected_relics >= dungeon.relic_goal
    room.grid[y, x] = DOOR_OPEN if door_open else DOOR_LOCKED


def _mutate_room_memory(room: DungeonRoom) -> None:
    if room.memory_shifted:
        return
    reserved = set(room.exits.keys()) | set(room.spawn_points.values()) | set(room.relic_positions)
    if room.objective.altar_pos is not None:
        reserved.add(room.objective.altar_pos)
    candidates = _available_floor_positions(room.grid, reserved | set(room.enemies.keys()) | set(room.powerups.keys()))
    for _ in range(min(4, len(candidates))):
        x, y = candidates.pop()
        room.grid[y, x] = WALL if random.random() < 0.45 else FLOOR
    for _ in range(min(2, len(candidates))):
        pos = candidates.pop()
        room.enemies[pos] = EnemyActor(role=random.choice(["stalker", "charger", "ranged"]))
    if candidates and len(room.powerups) < 4:
        room.powerups[candidates.pop()] = random.choice([POWER_HEAL, POWER_SPEED, POWER_SHIELD])
    room.memory_shifted = True


def _rotate_dungeon_event(dungeon: DungeonState, room_step: int) -> None:
    dungeon.active_event = "calm"


def _advance_predator(dungeon: DungeonState, player_room_id: int) -> None:
    return


def _apply_room_entry_effects(state: GameState, dungeon: DungeonState, room: DungeonRoom) -> None:
    state.room_special_type = room.special_room_type or "normal"
    state.elite_modifier = ""
    state.objective_kind = "none"


def _grant_objective_reward(state: GameState, room: DungeonRoom) -> None:
    objective = room.objective
    if objective.kind == "none" or objective.reward_granted or not objective.completed:
        return
    objective.reward_granted = True
    if objective.kind == "survive":
        state.shield_hits += 1
        _set_status(state, "Objective reward: +1 shield")
    elif objective.kind == "eliminate":
        state.speed_boost_steps += 4
        _set_status(state, "Objective reward: speed boost")
    elif objective.kind == "altar":
        state.hp = min(MAX_HP + state.overheal_hp, state.hp + 12)
        _set_status(state, "Objective reward: +12 HP")
    elif objective.kind == "protect_relic":
        state.overheal_hp = min(MAX_OVERHEAL, state.overheal_hp + 10)
        state.hp = min(MAX_HP + state.overheal_hp, state.hp + 10)
        _set_status(state, "Objective reward: overheal")


def _update_room_objective(state: GameState, room: DungeonRoom, dt_sec: float) -> None:
    objective = room.objective
    if objective.completed:
        return
    if objective.kind == "none":
        objective.completed = True
        return
    if objective.kind == "survive":
        objective.progress += dt_sec
        objective.completed = objective.progress >= objective.timer_target
    elif objective.kind == "eliminate":
        objective.completed = not room.enemies
    elif objective.kind == "altar":
        objective.completed = objective.altar_pos is not None and state.player_pos == objective.altar_pos
    elif objective.kind == "protect_relic":
        objective.progress += dt_sec
        for enemy_pos in room.enemies:
            for relic_pos in room.relic_positions:
                if abs(enemy_pos[0] - relic_pos[0]) + abs(enemy_pos[1] - relic_pos[1]) <= 1:
                    objective.progress = max(0.0, objective.progress - dt_sec * 2.0)
        objective.completed = objective.progress >= objective.timer_target
    if objective.completed:
        _grant_objective_reward(state, room)


def _room_exits_unlocked(room: DungeonRoom) -> bool:
    return True

def _create_room(
    room_id: int,
    dungeon: DungeonState,
    director: Director,
    stats: Optional[Tuple[float, int]],
    key_press_rate: float = 0.0,
    force_reward_room: bool = False,
) -> DungeonRoom:
    exits = _exit_layout(room_id, dungeon)
    
    grid, difficulty, attempts = director.generate_next_room(stats, key_press_rate=key_press_rate, exits=exits)
    
    engagement = director.last_engagement
    base_enemy_positions = _parse_generated_room(grid)
    
    for pos in exits:
        _carve_exit_access(grid, pos)
        
    spawn_points = _spawn_points_for_exits(exits)
    for sx, sy in spawn_points.values():
        grid[sy, sx] = FLOOR
        
    safe_start = list(spawn_points.values())[0] if spawn_points else (GRID_WIDTH // 2, GRID_HEIGHT // 2)
    all_starts = set(spawn_points.values()) if spawn_points else {safe_start}
    _remove_disconnected_floors(grid, all_starts)

    spawn_list = list(spawn_points.values())
    for sp in spawn_list[1:]:
        if not _is_reachable(grid, safe_start, sp):
            _carve_corridor(grid, safe_start, sp)

    _clear_small_wall_islands(grid)
    _remove_disconnected_floors(grid, all_starts)

    reserved_points = set(exits.keys()) | set(spawn_points.values())
    
    base_enemy_positions = {
        enemy for enemy in base_enemy_positions 
        if enemy not in reserved_points and int(grid[enemy[1], enemy[0]]) == FLOOR
    }

    special_room_type = None
    is_elite = False
    elite_modifier = None

    final_door_pos: Optional[Vec2] = None
    if room_id == dungeon.final_room:
        final_door_pos = _pick_final_door_pos(grid, safe_start, reserved_points)
        grid[final_door_pos[1], final_door_pos[0]] = DOOR_LOCKED

    extra_engagement_spawns = int(round(max(0.0, engagement - 0.45) * 4))
    if extra_engagement_spawns > 0:
        pool = _available_floor_positions(grid, reserved_points | base_enemy_positions)
        for _ in range(min(extra_engagement_spawns, len(pool))):
            base_enemy_positions.add(pool.pop())

    powerups, teleports = spawn_room_features(
        grid,
        base_enemy_positions,
        exits,
        difficulty,
        reserved_points,
        bonus_powerups=0,
        force_heal=False,
    )

    relic_positions: Set[Vec2] = set()
    relic_count = 1 if room_id in dungeon.relic_rooms else 0
    relic_positions = _spawn_relics(grid, reserved_points | base_enemy_positions | set(powerups.keys()), relic_count)

    include_predator = False
    enemies = _assign_enemy_roles(
        base_enemy_positions,
        difficulty,
        special_room_type,
        elite_modifier,
        dungeon.active_event,
        include_predator,
    )

    room = DungeonRoom(
        room_id=room_id,
        grid=grid,
        difficulty=difficulty,
        verify_attempts=attempts,
        profile=director.generator.last_room_profile,
        exits=exits,
        spawn_points=spawn_points,
        risk_exits={pos for pos, nxt in exits.items() if nxt in dungeon.risk_reward_rooms},
        enemies=enemies,
        powerups=powerups,
        teleports=teleports,
        relic_positions=relic_positions,
        objective=RoomObjective(kind="eliminate"),
        final_door_pos=final_door_pos,
        special_room_type=special_room_type,
        is_elite=is_elite,
        elite_modifier=elite_modifier,
        is_reward_room=force_reward_room or special_room_type == "reward",
    )
    room.objective = _build_room_objective(room)
    dungeon.rooms[room_id] = room
    _sync_final_door(dungeon)
    return room

def _remove_disconnected_floors(grid: np.ndarray, start_positions) -> None:
    height, width = grid.shape
    reachable: Set[Vec2] = set()
    if isinstance(start_positions, tuple):
        queue: list = [start_positions]
        reachable.add(start_positions)
    else:
        queue = [p for p in start_positions if 0 <= p[0] < width and 0 <= p[1] < height]
        reachable.update(queue)

    while queue:
        x, y = queue.pop(0)
        for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            if 0 <= nx < width and 0 <= ny < height:
                if int(grid[ny, nx]) in {FLOOR, EXIT} and (nx, ny) not in reachable:
                    reachable.add((nx, ny))
                    queue.append((nx, ny))

    for y in range(height):
        for x in range(width):
            if int(grid[y, x]) == FLOOR and (x, y) not in reachable:
                grid[y, x] = WALL


def _is_reachable(grid: np.ndarray, src: Vec2, dst: Vec2) -> bool:
    height, width = grid.shape
    walkable = {FLOOR, EXIT, DOOR_OPEN}
    visited: Set[Vec2] = {src}
    queue: deque = deque([src])
    while queue:
        x, y = queue.popleft()
        if (x, y) == dst:
            return True
        for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                if int(grid[ny, nx]) in walkable:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return False


def _carve_corridor(grid: np.ndarray, src: Vec2, dst: Vec2) -> None:
    x, y = src
    tx, ty = dst
    _carve_corridor_brush(grid, src)
    while x != tx:
        x += 1 if tx > x else -1
        if 0 < x < GRID_WIDTH - 1 and 0 < y < GRID_HEIGHT - 1:
            _carve_corridor_brush(grid, (x, y))
    while y != ty:
        y += 1 if ty > y else -1
        if 0 < x < GRID_WIDTH - 1 and 0 < y < GRID_HEIGHT - 1:
            _carve_corridor_brush(grid, (x, y))
    _carve_corridor_brush(grid, dst)


def _carve_corridor_brush(grid: np.ndarray, center: Vec2) -> None:
    cx, cy = center
    for y in range(max(1, cy - 1), min(GRID_HEIGHT - 1, cy + 2)):
        for x in range(max(1, cx - 1), min(GRID_WIDTH - 1, cx + 2)):
            if int(grid[y, x]) not in {EXIT, DOOR_LOCKED, DOOR_OPEN}:
                grid[y, x] = FLOOR


def _clear_small_wall_islands(grid: np.ndarray) -> None:
    to_clear: Set[Vec2] = set()

    for y in range(1, GRID_HEIGHT - 1):
        for x in range(1, GRID_WIDTH - 1):
            if int(grid[y, x]) != WALL:
                continue
            neighbors = [
                int(grid[y - 1, x]),
                int(grid[y + 1, x]),
                int(grid[y, x - 1]),
                int(grid[y, x + 1]),
            ]
            floorish = sum(value in {FLOOR, EXIT, DOOR_OPEN} for value in neighbors)
            if floorish >= 3:
                to_clear.add((x, y))

    for y in range(1, GRID_HEIGHT - 2):
        for x in range(1, GRID_WIDTH - 2):
            block = ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1))
            if not all(int(grid[by, bx]) == WALL for bx, by in block):
                continue
            ring = [
                (x - 1, y), (x - 1, y + 1),
                (x + 2, y), (x + 2, y + 1),
                (x, y - 1), (x + 1, y - 1),
                (x, y + 2), (x + 1, y + 2),
            ]
            floorish_ring = sum(int(grid[ry, rx]) in {FLOOR, EXIT, DOOR_OPEN} for rx, ry in ring)
            if floorish_ring >= 6:
                to_clear.update(block)

    for x, y in to_clear:
        grid[y, x] = FLOOR


def _pick_final_door_pos(grid: np.ndarray, safe_start: Vec2, forbidden: Set[Vec2]) -> Vec2:
    candidates = [
        (1, 1),
        (GRID_WIDTH - 2, 1),
        (1, GRID_HEIGHT - 2),
        (GRID_WIDTH - 2, GRID_HEIGHT - 2),
        (GRID_WIDTH // 2, 1),
        (GRID_WIDTH // 2, GRID_HEIGHT - 2),
        (1, GRID_HEIGHT // 2),
        (GRID_WIDTH - 2, GRID_HEIGHT // 2),
    ]

    valid: list[Vec2] = []
    for pos in candidates:
        x, y = pos
        if pos in forbidden:
            continue
        if int(grid[y, x]) not in {FLOOR, EXIT}:
            continue
        if _is_reachable(grid, safe_start, pos):
            valid.append(pos)

    if not valid:
        for y in range(1, GRID_HEIGHT - 1):
            for x in range(1, GRID_WIDTH - 1):
                pos = (x, y)
                if pos in forbidden:
                    continue
                if int(grid[y, x]) != FLOOR:
                    continue
                if _is_reachable(grid, safe_start, pos):
                    valid.append(pos)

    if not valid:
        return safe_start

    return max(valid, key=lambda p: abs(p[0] - safe_start[0]) + abs(p[1] - safe_start[1]))


def _enter_room(state: GameState, dungeon: DungeonState, room: DungeonRoom) -> None:
    state.current_room_id = room.room_id
    state.difficulty = room.difficulty
    state.verify_attempts = room.verify_attempts
    state.room_profile = room.profile
    state.enemies = room.enemies
    state.enemies_spawned = len(room.enemies)
    state.powerups = room.powerups
    state.powerups_spawned = len(room.powerups)
    state.teleports = room.teleports
    state.player_pos = _spawn_position_for_entry(room, state.previous_room_id, dungeon)
    state.enemy_delay = _enemy_delay(state.difficulty, state.flow_score)
    if room.is_elite and room.elite_modifier == "fast_enemies":
        state.enemy_delay = max(0.10, state.enemy_delay * 0.72)
    if room.special_room_type == "sanctuary":
        state.enemy_delay *= 1.25
    state.enemy_timer = 0.0
    state.room_start_time = time.time()
    state.room_start_hp = state.hp
    state.room_entries += 1
    _reset_room_counters(state)
    if room.visit_count > 0:
        _mutate_room_memory(room)
    room.visit_count += 1
    dungeon.discovered.add(room.room_id)
    _apply_room_entry_effects(state, dungeon, room)


def _collect_relic_if_any(state: GameState, dungeon: DungeonState, room: DungeonRoom) -> None:
    x, y = state.player_pos
    if int(room.grid[y, x]) != RELIC or (x, y) not in room.relic_positions:
        return
    room.grid[y, x] = FLOOR
    room.relic_positions.discard((x, y))
    dungeon.collected_relics += 1
    if room.room_id in dungeon.relic_rooms and not room.relic_positions:
        dungeon.relic_collected_rooms.add(room.room_id)
    _sync_final_door(dungeon)


def _update_caption(state: GameState, dungeon: DungeonState) -> None:
    dname = DIFFICULTY_NAMES.get(state.difficulty, str(state.difficulty))
    engagement_indicator = "↑" if state.flow_score > 0.6 else ("↓" if state.flow_score < 0.4 else "→")
    pygame.display.set_caption(
        f"Evolving Depths | Room #{state.current_room_id} | Step {state.room_step} | "
        f"Diff {dname} | Engagement {state.flow_score:.2f} {engagement_indicator} | "
        f"Relics {dungeon.collected_relics}/{dungeon.relic_goal} | Speed {state.speed_boost_steps} | Shield {state.shield_hits} | Combo {state.combo_streak}"
    )


def _draw_powerup_icon(screen: pygame.Surface, p_type: int, center: Vec2, radius: int) -> None:
    pygame.draw.circle(screen, COLORS[p_type], center, radius)
    pygame.draw.circle(screen, (18, 18, 18), center, radius, 2)

    if p_type == POWER_HEAL:
        pygame.draw.rect(screen, (245, 245, 245), pygame.Rect(center[0] - 2, center[1] - radius + 4, 4, 2 * radius - 8))
        pygame.draw.rect(screen, (245, 245, 245), pygame.Rect(center[0] - radius + 4, center[1] - 2, 2 * radius - 8, 4))
    elif p_type == POWER_SPEED:
        points = [
            (center[0] - radius // 2, center[1] - radius // 2),
            (center[0] + radius // 4, center[1]),
            (center[0] - radius // 2, center[1] + radius // 2),
            (center[0] + radius // 2, center[1]),
        ]
        pygame.draw.polygon(screen, (245, 245, 245), points)
    elif p_type == POWER_SHIELD:
        shield = [
            (center[0], center[1] - radius + 3),
            (center[0] + radius - 5, center[1] - radius // 3),
            (center[0] + radius // 2, center[1] + radius - 4),
            (center[0] - radius // 2, center[1] + radius - 4),
            (center[0] - radius + 5, center[1] - radius // 3),
        ]
        pygame.draw.polygon(screen, (245, 245, 245), shield, 2)


def draw_minimap(screen: pygame.Surface, state: GameState, dungeon: DungeonState, font: Optional[pygame.font.Font]) -> None:
    panel_w = 360
    panel_h = SCREEN_HEIGHT - GRID_HEIGHT * TILE_SIZE - 16
    panel = pygame.Rect(SCREEN_WIDTH - panel_w - 12, GRID_HEIGHT * TILE_SIZE + 8, panel_w, panel_h)
    pygame.draw.rect(screen, MINIMAP_BG, panel)
    pygame.draw.rect(screen, (60, 65, 75), panel, 1)

    cell = 24
    ox = panel.x + 12
    oy = panel.y + 28

    for rid, (mx, my) in dungeon.coords_by_room.items():
        if rid not in dungeon.discovered and rid != state.current_room_id:
            continue
        for n in dungeon.adjacency[rid]:
            if n not in dungeon.discovered and n != state.current_room_id:
                continue
            nx, ny = dungeon.coords_by_room[n]
            x1 = ox + mx * cell + cell // 2
            y1 = oy + my * cell + cell // 2
            x2 = ox + nx * cell + cell // 2
            y2 = oy + ny * cell + cell // 2
            pygame.draw.line(screen, (72, 78, 90), (x1, y1), (x2, y2), 2)

    for rid, (mx, my) in dungeon.coords_by_room.items():
        rect = pygame.Rect(ox + mx * cell, oy + my * cell, cell - 2, cell - 2)
        discovered = rid in dungeon.discovered
        color = (58, 62, 72) if discovered else (35, 38, 44)

        if rid == dungeon.final_room:
            color = (115, 86, 150) if discovered else (50, 43, 60)
        if rid in dungeon.relic_rooms and discovered:
            color = COLORS[RELIC]
        if rid == state.current_room_id:
            color = PLAYER_COLOR

        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (15, 15, 15), rect, 1)

    if font is not None:
        screen.blit(font.render("Room Map", True, TEXT_COLOR), (panel.x + 12, panel.y + 4))
        room_elapsed = max(0.0, time.time() - state.room_start_time)
        total_elapsed = max(0.0, time.time() - state.game_start_time - state.time_menu_open)
        timer_text = f"Room {room_elapsed:.1f}s  |  Total {total_elapsed:.1f}s"
        screen.blit(font.render(timer_text, True, (255, 255, 255)), (panel.x + 170, panel.y + 4))


def draw_pause_screen(screen: pygame.Surface, font: Optional[pygame.font.Font], state: GameState, dungeon: DungeonState, room: DungeonRoom) -> None:
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    screen.blit(overlay, (0, 0))
    
    if font is None:
        return
    
    panel_width = 500
    panel_height = 400
    panel_x = (SCREEN_WIDTH - panel_width) // 2
    panel_y = (SCREEN_HEIGHT - panel_height) // 2
    panel = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
    pygame.draw.rect(screen, HUD_BG, panel)
    pygame.draw.rect(screen, (120, 120, 120), panel, 3)
    
    y_offset = panel_y + 20
    
    title = font.render("PAUSED", True, (255, 200, 100))
    screen.blit(title, (panel_x + (panel_width - title.get_width()) // 2, y_offset))
    y_offset += 40
    
    dname = DIFFICULTY_NAMES.get(state.difficulty, str(state.difficulty))
    engagement = f"{state.flow_score:.2f}"
    line1 = f"Engagement: {engagement} | Difficulty: {dname}"
    screen.blit(font.render(line1, True, TEXT_COLOR), (panel_x + 20, y_offset))
    y_offset += 25
    
    line2 = f"Room: #{state.current_room_id}"
    screen.blit(font.render(line2, True, TEXT_COLOR), (panel_x + 20, y_offset))
    y_offset += 25
    
    line3 = f"HP: {state.hp}/{MAX_HP}"
    screen.blit(font.render(line3, True, TEXT_COLOR), (panel_x + 20, y_offset))
    y_offset += 25
    
    line4 = f"Relics: {dungeon.collected_relics}/{dungeon.relic_goal}"
    screen.blit(font.render(line4, True, TEXT_COLOR), (panel_x + 20, y_offset))
    y_offset += 25
    
    line5 = "Event: calm"
    screen.blit(font.render(line5, True, TEXT_COLOR), (panel_x + 20, y_offset))
    y_offset += 25
    
    line6 = "Objective: none"
    screen.blit(font.render(line6, True, TEXT_COLOR), (panel_x + 20, y_offset))
    y_offset += 40
    
    resume_text = "Press ESC or P to resume"
    screen.blit(font.render(resume_text, True, (150, 200, 150)), (panel_x + (panel_width - len(resume_text) * 8) // 2, panel_y + panel_height - 30))


def draw_grid(screen: pygame.Surface, room: DungeonRoom, player_pos: Vec2) -> None:
    grid = room.grid
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            cell = int(grid[y, x])
            base = FLOOR if cell not in COLORS else cell
            pygame.draw.rect(screen, COLORS[base], rect)
            pygame.draw.rect(screen, (18, 18, 18), rect, 1)

            if cell == EXIT and (x, y) in room.risk_exits:
                pygame.draw.rect(screen, RISK_EXIT_COLOR, rect, 3)

    for (x, y), p_type in room.powerups.items():
        center = (x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2)
        _draw_powerup_icon(screen, p_type, center, TILE_SIZE // 4)

    for spawn_pos in room.spawn_points.values():
        pass

    for rx, ry in room.relic_positions:
        if int(grid[ry, rx]) != RELIC:
            continue
        center = (rx * TILE_SIZE + TILE_SIZE // 2, ry * TILE_SIZE + TILE_SIZE // 2)
        points = [
            (center[0], center[1] - 10),
            (center[0] + 8, center[1]),
            (center[0], center[1] + 10),
            (center[0] - 8, center[1]),
        ]
        pygame.draw.polygon(screen, COLORS[RELIC], points)
        pygame.draw.polygon(screen, (30, 30, 30), points, 1)

    if room.objective.kind == "altar" and room.objective.altar_pos is not None:
        ax, ay = room.objective.altar_pos
        center = (ax * TILE_SIZE + TILE_SIZE // 2, ay * TILE_SIZE + TILE_SIZE // 2)
        pygame.draw.circle(screen, ALTAR_COLOR, center, TILE_SIZE // 4)
        pygame.draw.circle(screen, (245, 235, 210), center, TILE_SIZE // 7)

    if room.teleports is not None:
        (ax, ay), (bx, by) = room.teleports
        pygame.draw.rect(screen, COLORS[TELEPORT_A], pygame.Rect(ax * TILE_SIZE + 8, ay * TILE_SIZE + 8, TILE_SIZE - 16, TILE_SIZE - 16), 3)
        pygame.draw.rect(screen, COLORS[TELEPORT_B], pygame.Rect(bx * TILE_SIZE + 8, by * TILE_SIZE + 8, TILE_SIZE - 16, TILE_SIZE - 16), 3)

    if room.final_door_pos is not None:
        fx, fy = room.final_door_pos
        door_color = COLORS[DOOR_OPEN] if int(grid[fy, fx]) == DOOR_OPEN else COLORS[DOOR_LOCKED]
        pygame.draw.rect(screen, door_color, pygame.Rect(fx * TILE_SIZE, fy * TILE_SIZE, TILE_SIZE * 2, TILE_SIZE * 2), 0)
        pygame.draw.rect(screen, (200, 200, 200), pygame.Rect(fx * TILE_SIZE, fy * TILE_SIZE, TILE_SIZE * 2, TILE_SIZE * 2), 3)

    for (x, y), enemy in room.enemies.items():
        rect = pygame.Rect(x * TILE_SIZE + 4, y * TILE_SIZE + 4, TILE_SIZE - 8, TILE_SIZE - 8)
        pygame.draw.rect(screen, COLORS[ENEMY], rect)

    px, py = player_pos
    pygame.draw.rect(screen, PLAYER_COLOR, pygame.Rect(px * TILE_SIZE, py * TILE_SIZE, TILE_SIZE, TILE_SIZE))


def _draw_blackout_overlay(screen: pygame.Surface, state: GameState, room: DungeonRoom, dungeon: DungeonState) -> None:
    return


def draw_hud(screen: pygame.Surface, font: Optional[pygame.font.Font], state, dungeon) -> None:
    hud_height = SCREEN_HEIGHT - (GRID_HEIGHT * TILE_SIZE)
    hud_rect = pygame.Rect(0, GRID_HEIGHT * TILE_SIZE, SCREEN_WIDTH, hud_height)
    pygame.draw.rect(screen, HUD_BG, hud_rect)

    base_y = GRID_HEIGHT * TILE_SIZE
    hp_ratio = max(0.0, min(1.0, state.hp / max(1, MAX_HP)))
    pygame.draw.rect(screen, (52, 56, 66), pygame.Rect(16, base_y + 12, 220, 18))
    pygame.draw.rect(screen, (208, 77, 77), pygame.Rect(16, base_y + 12, int(220 * hp_ratio), 18))

    relic_ratio = dungeon.collected_relics / max(1, dungeon.relic_goal)
    pygame.draw.rect(screen, (52, 56, 66), pygame.Rect(16, base_y + 40, 220, 14))
    pygame.draw.rect(screen, COLORS[RELIC], pygame.Rect(16, base_y + 40, int(220 * relic_ratio), 14))

    flow_ratio = max(0.0, min(1.0, state.flow_score))
    pygame.draw.rect(screen, (52, 56, 66), pygame.Rect(16, base_y + 62, 220, 14))
    pygame.draw.rect(screen, (98, 168, 230), pygame.Rect(16, base_y + 62, int(220 * flow_ratio), 14))

    panel_x, panel_y = 16, base_y + 84
    for idx, p_type in enumerate([POWER_HEAL, POWER_SPEED]):
        box = pygame.Rect(panel_x + idx * 76, panel_y, 64, 28)
        pygame.draw.rect(screen, PANEL_BG, box, border_radius=6)
        pygame.draw.rect(screen, COLORS[p_type], box, 2, border_radius=6)
        _draw_powerup_icon(screen, p_type, (box.x + 16, box.y + 14), 9)
        
        if p_type == POWER_SPEED and state.speed_boost_steps > 0:
            pygame.draw.rect(screen, (245, 245, 245), box, 2, border_radius=6)

    if font is not None:
        text_x = 260 
        line_height = 20
        current_y = base_y + 10
        
        dname = DIFFICULTY_NAMES.get(state.difficulty, str(state.difficulty))
        engagement = "↑ RISING" if state.flow_score > 0.6 else ("↓ FALLING" if state.flow_score < 0.4 else "→ STABLE")
        
        lines = [
            f"Room: #{state.current_room_id} | Diff: {dname} | Eng: {state.flow_score:.2f} {engagement}",
            f"Relics: {dungeon.collected_relics}/{dungeon.relic_goal} — Reach the exit",
            f"HP: {state.hp}/{MAX_HP}",
            "Legend: Red=Enemy | Blue=Door | White=Relic"
        ]
        
        for line in lines:
            screen.blit(font.render(line, True, TEXT_COLOR), (text_x, current_y))
            current_y += line_height

        if state.status_timer > 0 and state.status_message:
            screen.blit(font.render(state.status_message, True, (255, 232, 160)), (text_x, current_y + 10))

    draw_minimap(screen, state, dungeon, font)


def _update_in_room_flow(state: GameState, director: Director) -> None:
    elapsed = max(0.5, time.time() - state.room_start_time)
    kpr = state.moves_made / elapsed
    avg_enemy_distance = 0.0
    if state.enemy_distance_samples > 0:
        avg_enemy_distance = state.enemy_distance_accum / state.enemy_distance_samples

    # Mid-room: time-to-complete is unknown so we cannot use reward_from_metrics.
    # Use only instantly-available signals: KPR (active input) and enemy proximity.
    # Both are monotonic: more input and closer enemies = higher engagement.
    kpr_signal = min(1.0, kpr / 3.0)                                    # saturates at 3 kp/s
    proximity_signal = max(0.0, 1.0 - min(avg_enemy_distance, 12.0) / 12.0)  # closer = higher
    inferred = 0.55 * kpr_signal + 0.45 * proximity_signal

    # Slow drift: flow score moves gradually, not frame-to-frame.
    state.flow_score = 0.92 * state.flow_score + 0.08 * inferred
    state.enemy_delay = _enemy_delay(state.difficulty, state.flow_score)

    if state.enemies:
        px, py = state.player_pos
        min_dist = min(abs(px - ex) + abs(py - ey) for ex, ey in state.enemies.keys())
        state.enemy_distance_accum += float(min_dist)
        state.enemy_distance_samples += 1


def _bfs_next_step(start: Vec2, goal: Vec2, room: DungeonRoom, blocked: Set[Vec2]) -> Vec2:
    if start == goal:
        return start

    queue: deque[Vec2] = deque([start])
    parent: Dict[Vec2, Optional[Vec2]] = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        x, y = cur
        for nxt in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            if nxt in parent:
                continue
            if nxt in blocked and nxt != goal:
                continue
            if not is_walkable(room.grid, nxt):
                continue
            parent[nxt] = cur
            queue.append(nxt)

    if goal not in parent:
        return start
    step = goal
    while parent[step] is not None and parent[step] != start:
        step = parent[step]
    return step


def _apply_enemy_damage(state: GameState, amount: int) -> None:
    state.hp = max(0, state.hp - amount)
    state.enemy_hits_taken += 1


def _apply_powerup(state: GameState, room: DungeonRoom, ptype: int) -> None:
    state.powerups_collected += 1
    if ptype == POWER_HEAL:
        state.hp = min(MAX_HP, state.hp + HEAL_AMOUNT)
        state.heal_collected += 1
    elif ptype == POWER_SPEED:
        state.speed_boost_steps += SPEED_BOOST_STEPS
        state.speed_collected += 1
    elif ptype == POWER_SHIELD:
        state.hp = min(MAX_HP, state.hp + HEAL_AMOUNT // 2)


def _try_teleport(state: GameState, room: DungeonRoom) -> None:
    if room.teleports is None:
        return
    a, b = room.teleports
    if state.player_pos == a:
        state.player_pos = b
        state.teleport_uses += 1
    elif state.player_pos == b:
        state.player_pos = a
        state.teleport_uses += 1


def _spawn_split_children(room: DungeonRoom, pos: Vec2) -> None:
    x, y = pos
    for child in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
        if child in room.enemies:
            continue
        if not is_walkable(room.grid, child):
            continue
        room.enemies[child] = EnemyActor(role="stalker")
        if len(room.enemies) >= 16:
            break


def _resolve_player_tile(state: GameState, room: DungeonRoom, dungeon: DungeonState) -> None:
    if state.player_pos in room.enemies:
        room.enemies.pop(state.player_pos)
        state.enemy_kills += 1
        _apply_enemy_damage(state, ENEMY_CONTACT_DAMAGE // 2)

    p = room.powerups.pop(state.player_pos, None)
    if p is not None:
        _apply_powerup(state, room, p)

    _collect_relic_if_any(state, dungeon, room)
    _try_teleport(state, room)


def _move_player(state: GameState, room: DungeonRoom, dungeon: DungeonState, dx: int, dy: int) -> None:
    if dx == 0 and dy == 0:
        return

    steps = 2 if state.speed_boost_steps > 0 else 1
    for _ in range(steps):
        nx, ny = state.player_pos[0] + dx, state.player_pos[1] + dy
        nxt = (nx, ny)
        if not is_walkable(room.grid, nxt):
            break
        state.player_pos = nxt
        state.moves_made += 1
        _resolve_player_tile(state, room, dungeon)

    if state.speed_boost_steps > 0:
        state.speed_boost_steps -= 1


def _line_distance(a: Vec2, b: Vec2) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _move_enemies(state: GameState, room: DungeonRoom) -> None:
    if not room.enemies:
        return

    moved: Dict[Vec2, EnemyActor] = {}
    remaining = set(room.enemies.keys())
    for enemy_pos, enemy in sorted(room.enemies.items()):
        remaining.discard(enemy_pos)
        enemy.cooldown = max(0, enemy.cooldown - 1)

        if enemy.cooldown > 0:
            moved[enemy_pos] = enemy
            continue

        steps = 1
        next_pos = enemy_pos
        for _ in range(steps):
            blocked = set(moved.keys()) | remaining
            candidate = _bfs_next_step(next_pos, state.player_pos, room, blocked)
            if candidate == state.player_pos:
                damage = ENEMY_CONTACT_DAMAGE // 2
                _apply_enemy_damage(state, damage)
                enemy.cooldown = 4
                break
            if candidate == next_pos:
                break
            next_pos = candidate

        moved[next_pos] = enemy

    room.enemies = moved
    state.enemies = room.enemies


def _find_transition_room(state: GameState, room: DungeonRoom) -> Optional[int]:
    if not _room_exits_unlocked(room):
        return None
    return room.exits.get(state.player_pos)


def _create_font() -> Optional[pygame.font.Font]:
    try:
        if not pygame.get_init():
            pygame.init()
        if hasattr(pygame, 'font'):
            return pygame.font.Font(None, 24)
        return None
    except Exception:
        return None

def _new_game_state(start_room: int) -> GameState:
    return GameState(
        room_step=1,
        hp=MAX_HP,
        player_pos=(0, 0),
        current_room_id=start_room,
        previous_room_id=None,
        difficulty=1,
        verify_attempts=0,
        room_profile="unknown",
        room_special_type="normal",
        elite_modifier="",
        objective_kind="",
        flow_score=0.5,
        enemy_delay=0.50,
        enemy_timer=0.0,
        speed_boost_steps=0,
        shield_hits=0,
        overheal_hp=0,
        combo_streak=0,
        combo_reward_pending=False,
        status_message="",
        status_timer=0.0,
        enemies={},
        enemies_spawned=0,
        powerups={},
        powerups_spawned=0,
        teleports=None,
        room_start_time=time.time(),
        room_start_hp=MAX_HP,
        moves_made=0,
        enemy_kills=0,
        enemy_hits_taken=0,
        damage_blocked=0,
        powerups_collected=0,
        heal_collected=0,
        speed_collected=0,
        shield_collected=0,
        teleport_uses=0,
        enemy_distance_accum=0.0,
        enemy_distance_samples=0,
        room_entries=0,
        game_over=False,
        is_win=False,
        paused=False,
        time_menu_open=0.0,
        game_start_time=time.time(),
    )


def _build_dungeon() -> DungeonState:
    n = max(3, min(MAX_ROOMS, DUNGEON_MAP_W * DUNGEON_MAP_H))
    coords, adjacency, start, final = _build_dungeon_layout(n)
    candidates = [rid for rid in coords if rid not in {start, final}]
    random.shuffle(candidates)
    goal = min(RELIC_GOAL, len(candidates))
    relic_rooms = set(candidates[:goal])
    elite_rooms, merchant_rooms, sanctuary_rooms, risk_reward_rooms, predator_room_id = _pick_special_rooms(adjacency, start, final)
    active_event = "calm"
    return DungeonState(
        rooms={},
        coords_by_room=coords,
        adjacency=adjacency,
        discovered={start},
        start_room=start,
        final_room=final,
        relic_rooms=relic_rooms,
        relic_collected_rooms=set(),
        relic_goal=goal,
        collected_relics=0,
        elite_rooms=elite_rooms,
        merchant_rooms=merchant_rooms,
        sanctuary_rooms=sanctuary_rooms,
        risk_reward_rooms=risk_reward_rooms,
        active_event=active_event,
        predator_room_id=predator_room_id,
    )


def run_self_test() -> int:
    director = Director(use_bandit=USE_BANDIT)
    observer = Observer()
    dungeon = _build_dungeon()
    state = _new_game_state(dungeon.start_room)

    room = _create_room(state.current_room_id, dungeon, director, None)
    _enter_room(state, dungeon, room)
    observer.start_room(state.room_step, state.difficulty, USE_BANDIT, state.hp)

    for _ in range(3):
        if room.exits:
            nxt = next(iter(room.exits.values()))
            metrics = observer.complete_room(state.hp, _engagement_snapshot(state, dungeon))

            if director.bandit is not None:
                state.flow_score = director.bandit.reward_from_metrics(
                    metrics.time_taken,
                    metrics.enemy_hits_taken,
                    metrics.avg_enemy_distance,
                    0.0,
                )
            else:
                state.flow_score = 0.5
            
            _rotate_dungeon_event(dungeon, state.room_step + 1)
            _advance_predator(dungeon, nxt)
            if nxt not in dungeon.rooms:
                room = _create_room(
                    nxt,
                    dungeon,
                    director,
                    (metrics.time_taken, metrics.enemy_hits_taken,
                     metrics.avg_enemy_distance, 0.0),
                    force_reward_room=False,
                )
            else:
                room = dungeon.rooms[nxt]
            state.room_step += 1
            state.previous_room_id = state.current_room_id
            _enter_room(state, dungeon, room)
            observer.start_room(state.room_step, state.difficulty, USE_BANDIT, state.hp)

    log_event("SELF_TEST success")
    return 0

def run(max_frames: Optional[int] = None) -> None:
    pygame.init()
    pygame.font.init()
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
    clock = pygame.time.Clock()
    
    font = _create_font()
    if font is None:
        font = pygame.font.SysFont("Arial", 17)

    cfg = _show_startup_menu(screen)
    log_event(f"MENU_CONFIG: {cfg}")
    
    import config
    config.USE_BANDIT = cfg['bandit']
    config.ENEMY_DENSITY_MULT = cfg['density']
    
    director = Director(use_bandit=cfg['bandit'])
    observer = Observer()
    dungeon = _build_dungeon()
    
    state = _new_game_state(dungeon.start_room)
    state.relic_goal = cfg['relics']
    state.max_rooms = cfg['rooms']

    room = _create_room(state.current_room_id, dungeon, director, None)
    _enter_room(state, dungeon, room)
    
    observer.start_room(state.room_step, state.difficulty, cfg['bandit'], state.hp)

    running = True
    frames = 0
    while running:
        dt = clock.tick(FPS)
        dt_sec = dt / 1000.0
        frames += 1
        
        if state.status_timer > 0:
            state.status_timer = max(0.0, state.status_timer - dt_sec)

        if max_frames is not None and frames >= max_frames:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_p):
                    state.paused = not state.paused
                elif not state.paused:
                    dx, dy = 0, 0
                    if event.key == pygame.K_UP: dy = -1
                    elif event.key == pygame.K_DOWN: dy = 1
                    elif event.key == pygame.K_LEFT: dx = -1
                    elif event.key == pygame.K_RIGHT: dx = 1
                    if dx != 0 or dy != 0:
                        _move_player(state, room, dungeon, dx, dy)

        if state.paused:
            state.time_menu_open += dt_sec
        else:
            state.enemy_timer += dt_sec
            _update_in_room_flow(state, director)
            _update_room_objective(state, room, dt_sec)
            if state.enemy_timer >= state.enemy_delay:
                _move_enemies(state, room)
                state.enemy_timer = 0.0

        target_room = _find_transition_room(state, room)
        if target_room is not None:
            snap = _engagement_snapshot(state, dungeon)
            metrics = observer.complete_room(state.hp, snap)
            _kpr = snap.get("key_press_rate", 0.0)

            if director.bandit is not None:
                state.flow_score = director.bandit.reward_from_metrics(
                    metrics.time_taken,
                    metrics.enemy_hits_taken,
                    metrics.avg_enemy_distance,
                    _kpr,
                )
            else:
                state.flow_score = 0.5 

            _rotate_dungeon_event(dungeon, state.room_step + 1)
            _advance_predator(dungeon, target_room)
            
            if target_room not in dungeon.rooms:
                room = _create_room(
                    target_room,
                    dungeon,
                    director,
                    (metrics.time_taken, metrics.enemy_hits_taken,
                     metrics.avg_enemy_distance, _kpr),
                    key_press_rate=_kpr,
                    force_reward_room=False,
                )
            else:
                room = dungeon.rooms[target_room]
                
            state.room_step += 1
            state.previous_room_id = state.current_room_id
            _enter_room(state, dungeon, room)
            observer.start_room(state.room_step, state.difficulty, cfg['bandit'], state.hp)

        if room.final_door_pos is not None and state.player_pos == room.final_door_pos:
            if int(room.grid[room.final_door_pos[1], room.final_door_pos[0]]) == DOOR_OPEN:
                if dungeon.collected_relics >= dungeon.relic_goal:
                    state.is_win = True
                    running = False

        if state.hp <= 0:
            state.game_over = True
            running = False

        _update_caption(state, dungeon)
        screen.fill((0, 0, 0))
        draw_grid(screen, room, state.player_pos)
        _draw_blackout_overlay(screen, state, room, dungeon)
        
        if font:
            draw_hud(screen, font, state, dungeon)
            if state.paused:
                draw_pause_screen(screen, font, state, dungeon, room)
        
        pygame.display.flip()

    if state.is_win:
        log_event("MAIN win_dungeon_exit")
    elif state.game_over:
        log_event("MAIN game_over")
        
    log_event("MAIN shutdown")
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolving Depths prototype")
    parser.add_argument("--self-test", action="store_true", help="Run non-interactive component test")
    parser.add_argument("--max-frames", type=int, default=None, help="Auto-stop main loop after N frames")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(run_self_test())

    run(max_frames=args.max_frames)