from __future__ import annotations

import random
from typing import Optional, Tuple, List

import numpy as np

from config import ENEMY, EXIT, FLOOR, GRID_HEIGHT, GRID_WIDTH, WALL

class RoomGenerator:
    """Search-based PCG via cellular automata refinement."""

    def __init__(self, width: int = GRID_WIDTH, height: int = GRID_HEIGHT) -> None:
        self.width = width
        self.height = height
        # Inizializziamo l'attributo richiesto dal main.py
        self.last_room_profile = "none" 
        # Definiamo i profili disponibili
        self.profiles = ["arena", "pillars", "crossroads", "cave", "corridor"]

    def _get_profile_weights(self, difficulty: int) -> List[float]:
        # Profili più chiusi nelle difficoltà alte, più aperti solo nel safe.
        if difficulty == 0:  # Safe
            return [0.28, 0.20, 0.18, 0.16, 0.18]
        elif difficulty == 1:  # Medium
            return [0.16, 0.18, 0.18, 0.24, 0.24]
        else:  # Swarm (difficulty == 2)
            return [0.08, 0.15, 0.17, 0.30, 0.30]

    def _wall_probability(self, difficulty_level: int) -> float:
        # Più muri = stanze più strette, chiuse e articolate.
        return {0: 0.36, 1: 0.44, 2: 0.52}.get(difficulty_level, 0.44)

    def _automata_steps(self, difficulty_level: int, profile: str) -> int:
        base_steps = {0: 3, 1: 4, 2: 5}.get(difficulty_level, 4)
        if profile in {"cave", "corridor"}:
            return base_steps + 1
        if profile == "arena":
            return max(2, base_steps - 1)
        if profile == "pillars":
            return base_steps
        return base_steps

    def _profile_wall_bonus(self, profile: str) -> float:
        if profile == "arena":
            return -0.04
        if profile == "pillars":
            return 0.02
        if profile == "crossroads":
            return 0.04
        if profile == "cave":
            return 0.08
        if profile == "corridor":
            return 0.10
        return 0.0

    def _target_wall_ratio(self, difficulty_level: int, profile: str) -> float:
        base = {0: 0.34, 1: 0.42, 2: 0.50}.get(difficulty_level, 0.42)
        if profile in {"cave", "corridor"}:
            return base + 0.06
        if profile == "crossroads":
            return base + 0.03
        if profile == "arena":
            return max(0.22, base - 0.08)
        return base

    def _place_organic_blob(self, grid: np.ndarray, cx: int, cy: int, reserved: set) -> int:
        """Place a randomly-shaped wall blob centred at (cx, cy). Returns cells added."""
        shape = random.choice(["rect", "L", "T", "plus", "diagonal"])
        cells = []
        w = random.randint(2, 5)
        h = random.randint(2, 4)
        if shape == "rect":
            cells = [(cx + dx, cy + dy) for dx in range(w) for dy in range(h)]
        elif shape == "L":
            cells  = [(cx + dx, cy) for dx in range(w)]
            cells += [(cx, cy + dy) for dy in range(1, h)]
        elif shape == "T":
            cells  = [(cx + dx, cy) for dx in range(w)]
            mid = w // 2
            cells += [(cx + mid, cy + dy) for dy in range(1, h)]
        elif shape == "plus":
            mid = w // 2
            cells  = [(cx + mid, cy + dy) for dy in range(h)]
            cells += [(cx + dx, cy + h // 2) for dx in range(w)]
        else:  # diagonal staircase
            cells = [(cx + i, cy + i) for i in range(min(w, h))]
            cells += [(cx + i + 1, cy + i) for i in range(min(w, h) - 1)]

        # Random rotation: swap x/y axes ~half the time
        if random.random() < 0.5:
            cells = [(cy_off + cx - cy, cx_off + cy - cx) for cx_off, cy_off in cells]

        added = 0
        for fx, fy in cells:
            if (fx, fy) in reserved:
                continue
            if 1 <= fx < self.width - 1 and 1 <= fy < self.height - 1:
                if int(grid[fy, fx]) == FLOOR:
                    grid[fy, fx] = WALL
                    added += 1
        return added

    def _enforce_wall_density(self, grid: np.ndarray, difficulty_level: int, entrance: Tuple[int, int], exit_cell: Tuple[int, int]) -> np.ndarray:
        target_ratio = self._target_wall_ratio(difficulty_level, self.last_room_profile)
        reserved = {entrance, exit_cell}
        total_cells = self.width * self.height
        target_walls = int(total_cells * target_ratio)
        current_walls = int(np.count_nonzero(grid == WALL))
        if current_walls >= target_walls:
            return grid

        deficit = target_walls - current_walls
        attempts = 0
        while deficit > 0 and attempts < 200:
            attempts += 1
            cx = random.randint(3, self.width - 6)
            cy = random.randint(3, self.height - 6)
            added = self._place_organic_blob(grid, cx, cy, reserved)
            deficit -= added
        return grid

    def _enemy_count(self, difficulty_level: int) -> int:
        return {0: 5, 1: 7, 2: 10}.get(difficulty_level, 5)

    def _count_wall_neighbors(self, grid: np.ndarray, x: int, y: int) -> int:
        count = 0
        for ny in range(max(0, y - 1), min(self.height, y + 2)):
            for nx in range(max(0, x - 1), min(self.width, x + 2)):
                if nx == x and ny == y:
                    continue
                if grid[ny, nx] == WALL:
                    count += 1
        return count

    def _step_automata(self, grid: np.ndarray) -> np.ndarray:
        # Vectorised 8-neighbour wall count using numpy slicing (no scipy needed).
        wall = (grid == WALL).astype(np.int32)
        h, w = wall.shape
        count = np.zeros((h, w), dtype=np.int32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                y0s, y0e = max(0, -dy), h - max(0, dy)
                x0s, x0e = max(0, -dx), w - max(0, dx)
                y1s, y1e = max(0,  dy), h - max(0, -dy)
                x1s, x1e = max(0,  dx), w - max(0, -dx)
                count[y0s:y0e, x0s:x0e] += wall[y1s:y1e, x1s:x1e]
        next_grid = grid.copy()
        interior = np.zeros((h, w), dtype=bool)
        interior[1:-1, 1:-1] = True
        next_grid[interior & (count >= 5)] = WALL
        next_grid[interior & (count < 5)] = FLOOR
        return next_grid

    def create_map(self, difficulty_level: int) -> np.ndarray:
        # 1. Selezioniamo il profilo basato sulla difficoltà
        weights = self._get_profile_weights(difficulty_level)
        self.last_room_profile = random.choices(self.profiles, weights=weights)[0]

        # 2. Generazione mappa
        wall_prob = min(0.62, max(0.22, self._wall_probability(difficulty_level) + self._profile_wall_bonus(self.last_room_profile)))
        grid = np.where(
            np.random.rand(self.height, self.width) < wall_prob,
            WALL,
            FLOOR,
        ).astype(np.int32)

        # Manteniamo i bordi chiusi
        grid[0, :] = WALL
        grid[:, 0] = WALL
        grid[self.height - 1, :] = WALL
        grid[:, self.width - 1] = WALL

        entrance = (0, 0)
        exit_cell = (self.width - 1, self.height - 1)
        grid[entrance[1], entrance[0]] = FLOOR
        grid[exit_cell[1], exit_cell[0]] = EXIT

        for _ in range(self._automata_steps(difficulty_level, self.last_room_profile)):
            grid = self._step_automata(grid)
            grid[entrance[1], entrance[0]] = FLOOR
            grid[exit_cell[1], exit_cell[0]] = EXIT

        grid = self._enforce_wall_density(grid, difficulty_level, entrance, exit_cell)

        # Assicuriamo la traversabilità base
        if self.width > 1: grid[0, 1] = FLOOR
        if self.height > 1: grid[1, 0] = FLOOR
        if self.width > 1: grid[self.height - 1, self.width - 2] = FLOOR
        if self.height > 1: grid[self.height - 2, self.width - 1] = FLOOR

        # 3. Spawn nemici
        enemy_count = self._enemy_count(difficulty_level)
        floor_positions = list(zip(*np.where(grid == FLOOR)))
        random.shuffle(floor_positions)

        placed = 0
        for y, x in floor_positions:
            if (x, y) in [entrance, exit_cell]: continue
            grid[y, x] = ENEMY
            placed += 1
            if placed >= enemy_count: break

        grid[exit_cell[1], exit_cell[0]] = EXIT
        return grid