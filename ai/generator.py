from __future__ import annotations

import random

import numpy as np

from config import ENEMY, EXIT, FLOOR, GRID_HEIGHT, GRID_WIDTH, WALL


class RoomGenerator:
    def __init__(self, width: int = GRID_WIDTH, height: int = GRID_HEIGHT) -> None:
        self.width = width
        self.height = height
        self.last_room_profile = "cave"
        self.last_wall_prob = 0.0
        self.last_enemy_count = 0
        self.last_engagement = 0.5

    def _weighted_choice(self, weights: dict[str, float], fallback: str) -> str:
        total = sum(max(0.05, value) for value in weights.values())
        roll = random.random() * total
        acc = 0.0
        for profile, value in weights.items():
            acc += max(0.05, value)
            if roll <= acc:
                return profile
        return fallback

    def _wall_probability(self, difficulty_level: int, engagement: float) -> float:
        base = {
            0: 0.30,
            1: 0.36,
            2: 0.42,
        }.get(difficulty_level, 0.36)
        adaptive = (engagement - 0.5) * 0.16
        jitter = random.uniform(-0.03, 0.03)
        return max(0.16, min(0.58, base + adaptive + jitter))

    def _enemy_count(self, difficulty_level: int, engagement: float) -> int:
        base = {
            0: 4,
            1: 7,
            2: 12,
        }.get(difficulty_level, 4)
        adaptive = round((engagement - 0.5) * 6)
        jitter = random.choice([-2, -1, 0, 1, 2])
        return max(1, base + adaptive + jitter)

    def _choose_profile(self, difficulty_level: int, engagement: float) -> str:
        engagement = max(0.0, min(1.0, engagement))

        if difficulty_level == 0:
            weights = {
                "arena":      0.50,
                "pillars":    0.40,
                "cave":       0.10,
                "corridor":   0.05,
                "crossroads": 0.10,
                "islands":    0.15,
            }
        elif difficulty_level == 1:
            weights = {
                "arena":      0.20,
                "pillars":    0.20,
                "cave":       0.30,
                "corridor":   0.25,
                "crossroads": 0.15,
                "islands":    0.10,
            }
        else:
            weights = {
                "arena":      0.05,
                "pillars":    0.05,
                "cave":       0.40,
                "corridor":   0.40,
                "crossroads": 0.15,
                "islands":    0.10,
            }

        weights["arena"]    += (0.5 - engagement) * 0.40
        weights["pillars"]  += (0.5 - engagement) * 0.20
        weights["corridor"] += (engagement - 0.5) * 0.50
        weights["cave"]     += (engagement - 0.5) * 0.30

        return self._weighted_choice(weights, fallback="cave")

    def _init_by_profile(self, profile: str, wall_prob: float) -> np.ndarray:
        if profile == "arena":
            grid = np.full((self.height, self.width), FLOOR, dtype=np.int32)
            obstacle_prob = max(0.18, min(0.30, wall_prob * 0.75))
            mask = np.random.rand(self.height, self.width) < obstacle_prob
            grid[mask] = WALL
            return grid

        if profile == "pillars":
            grid = np.full((self.height, self.width), FLOOR, dtype=np.int32)
            for y in range(2, self.height - 2, 3):
                for x in range(2, self.width - 2, 4):
                    if random.random() < 0.8:
                        grid[y, x] = WALL
                        if random.random() < 0.55 and x + 1 < self.width - 1:
                            grid[y, x + 1] = WALL
            return grid

        if profile == "crossroads":
            grid = np.full((self.height, self.width), WALL, dtype=np.int32)
            center_x = self.width // 2
            center_y = self.height // 2
            grid[:, max(1, center_x - 1):min(self.width - 1, center_x + 2)] = FLOOR
            grid[max(1, center_y - 1):min(self.height - 1, center_y + 2), :] = FLOOR
            for _ in range(4):
                room_w = random.randint(3, 5)
                room_h = random.randint(3, 4)
                x0 = random.randint(1, max(1, self.width - room_w - 1))
                y0 = random.randint(1, max(1, self.height - room_h - 1))
                grid[y0:y0 + room_h, x0:x0 + room_w] = FLOOR
            return grid

        if profile == "islands":
            grid = np.full((self.height, self.width), FLOOR, dtype=np.int32)
            island_count = max(3, (self.width * self.height) // 45)
            for _ in range(island_count):
                x0 = random.randint(2, self.width - 4)
                y0 = random.randint(2, self.height - 4)
                w = random.randint(2, 4)
                h = random.randint(2, 3)
                grid[y0:y0 + h, x0:x0 + w] = WALL
            return grid

        if profile == "corridor":
            grid = np.full((self.height, self.width), WALL, dtype=np.int32)
            x, y = 0, 0
            grid[y, x] = FLOOR
            for _ in range(int(self.width * self.height * 1.5)):
                dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                x = max(0, min(self.width - 1, x + dx))
                y = max(0, min(self.height - 1, y + dy))
                grid[y, x] = FLOOR
                if random.random() < 0.35:
                    nx = max(0, min(self.width - 1, x + random.choice([-1, 0, 1])))
                    ny = max(0, min(self.height - 1, y + random.choice([-1, 0, 1])))
                    grid[ny, nx] = FLOOR
            return grid

        return np.where(
            np.random.rand(self.height, self.width) < wall_prob,
            WALL,
            FLOOR,
        ).astype(np.int32)

    def _step_automata_light(self, grid: np.ndarray) -> np.ndarray:
        next_grid = grid.copy()
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if grid[y, x] == WALL and self._count_wall_neighbors(grid, x, y) == 0:
                    next_grid[y, x] = FLOOR
        return next_grid

    def _apply_profile_post(self, profile: str, grid: np.ndarray) -> np.ndarray:
        if profile == "cave":
            for _ in range(4):
                grid = self._step_automata(grid)
            return grid

        if profile == "arena":
            return self._step_automata_light(grid)

        if profile == "pillars":
            return grid

        if profile == "crossroads":
            return grid

        if profile == "islands":
            return self._step_automata(grid)

        return self._step_automata(grid)

    def _apply_random_transform(self, grid: np.ndarray) -> np.ndarray:
        k = random.choice([0, 2])
        transformed = np.rot90(grid, k=k)
        if random.random() < 0.5:
            transformed = np.fliplr(transformed)
        return transformed.copy().astype(np.int32)

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
        next_grid = grid.copy()
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                neighbors = self._count_wall_neighbors(grid, x, y)
                if neighbors >= 5:
                    next_grid[y, x] = WALL
                else:
                    next_grid[y, x] = FLOOR
        return next_grid

    def create_map(self, difficulty_level: int, engagement: float = 0.5) -> np.ndarray:
        wall_prob = self._wall_probability(difficulty_level, engagement)
        profile = self._choose_profile(difficulty_level, engagement)
        self.last_room_profile = profile
        self.last_wall_prob = wall_prob
        self.last_engagement = engagement

        grid = self._init_by_profile(profile, wall_prob)
        grid = self._apply_profile_post(profile, grid)
        grid = self._apply_random_transform(grid)

        grid[0, :] = WALL
        grid[:, 0] = WALL
        grid[self.height - 1, :] = WALL
        grid[:, self.width - 1] = WALL

        entrance = (0, 0)
        exit_cell = (self.width - 1, self.height - 1)
        grid[entrance[1], entrance[0]] = FLOOR
        grid[exit_cell[1], exit_cell[0]] = EXIT

        grid[entrance[1], entrance[0]] = FLOOR
        grid[exit_cell[1], exit_cell[0]] = EXIT

        if self.width > 1:
            grid[0, 1] = FLOOR
        if self.height > 1:
            grid[1, 0] = FLOOR
        if self.width > 1:
            grid[self.height - 1, self.width - 2] = FLOOR
        if self.height > 1:
            grid[self.height - 2, self.width - 1] = FLOOR

        enemy_count = self._enemy_count(difficulty_level, engagement)
        self.last_enemy_count = enemy_count
        floor_positions = list(zip(*np.where(grid == FLOOR)))
        random.shuffle(floor_positions)

        placed = 0
        for y, x in floor_positions:
            if (x, y) in [entrance, exit_cell]:
                continue
            if abs(x - entrance[0]) + abs(y - entrance[1]) < 4:
                continue
            if abs(x - exit_cell[0]) + abs(y - exit_cell[1]) < 3:
                continue
            grid[y, x] = ENEMY
            placed += 1
            if placed >= enemy_count:
                break

        grid[exit_cell[1], exit_cell[0]] = EXIT
        return grid
