from __future__ import annotations

import heapq
from typing import List, Tuple

import numpy as np

from config import ENEMY, EXIT, FLOOR, WALL


class AStarVerifier:
    def __init__(self, start: Tuple[int, int] = (0, 0)) -> None:
        self.start = start

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _neighbors(node: Tuple[int, int], width: int, height: int) -> List[Tuple[int, int]]:
        x, y = node
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [(nx, ny) for nx, ny in candidates if 0 <= nx < width and 0 <= ny < height]

    @staticmethod
    def _is_passable(cell_value: int) -> bool:
        return cell_value in {FLOOR, ENEMY, EXIT, 12}

    def check_path(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        height, width = grid.shape
        
        if not self._is_passable(int(grid[start[1], start[0]])): return False
        if not self._is_passable(int(grid[goal[1], goal[0]])): return False

        open_heap: List[Tuple[int, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0, start))
        g_score = {start: 0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                return True

            for neighbor in self._neighbors(current, width, height):
                if not self._is_passable(int(grid[neighbor[1], neighbor[0]])):
                    continue

                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, 10**9):
                    g_score[neighbor] = tentative_g
                    score = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (score, neighbor))

        return False

    def check_playability(self, grid: np.ndarray, exits: dict = None) -> bool:
        if grid.size == 0:
            return False

        height, width = grid.shape
        test_grid = grid.copy()
        mid_x, mid_y = width // 2, height // 2

        points_to_connect = []

        if exits:
            for (x, y) in exits.keys():
                if y == 0:
                    for nx in range(mid_x - 1, mid_x + 2):
                        test_grid[0, nx] = EXIT if nx == mid_x else FLOOR
                        test_grid[1, nx] = FLOOR
                        test_grid[2, nx] = FLOOR
                    points_to_connect.append((mid_x, 2))
                elif y == height - 1:
                    for nx in range(mid_x - 1, mid_x + 2):
                        test_grid[height - 1, nx] = EXIT if nx == mid_x else FLOOR
                        test_grid[height - 2, nx] = FLOOR
                        test_grid[height - 3, nx] = FLOOR
                    points_to_connect.append((mid_x, height - 3))
                elif x == 0:
                    for ny in range(mid_y - 1, mid_y + 2):
                        test_grid[ny, 0] = EXIT if ny == mid_y else FLOOR
                        test_grid[ny, 1] = FLOOR
                        test_grid[ny, 2] = FLOOR
                    points_to_connect.append((2, mid_y))
                elif x == width - 1:
                    for ny in range(mid_y - 1, mid_y + 2):
                        test_grid[ny, width - 1] = EXIT if ny == mid_y else FLOOR
                        test_grid[ny, width - 2] = FLOOR
                        test_grid[ny, width - 3] = FLOOR
                    points_to_connect.append((width - 3, mid_y))

        if not points_to_connect:
            points_to_connect.append((mid_x, mid_y))

        test_grid[height - 2, width - 2] = FLOOR
        points_to_connect.append((width - 2, height - 2))

        reference_point = points_to_connect[0]
        for target in points_to_connect[1:]:
            if not self.check_path(test_grid, reference_point, target):
                return False

        return True