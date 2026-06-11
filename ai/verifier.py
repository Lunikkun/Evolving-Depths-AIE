from __future__ import annotations

import heapq
from typing import Dict, List, Tuple

import numpy as np

from config import ENEMY, EXIT, FLOOR, WALL


class AStarVerifier:
    """Verifies room reachability with A* and Manhattan heuristic."""

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
        return cell_value in {FLOOR, ENEMY, EXIT}

    def check_playability(self, grid: np.ndarray) -> bool:
        if grid.size == 0:
            return False

        height, width = grid.shape
        start = self.start

        exit_positions = np.argwhere(grid == EXIT)
        if len(exit_positions) == 0:
            return False

        exit_y, exit_x = exit_positions[0]
        goal = (int(exit_x), int(exit_y))

        if not self._is_passable(int(grid[start[1], start[0]])):
            return False

        open_heap: List[Tuple[int, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0, start))

        g_score: Dict[Tuple[int, int], int] = {start: 0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                return True

            for neighbor in self._neighbors(current, width, height):
                nx, ny = neighbor
                if not self._is_passable(int(grid[ny, nx])):
                    continue

                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, 10**9):
                    g_score[neighbor] = tentative_g
                    score = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (score, neighbor))

        return False
