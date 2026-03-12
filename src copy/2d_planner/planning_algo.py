import numpy as np
import math
import heapq
import cv2


def is_waypoint_pipe_safe(grid_map, row, col, radius):
    """
    判断以航路点 (row, col) 为圆心、半径为 radius 的管道内是否无障碍（安全）。
    :param grid_map: 二维，0 可通行，1 障碍
    :param row, col: 航路点栅格坐标
    :param radius: 管道半径（栅格数）
    :return: True 表示管道内无障碍
    """
    if radius <= 0:
        return True
    grid = np.asarray(grid_map)
    h, w = grid.shape[0], grid.shape[1]
    r0, c0 = int(row), int(col)
    r_lo = max(0, r0 - int(np.ceil(radius)))
    r_hi = min(h, r0 + int(np.ceil(radius)) + 1)
    c_lo = max(0, c0 - int(np.ceil(radius)))
    c_hi = min(w, c0 + int(np.ceil(radius)) + 1)
    for r in range(r_lo, r_hi):
        for c in range(c_lo, c_hi):
            if math.dist((r, c), (r0, c0)) <= radius and grid[r, c] == 1:
                return False
    return True


def filter_path_by_safe_pipe(path, grid_map, path_safety_radius):
    """
    只保留“管道内无障碍”的航路点：每个点以 path_safety_radius 为半径的圆内不能有障碍。
    起点和终点始终保留；中间点若不安全则剔除。
    :return: 过滤后的路径 [(row,col), ...]
    """
    if not path or path_safety_radius <= 0:
        return path
    if len(path) <= 2:
        return path
    out = [path[0]]
    for i in range(1, len(path) - 1):
        r, c = path[i][0], path[i][1]
        if is_waypoint_pipe_safe(grid_map, r, c, path_safety_radius):
            out.append(path[i])
    out.append(path[-1])
    return out


class CostMap:
    """
    代价地图类：根据栅格地图在内部计算各类代价（如到障碍物距离），供规划器使用。
    输入：栅格地图。
    """

    def __init__(self, grid_map, obstacle_dist_weight=2.0, obstacle_dist_epsilon=0.5):
        """
        :param grid_map: 二维数组，0 可通行，1 障碍
        :param obstacle_dist_weight: 障碍距离代价权重
        :param obstacle_dist_epsilon: 防止除零的小常数
        """
        self.grid_map = np.asarray(grid_map) if grid_map is not None else None
        self.obstacle_dist_weight = obstacle_dist_weight
        self.obstacle_dist_epsilon = obstacle_dist_epsilon
        self._obstacle_dist_map = None
        self._recompute()

    def _recompute(self):
        """内部计算：根据当前 grid_map 更新障碍物距离图等。"""
        if self.grid_map is None or self.grid_map.size == 0:
            self._obstacle_dist_map = None
            return
        # 可通行=255，障碍=0，便于 distanceTransform 得到“到最近障碍的距离”
        binary = np.where(self.grid_map == 1, 0, 255).astype(np.uint8)
        self._obstacle_dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    def update_grid_map(self, grid_map):
        """更新栅格地图并重新计算代价。"""
        self.grid_map = np.asarray(grid_map) if grid_map is not None else None
        self._recompute()

    def get_obstacle_dist_map(self):
        """返回到最近障碍物的距离图"""
        return self._obstacle_dist_map

    def get_step_cost(self, row, col, base_cost):
        """
        根据代价地图得到 (row, col) 的步进代价。
        :param base_cost: 基础移动代价（如 1 或 1.414）
        :return: base_cost + 障碍距离惩罚项
        """
        cost = base_cost
        if self._obstacle_dist_map is not None:
            r, c = int(row), int(col)
            if 0 <= r < self._obstacle_dist_map.shape[0] and 0 <= c < self._obstacle_dist_map.shape[1]:
                d = float(self._obstacle_dist_map[r, c])
                cost += self.obstacle_dist_weight / (d + self.obstacle_dist_epsilon)
        return cost


class AstarPlanner:
    def __init__(self, obstacle_dist_weight=2.0, obstacle_dist_epsilon=0.5, path_safety_radius=2):
        """
        :param path_safety_radius: 航路点安全管道半径（栅格数）。每个航路点带该半径的管道，只保留管道内无障碍的航路点。0 表示不做管道过滤。
        """
        self.obstacle_dist_weight = obstacle_dist_weight
        self.obstacle_dist_epsilon = obstacle_dist_epsilon
        self.path_safety_radius = max(0, float(path_safety_radius))
        self.neighbors = [
            (0, 1, 1), (0, -1, 1), (1, 0, 1), (-1, 0, 1),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]

    def heuristic(self, a, b):
        """启发函数：欧几里得距离。a、b 为 (row, col)。"""
        return math.dist(a, b)

    def plan(self, grid_map, start, goal):
        """
        执行 A* 规划。输入：栅格地图、起点、终点；代价由 CostMap 内部计算。
        :param grid_map: 二维整数矩阵，0 可通行，1 障碍
        :param start: (row, col) 起点
        :param goal: (row, col) 终点
        :param cost_map: 可选，CostMap 实例；不传则内部根据 grid_map 创建并计算
        :return: [(row, col), ...] 或 None
        """
        if grid_map is None or start is None or goal is None:
            return None

        grid = np.asarray(grid_map)
        rows, cols = grid.shape[0], grid.shape[1]
        start_idx = (max(0, min(int(start[0]), rows - 1)), max(0, min(int(start[1]), cols - 1)))
        goal_idx = (max(0, min(int(goal[0]), rows - 1)), max(0, min(int(goal[1]), cols - 1)))

        cost_map = CostMap(
                grid_map,
                obstacle_dist_weight=self.obstacle_dist_weight,
                obstacle_dist_epsilon=self.obstacle_dist_epsilon,
            )

        open_list = []
        heapq.heappush(open_list, (0, start_idx))
        came_from = {}
        g_score = {start_idx: 0}

        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal_idx:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_idx)
                return path[::-1]

            for dx, dy, base_cost in self.neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                    continue
                if grid[neighbor[0], neighbor[1]] == 1:  # 障碍不可通行
                    continue

                step_cost = cost_map.get_step_cost(neighbor[0], neighbor[1], base_cost)
                tentative_g = g_score[current] + step_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal_idx)
                    heapq.heappush(open_list, (f, neighbor))

        return None

    def plan_with_safety_corridor(self, grid_map, start, goal, path_safety_radius=None):
        """
        执行 A* 规划，为每个航路点附加可设置的安全管道半径（膨胀航路点）；只保留管道内无障碍的航路点。
        不膨胀障碍物，仅在原始地图上规划后按管道半径过滤航路点。
        :param grid_map: 栅格地图，0 可通行，1 障碍，-1 终点
        :param start: (row, col) 起点
        :param goal: (row, col) 终点
        :param path_safety_radius: 安全管道半径（栅格数）；不传则用构造时的 self.path_safety_radius
        :return: (path, corridor_radii) 或 (None, None)。path 为过滤后航路点，corridor_radii 为每点安全半径。
        """
        r = path_safety_radius if path_safety_radius is not None else self.path_safety_radius
        r = max(0, float(r))

        path = self.plan(grid_map, start, goal)
        if path is None or len(path) == 0:
            return None, None

        if r > 0:
            path = filter_path_by_safe_pipe(path, grid_map, r)
            corridor_radii = [r] * len(path)
            return path, corridor_radii

        cost_map = CostMap(
            grid_map,
            obstacle_dist_weight=self.obstacle_dist_weight,
            obstacle_dist_epsilon=self.obstacle_dist_epsilon,
        )
        corridor_radii = compute_safety_corridor_radii(path, cost_map)
        return path, corridor_radii


def compute_safety_corridor_radii(path, cost_map):
    """
    根据路径与代价地图，计算路径上每点处的安全管道半径（到最近障碍物的距离，栅格单位）。
    :param path: [(row, col), ...]
    :param cost_map: CostMap 实例，需有 get_obstacle_dist_map()
    :return: [float, ...] 与 path 等长；无距离图时返回全 0。
    """
    if cost_map is None:
        return [0.0] * len(path)
    dist_map = cost_map.get_obstacle_dist_map()
    if dist_map is None:
        return [0.0] * len(path)
    h, w = dist_map.shape[0], dist_map.shape[1]
    radii = []
    for (r, c) in path:
        ri, ci = int(r), int(c)
        if 0 <= ri < h and 0 <= ci < w:
            radii.append(float(dist_map[ri, ci]))
        else:
            radii.append(0.0)
    return radii
