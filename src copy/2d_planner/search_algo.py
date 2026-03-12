import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
class SearchAlgo(Node):
    def __init__(self):
        super().__init__('search_algo')

        self.camera_sub = self.create_subscription(Image,'/camera/image_raw', self.image_callback, 10)
        self.grid_map = None
        self.search_area_size = 5.0
        self.occupy_threshold = 0.6
        self.dt = 0.1
        self.safe_radis = 0.5
        self.target_grid_x = 25
        self.target_grid_y = 25

        self.timer = self.create_timer(self.dt, self.search_loop)

    def image_callback(self, msg):
        self.image = msg
    
    def search_best_landing_point(self):
        if self.grid_map is None or len(self.grid_map) == 0:
            return None
        # 积分图：integral[i,j] = sum(grid_map[0:i, 0:j])，便于 O(1) 矩形和
        integral = np.pad(self.grid_map.astype(np.float32), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
        half_win = int(self.search_area_size)
        h, w = self.grid_map.shape[0], self.grid_map.shape[1]
        window_area = (2 * half_win + 1) ** 2
        drone_center_x, drone_center_y = w // 2, h // 2

        min_obstacles = float('inf')
        best_grid = None

        for cy in range(half_win, h - half_win):
            for cx in range(half_win, w - half_win):
                y1, y2 = cy - half_win, cy + half_win
                x1, x2 = cx - half_win, cx + half_win
                # 矩形和：integral 多了一行一列前缀 0
                obstacle_count = (
                    integral[y2 + 1, x2 + 1]
                    - integral[y1, x2 + 1]
                    - integral[y2 + 1, x1]
                    + integral[y1, x1]
                )
                occupancy = obstacle_count / window_area
                if occupancy >= self.occupy_threshold:
                    continue
                # 在满足占用率的前提下，选障碍最少；同障碍时选离当前中心最近的
                if obstacle_count < min_obstacles:
                    min_obstacles = obstacle_count
                    best_grid = (cx, cy)
                elif obstacle_count == min_obstacles:
                    if best_grid is None:
                        best_grid = (cx, cy)
                    else:
                        d_new = (cx - drone_center_x) ** 2 + (cy - drone_center_y) ** 2
                        d_old = (best_grid[0] - drone_center_x) ** 2 + (best_grid[1] - drone_center_y) ** 2
                        if d_new < d_old:
                            best_grid = (cx, cy)

        if best_grid is None:
            return None
        cx, cy = best_grid[0], best_grid[1]
        return (cx, cy)   