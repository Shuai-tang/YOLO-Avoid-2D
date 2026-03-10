# 姿态角变化（imu 话题）考虑图像变化：
# 建图侧（sensing_node）应对图像做 roll/pitch 纠正后再建图，使栅格尺度仅依赖高度；
# 本节点假定 /grid_map 来自“水平朝下”等价视图，grid_to_world/world_to_grid 仅用高度计算 GSD。
#


from re import search
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
import numpy as np  
import json
from std_msgs.msg import Int8MultiArray, Int32
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from enum import IntEnum
from nav_msgs.msg import Path as NavPath
import time
import os
from pathlib import Path
from datetime import datetime
from planning_algo import AstarPlanner
try:
    # 用于B样条轨迹平滑
    from scipy.interpolate import splprep, splev
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

class FlightMode(IntEnum):
    INIT = 0         # 解锁/模式切换阶段
    TAKEOFF = 1      # 起飞阶段
    NAVIGATE = 2     # 导航阶段
    REACHED = 3     # 到达目标点阶段
    # TODO:可以加一个降落搜索阶段，专门悬停于dilivery_point上方用于搜索最优降落点
    LANDING = 4      # 降落阶段
    LANDED = 5      # 降落完成阶段

class LandingNavigation(Node):
    def __init__(self):
        super().__init__('landing_navigator')
        qos_cmd = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.TRANSIENT_LOCAL,history=HistoryPolicy.KEEP_LAST,depth=1) # 规划器配置
        qos_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE,history=HistoryPolicy.KEEP_LAST,depth=5)  # 感知器配置

        # 订阅
        self.grid_map_sub = self.create_subscription(Int8MultiArray, '/grid_map', self.map_callback, 10) # 订阅栅格地图
        self.local_pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos_sensor) # 订阅无人机当前位置
        self.state_sub = self.create_subscription(State,'/mavros/state',self.state_callback,qos_sensor) # 订阅无人机状态
        # 发布
        self.position_setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos_cmd) # 发布航路点，供PX4轨迹跟踪
        # 规划路径发布，供可视化节点订阅
        self.planned_path_pub = self.create_publisher(NavPath, '/planned_path', 10) # 发布可视化航路点路径
        self.path_progress_pub = self.create_publisher(Int32, '/path_progress', 10) # 发布可视化
        self.safety_area_pub_ = self.create_publisher(PointStamped, '/safety_area', 10) # 发布安全区域中心与半径
        # 服务
        self.arming_client = self.create_client(CommandBool,'/mavros/cmd/arming') # 解锁/上锁服务
        self.set_mode_client = self.create_client(SetMode,'/mavros/set_mode') # 模式切换服务
        
        # 初始化参数
        self.camera_intrinsics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cfg', 'camera_intrinsics.json') # 加载相机配置文件
        self._load_camera_intrinsics()  # 启动时加载，供grid_to_world/world_to_grid等使用
        # K / dist 由 _load_camera_intrinsics() 从 JSON 设置，勿再赋为 None
        self.scale_ratio = 5   # 栅格-像素坐标扩大比例，输入图像是1024*768,构建1024*768栅格太慢了，因此缩放像素和栅格比例
        self.path_safety_radius = 5  # 安全管道半径（单位：栅格）
        self.a_star_path = AstarPlanner(path_safety_radius=self.path_safety_radius) # 加载A*规划器
        self.phase = FlightMode.INIT  # 飞行状态机
        self.current_pose = PoseStamped() # 飞机当前位姿
        self.current_state = State() # 飞机当前状态
        self.grid_map = None # 当前栅格地图，-1 终点，0 空余，1 障碍
        self.map_width = 1024  # 1024/5 = 204.8 768/5 = 153.6
        self.map_height = 768  # 图像像素大小
        self.flight_altitude = 20.0  # 起飞飞行高度（米）
        self.landing_velocity = 3.0  # 降落下降速度（米/秒）
        self.search_area_size = 20.0  # 栅格占有率搜索范围
        self.occupy_threshold = 0.6  # 占有率阈值：搜索区域内障碍占比超过该值则认为不安全
        self.dt = 0.1  # 控制周期(s)：越小轨迹跟踪越快，0.1=10Hz 推荐，0.05=20Hz 更跟
        # self.safe_radis = 0.5  # 安全半径，未使用，与安全管道同样道理
        self.target_grid_x = None
        self.target_grid_y = None

        # 规划出的航路点（ENU 世界坐标），供发布给飞机
        self.planned_path_world = None  # list of (x, y) in local ENU
        self.waypoint_index = 0
        # INIT 阶段：持续发布 setpoint 计数，以便进入 OFFBOARD
        self.setpoint_stream_count = 0
        self.offboard_ready_count = 5
        self.direct_fly_radius_grid = 10.0  # 直接飞向目标点的最小栅格距离阈值（避免末端一直搜索）
        # 栅格 → ENU 方向（若航线方向错可逐项改）
        # 试参顺序：1) 前后反 → grid_row_to_enu_y_sign=1  2) 左右反 → grid_col_to_enu_x_sign=-1  3) 轴装反 → grid_swap_xy=True
        self.grid_col_to_enu_x_sign = -1
        self.grid_row_to_enu_y_sign = -1
        self.grid_swap_xy = True   # 应往前飞却往左飞 → 轴装反：图像前→ENU y(北)，左→ENU x
        
        # 规划间隔：每 plan_interval 个控制周期重规划一次，其余周期复用当前路径发 setpoint（减轻 CPU）
        self.plan_interval = 3
        self.control_loop_count = 0
        # 前视航路点数：setpoint 取路径上当前点往前第 lookahead 个点，越大飞得越快（不会每格减速）
        self.path_lookahead = 5

        # 规划时间，用于记录每次规划耗时，输出到日志文件
        self.planning_time_dir = os.path.join(os.getcwd(), 'planning_time')
        os.makedirs(self.planning_time_dir, exist_ok=True)
        self.planning_log_name = f'planning_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("无人机自主导航-降落节点已启动，初始化完成！")

    # 加载相机配置文件
    def _load_camera_intrinsics(self):
        with open(self.camera_intrinsics_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self.K = np.array(cfg['K'], dtype=np.float64)
        self.dist = np.array(cfg['dist'], dtype=np.float64)

    # 栅格转世界坐标（假定 /grid_map 由姿态纠正后的图像生成，尺度仅依赖高度）
    def grid_to_world(self, grid_x, grid_y):
        if self.K is None:
            self.get_logger().error("相机内参未加载")
            return 0.0, 0.0
        height = self.current_pose.pose.position.z
        gsd_x = height / self.K[0, 0]
        gsd_y = height / self.K[1, 1]
        world_x = grid_x * self.scale_ratio * gsd_x
        world_y = grid_y * self.scale_ratio * gsd_y
        return world_x, world_y
    
    # 世界转栅格坐标（与 grid_to_world 对应，依赖姿态纠正后的建图）
    def world_to_grid(self, world_x, world_y):
        if self.K is None:
            return 0.0, 0.0
        height = self.current_pose.pose.position.z
        if height <= 0:
            return 0.0, 0.0
        gsd_x = height / self.K[0, 0]
        gsd_y = height / self.K[1, 1]
        grid_x = world_x / (self.scale_ratio * gsd_x)
        grid_y = world_y / (self.scale_ratio * gsd_y)
        return grid_x, grid_y
            
    # 当前位置回调函数
    def pose_callback(self, msg):
        self.current_pose = msg

    # 当前状态回调函数
    def state_callback(self,msg):
        self.current_state = msg

    # 解锁无人机
    def arm(self):
        if self.arming_client.service_is_ready():
            req = CommandBool.Request()
            req.value = True
            self.arming_client.call_async(req)
            self.get_logger().info("无人机已解锁") 

    # 上锁无人机
    def disarm(self):
        if self.arming_client.service_is_ready():  
            req = CommandBool.Request()  
            req.value = False            
            self.arming_client.call_async(req)  
            self.get_logger().info("无人机已上锁") 

    # OFFBOARD模式切换 
    def set_offboard_mode(self):
        if self.set_mode_client.service_is_ready(): 
            req = SetMode.Request()  
            req.custom_mode = "OFFBOARD"  
            self.set_mode_client.call_async(req) 
            self.get_logger().info("已设置offboard模式") 

    # 位置发布函数
    def publish_position_setpoint(self, x: float, y: float, z: float):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        self.position_setpoint_pub.publish(msg)
    
    # 地图数据回调
    def map_callback(self, msg):
        new_h = self.map_height // self.scale_ratio
        new_w = self.map_width // self.scale_ratio
        self.grid_map = np.array(msg.data, dtype=np.int32).reshape(new_h, new_w)
        ys, xs = np.where(self.grid_map == -1)
        if ys.size > 0 and xs.size > 0:
            self.target_grid_x = int(round(float(np.mean(xs))))
            self.target_grid_y = int(round(float(np.mean(ys))))
        else:
            self._clear_target_grid()

    # 安全判断函数
    def is_safe(self):
        if self.grid_map is None:
            return False
        gh, gw = self.grid_map.shape[0], self.grid_map.shape[1]
        grip_center_x = gw // 2
        grip_center_y = gh // 2
        if self.grid_map[grip_center_y, grip_center_x] == 1:
            return False
        return True
    
    # 悬停函数
    def hover(self):
        self.publish_position_setpoint(
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        )

    # 降落函数
    def landing_step(self):
        if self.is_safe():
            new_altitude = max(0, self.current_pose.pose.position.z - self.landing_velocity * self.dt)
            self.publish_position_setpoint(
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y,
                new_altitude
            )
            self.get_logger().info(f"安全区域，下降至高度: {new_altitude:.2f} 米")
            if new_altitude <= 0.1:
                self.phase = FlightMode.LANDED
                self.get_logger().info("无人机已着陆")

    # 计算搜索区域占用率（使用缩略栅格尺寸
    # 参数：搜索范围self.search_area_size 返回：栅格占有率
    def compute_grid_occpancy(self):
        if self.grid_map is None:
            return 1.0
        gh, gw = self.grid_map.shape[0], self.grid_map.shape[1]
        grip_center_x = gw // 2
        grip_center_y = gh // 2
        grid_map = self.grid_map
        half_win = int(self.search_area_size)

        min_x = max(0, grip_center_x - half_win)
        max_x = min(gw, grip_center_x + half_win)
        min_y = max(0, grip_center_y - half_win)
        max_y = min(gh, grip_center_y + half_win)

        obstacle_count = np.sum(grid_map[min_y:max_y, min_x:max_x] == 1)
        total_count = (max_x - min_x) * (max_y - min_y)

        if total_count > 0:
            occupancy_ratio = obstacle_count / total_count
            self.get_logger().info(f"搜索区域占用率: {occupancy_ratio:.2f}")
            return occupancy_ratio
        else:
            return 1.0

    # 搜索最佳降落点，返回其中心栅格坐标
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

    # 安全区域发布函数，用于可视化
    def safety_area_pub(self):
        """发布最佳降落栅格坐标与搜索区域半宽，供 visualisation_node 在地图上绘制。"""
        safety_area = self.search_best_landing_point()
        search_area = self.search_area_size
        if safety_area is None:
            return
        cx, cy = safety_area[0], safety_area[1]
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'grid'
        msg.point.x = float(cx)
        msg.point.y = float(cy)
        msg.point.z = float(search_area)  # 搜索区域半宽（栅格数）
        self.safety_area_pub_.publish(msg)

    # 新降落点函数
    def new_landing_point(self):
        if self.grid_map is None or len(self.grid_map) == 0:
            return None
        best = self.search_best_landing_point()
        if best is not None:
            self.target_grid_x, self.target_grid_y = best[0], best[1]
            self.get_logger().info("找到更优降落点，重新导航")
            self.planned_path_world = None  # 触发重新规划
            self.phase = FlightMode.NAVIGATE
        else:
            self.hover()
            self.get_logger().info("地图无最佳降落点，悬停等待")
        return self.target_grid_x, self.target_grid_y

    def navigate(self):
        # t1,t2用于time日志记录，评估规划耗时
        t1 = time.perf_counter()
        goal_xy = self.compute_target_point()  # 计算终点坐标（栅格）
        if goal_xy is None:
            self.get_logger().warn("无法规划：当前栅格地图中无目标点(-1)，跳过本周期规划")
            self.planned_path_world = None
            self.hover()
            return
        goal_col, goal_row = goal_xy[0], goal_xy[1]
        goal = (goal_row, goal_col)  # 转为(row, col)给A*，终点=投放点
        gh, gw = self.grid_map.shape[0], self.grid_map.shape[1]
        start = (self.map_height // self.scale_ratio // 2, self.map_width // self.scale_ratio // 2)  # 起点=栅格中心
        start_row, start_col = start[0], start[1]
        dist_grid = np.hypot(goal_row - start_row, goal_col - start_col)

        # 靠近目标点时不再跑 A*，直接飞向目标，避免末端一直搜索
        if dist_grid <= self.direct_fly_radius_grid:
            path = [start, goal]
            corridor_radii = [self.path_safety_radius] * 2
            path_len_str = 2
            self.get_logger().info(f"靠近目标(栅格距离={dist_grid:.0f}<{self.direct_fly_radius_grid})，直接飞往目标点，不调用A*")
        else:
            path, corridor_radii = self.a_star_path.plan_with_safety_corridor(
                self.grid_map, start, goal, path_safety_radius=self.path_safety_radius
            )
            path_len_str = len(path) if path is not None else '无路径'
            self.get_logger().info(f"起点: {start}, 终点(投放点): {goal}, 路径长度: {path_len_str}")
        self.planned_path_world = [] # 规划出的航路点（ENU世界坐标），供发布给飞机
        # 可视化用路径改成平滑后的栅格轨迹，飞行仍使用下方 ENU 平滑轨迹
        viz_path = self.smooth_grid_path_for_viz(path) if path is not None else None
        self.publish_planned_path(viz_path, corridor_radii=corridor_radii)
        # 将栅格航路点转为 ENU 世界坐标并缓存，供 control_loop 发布给飞机
        # 使用 grid_col_to_enu_x_sign / grid_row_to_enu_y_sign / grid_swap_xy 修正方向
        if path is not None and len(path) > 0:
            row_center, col_center = gh // 2, gw // 2
            wx_center, wy_center = self.grid_to_world(col_center, row_center)
            cur_x = self.current_pose.pose.position.x
            cur_y = self.current_pose.pose.position.y
            for (row, col) in path:
                wx, wy = self.grid_to_world(col, row)
                dx = self.grid_col_to_enu_x_sign * (wx - wx_center)
                dy = self.grid_row_to_enu_y_sign * (wy - wy_center)
                if self.grid_swap_xy:
                    local_x = cur_x + dy
                    local_y = cur_y + dx
                else:
                    local_x = cur_x + dx
                    local_y = cur_y + dy
                self.planned_path_world.append((local_x, local_y))
            # 使用B样条对规划轨迹进行平滑（若Scipy可用）
            self.planned_path_world = self.smooth_path_bspline(self.planned_path_world)
            self.waypoint_index = 0
        else:
            self.planned_path_world = None
        
        # 记录规划时间日志
        t2 = time.perf_counter()
        planning_time_sec = t2 - t1
        self.get_logger().info(f"A* 路径规划耗时: {planning_time_sec:.3f} 秒")
        log_line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | 起点: {start}, 终点(投放点): {goal}, 路径长度: {path_len_str} | 耗时: {planning_time_sec:.3f} 秒\n"
        log_path = os.path.join(self.planning_time_dir, self.planning_log_name)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_line)

    # 发布路径函数，用于可视化
    def publish_planned_path(self, path, corridor_radii=None):
        if path is None or len(path) == 0:
            return
        msg = NavPath()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'grid'
        for i, (row, col) in enumerate(path):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(col)
            pose.pose.position.y = float(row)
            if corridor_radii is not None and i < len(corridor_radii):
                pose.pose.position.z = float(corridor_radii[i])  # 安全管道半径（栅格）
            else:
                pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.planned_path_pub.publish(msg)
    
    # 到达目标点判断函数
    def is_at_goal(self, current_x, current_y, goal_x, goal_y, threshold=0.5):
        return abs(current_x - goal_x) < threshold and abs(current_y - goal_y) < threshold
    
    # 控制循环函数
    def control_loop(self):
        if self.phase == FlightMode.INIT:
            # PX4 要求：必须先持续发布 setpoint，飞控才会接受 OFFBOARD
            self.publish_position_setpoint(
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y,
                self.current_pose.pose.position.z,
            )
            self.setpoint_stream_count += 1
            if self.setpoint_stream_count < self.offboard_ready_count:
                return
            self.set_offboard_mode()
            self.arm()
            if self.current_state.armed and self.current_state.mode == 'OFFBOARD':
                self.get_logger().info("已解锁并进入 OFFBOARD，开始起飞")
                self.phase = FlightMode.TAKEOFF

        elif self.phase == FlightMode.TAKEOFF:
            # OFFBOARD 下必须每周期都发 setpoint，否则会触发 No offboard signal
            if self.current_pose.pose.position.z < self.flight_altitude - 0.5:
                self.publish_position_setpoint(
                    self.current_pose.pose.position.x,
                    self.current_pose.pose.position.y,
                    self.flight_altitude
                )
            else:
                self.publish_position_setpoint(
                    self.current_pose.pose.position.x,
                    self.current_pose.pose.position.y,
                    self.flight_altitude
                )
                self.get_logger().info("达到巡航高度，进入导航阶段")
                self.phase = FlightMode.NAVIGATE

        elif self.phase == FlightMode.NAVIGATE:
            self.compute_target_point()
            if self.target_grid_x is None or self.target_grid_y is None:
                if self.planned_path_world is None or len(self.planned_path_world) == 0:
                    self.get_logger().info("没有识别到投放点，无路径规划")
                    self.hover()
                    return
            # 按 plan_interval 间隔重规划，其余周期复用当前路径，保证 setpoint 高频发布
            do_plan = (self.control_loop_count % self.plan_interval == 0) or self.planned_path_world is None or len(self.planned_path_world) == 0
            self.control_loop_count += 1
            if do_plan:
                self.navigate()
            if self.planned_path_world is None or len(self.planned_path_world) == 0:
                self.hover()
                return
            # 取路径上“前视”航路点作为 setpoint
            next_idx = min(self.path_lookahead, len(self.planned_path_world) - 1)
            next_idx = max(1, next_idx)
            wx, wy = self.planned_path_world[next_idx]
            self.publish_position_setpoint(wx, wy, self.flight_altitude)
            cur_x = self.current_pose.pose.position.x
            cur_y = self.current_pose.pose.position.y
            goal_x, goal_y = self.planned_path_world[-1]
            if self.is_at_goal(cur_x, cur_y, goal_x, goal_y):
                self.phase = FlightMode.REACHED
            # 每周期新路径，已飞段视为 0（从当前点出发）
            progress_msg = Int32()
            progress_msg.data = 0
            self.path_progress_pub.publish(progress_msg)

        elif self.phase == FlightMode.REACHED:
            # 到达后先悬停一个周期，再进入降落
            self.hover()
            self.get_logger().info("到达目标点，进入降落阶段")
            self.phase = FlightMode.LANDING

        elif self.phase == FlightMode.LANDING:
            if self.compute_grid_occpancy() < self.occupy_threshold:
                if self.is_safe():
                    self.landing_step()
                else:
                    self.hover()
                    self.get_logger().info("当前区域不安全，悬停等待")
            else:
                best = self.search_best_landing_point()
                if best is not None:
                    self.target_grid_x, self.target_grid_y = best[0], best[1]
                    self.safety_area_pub()
                    self.get_logger().info("找到更优降落点，重新导航")
                    self.planned_path_world = None  # 触发重新规划
                    self.phase = FlightMode.NAVIGATE
                else:
                    self.hover()
                    self.get_logger().info("地图无最佳降落点，悬停等待，等待手动降落")
                    self.phase = FlightMode.LANDING
        elif self.phase == FlightMode.LANDED:
            self.disarm()
            self.get_logger().info("任务完成，无人机已上锁")
    
    def smooth_path_bspline(self, path_world):
        """
        使用B样条对规划出的ENU航路点进行平滑。
        - 输入:  path_world 为[(x,y), ...] 列表
        - 输出:  平滑后的[(x,y), ...] 列表
        若scipy不可用/路径太短，则直接返回原路径。
        """
        if not _SCIPY_OK:
            return path_world
        if path_world is None or len(path_world) < 4:
            return path_world
        try:
            xs = [p[0] for p in path_world]
            ys = [p[1] for p in path_world]
            # 以弧长为参数，使采样更均匀
            ds = [0.0]
            for i in range(1, len(xs)):
                dx = xs[i] - xs[i - 1]
                dy = ys[i] - ys[i - 1]
                ds.append(ds[-1] + float(np.hypot(dx, dy)))
            if ds[-1] <= 0.0:
                return path_world
            ts = [d / ds[-1] for d in ds]
            # s 为平滑因子，可根据抖动程度适当调大，比如 0.5~2.0
            tck, u = splprep([xs, ys], u=ts, s=1.0, k=3)
            # 重新采样更多点，让轨迹更圆滑
            num_samples = max(len(xs) * 3, 20)
            u_new = np.linspace(0.0, 1.0, num_samples)
            x_new, y_new = splev(u_new, tck)
            smoothed = list(zip([float(x) for x in x_new], [float(y) for y in y_new]))
            return smoothed
        except Exception as e:
            self.get_logger().warn(f"B样条平滑失败，使用原始路径: {e}")
            return path_world
    
    def smooth_grid_path_for_viz(self, path_grid):
        """
        对A*输出的栅格路径进行B样条平滑，仅用于可视化（/planned_path）。
        输入: [(row, col), ...]，输出: [(row, col), ...]，坐标可为浮点但在visualisation中会取round。
        """
        if not _SCIPY_OK:
            return path_grid
        if path_grid is None or len(path_grid) < 4:
            return path_grid
        try:
            rows = [float(p[0]) for p in path_grid]
            cols = [float(p[1]) for p in path_grid]
            ds = [0.0]
            for i in range(1, len(rows)):
                dr = rows[i] - rows[i - 1]
                dc = cols[i] - cols[i - 1]
                ds.append(ds[-1] + float(np.hypot(dr, dc)))
            if ds[-1] <= 0.0:
                return path_grid
            ts = [d / ds[-1] for d in ds]
            tck, u = splprep([rows, cols], u=ts, s=1.0, k=3)
            num_samples = max(len(rows) * 3, 20)
            u_new = np.linspace(0.0, 1.0, num_samples)
            r_new, c_new = splev(u_new, tck)
            smoothed = list(zip(r_new, c_new))
            return smoothed
        except Exception as e:
            self.get_logger().warn(f"栅格路径B样条平滑失败，使用原始路径: {e}")
            return path_grid
    
    def compute_target_point(self):
        """返回 map_callback 中已算好的栅格目标点；无目标时返回 None。"""
        if self.grid_map is None:
            self._clear_target_grid()
            return None
        if not np.any(self.grid_map == -1):
            self.get_logger().warn("未找到目标点，栅格地图中没有值为 -1 的格子")
            self._clear_target_grid()
            return None
        if self.target_grid_x is not None and self.target_grid_y is not None:
            return self.target_grid_x, self.target_grid_y
        ys, xs = np.where(self.grid_map == -1)
        self.target_grid_x = int(round(float(np.mean(xs))))
        self.target_grid_y = int(round(float(np.mean(ys))))
        return self.target_grid_x, self.target_grid_y

    # 清空终点函数；防止飞行过程中找不到下一个点后仍使用上一帧的旧值
    def _clear_target_grid(self):
        """清空栅格目标与 ENU 目标，避免目标消失后仍使用上一帧的旧值。"""
        self.target_grid_x = None
        self.target_grid_y = None
        self.target_x = None
        self.target_y = None

def main(args=None):
    rclpy.init(args=args)
    landing_navigator = LandingNavigation()
    rclpy.spin(landing_navigator)
    landing_navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()  