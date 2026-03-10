"""
可视化节点：在栅格图与检测图上绘制规划路径、已飞轨迹、安全区域，供 rqt_image_view 查看验证飞行。

- /grid_map_path：栅格地图 + 规划路径（待飞=绿色）+ 从导航阶段开始的已飞轨迹（蓝色）+ 安全区域
- /yolo_with_path：YOLO 检测图 + 同上叠加
- /camera/image_rectified：姿态纠正后的相机图（与建图尺度一致，便于对照）
- 已飞轨迹来自 /flown_trajectory（navigation 节点发布），在每帧图片上均用蓝色标注
- 原始相机图保存时可保存纠正后的版本，与栅格地图几何一致
"""

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import os
import math
import json
from pathlib import Path
from std_msgs.msg import Int8MultiArray, Int32
from sensor_msgs.msg import Image
from nav_msgs.msg import Path as NavPath
from geometry_msgs.msg import PointStamped, PoseStamped
from cv_bridge import CvBridge
from datetime import datetime


def euler_from_quaternion(qx, qy, qz, qw):
    """从四元数计算欧拉角 (roll, pitch, yaw) 弧度，ZYX 顺序。"""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rotation_matrix_from_euler(roll, pitch, yaw):
    """从欧拉角构建 3x3 旋转矩阵 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ], dtype=np.float64)
    return R


def get_rectification_homography(roll, pitch, K):
    """根据 roll、pitch 计算水平朝下纠正的单应矩阵 H = K * R_rect * inv(K)。"""
    R_rect = rotation_matrix_from_euler(-roll, -pitch, 0.0)
    K = np.array(K, dtype=np.float64)
    H = K @ R_rect @ np.linalg.inv(K)
    return H


class visualisation_node(Node):
    """订阅规划路径、已飞轨迹、栅格图、检测图；在每帧上绘制蓝色已飞轨迹与绿色待飞路径，发布图像供 rqt_image_view 查看。"""

    def __init__(self):
        super().__init__('visualisation_node')

        self.br = CvBridge()
        self.map_width = 1024
        self.map_height = 768
        self.grid_scale = 10  # 栅格下采样倍数，与 navigation/sensing 一致

        # 订阅
        self.planned_path_sub = self.create_subscription(NavPath, '/planned_path', self.planned_path_callback, 10)
        self.path_progress_sub = self.create_subscription(Int32, '/path_progress', self.path_progress_callback, 10)
        self.flown_trajectory_sub = self.create_subscription(NavPath, '/flown_trajectory', self.flown_trajectory_callback, 10)
        self.safety_area_sub = self.create_subscription(PointStamped, '/safety_area', self.safety_area_callback, 10)
        self.grid_map_sub = self.create_subscription(Int8MultiArray, '/grid_map', self.grid_map_callback, 10)
        self.viz_yolo_sub = self.create_subscription(Image, '/viz_yolo_detections', self.yolo_viz_callback, 10)
        # 与 sensing_node 使用同一相机话题，否则 camera_rectified 无图可保存
        self.camera_topic = '/world/myworld/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image'
        self.viz_camera_sub = self.create_subscription(Image, self.camera_topic, self.camera_callback, 10)
        self.local_pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, 10)

        # 发布（供 rqt_image_view 查看）
        self.grid_map_path_pub = self.create_publisher(Image, '/grid_map_path', 10)
        self.yolo_with_path_pub = self.create_publisher(Image, '/yolo_with_path', 10)
        self.camera_rectified_pub = self.create_publisher(Image, '/camera/image_rectified', 10)

        self.planned_path = None  # nav_msgs/Path，路径点 pose.position.x=col, y=row
        self.path_progress = 0    # 已飞航路点数量（与 navigation 的 waypoint_index 一致）
        self.flown_trajectory = []  # 从导航阶段开始所有已走过的航路点 [(row,col), ...]，用于蓝色标注
        self.safety_area = None   # PointStamped: x=best_col, y=best_row, z=search_area_half（均为 navigation 栅格坐标）
        self.grid_map = None     # 下采样后的栅格 (new_h, new_w)
        # navigation 使用的栅格尺寸（scale_ratio=5），用于将 safety_area 坐标缩放到本节点栅格
        self.nav_grid_w = self.map_width // 5
        self.nav_grid_h = self.map_height // 5
        self.yolo_rgb = None     # 检测结果图（带框）
        self.raw_rgb = None      # 原始相机图
        self.rectified_rgb = None  # 姿态纠正后的相机图（与建图尺度一致）
        self.grid_map_path_viz_rgb = None  # 栅格地图 + 路径
        self.path_viz_rgb = None            # 检测图 + 路径
        self.local_pose = None   # 当前姿态，用于图像纠正
        self.K = None            # 相机内参 3x3，用于姿态纠正
        self._load_camera_intrinsics()
        # 路径可视化颜色（BGR）：飞过的路径=蓝色，待飞航路=绿色
        self.color_flown = (255, 0, 0)      # 飞过的路径 - 蓝色
        self.color_remaining = (0, 255, 0)   # 待飞航路 - 绿色

        script_dir = os.path.dirname(os.path.abspath(__file__))
        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.viz_save_base = os.path.join(script_dir, 'log', date_str)
        self.viz_save_dirs = {
            'raw_rgb': os.path.join(self.viz_save_base, 'raw_rgb'),
            'camera_rectified': os.path.join(self.viz_save_base, 'camera_rectified'),
            'grid_map_path': os.path.join(self.viz_save_base, 'grid_map_path'),
            'yolo_with_path': os.path.join(self.viz_save_base, 'yolo_with_path'),
            'yolo_result': os.path.join(self.viz_save_base, 'yolo_result'),
        }
        for d in self.viz_save_dirs.values():
            os.makedirs(d, exist_ok=True)

        self.dt = 1.0
        self.viz_save_count = 0
        self.viz_save_timer = self.create_timer(self.dt, self.save_viz_images_callback)

    def _load_camera_intrinsics(self):
        """加载相机内参，用于姿态纠正可视化（与 sensing/navigation 共用 cfg）。"""
        cfg_path = Path(__file__).resolve().parent.parent / 'cfg' / 'camera_intrinsics.json'
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            self.K = np.array(cfg['K'], dtype=np.float64)
            self.get_logger().info('已加载相机内参，可视化将发布纠正图 /camera/image_rectified')
        except Exception as e:
            self.get_logger().warn(f'未加载相机内参 ({cfg_path}): {e}，不发布纠正图')

    def pose_callback(self, msg):
        self.local_pose = msg

    def _rectify_image(self, cv_image):
        """根据当前姿态将图像纠正为水平朝下视图，与建图一致。"""
        if self.K is None or cv_image is None or self.local_pose is None:
            return cv_image
        q = self.local_pose.pose.orientation
        roll, pitch, _ = euler_from_quaternion(q.x, q.y, q.z, q.w)
        if abs(roll) < 1e-4 and abs(pitch) < 1e-4:
            return cv_image
        H = get_rectification_homography(roll, pitch, self.K)
        h, w = cv_image.shape[:2]
        M = np.linalg.inv(H)
        rectified = cv2.warpPerspective(cv_image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return rectified

    def planned_path_callback(self, msg):
        self.planned_path = msg
        self.path_progress = 0  # 新路径到达时先置 0，由 /path_progress 后续更新
        self._draw_path_on_grid_map()
        self._draw_path_on_detection_map()

    def path_progress_callback(self, msg):
        self.path_progress = max(0, int(msg.data))
        self._draw_path_on_grid_map()
        self._draw_path_on_detection_map()

    def flown_trajectory_callback(self, msg):
        """接收从导航阶段开始的所有已走过航路点（栅格），用于在每帧图片上用蓝色标注。"""
        self.flown_trajectory = []
        for p in msg.poses:
            col = int(round(p.pose.position.x))
            row = int(round(p.pose.position.y))
            self.flown_trajectory.append((row, col))
        self._draw_path_on_grid_map()
        self._draw_path_on_detection_map()

    def safety_area_callback(self, msg):
        """接收最佳降落栅格坐标与搜索区域，触发重绘。"""
        self.safety_area = msg
        self._draw_path_on_grid_map()
        self._draw_path_on_detection_map()

    def grid_map_callback(self, msg):
        # sensing 已下发整形栅格 153×204，直接 reshape 使用（与 navigation 一致）
        new_h = self.map_height // 5   # 153
        new_w = self.map_width // 5    # 204
        expected_size = new_h * new_w  # 31212
        data = np.array(msg.data, dtype=np.int8)
        if data.size != expected_size:
            self.get_logger().warn(
                f"grid_map 长度不符: 收到 {data.size}，期望 {expected_size} (形状 {new_h}×{new_w})，跳过本帧"
            )
            return
        self.grid_map = data.reshape(new_h, new_w)
        self._draw_path_on_grid_map()

    def yolo_viz_callback(self, msg):
        self.yolo_rgb = self.br.imgmsg_to_cv2(msg, 'bgr8')
        self._draw_path_on_detection_map()

    def camera_callback(self, msg):
        self.raw_rgb = self.br.imgmsg_to_cv2(msg, 'bgr8')
        # 姿态纠正后发布，便于与栅格地图对照查看（无内参/姿态时即为原图）
        rectified = self._rectify_image(self.raw_rgb)
        self.rectified_rgb = rectified
        out_msg = self.br.cv2_to_imgmsg(rectified, encoding='bgr8')
        out_msg.header = msg.header
        out_msg.header.frame_id = 'camera_rectified'
        self.camera_rectified_pub.publish(out_msg)

    def _path_to_rows_cols(self):
        """路径点 (row, col)；若 Path 中带安全管道，则 position.z 为安全半径（栅格）。"""
        if self.planned_path is None or len(self.planned_path.poses) == 0:
            return [], []
        rows_cols = []
        radii = []
        for p in self.planned_path.poses:
            rows_cols.append((int(round(p.pose.position.y)), int(round(p.pose.position.x))))
            radii.append(max(0.0, float(p.pose.position.z)))
        return rows_cols, radii

    def _draw_path_on_grid_map(self):
        if self.grid_map is None:
            return
        gh, gw = self.grid_map.shape[0], self.grid_map.shape[1]
        img = np.ones((gh, gw, 3), dtype=np.uint8) * 255
        img[self.grid_map == 1] = (0, 0, 255)
        img[self.grid_map == -1] = (128, 128, 128)

        path, corridor_radii = self._path_to_rows_cols()
        pt_r = max(1, min(gh, gw) // 40)
        line_w = max(1, min(gh, gw) // 50)
        if path:
            n_flown = min(max(0, self.path_progress), len(path))
            flown = path[:n_flown]
            remaining = path[n_flown:]
            # 飞过的路径：图片中标记为蓝色
            for i, (row, col) in enumerate(flown):
                if 0 <= row < gh and 0 <= col < gw:
                    if i < len(corridor_radii) and corridor_radii[i] > 0:
                        r = int(round(corridor_radii[i]))
                        cv2.circle(img, (col, row), r, (255, 200, 200), max(1, line_w))
                    cv2.circle(img, (col, row), pt_r, self.color_flown, -1)
            for i in range(len(flown) - 1):
                r0, c0 = flown[i][0], flown[i][1]
                r1, c1 = flown[i + 1][0], flown[i + 1][1]
                if 0 <= r0 < gh and 0 <= c0 < gw and 0 <= r1 < gh and 0 <= c1 < gw:
                    cv2.line(img, (c0, r0), (c1, r1), self.color_flown, line_w)
            # 待飞航路：绿色
            for i, (row, col) in enumerate(remaining):
                if 0 <= row < gh and 0 <= col < gw:
                    j = n_flown + i
                    if j < len(corridor_radii) and corridor_radii[j] > 0:
                        r = int(round(corridor_radii[j]))
                        cv2.circle(img, (col, row), r, (200, 255, 200), max(1, line_w))
                    cv2.circle(img, (col, row), pt_r, self.color_remaining, -1)
            for i in range(len(remaining) - 1):
                r0, c0 = remaining[i][0], remaining[i][1]
                r1, c1 = remaining[i + 1][0], remaining[i + 1][1]
                if 0 <= r0 < gh and 0 <= c0 < gw and 0 <= r1 < gh and 0 <= c1 < gw:
                    cv2.line(img, (c0, r0), (c1, r1), self.color_remaining, line_w)
            if flown and remaining:
                r0, c0 = flown[-1][0], flown[-1][1]
                r1, c1 = remaining[0][0], remaining[0][1]
                if 0 <= r0 < gh and 0 <= c0 < gw and 0 <= r1 < gh and 0 <= c1 < gw:
                    cv2.line(img, (c0, r0), (c1, r1), (180, 180, 0), line_w)
        # 从导航阶段开始所有已走过的航路点：蓝色标注，便于 rqt_image_view 验证飞行
        if self.flown_trajectory:
            traj_pt_r = max(1, pt_r)
            for i, (row, col) in enumerate(self.flown_trajectory):
                if 0 <= row < gh and 0 <= col < gw:
                    cv2.circle(img, (col, row), traj_pt_r, self.color_flown, -1)
            for i in range(len(self.flown_trajectory) - 1):
                r0, c0 = self.flown_trajectory[i][0], self.flown_trajectory[i][1]
                r1, c1 = self.flown_trajectory[i + 1][0], self.flown_trajectory[i + 1][1]
                if 0 <= r0 < gh and 0 <= c0 < gw and 0 <= r1 < gh and 0 <= c1 < gw:
                    cv2.line(img, (c0, r0), (c1, r1), self.color_flown, max(1, line_w))
        # 绘制最佳降落栅格与搜索区域
        if self.safety_area is not None:
            sx = self.safety_area.point.x * gw / self.nav_grid_w
            sy = self.safety_area.point.y * gh / self.nav_grid_h
            half = self.safety_area.point.z * min(gw / self.nav_grid_w, gh / self.nav_grid_h)
            c_pt = (int(round(sx)), int(round(sy)))
            if 0 <= c_pt[0] < gw and 0 <= c_pt[1] < gh:
                r_rect = max(1, int(round(half)))
                cv2.rectangle(img, (c_pt[0] - r_rect, c_pt[1] - r_rect),
                              (c_pt[0] + r_rect, c_pt[1] + r_rect), (255, 165, 0), max(1, line_w))
                cv2.circle(img, c_pt, max(1, pt_r + 1), (0, 0, 255), -1)
        self.grid_map_path_viz_rgb = img
        # 放大后再发布/保存，否则 76x102 过小
        if self.grid_map_path_viz_rgb is not None:
            big = cv2.resize(self.grid_map_path_viz_rgb, (gw * self.grid_scale, gh * self.grid_scale), interpolation=cv2.INTER_NEAREST)
            msg = self.br.cv2_to_imgmsg(big, encoding='bgr8')
            self.grid_map_path_pub.publish(msg)

    def _draw_path_on_detection_map(self):
        if self.yolo_rgb is None:
            return
        viz = self.yolo_rgb.copy()
        img_h, img_w = viz.shape[0], viz.shape[1]
        path, corridor_radii = self._path_to_rows_cols()
        if not path or self.grid_map is None:
            # 无规划路径时仍绘制已飞轨迹（若有）
            if self.flown_trajectory and self.grid_map is not None:
                gh, gw = self.grid_map.shape[0], self.grid_map.shape[1]
                for i, (row, col) in enumerate(self.flown_trajectory):
                    px = int(col * img_w / gw)
                    py = int(row * img_h / gh)
                    px = max(0, min(px, img_w - 1))
                    py = max(0, min(py, img_h - 1))
                    cv2.circle(viz, (px, py), 4, self.color_flown, -1)
                for i in range(len(self.flown_trajectory) - 1):
                    r0, c0 = self.flown_trajectory[i][0], self.flown_trajectory[i][1]
                    r1, c1 = self.flown_trajectory[i + 1][0], self.flown_trajectory[i + 1][1]
                    p0 = (int(c0 * img_w / gw), int(r0 * img_h / gh))
                    p1 = (int(c1 * img_w / gw), int(r1 * img_h / gh))
                    cv2.line(viz, p0, p1, self.color_flown, 2)
            self.path_viz_rgb = viz
            if self.path_viz_rgb is not None:
                msg = self.br.cv2_to_imgmsg(self.path_viz_rgb, encoding='bgr8')
                self.yolo_with_path_pub.publish(msg)
            return
        gh, gw = self.grid_map.shape[0], self.grid_map.shape[1]
        n_flown = min(max(0, self.path_progress), len(path))
        flown = path[:n_flown]
        remaining = path[n_flown:]
        # 飞过的路径：图片中标记为蓝色
        for i, (row, col) in enumerate(flown):
            px = int(col * img_w / gw)
            py = int(row * img_h / gh)
            px = max(0, min(px, img_w - 1))
            py = max(0, min(py, img_h - 1))
            if i < len(corridor_radii) and corridor_radii[i] > 0:
                rr = int(round(corridor_radii[i] * img_w / gw))
                cv2.circle(viz, (px, py), max(2, rr), (255, 200, 200), 1)
            cv2.circle(viz, (px, py), 5, self.color_flown, 2)
        for i in range(len(flown) - 1):
            r0, c0 = flown[i][0], flown[i][1]
            r1, c1 = flown[i + 1][0], flown[i + 1][1]
            p0 = (int(c0 * img_w / gw), int(r0 * img_h / gh))
            p1 = (int(c1 * img_w / gw), int(r1 * img_h / gh))
            cv2.line(viz, p0, p1, self.color_flown, 2)
        # 待飞航路：绿色
        for i, (row, col) in enumerate(remaining):
            px = int(col * img_w / gw)
            py = int(row * img_h / gh)
            px = max(0, min(px, img_w - 1))
            py = max(0, min(py, img_h - 1))
            j = n_flown + i
            if j < len(corridor_radii) and corridor_radii[j] > 0:
                rr = int(round(corridor_radii[j] * img_w / gw))
                cv2.circle(viz, (px, py), max(2, rr), (200, 255, 200), 1)
            cv2.circle(viz, (px, py), 5, self.color_remaining, 2)
        for i in range(len(remaining) - 1):
            r0, c0 = remaining[i][0], remaining[i][1]
            r1, c1 = remaining[i + 1][0], remaining[i + 1][1]
            p0 = (int(c0 * img_w / gw), int(r0 * img_h / gh))
            p1 = (int(c1 * img_w / gw), int(r1 * img_h / gh))
            cv2.line(viz, p0, p1, self.color_remaining, 2)
        if flown and remaining:
            r0, c0 = flown[-1][0], flown[-1][1]
            r1, c1 = remaining[0][0], remaining[0][1]
            p0 = (int(c0 * img_w / gw), int(r0 * img_h / gh))
            p1 = (int(c1 * img_w / gw), int(r1 * img_h / gh))
            cv2.line(viz, p0, p1, (0, 180, 180), 2)
        # 从导航阶段开始所有已走过的航路点：蓝色标注，便于 rqt_image_view 验证飞行
        if self.flown_trajectory and self.grid_map is not None:
            gh, gw = self.grid_map.shape[0], self.grid_map.shape[1]
            for i, (row, col) in enumerate(self.flown_trajectory):
                px = int(col * img_w / gw)
                py = int(row * img_h / gh)
                px = max(0, min(px, img_w - 1))
                py = max(0, min(py, img_h - 1))
                cv2.circle(viz, (px, py), 4, self.color_flown, -1)
            for i in range(len(self.flown_trajectory) - 1):
                r0, c0 = self.flown_trajectory[i][0], self.flown_trajectory[i][1]
                r1, c1 = self.flown_trajectory[i + 1][0], self.flown_trajectory[i + 1][1]
                p0 = (int(c0 * img_w / gw), int(r0 * img_h / gh))
                p1 = (int(c1 * img_w / gw), int(r1 * img_h / gh))
                cv2.line(viz, p0, p1, self.color_flown, 2)
        if self.safety_area is not None and self.grid_map is not None:
            sx = self.safety_area.point.x * gw / self.nav_grid_w
            sy = self.safety_area.point.y * gh / self.nav_grid_h
            half = self.safety_area.point.z * min(gw / self.nav_grid_w, gh / self.nav_grid_h)
            px = int(sx * img_w / gw)
            py = int(sy * img_h / gh)
            half_px = max(2, int(half * img_w / gw))
            cv2.rectangle(viz, (px - half_px, py - half_px), (px + half_px, py + half_px), (255, 165, 0), 2)
            cv2.circle(viz, (px, py), 6, (0, 0, 255), 2)
        self.path_viz_rgb = viz
        if self.path_viz_rgb is not None:
            msg = self.br.cv2_to_imgmsg(self.path_viz_rgb, encoding='bgr8')
            self.yolo_with_path_pub.publish(msg)

    def save_viz_images_callback(self):
        self.viz_save_count += 1
        name = f'{self.viz_save_count:06d}.png'
        if self.raw_rgb is not None:
            path = os.path.join(self.viz_save_dirs['raw_rgb'], name)
            cv2.imwrite(path, self.raw_rgb)
            self.get_logger().info(f'已保存 原始图像: {path}')
        if self.rectified_rgb is not None and self.rectified_rgb.size > 0:
            path = os.path.join(self.viz_save_dirs['camera_rectified'], name)
            cv2.imwrite(path, self.rectified_rgb)
            self.get_logger().info(f'已保存 纠正图像: {path}')
        if self.grid_map_path_viz_rgb is not None:
            path = os.path.join(self.viz_save_dirs['grid_map_path'], name)
            big = cv2.resize(self.grid_map_path_viz_rgb, (
                self.grid_map_path_viz_rgb.shape[1] * self.grid_scale,
                self.grid_map_path_viz_rgb.shape[0] * self.grid_scale,
            ), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(path, big)
            self.get_logger().info(f'已保存 栅格地图+路径: {path}')
        if self.path_viz_rgb is not None:
            path = os.path.join(self.viz_save_dirs['yolo_with_path'], name)
            cv2.imwrite(path, self.path_viz_rgb)
            self.get_logger().info(f'已保存 检测图+路径: {path}')
        if self.yolo_rgb is not None:
            path = os.path.join(self.viz_save_dirs['yolo_result'], name)
            cv2.imwrite(path, self.yolo_rgb)
            self.get_logger().info(f'已保存 YOLO结果: {path}')


def main(args=None):
    rclpy.init(args=args)
    node = visualisation_node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
