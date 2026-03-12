import rclpy
from rclpy.node import Node
import pathlib
import json
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from std_msgs.msg import Int8MultiArray, Header
import cv2
import numpy as np
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
import math

temp = pathlib.PosixPath
pathlib.WindowsPath = temp

def quaternion_from_euler(roll, pitch, yaw):
    """
    从欧拉角计算四元数
    欧拉角转四元数是ROS中姿态表示的常用转换
    
    参数:
        roll (float): 横滚角 (弧度)
        pitch (float): 俯仰角 (弧度)
        yaw (float): 偏航角 (弧度)
    
    返回:
        tuple: (qx, qy, qz, qw) 四元数四个分量
    """
    # 半角计算，减少三角函数计算次数
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    # 四元数计算公式
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def euler_from_quaternion(qx, qy, qz, qw):
    """
    从四元数计算欧拉角 (ZYX 顺序，与 quaternion_from_euler 互逆)
    ROS 中常用 (qx, qy, qz, qw) 表示四元数
    
    参数:
        qx, qy, qz, qw (float): 四元数四个分量
    
    返回:
        tuple: (roll, pitch, yaw) 弧度制欧拉角
    """
    # roll (x 轴旋转)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # pitch (y 轴旋转)，处理万向节锁
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # yaw (z 轴旋转)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def rotation_matrix_from_euler(roll, pitch, yaw):
    """
    从欧拉角 (ZYX) 构建 3x3 旋转矩阵 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。
    用于图像纠正时用 R_rect = Rx(-roll) * Ry(-pitch) 将倾斜视图纠正为水平朝下。
    """
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
    """
    根据当前姿态的 roll、pitch 计算“水平朝下”纠正的单应矩阵。
    H = K * R_rect * inv(K)，其中 R_rect = Rx(-roll) * Ry(-pitch)，将倾斜相机平面映射到虚拟水平朝下平面。
    """
    R_rect = rotation_matrix_from_euler(-roll, -pitch, 0.0)
    K = np.array(K, dtype=np.float64)
    H = K @ R_rect @ np.linalg.inv(K)
    return H


class ImageMapping(Node):
    def __init__(self):
        super().__init__('image_mapping_node')
        
        self.model_path = '/home/tangsh/20260216tuhu_ws/yolov5'
        self.model_file = self.model_path + '/self_models/20260213best.pt'
        self.br = CvBridge()
        self.target_width = 1024
        self.target_height = 768
        self.image = None
        self.detections = None
        self.yolo_detections = None
        self.grip_mapping = None
        self.last_image_msg = None
        self.dt = 1/30.0
        self.scale_ratio = 5
        self.local_pose = PoseStamped()
        self.yolo_3d_cfg_path = pathlib.Path(__file__).resolve().parent.parent / 'cfg' / 'yolo_3d.json'
        self.camera_intrinsics_path = pathlib.Path(__file__).resolve().parent.parent / 'cfg' / 'camera_intrinsics.json'
        self.class_height = self._load_yolo_3d_config()
        self.K = None
        self._load_camera_intrinsics()

        try:
            self.model = torch.hub.load(self.model_path, 'custom', path=self.model_file, source='local')
            self.get_logger().info('Model loaded successfully.')
        except Exception as e:
            self.get_logger().error(f'Error loading model: {e}')
            raise e
        
        self.camera_sub= self.create_subscription(Image, '/world/myworld/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image', self.image_callback,10)
        # self.camera_sub = self.create_subscription(Image,'/camera/image_raw', self.image_callback, 10)
        self.local_pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, 10)
        self.mapping_pub = self.create_publisher(Int8MultiArray, '/grid_map', 10)                       # 栅格地图
        self.yolo_det_pub = self.create_publisher(Image, '/viz_yolo_detections', 10)                   # 可视化发布
        self.detection_pub = self.create_publisher(Detection2DArray, '/yolo_detections_array', 10)  # 标准 yolo 信息发布                      # 状态发布
        self.timer = self.create_timer(self.dt, self.sensing_loop)  
        self.get_logger().info('无人机YOLO感知建图节点已启动！')

    # 位置回调函数
    def pose_callback(self, msg):
        self.local_pose = msg

    def _load_yolo_3d_config(self):
        """从 self.yolo_3d_cfg_path 加载类别->高度(m)映射，未配置的类别默认为 0。"""
        class_height = {}
        try:
            with open(self.yolo_3d_cfg_path, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    class_height[int(item['class'])] = float(item['height'])
            self.get_logger().info(f'从 {self.yolo_3d_cfg_path}: {class_height} 成功加载yolo3d先验高度配置文件')
        except Exception as e:
            self.get_logger().warn(f'找不到配置文件 ({self.yolo_3d_cfg_path}): {e}, using default height 0')
        return class_height

    def _load_camera_intrinsics(self):
        """加载相机内参 K（3x3），用于基于 roll/pitch 的图像纠正，使建图不随姿态变化。"""
        try:
            with open(self.camera_intrinsics_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            self.K = np.array(cfg['K'], dtype=np.float64)
            self.get_logger().info('已加载相机内参，姿态纠正建图已启用')
        except Exception as e:
            self.get_logger().warn(f'未加载相机内参 ({self.camera_intrinsics_path}): {e}，建图不做姿态纠正')
            self.K = None

    def _rectify_image(self, cv_image):
        """根据当前机体姿态（roll, pitch）将图像纠正为“水平朝下”视图，保证像素与地面尺度一致。"""
        if self.K is None or cv_image is None:
            return cv_image
        qx = self.local_pose.pose.orientation.x
        qy = self.local_pose.pose.orientation.y
        qz = self.local_pose.pose.orientation.z
        qw = self.local_pose.pose.orientation.w
        roll, pitch, _ = euler_from_quaternion(qx, qy, qz, qw)
        if abs(roll) < 1e-4 and abs(pitch) < 1e-4:
            return cv_image
        H = get_rectification_homography(roll, pitch, self.K)
        h, w = cv_image.shape[:2]
        M = np.linalg.inv(H)
        rectified = cv2.warpPerspective(cv_image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return rectified

    # 图像回调函数
    def image_callback(self, msg):
        cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.resize(cv_image, (self.target_width, self.target_height))
        self.image = cv_image
        self.last_image_msg = msg

    # 模型推理（可在纠正后的图像上推理，使检测框与建图尺度一致）
    def model_reasoning(self, image_for_inference=None):
        img = image_for_inference if image_for_inference is not None else self.image
        results = self.model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        qx, qy, qz, qw = self.local_pose.pose.orientation.x, self.local_pose.pose.orientation.y, self.local_pose.pose.orientation.z, self.local_pose.pose.orientation.w
        roll, pitch, yaw = euler_from_quaternion(qx, qy, qz, qw)
        detections = results.xyxy[0].cpu().numpy()
        self._rectified_image = img
        self.detections = detections

    # 构建yolo标准消息Detection2DArray并发布 yolo 检测标准信息
    def detections_pub(self, msg):
        detection_array = Detection2DArray()
        for det in self.detections:
            x1, y1, x2, y2, conf, cls = det
            detection = Detection2D()
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = self.model.names[int(cls)]
            hypothesis.hypothesis.score = float(conf)
            cls_id= int(cls)
            # 用先验类别高度作为 pose 的 z (m)
            hypothesis.pose.pose.position.z = float(self.class_height.get(cls_id, 0.0))
            detection.results.append(hypothesis)
            detection.bbox.center.position.x = float((x1 + x2) / 2.0)
            detection.bbox.center.position.y = float((y1 + y2) / 2.0)
            detection.bbox.center.theta = 0.0
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            detection_array.detections.append(detection)
        detection_array.header = msg.header
        self.yolo_detections = detection_array
        self.detection_pub.publish(detection_array)

    # 发布检测结果可视化（在纠正后的图像上画框，与建图一致）
    def viz_image_pub(self, msg):
        base = getattr(self, '_rectified_image', None)
        if base is None:
            base = self.image
        vis = base.copy()
        for det in self.detections:
            x1, y1, x2, y2, conf, cls = det
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            z = self.class_height.get(int(cls), 0.0)
            label = f"{self.model.names[int(cls)]}:{conf:.2f} z={z}m"
            cv2.putText(vis, label, (x1i, max(15, y1i - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        vis_msg = self.br.cv2_to_imgmsg(vis, encoding='bgr8')
        vis_msg.header = msg.header
        self.yolo_det_pub.publish(vis_msg)

    # 建图模块
    def mapping(self,detections):
        pixel_map = np.zeros((self.target_height, self.target_width), dtype=np.int8)
        for det in detections:
            cls_id = int(det[5])
            if cls_id == 0: 
                x1, y1, x2, y2 = map(int, det[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.target_width, x2), min(self.target_height, y2)
                pixel_map[y1:y2, x1:x2] = -1 
            else:
                z = self.class_height.get(int(cls_id), 0.0)
                h = self.local_pose.pose.position.z
                x1, y1, x2, y2, conf, cls_id = det[:6]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(self.target_width, int(x2)), min(self.target_height, int(y2))
                self.get_logger().info(f"Detected class {int(cls_id)} at [{x1}, {y1}, {x2}, {y2}]")
                # 判断z-h>0,则认为该物体在飞行器上方，否则认为该物体在飞行器下方
                if z-h>0:
                    pixel_map[y1:y2, x1:x2] = 1
                else:
                    pixel_map[y1:y2, x1:x2] = 0
        # 将像素地图降采样为栅格地图（与 navigation 约定：整形栅格 153×204，直接使用）
        new_h = self.target_height // self.scale_ratio
        new_w = self.target_width // self.scale_ratio
        trimmed = pixel_map[: new_h * self.scale_ratio, : new_w * self.scale_ratio]
        blocks = trimmed.reshape(new_h, self.scale_ratio, new_w, self.scale_ratio)
        has_obstacle = (blocks == 1).any(axis=(1, 3))
        has_goal = (blocks == -1).any(axis=(1, 3))
        grid_map = np.zeros((new_h, new_w), dtype=np.int8)
        grid_map[has_obstacle] = 1
        grid_map[~has_obstacle & has_goal] = -1
        grid_map_msg = Int8MultiArray()
        grid_map_msg.data = grid_map.ravel().tolist()
        self.mapping_pub.publish(grid_map_msg)

    # 感知循环：先按姿态纠正图像再检测与建图，避免姿态变化导致建图几何失真
    def sensing_loop(self):
        if self.image is None:
            return
        try:
            rectified = self._rectify_image(self.image)
            self.model_reasoning(rectified)
            if hasattr(self, 'last_image_msg') and self.last_image_msg is not None:
                ref_msg = self.last_image_msg
            else:
                ref_msg = Image()
                ref_msg.header = Header()
                ref_msg.header.stamp = self.get_clock().now().to_msg()
                ref_msg.header.frame_id = 'camera'
            if self.detections is not None:
                self.detections_pub(ref_msg)
                self.viz_image_pub(ref_msg)
                self.mapping(self.detections)
        except Exception as e:
            self.get_logger().error(f"sensing_loop error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageMapping()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


    # 