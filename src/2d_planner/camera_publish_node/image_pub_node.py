#!/usr/bin/env python3
"""
图片发布节点：从本地文件读取一张照片，按可配置路径发布到 /camera/image_raw，
供 detect_mapping 与 navigation_2 使用（可替代相机节点做离线/单帧测试）。
支持通过参数修改图片路径、分辨率、发布频率。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os


class ImagePubNode(Node):
    def __init__(self):
        super().__init__('image_pub_node')

        # --- 可修改的参数：图片路径、话题、分辨率、发布频率 ---
        self.declare_parameter('image_path', '/home/tangsh/20260216tuhu_ws/video_and_photo/demo_photo_2.png')
        self.declare_parameter('topic', '/camera/image_raw')
        self.declare_parameter('width', 1024)
        self.declare_parameter('height', 768)
        self.declare_parameter('publish_rate', 10.0)  # Hz，与 detect_mapping 使用习惯一致

        self._image_path = self.get_parameter('image_path').value
        topic = self.get_parameter('topic').value
        self._width = self.get_parameter('width').value
        self._height = self.get_parameter('height').value
        rate = self.get_parameter('publish_rate').value

        self._publisher = self.create_publisher(Image, topic, 10)
        self._bridge = CvBridge()
        self._current_image = None  # 当前要发布的图像 (BGR, 已 resize)
        self._last_loaded_path = None  # 用于检测参数修改后重新加载

        if self._image_path and os.path.isfile(self._image_path):
            self._load_and_resize(self._image_path)
            self._last_loaded_path = self._image_path
        else:
            self.get_logger().warn(
                f'未设置有效图片路径 (image_path="{self._image_path}")，节点将发布空白图；'
                '请通过参数设置，例如: --ros-args -p image_path:=/path/to/your/image.png'
            )
            self._current_image = self._blank_image()

        period = 1.0 / rate if rate > 0 else 1.0
        self._timer = self.create_timer(period, self._timer_callback)
        self.get_logger().info(
            f'图片发布节点已启动: topic={topic}, size={self._width}x{self._height}, rate={rate} Hz'
        )

    def _blank_image(self):
        """返回一张空白图（灰度 128），尺寸为配置的 width x height。"""
        return np.ones((self._height, self._width, 3), dtype=np.uint8) * 128

    def _load_and_resize(self, path):
        """从 path 加载图片并 resize 到 (width, height)，存到 self._current_image。"""
        try:
            img = cv2.imread(path)
            if img is None:
                self.get_logger().error(f'无法解码图片: {path}')
                self._current_image = self._blank_image()
                return
            self._current_image = cv2.resize(
                img, (self._width, self._height), interpolation=cv2.INTER_AREA
            )
            self._last_loaded_path = path
            self.get_logger().info(f'已加载并缩放图片: {path} -> {self._width}x{self._height}')
        except Exception as e:
            self.get_logger().error(f'加载图片失败 {path}: {e}')
            self._current_image = self._blank_image()

    def _timer_callback(self):
        # 支持运行时通过 ros2 param set 修改路径后自动重新加载
        self._image_path = self.get_parameter('image_path').value
        if self._image_path and os.path.isfile(self._image_path) and self._image_path != self._last_loaded_path:
            self._load_and_resize(self._image_path)
        if self._current_image is None:
            self._current_image = self._blank_image()

        msg = self._bridge.cv2_to_imgmsg(self._current_image, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        self._publisher.publish(msg)

    def reload_image(self, path):
        """供外部调用：重新加载并发布新路径的图片。"""
        if path and os.path.isfile(path):
            self._image_path = path
            self._load_and_resize(path)
            return True
        return False


def main(args=None):
    rclpy.init(args=args)
    node = ImagePubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
