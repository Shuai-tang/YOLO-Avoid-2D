#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class VideoPublisherNode(Node):
    def __init__(self):
        super().__init__('video_publisher_node')

        # 参数：视频路径与话题名
        self.declare_parameter('video_path', '/home/tangsh/20260216tuhu_ws/video_and_photo/demo_video.mp4')
        self.declare_parameter('publish_topic', '/camera/image_raw')
        self.declare_parameter('frame_id', 'camera')

        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.publish_topic = self.get_parameter('publish_topic').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self.publisher = self.create_publisher(Image, self.publish_topic, 10)
        self.br = CvBridge()

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open video: {self.video_path}')
            raise RuntimeError('Video open failed')

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            period = 1.0 / fps
        else:
            period = 1.0 / 30.0
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f'Publishing video frames from {self.video_path} on {self.publish_topic} (period={period:.3f}s)'
        )

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            # 循环播放：回到开头
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warning('Failed to read frame after reset.')
                return

        msg = self.br.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.publisher.publish(msg)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
