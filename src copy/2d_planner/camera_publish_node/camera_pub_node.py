#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import time

class DecxinCamPublisher(Node):
    def __init__(self):
        super().__init__('decxin_cam_node')
        
        # --- 参数声明 ---
        self.declare_parameter('device_path', '/dev/video0')
        self.declare_parameter('fps', 30)  # Orange Pi 5 建议跑 30fps
        self.declare_parameter('width', 1024)
        self.declare_parameter('height', 768)

        self.device_path = self.get_parameter('device_path').value
        self.target_fps = self.get_parameter('fps').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value

        # --- 初始化摄像头 ---
        # 使用 CAP_V4L2 提高在 Linux 上的启动速度和稳定性
        self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 关键：设置缓冲区为 1

        if not self.cap.isOpened():
            self.get_logger().error(f"❌ 无法打开摄像头: {self.device_path}")
            return

        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        
        # --- 线程安全变量 ---
        self.frame = None
        self.ret = False
        self.running = True
        
        # --- 启动硬件读取线程 ---
        # 这样 read() 不会阻塞 ROS 的定时器
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()

        # --- 设置发布定时器 ---
        timer_period = 1.0 / self.target_fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f"✅ 优化版节点启动! 模式: 多线程异步抓取")

    def _reader_thread(self):
        """专门负责清空缓冲区并获取最新帧的线程"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                self.ret = True
            else:
                self.get_logger().warn("读取线程丢失帧...")
            # 这里的延时极小，确保全力抢夺最新帧
            time.sleep(0.001)

    def timer_callback(self):
        # 只有当线程拿到了新帧才发布
        if self.ret and self.frame is not None:
            # 转换为 ROS 消息
            msg = self.bridge.cv2_to_imgmsg(self.frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_link'
            
            self.publisher_.publish(msg)
            # 发布后可以不重置 self.ret，保证高频输出
        else:
            self.get_logger().debug("等待图像输入...")

    def destroy_node(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DecxinCamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()