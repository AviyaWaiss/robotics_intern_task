#!/usr/bin/env python3
import rclpy
import numpy as np
import cv2
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header

WIDTH, HEIGHT = 640, 480
LIDAR_PTS = 2048

# ---------- Helper Functions ----------
def make_rgb8(w, h):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.rectangle(img, (0, 0), (w, h//2), (30, 30, 30), -1)
    cv2.circle(img, (w//2, h//2), 60, (0, 0, 255), -1)
    return img

def make_depth(w, h):
    depth = np.full((h, w), 1.0, np.float32)
    depth[:, :w//3] = 2.0
    return depth  # in meters (32FC1)

def make_point_cloud(n):
    a = np.linspace(-np.pi/4, np.pi/4, n)
    xs = np.cos(a) * 2.0
    ys = np.sin(a) * 2.0
    # zs = np.random.uniform(-0.2, 0.2, n)
    zs = np.random.uniform(0.4, 0.6, n)
    return np.vstack((xs, ys, zs)).T.astype(np.float32)

# ---------- Node Class ----------
class FakeSensors(Node):
    def __init__(self):
        super().__init__('fake_sensors')
        self.pub_rgb   = self.create_publisher(Image,       '/camera/color/image_raw', 10)
        self.pub_depth = self.create_publisher(Image,       '/camera/depth/image_rect_raw', 10)
        self.pub_cloud = self.create_publisher(PointCloud2, '/lidar_points', 10)
        self.create_timer(0.1, self.publish_frame)  # 10 Hz

    def publish_frame(self):
        now = self.get_clock().now().to_msg()
        hdr_cam = Header(stamp=now, frame_id='camera')
        hdr_lid = Header(stamp=now, frame_id='lidar')

        # ---------- RGB Image ----------
        rgb = make_rgb8(WIDTH, HEIGHT)
        msg_rgb = Image()
        msg_rgb.header = hdr_cam
        msg_rgb.height = HEIGHT
        msg_rgb.width = WIDTH
        msg_rgb.encoding = 'rgb8'
        msg_rgb.step = WIDTH * 3
        msg_rgb.data = rgb.tobytes()
        self.pub_rgb.publish(msg_rgb)

        # ---------- Depth Image ----------
        depth = make_depth(WIDTH, HEIGHT)
        msg_d = Image()
        msg_d.header = hdr_cam
        msg_d.height = HEIGHT
        msg_d.width = WIDTH
        msg_d.encoding = '32FC1'
        msg_d.step = WIDTH * 4
        msg_d.data = depth.tobytes()
        self.pub_depth.publish(msg_d)

        # ---------- Point Cloud ----------
        pts = make_point_cloud(LIDAR_PTS)
        cloud = PointCloud2()
        cloud.header = hdr_lid
        cloud.height = 1
        cloud.width = LIDAR_PTS
        cloud.is_dense = True
        cloud.is_bigendian = False
        cloud.point_step = 12  # 3 × float32
        cloud.row_step = 12 * LIDAR_PTS
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.data = pts.tobytes()
        self.pub_cloud.publish(cloud)

# ---------- Main ----------
def main():
    rclpy.init()
    rclpy.spin(FakeSensors())

if __name__ == '__main__':
    main()
