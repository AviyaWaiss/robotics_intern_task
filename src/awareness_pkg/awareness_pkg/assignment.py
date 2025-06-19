import rclpy, numpy as np, torch, ros2_numpy as rnp, cv2
import torchvision.transforms.functional as TF
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import OccupancyGrid
RES, SZ, FX, CX = 0.05, 200,  615.0, 320.0  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # use GPU if available

class FuseMap(Node):
    def __init__(self):
        super().__init__("fuse_map")
        self.create_subscription(PointCloud2, "/lidar_points", self.lidar_cb, 10)
        self.create_subscription(Image, "/camera/color/image_raw", self.rgb_cb, 10)
        self.create_subscription(Image, "/camera/depth/image_rect_raw", self.depth_cb, 10)
        self.pub = self.create_publisher(OccupancyGrid, "/world_grid", 1)
        self.grid = np.full((SZ, SZ), -1, np.int8)    # numpy buffer (not a ROS msg!)
        self.pts = self.mask = self.depth = None
        self.net = torch.hub.load("pytorch/vision", "deeplabv3_resnet50", weights="DeepLabV3_ResNet50_Weights.DEFAULT").to(DEVICE).eval()
        self.create_timer(0.15, self.tick) # ~6.7 Hz

    def lidar_cb(self, msg):
        self.pts = rnp.numpify(msg)["xyz"] # extract LiDAR XYZ points

    def rgb_cb(self, msg):
        rgb = cv2.cvtColor(rnp.numpify(msg), cv2.COLOR_BGR2RGB) # convert BGR to RGB
        logits = self.net(TF.to_tensor(rgb).unsqueeze(0).to(DEVICE))["out"]
        self.mask = logits.argmax(1)[0].cpu().numpy() == 0

    def depth_cb(self, msg):
        self.depth = rnp.numpify(msg).astype(np.float32)  # convert depth to float32 in meters

    def tick(self):
        if self.pts is None or self.mask is None or self.depth is None: return                                # wait until all three arrive
        self.grid.fill(-1)  # Reset grid
        ground = self.pts[(self.pts[:, 2] > -1) & (self.pts[:, 2] < 0.5)] # filter near-ground points
        cols = (ground[:, 0] / RES + SZ // 2).astype(int) # X to column
        rows = (ground[:, 1] / RES + SZ // 2).astype(int) # Y to row
        inside = (rows >= 0) & (rows < SZ) & (cols >= 0) & (cols < SZ)
        self.grid[rows[inside], cols[inside]] = 100 # mark occupied cells
        mask_r, mask_c = np.where(self.mask[::8, ::8]) # downsample mask
        depth_sample = self.depth[mask_r * 8, mask_c * 8] # sample corresponding depths
        fwd, side   = depth_sample, (mask_c * 8 - CX) * depth_sample / FX
        free_c, free_r = (fwd  / RES + SZ // 2).astype(int), (side / RES + SZ // 2).astype(int)
        good = (free_r >= 0) & (free_r < SZ) & (free_c >= 0) & (free_c < SZ)
        self.grid[free_r[good], free_c[good]] = 0 # mark free cells
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"                  
        msg.info.resolution = RES
        msg.info.width = msg.info.height = SZ
        msg.data = self.grid.flatten().tolist()
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = FuseMap()
    rclpy.spin(node)

if __name__ == "__main__":
    main()