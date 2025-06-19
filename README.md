# Robotics Intern Task – Situational Awareness PoC

End-to-end demo that fuses a **128-beam LiDAR** and an **Intel RealSense RGB-D** camera to create a 2-D occupancy grid in real time.  
The grid marks **free ground**, **obstacles** (people, rocks, debris), and **unknown** cells, updating at **≥ 6 Hz** on a Jetson Orin NX.  
All fusion, segmentation, and mapping logic lives in **≃ 60 Python lines** (`sensors_fusion.py`) and runs on stock **ROS 2 Humble**.  
The original task description is included in the accompanying PDF assignment.

<div align="center">

![Occupancy-grid demo](rviz_grid_demo.png)

*RViz snapshot:*   
*• Gray arc — simulated LiDAR obstacle points projected into the occupancy grid*  
*• White background — free ground cells projected from RGB-D and DeepLabV3 mask*  
*• Faint vertical lines — expected artifacts from sparse segmentation after mask downsampling*



</div>

---

## Repository Layout

robotics_intern/
├── src/
│ └── sensors_fusion_pkg/
│ ├── sensors_fusion.py # Fusion + mapping node (main task answer)
│ ├── fake_sensors.py # Publishes mock LiDAR + RGB-D data
│ ├── init.py
│ └── package.xml
├── .gitignore
└── README.md 

---

## Quick start (ROS 2 Humble, Ubuntu 22.04)

```bash
# 1. cd into the workspace and build it
cd robotics_intern_task
colcon build

# 2. start the fake sensor (terminal 1)
source install/setup.bash && ros2 run sensors_fusion_pkg fake_sensors

# 3. run the fusion / mapping node (terminal 2)
source install/setup.bash && ros2 run sensors_fusion_pkg sensors_fusion

# 4. visualize in RViz (terminal 3)
source install/setup.bash && rviz2
```

### RViZ configuration

To view the occupancy grid:
* Add “OccupancyGrid” display
* Set topic to /world_grid
* Fixed frame: map
* Adjust camera view to see the full grid area

---
## Algorithm Outline

1. **LiDAR filtering**  
   Discard ground points (< 0.3 m).
   Mark points above as obstacles with value 100.

2. **Segmentation**  
   Run DeepLabV3-ResNet50 on the RGB image.  
   Keep only the pixels classified as "ground".

3. **RGB-D projection**  
   For every "ground" pixel, use the depth value to project it into 2-D grid space.  
   Mark the corresponding grid cells with value `0` (free).

4. **Unknown regions**  
   Any grid cell not updated by LiDAR or RGB-D is marked as `-1` (unknown).

5. **Grid publishing**  
   The final occupancy grid is published as a `nav_msgs/OccupancyGrid` message  
   on the `/world_grid` topic at approximately 6.7 Hz.


---
## Dependencies
```bash
python3 -m pip install torch torchvision opencv-python ros2-numpy
sudo apt install ros-humble-image-transport ros-humble-message-filters ros-humble-rviz2
```

---
## Possible extensions
* Replace fake_sensors.py with real hardware drivers
* Add launch files + YAML config for parameters
* Add unit tests and continuous integration