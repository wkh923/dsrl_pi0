# Quest3 Setup

## Connect Ubuntu PC to the Quest3 via USB

Connect the Quest3 to the Ubuntu PC via USB  
Inside Quest3 VR, enable USB debugging via a notification

## Set up reverse port forwarding

    adb reverse tcp:10000 tcp:10000



# Ros2 TCP Endpoint setup

cd /home/jpy/RM/Airbot-VLA-RL/Airbot/vr/unity_tcp_ws/ROS-TCP-Endpoint
source /home/jpy/RM/Airbot-VLA-RL/Airbot/vr/unity_tcp_ws/install/setup.bash
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0 -p ROS_TCP_PORT:=10000



# Open APK through Quest3 Interface

Library --> Unknown Sources --> teleoperation   
On start up, the APK will automatically attempt to connect onto the ROS2 TCP Endpoint established. If successful, the VR Menu would turn green. 

![alt text](Menu.png)  

The VRConsole Debug system is also active on start up. To disable it, press the middle robot icon on the Menu.  



# Testing Quest 3 (no robot)

conda activate airbot_data
cd /home/jpy/RM/Airbot-VLA-RL/Airbot/vr/scripts
/usr/bin/python3 test_vr_dual_robot.py --no-robot

# Testing with Airbot Arms

cd /home/jpy/RM/Airbot-VLA-RL/Airbot/vr/scripts
conda activate airbot_data
python3 test_vr_dual_robot.py \
    --left-port 50051 \
    --right-port 50053 \
    --pos-scale 5 \
    --rot-scale 5 \
    --smooth-alpha 0.4

    

_Available parameters:_  
Parameter           Description                     Default  
--left-port         Left arm gRPC port	            50051  
--right-port	    Right arm gRPC port	            50053  
--no-robot	        Test VR only (no robot)	        Off  
--pos-scale	        Position sensitivity	        5.0  
--rot-scale	        Rotation sensitivity	        5.0  
--smooth-alpha      Controls Responsiveness         0.4  
--settle-duration   Time robot stays after switch   0.5



# Debug

## Test if the device has been connected and authorised

adb devices 2>&1

Output:
List of devices attached
2G0YC1ZG3Q0056	device

## Test if the Ros2 TCP is listening

ss -tlnp | grep 10000

Output:
LISTEN 0      10                         0.0.0.0:10000      0.0.0.0:*    users:(("default_server_",pid=1620551,fd=20))

## Test Ros2 Endpoint Subscribing from Quest3 Publishing

ros2 topic list

ros2 topic echo /tf  
ros2 topic echo /quest/joystick

