#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import Image, LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from nav_msgs.msg import OccupancyGrid

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from std_msgs.msg import Float32MultiArray
from collections import deque
from dataclasses import dataclass
import time
import json
import os
import subprocess
from datetime import datetime

import math
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
import tf2_geometry_msgs as tfg


@dataclass
class RGBDetection:
    stamp: rclpy.time.Time
    faces: list  # List of (cx, cy, width, height) for detected faces
    image: np.ndarray = None  # Optional, for visualization


class FaceData:
    def __init__(self, face_id, position, is_new=True, normal=None):
        self.face_id = face_id
        self.position = position  # numpy array [x, y, z]
        self.is_new = is_new      # Flag to track if this is a newly detected face
        self.last_seen = time.time()
        # Add normal vector for orientation (default facing forward)
        if normal is None:
            self.normal = np.array([1.0, 0.0, 0.0])
        else:
            self.normal = normal / np.linalg.norm(normal)  # Normalize

    def update_position(self, new_position, new_normal=None, smoothing_factor=0.3):
        """Update position with smoothing"""
        self.position = (1 - smoothing_factor) * self.position + smoothing_factor * new_position
        if new_normal is not None:
            new_normal = new_normal / np.linalg.norm(new_normal)  # Ensure normalized
            self.normal = (1 - smoothing_factor) * self.normal + smoothing_factor * new_normal
            # Renormalize after smoothing
            self.normal = self.normal / np.linalg.norm(self.normal)
        self.last_seen = time.time()
        self.is_new = False

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'face_id': int(self.face_id),
            'position': self.position.tolist(),
            'normal': self.normal.tolist() if hasattr(self, 'normal') else [1.0, 0.0, 0.0],
            'last_seen': self.last_seen
        }

    @classmethod
    def from_dict(cls, data):
        """Create FaceData from dictionary"""
        face = cls(
            face_id=data['face_id'],
            position=np.array(data['position']),
            is_new=False,
            normal=np.array(data.get('normal', [1.0, 0.0, 0.0]))
        )
        face.last_seen = data.get('last_seen', time.time())
        return face


class DetectFaces(Node):
    def __init__(self):
        super().__init__('detect_faces')

        self.declare_parameters(namespace='', parameters=[
            ('device', ''),
            ('save_file', ''),
            ('save_interval', 5.0),  # Save every 5 seconds
            ('marker_lifetime', 0.0),  # 0 means forever
            ('greeting_text', 'Hello!')
        ])

        marker_topic = "/people_marker"
        array_topic = "/people_array"
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.save_file = self.get_parameter('save_file').get_parameter_value().string_value
        self.greeting_text = self.get_parameter('greeting_text').get_parameter_value().string_value

        if not self.save_file:
            # Default save file in home directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_file = os.path.expanduser(f"~/colcon_ws/src/dis_tutorial3/detected_faces_{timestamp}.json")

        self.save_interval = self.get_parameter('save_interval').get_parameter_value().double_value
        self.marker_lifetime = self.get_parameter('marker_lifetime').get_parameter_value().double_value

        self.detection_color = (0, 0, 255)  # BGR: Red color

        self.bridge = CvBridge()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Message synchronization
        self.rgb_buffer = deque(maxlen=30)  # Store recent RGB detections
        
        # Setup for LiDAR-based detection
        self.faces = []  # Store detected faces for LiDAR processing
        self.latest_lidar_data = None
        self.detected_positions = []
        self.goal_positions = []
        self.latest_cost_map = None
        self.people_clusters = {}
        self.cluster_id_counter = 0
        self.cluster_radius = 0.5
        self.min_detections_threshold = 1

        # TF Buffer for coordinate transformations
        self.tf_buffer = Buffer(Duration(seconds=2))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscriptions
        self.rgb_image_sub = self.create_subscription(
            Image, "/oak/rgb/image_raw", self.rgb_callback, qos_profile_sensor_data)
        
        self.lidar_sub = self.create_subscription(
            LaserScan, "/scan", self.lidar_callback, qos_profile_sensor_data)
            
        self.costmap_subscription = self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap', self.costmap_callback, 10)
                                                      
        # Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)
        
        self.positions_pub = self.create_publisher(
            PoseArray, '/detected_people_poses', QoSReliabilityPolicy.BEST_EFFORT)

        # Path to the TTS script
        self.tts_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speak.py")

        # Load any previously saved faces
        self.load_faces()

        # Parameters
        self.face_distance_threshold = 0.5  # meters for determining if a face is new
        self.next_face_id = 0  # Counter for face IDs
        
        # Store state
        self.persistent_faces = {}  # Dictionary of face_id -> FaceData
        self.last_save_time = time.time()

        # Camera intrinsics (adjust these based on your calibration)
        self.fx = 306.00787353515625
        self.fy = 306.00787353515625
        self.cx = 188.68125915527344
        self.cy = 105.0

        self.get_logger().info(f"Node initialized! Detecting faces and publishing markers to {marker_topic}.")
        self.get_logger().info(f"Saving detected faces to {self.save_file}")
        self.get_logger().info(f"Will say '{self.greeting_text}' when a new face is detected")

    def say_greeting(self):
        """Use the text-to-speech script to say greeting"""
        try:
            # Print to terminal
            print(f"{self.greeting_text}")

            # Run the TTS script
            subprocess.Popen(["python3", self.tts_script_path, self.greeting_text])
            self.get_logger().info(f"Speaking greeting: {self.greeting_text}")
        except Exception as e:
            self.get_logger().error(f"Error running TTS script: {e}")

    def load_faces(self):
        """Load previously saved faces from file"""
        if os.path.exists(self.save_file):
            try:
                with open(self.save_file, 'r') as f:
                    data = json.load(f)
                    for face_dict in data:
                        face = FaceData.from_dict(face_dict)
                        self.persistent_faces[face.face_id] = face
                        # Update next_face_id to be higher than any loaded ID
                        self.next_face_id = max(self.next_face_id, face.face_id + 1)

                self.get_logger().info(f"Loaded {len(self.persistent_faces)} faces from {self.save_file}")
            except Exception as e:
                self.get_logger().error(f"Error loading faces from {self.save_file}: {e}")

    def save_faces(self):
        """Save detected faces to file"""
        try:
            # Convert faces to list of dictionaries
            faces_data = [face.to_dict() for face in self.persistent_faces.values()]

            # Make sure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.save_file)), exist_ok=True)

            with open(self.save_file, 'w') as f:
                json.dump(faces_data, f, indent=2)

            self.last_save_time = time.time()
            self.get_logger().debug(f"Saved {len(faces_data)} faces to {self.save_file}")
        except Exception as e:
            self.get_logger().error(f"Error saving faces to {self.save_file}: {e}")

    def rgb_callback(self, data):
        """Process RGB image to detect faces using Haar Cascade"""
        try:
            stamp = data.header.stamp
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.get_logger().info("Received RGB image")

            # Convert to grayscale for Haar detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            self.get_logger().info(f"Detected {len(faces)} faces")

            # Process detections
            detected_faces = []
            detect_image = cv_image.copy()
            
            # Clear previous faces for LiDAR processing
            self.faces = []

            for (x, y, w, h) in faces:
                cx = x + w // 2
                cy = y + h // 2
                
                # Add to faces list for LiDAR processing
                right = int((cx + (x+w)) / 2)
                bottom = int((cy + y) / 2)
                self.faces.append((cx, cy, right, bottom))
                
                detected_faces.append((cx, cy, w, h))
                detect_image = cv2.rectangle(detect_image, (x, y), (x + w, y + h), self.detection_color, 2)
                detect_image = cv2.circle(detect_image, (cx, cy), 5, self.detection_color, -1)

            # Log detections
            if detected_faces:
                self.get_logger().info(f"Detected {len(detected_faces)} people/faces")

            # Store detection in buffer
            self.rgb_buffer.append(RGBDetection(stamp=stamp, faces=detected_faces, image=detect_image))

            # Display detections
            cv2.imshow("Detections", detect_image)
            cv2.waitKey(1)

            # If we have LiDAR data already, process faces immediately
            if self.latest_lidar_data:
                self.process_faces_with_lidar(self.latest_lidar_data)

        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")

    def lidar_callback(self, data):
        """Process LiDAR scan data"""
        self.latest_lidar_data = data
        self.get_logger().info(f"Received LiDAR data ranges")
        if self.faces:
            self.get_logger().info(f"Processing {len(self.faces)} faces with LiDAR data")
            self.process_faces_with_lidar(data)

    def process_faces_with_lidar(self, data):
        """Process detected faces with LiDAR data to get 3D positions"""
        if not self.faces:
            return
            
        for x, y, r, b in self.faces:
            try:
                # Calculate angle from image coordinates
                dx = 1.0
                dy = -(x - self.cx) / self.fx
                camera_angle = math.atan2(dy, dx)
                target_angle = -math.pi / 2 + camera_angle
                
                # Find corresponding LiDAR data
                index = int((target_angle - data.angle_min) / data.angle_increment)
                if 0 <= index < len(data.ranges):
                    angle = data.angle_min + index * data.angle_increment
                    depth = data.ranges[index]
                    x_centre = depth * math.cos(angle)
                    y_centre = depth * math.sin(angle)
                    
                    # Calculate right point coordinates
                    dy_right = -(r - self.cx) / self.fx
                    camera_angle_right = math.atan2(dy_right, dx)
                    target_angle_right = -math.pi / 2 + camera_angle_right
                    index_right = int((target_angle_right - data.angle_min) / data.angle_increment)
                    
                    if 0 <= index_right < len(data.ranges):
                        angle_right = data.angle_min + index_right * data.angle_increment
                        depth_right = data.ranges[index_right]
                        x_right = depth_right * math.cos(angle_right)
                        y_right = depth_right * math.sin(angle_right)
                        
                        # Transform points to map frame
                        point_in_lidar = PointStamped()
                        point_in_lidar.header.frame_id = 'rplidar_link'
                        point_in_lidar.header.stamp = self.get_clock().now().to_msg()
                        point_in_lidar.point.x = x_centre
                        point_in_lidar.point.y = y_centre
                        point_in_lidar.point.z = 0.0

                        try:
                            # Use latest available transform
                            transform_time = rclpy.time.Time()  # Get latest available transform
                            trans = self.tf_buffer.lookup_transform(
                                "map", 
                                "rplidar_link",
                                transform_time,
                                Duration(seconds=2.0)  # Increase timeout to be more forgiving
                            )
                            
                            # Set up right point
                            point_in_lidar_right = PointStamped()
                            point_in_lidar_right.header.frame_id = 'rplidar_link'
                            point_in_lidar_right.header.stamp = self.get_clock().now().to_msg()
                            point_in_lidar_right.point.x = x_right
                            point_in_lidar_right.point.y = y_right
                            point_in_lidar_right.point.z = 0.0

                            # Transform both points using the same transform
                            map_point = tfg.do_transform_point(point_in_lidar, trans)
                            map_point_right = tfg.do_transform_point(point_in_lidar_right, trans)

                            # Process the detected person
                            self.process_person_for_clustering(map_point, map_point_right, data)
                            self.get_logger().info(f"Transformed point: {map_point.point.x}, {map_point.point.y}")
                            
                        except TransformException as e:
                            self.get_logger().warn(f"Transform failed: {e}")
                    else:
                        self.get_logger().warn(f"Right point index {index_right} out of LiDAR range bounds")
                else:
                    self.get_logger().warn(f"Center point index {index} out of LiDAR range bounds")
                    
            except Exception as e:
                self.get_logger().warn(f"Error processing LiDAR data for face: {e}")
                
    def costmap_callback(self, msg):
        """Store the latest costmap"""
        self.latest_cost_map = msg
        
    def process_person_for_clustering(self, person_point, right_point, data):
        """
        Process a person detection for clustering and update persistent faces
        """
        try:
            # Get coordinates in map frame
            map_x = person_point.point.x
            map_y = person_point.point.y
            map_z = person_point.point.z
            
            position = np.array([map_x, map_y, map_z])
            
            # Calculate normal vector for orientation
            point_right = np.array([right_point.point.x, right_point.point.y, right_point.point.z])
            point_bottom = np.array([right_point.point.x, right_point.point.y, 0])
            
            vector_right = point_right - position
            vector_right = vector_right / np.linalg.norm(vector_right)
            vector_bottom = point_bottom - position
            vector_bottom = vector_bottom / np.linalg.norm(vector_bottom)
            normal = np.cross(vector_right, vector_bottom)
            normal = normal / np.linalg.norm(normal)
            
            # Check if this is a previously seen face
            matched_face_id = None
            for face_id, face_data in self.persistent_faces.items():
                distance = np.linalg.norm(position[:2] - face_data.position[:2])  # 2D distance (x,y plane)
                if distance < self.face_distance_threshold:
                    matched_face_id = face_id
                    # Update position and normal with smoothing
                    face_data.update_position(position, normal)
                    self.get_logger().info(f"Updated face ID: {face_id}, Position: {position[:2]}")
                    
                    # Create goal position
                    self.calculate_and_store_goal_position(face_data)
                    break
            
            # If no match was found, add as a new detection to clusters
            if matched_face_id is None:
                # Add to clusters for verification
                assigned_to_cluster = False
                
                for cluster_id, cluster_data in self.people_clusters.items():
                    # Calculate average position of the cluster
                    cluster_positions = cluster_data["positions"]
                    if not cluster_positions:
                        continue
                        
                    avg_x = sum(pos[0] for pos in cluster_positions) / len(cluster_positions)
                    avg_y = sum(pos[1] for pos in cluster_positions) / len(cluster_positions)
                    avg_z = sum(pos[2] for pos in cluster_positions) / len(cluster_positions)
                    
                    # Calculate distance to cluster center
                    distance = ((map_x - avg_x)**2 + (map_y - avg_y)**2)**0.5
                    self.get_logger().info(f"Distance to cluster {cluster_id}: {distance:.2f} m")
                    
                    # Check if within clustering radius
                    if distance <= self.cluster_radius:
                        # Add to existing cluster
                        cluster_data["positions"].append(position)
                        cluster_data["timestamps"].append(self.get_clock().now())
                        assigned_to_cluster = True
                        
                        # Check if we have enough detections in this cluster
                        if len(cluster_data["positions"]) >= self.min_detections_threshold:
                            # Calculate average position and normal of the cluster
                            avg_position = np.mean(np.array(cluster_data["positions"]), axis=0)
                            
                            # Create a new persistent face
                            face_id = self.next_face_id
                            self.next_face_id += 1
                            self.persistent_faces[face_id] = FaceData(
                                face_id=face_id,
                                position=avg_position,
                                is_new=True,
                                normal=normal
                            )
                            self.get_logger().info(f"New face detected! ID: {face_id}, Position: {avg_position[:2]}")
                            
                            # Calculate goal position
                            self.calculate_and_store_goal_position(self.persistent_faces[face_id])
                            
                            # Say greeting for the first face only
                            # if len(self.persistent_faces) == 1:
                            #     self.say_greeting()
                        
                        break
                
                # If not assigned to any existing cluster, create a new one
                if not assigned_to_cluster:
                    self.people_clusters[self.cluster_id_counter] = {
                        "positions": [position],
                        "timestamps": [self.get_clock().now()]
                    }
                    self.cluster_id_counter += 1
                    
        except Exception as e:
            self.get_logger().error(f"Error processing person detection: {str(e)}")

    def calculate_and_store_goal_position(self, face_data):
        """Calculate and store the goal position for approaching a face"""
        position = face_data.position
        normal = face_data.normal
        
        # Calculate goal point at a safe distance (0.8m) in front of the person
        scale = 0.8  # 80cm away from the person
        goal_point = position - normal * scale
        
        # Get valid goal position respecting costmap
        valid_goal_point = self.get_valid_goal_position(goal_point[0], goal_point[1], goal_point[2])
        if valid_goal_point is None:
            return

        # Calculate direction to face the person
        direction = position - np.array(valid_goal_point)
        direction = direction / np.linalg.norm(direction)
        yaw = math.atan2(direction[1], direction[0])
        
        # Store the goal pose
        # Check if we already have a goal for this face
        existing_goal = next((i for i, g in enumerate(self.goal_positions) 
                            if g[4] == face_data.face_id), None)
        
        if existing_goal is not None:
            # Update existing goal
            self.goal_positions[existing_goal] = (
                valid_goal_point[0], valid_goal_point[1], valid_goal_point[2], 
                yaw, face_data.face_id
            )
        else:
            # Add new goal
            self.goal_positions.append((
                valid_goal_point[0], valid_goal_point[1], valid_goal_point[2], 
                yaw, face_data.face_id
            ))
        
        # Publish markers
        self.publish_person_markers(position, valid_goal_point, face_data.face_id)
                
    def get_valid_goal_position(self, x, y, z):
        """Find a valid goal position near the person that respects costmap constraints"""
        if self.latest_cost_map is None:
            self.get_logger().warn("No costmap data available yet")
            return (x, y, z)
            
        costmap_data = self.latest_cost_map.data
        costmap_info = self.latest_cost_map.info
        
        # Convert world coordinates to costmap grid coordinates
        grid_x = int((x - costmap_info.origin.position.x) / costmap_info.resolution)
        grid_y = int((y - costmap_info.origin.position.y) / costmap_info.resolution)
        
        # Check if the position is within costmap bounds
        if (0 <= grid_x < costmap_info.width and 0 <= grid_y < costmap_info.height):
            # Get the cost at this position (0-100, where 100 means occupied)
            idx = grid_y * costmap_info.width + grid_x
            cost = costmap_data[idx] if idx < len(costmap_data) else 100
            
            # If cost is too high (position not safe), find closest safe position
            if cost > 60:  # Threshold for considering a cell navigable
                # Search for nearest valid position using spiral search
                valid_pos = self.find_nearest_valid_position(
                    grid_x, grid_y, costmap_data, costmap_info)
                
                if valid_pos:
                    # Convert grid coordinates back to world coordinates
                    safe_x = valid_pos[0] * costmap_info.resolution + costmap_info.origin.position.x
                    safe_y = valid_pos[1] * costmap_info.resolution + costmap_info.origin.position.y
                    self.get_logger().info(f"Position in obstacle zone. Using safe position: {safe_x:.2f}, {safe_y:.2f}")
                    return (safe_x, safe_y, z)
            
            # Position is safe to use directly
            return (x, y, z)
        else:
            self.get_logger().warn(f"Position outside costmap bounds: {grid_x}, {grid_y}")
            return None
        
    def find_nearest_valid_position(self, center_x, center_y, costmap_data, costmap_info):
        """
        Use spiral search to find the nearest valid position in the costmap
        """
        # Define a spiral pattern search
        max_dist = 20  # Maximum search distance
        dx = [0, 1, 0, -1]  # Direction vectors for East, South, West, North
        dy = [1, 0, -1, 0]
        
        x, y = center_x, center_y
        dist = 0
        segment_length = 1
        segment_passed = 0
        direction = 0
        
        # Start spiral search
        while dist <= max_dist:
            # Check current position
            if (0 <= x < costmap_info.width and 0 <= y < costmap_info.height):
                idx = y * costmap_info.width + x
                if idx < len(costmap_data) and costmap_data[idx] < 60:
                    # Found valid position
                    return (x, y)
            
            # Move to next position in spiral
            x += dx[direction]
            y += dy[direction]
            dist = max(abs(x - center_x), abs(y - center_y))
            segment_passed += 1
            
            if segment_passed == segment_length:
                # Change direction
                direction = (direction + 1) % 4
                segment_passed = 0
                
                # Increase segment length after completing half a circle
                if direction == 0 or direction == 2:
                    segment_length += 1
        
        # If no valid position found, return None
        return None
        
    def publish_person_markers(self, person_position, goal_position, face_id):
        """Publish markers for person position and robot goal position"""
        
        marker_array = MarkerArray()

        # Person position marker (sphere)
        person_marker = Marker()
        person_marker.header.frame_id = "map"
        person_marker.header.stamp = self.get_clock().now().to_msg()
        person_marker.ns = "person_positions"
        person_marker.id = face_id
        person_marker.type = Marker.SPHERE
        person_marker.action = Marker.ADD
        person_marker.pose.position.x = person_position[0]
        person_marker.pose.position.y = person_position[1]
        person_marker.pose.position.z = person_position[2]
        person_marker.pose.orientation.w = 1.0
        person_marker.scale.x = person_marker.scale.y = person_marker.scale.z = 0.2
        
        # Different colors for new vs. known faces
        if self.persistent_faces[face_id].is_new:
            person_marker.color.r = 0.0
            person_marker.color.g = 1.0
            person_marker.color.b = 0.0  # Green for new faces
        else:
            person_marker.color.r = 0.0
            person_marker.color.g = 0.0
            person_marker.color.b = 1.0  # Blue for known faces
            
        person_marker.color.a = 0.8

        # Goal position marker (sphere)
        goal_marker = Marker()
        goal_marker.header.frame_id = "map"
        goal_marker.header.stamp = self.get_clock().now().to_msg()
        goal_marker.ns = "goal_positions"
        goal_marker.id = face_id
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = goal_position[0]
        goal_marker.pose.position.y = goal_position[1]
        goal_marker.pose.position.z = goal_position[2]
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = goal_marker.scale.y = goal_marker.scale.z = 0.15
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 1.0  # Purple for goal
        goal_marker.color.a = 0.8

        # Direction marker (arrow) showing face orientation
        face_data = self.persistent_faces[face_id]
        if hasattr(face_data, 'normal'):
            direction_marker = Marker()
            direction_marker.header.frame_id = "map"
            direction_marker.header.stamp = self.get_clock().now().to_msg()
            direction_marker.ns = "face_directions"
            direction_marker.id = face_id
            direction_marker.type = Marker.ARROW
            direction_marker.action = Marker.ADD
            direction_marker.pose.position.x = person_position[0]
            direction_marker.pose.position.y = person_position[1]
            direction_marker.pose.position.z = person_position[2]
            
            # Create quaternion from normal vector to reference vector [1,0,0]
            normal = face_data.normal
            reference = np.array([1.0, 0.0, 0.0])
            
            if np.allclose(normal, reference) or np.allclose(normal, -reference):
                rotation_axis = np.array([0.0, 0.0, 1.0])
                angle = 0.0 if np.allclose(normal, reference) else np.pi
            else:
                rotation_axis = np.cross(reference, normal)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.dot(reference, normal))
            
            # Convert to quaternion
            qx = rotation_axis[0] * np.sin(angle/2)
            qy = rotation_axis[1] * np.sin(angle/2)
            qz = rotation_axis[2] * np.sin(angle/2)
            qw = np.cos(angle/2)
            
            direction_marker.pose.orientation.x = -qx
            direction_marker.pose.orientation.y = -qy
            direction_marker.pose.orientation.z = -qz
            direction_marker.pose.orientation.w = qw
            
            direction_marker.scale.x = 0.3  # Length
            direction_marker.scale.y = 0.05  # Width
            direction_marker.scale.z = 0.05  # Height
            
            direction_marker.color.r = 0.0
            direction_marker.color.g = 0.0
            direction_marker.color.b = 1.0
            direction_marker.color.a = 0.8
            
            marker_array.markers.append(direction_marker)

        # Text marker for face ID
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = self.get_clock().now().to_msg()
        text_marker.ns = "face_ids"
        text_marker.id = face_id
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = person_position[0]
        text_marker.pose.position.y = person_position[1]
        text_marker.pose.position.z = person_position[2] + 0.2  # Above the sphere
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.1  # Text size
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 0.8
        text_marker.text = f"ID: {face_id}"

        marker_array.markers.append(person_marker)
        marker_array.markers.append(goal_marker)
        marker_array.markers.append(text_marker)

        # Publish the marker array
        self.marker_pub.publish(marker_array)
        
        # Publish to PoseArray for consistent tracking
        self.publish_people_poses()
    
    def publish_people_poses(self):
        """Publish positions of detected people as a PoseArray"""
        if not self.persistent_faces:
            return
            
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.header.stamp = self.get_clock().now().to_msg()
        
        for face_id, face_data in self.persistent_faces.items():
            # Only include faces seen recently (within last 60 seconds)
            if time.time() - face_data.last_seen > 60:
                continue
                
            # Create a pose for the actual person position
            person_pose = Pose()
            person_pose.position.x = face_data.position[0]
            person_pose.position.y = face_data.position[1]
            person_pose.position.z = face_data.position[2]
            
            # Calculate quaternion from normal vector
            normal = face_data.normal
            reference = np.array([1.0, 0.0, 0.0])
            
            if np.allclose(normal, reference) or np.allclose(normal, -reference):
                rotation_axis = np.array([0.0, 0.0, 1.0])
                angle = 0.0 if np.allclose(normal, reference) else np.pi
            else:
                rotation_axis = np.cross(reference, normal)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.dot(reference, normal))
            
            qx = rotation_axis[0] * np.sin(angle/2)
            qy = rotation_axis[1] * np.sin(angle/2)
            qz = rotation_axis[2] * np.sin(angle/2)
            qw = np.cos(angle/2)
            
            # Add orientation to the pose
            person_pose.orientation.x = qx
            person_pose.orientation.y = qy
            person_pose.orientation.z = qz
            person_pose.orientation.w = qw
            
            # Send only the person's position
            pose_array.poses.append(person_pose)
        
        # Publish the array
        self.positions_pub.publish(pose_array)
        self.get_logger().debug(f"Published {len(pose_array.poses)} people positions")

    def publish_all_markers(self):
        """Publish markers for all stored faces"""
        if not self.persistent_faces:
            return

        marker_array = MarkerArray()

        for face_id, face_data in self.persistent_faces.items():
            # Face position marker (sphere)
            face_marker = Marker()
            face_marker.header.frame_id = "map"
            face_marker.header.stamp = self.get_clock().now().to_msg()
            face_marker.ns = "face_positions"
            face_marker.id = face_id
            face_marker.type = Marker.SPHERE
            face_marker.action = Marker.ADD
            face_marker.pose.position.x = face_data.position[0]
            face_marker.pose.position.y = face_data.position[1]
            face_marker.pose.position.z = 0.0
            face_marker.pose.orientation.w = 1.0
            face_marker.scale.x = face_marker.scale.y = face_marker.scale.z = 0.15

            face_marker.color.r = 1.0
            face_marker.color.g = 0.7
            face_marker.color.b = 0.0
            face_marker.color.a = 0.8

            marker_array.markers.append(face_marker)

        # Publish the marker array
        self.marker_pub.publish(marker_array)
        self.get_logger().debug(f"Published markers for {len(self.persistent_faces)} faces")


def main():
    print('Face detection node starting.')
    rclpy.init(args=None)
    node = DetectFaces()
    
    # Timer for publishing markers periodically
    timer = node.create_timer(1.0, node.publish_all_markers)
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()