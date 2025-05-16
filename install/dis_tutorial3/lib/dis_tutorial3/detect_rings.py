#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs_py import point_cloud2 as pc2
import subprocess
import os
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
import tf2_geometry_msgs as tfg
from geometry_msgs.msg import PointStamped, Vector3Stamped
import rclpy.duration
import time
from collections import deque
from dataclasses import dataclass
from sensor_msgs.msg import CompressedImage

from rclpy.qos import qos_profile_sensor_data

@dataclass
class RingData:
    position: np.ndarray  # 3D position in map frame [x, y, z]
    radius: float         # Radius in meters
    color_name: str       # Color name (e.g., "red", "blue", "green", "yellow", "black")
    color_bgr: tuple      # BGR color tuple for visualization
    last_seen: float      # Timestamp when last detected
    announced: bool       # Whether this ring has been announced via speech

class RingDetector(Node):
    def __init__(self):
        super().__init__('ring_detector')
        self.bridge = CvBridge()
        
        # Subscriptions
        self.image_sub = self.create_subscription(Image, "/oak/rgb/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(CompressedImage, "/oak/stereo/image_raw/compressedDepth", self.depth_callback, qos_profile_sensor_data)
        # self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, 1)
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, "/ring_markers", 10)
        
        # TF2 buffer and listener for transformation
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Data storage
        self.depth_data = None
        self.depth_width = 0
        self.depth_height = 0
        self.pointcloud_data = None
        self.rings = {}  # Dictionary of detected rings by position hash
        
        # Parameters
        self.marker_lifetime = 0.0  # Marker lifetime in seconds
        self.ring_position_threshold = 0.3  # meters, threshold for considering a ring as the same
        # self.ring_timeout = 30.0  # seconds before removing a ring from tracking
        self.announce_cooldown = 2.0  # minimum seconds between announcing the same ring
        
        # Create timers
        self.marker_timer = self.create_timer(0.5, self.publish_ring_markers)
        # self.cleanup_timer = self.create_timer(5.0, self.cleanup_old_rings)
        
        # Path to the TTS script (assuming it's in the same directory as this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.tts_script_path = os.path.join(script_dir, "speak.py")
        if not os.path.exists(self.tts_script_path):
            self.get_logger().warn(f"TTS script not found at {self.tts_script_path}. Using default path.")
            self.tts_script_path = os.path.expanduser("~/colcon_ws/src/dis_tutorial3/speak.py")
        
        # Create windows
        cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected Rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
        # Add debug windows
        cv2.namedWindow("Ring Debug", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth Points", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Color Debug", cv2.WINDOW_NORMAL)
        
        self.get_logger().info("Ring detector node initialized. Publishing markers to /ring_markers")

    def depth_callback(self, data):
        try:
            # self.get_logger().info(f"Depth image received, {len(data.data)} bytes")
            data1 = data.data.tobytes()
            sig = b'\x89PNG\r\n\x1a\n'
            idx = data1.find(sig)
            if idx == -1:
                self.get_logger().warn("PNG signature not found in depth image data")
                return

            png = data1[idx:]
            arr = np.frombuffer(png, dtype=np.uint8)
            depth_image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # Decode the PNG image and convert to meters

            # Log basic information
            # self.get_logger().info(f"Depth image received, len: {len(data.data)} bytes")
            # self.get_logger().info(f"Depth image format: {data.format}")
            
            # This is a PNG compressed depth image
            # Convert to numpy array for processing
            # np_arr = np.frombuffer(data.data, np.uint8)
            # self.get_logger().info(f"Depth image array shape: {np_arr.shape}")
            # depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # Decode the PNG image and convert to meters
            # # Decode the PNG image
            cv2.imshow("Depth window", depth_image)
            self.depth_data = depth_image  # Store for ring detection
            self.depth_height, self.depth_width = depth_image.shape

            # Create a copy for visualization
            depth_display = depth_image.copy()
            
            # Replace invalid values (inf and nan)
            depth_display = np.nan_to_num(depth_display, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clip the depth values to a reasonable range (e.g., 0 to 3 meters)
            max_depth = 3.0  # meters
            depth_display = np.clip(depth_display, 0, max_depth)
            
            # Normalize to 0-255 range for visualization
            depth_normalized = cv2.normalize(depth_display, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Apply colormap for better visualization (TURBO gives better depth perception)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
            
            # Display the depth map
            cv2.imshow("Depth window", depth_colormap)
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
      

    def announce_ring_color(self, color_name):
        """Announce the detected ring color using TTS"""
        try:
            message = f"{color_name} ring detected"
            # Print to terminal
            self.get_logger().info(message)
            
            # Run the TTS script
            subprocess.Popen(["python3", self.tts_script_path, message])
        except Exception as e:
            self.get_logger().error(f"Error running TTS script: {e}")

    def position_hash(self, position):
        """Create a simple hash from a position to use as a ring identifier"""
        return f"{position[0]:.2f}_{position[1]:.2f}_{position[2]:.2f}"

    def transform_point_to_map(self, point_3d):
        """Transform a point from camera frame to map frame"""
        try:
            # Create PointStamped object
            point_stamped = PointStamped()
            point_stamped.header.frame_id = "oakd_rgb_camera_frame"
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point.x = float(point_3d[0])
            point_stamped.point.y = float(point_3d[1])
            point_stamped.point.z = float(point_3d[2])
            
            # Get latest transform
            transform = self.tf_buffer.lookup_transform(
                "map", 
                "oakd_rgb_camera_optical_frame",
                rclpy.time.Time(),  # Get latest transform
                rclpy.duration.Duration(seconds=1.0)
            )
            
            # Transform the point
            transformed_point = tfg.do_transform_point(point_stamped, transform)
            
            return np.array([
                transformed_point.point.x, 
                transformed_point.point.y, 
                transformed_point.point.z
            ])
            
        except TransformException as e:
            self.get_logger().warn(f"Could not transform point: {e}")
            return None

    def get_point_cloud_position(self, x, y, r):
        """Get 3D position of ring center from point cloud data"""
        if self.pointcloud_data is None:
            return None
            
        try:
            # Convert point cloud to numpy array
            pc_array = pc2.read_points_numpy(
                self.pointcloud_data, 
                field_names=("x", "y", "z")
            ).reshape((self.pointcloud_data.height, self.pointcloud_data.width, 3))
            
            # Sample points around the ring
            ring_points = []
            num_samples = 8
            for angle in np.linspace(0, 2*np.pi, num_samples, endpoint=False):
                px = int(x + r * 0.8 * np.cos(angle))  # Sample at 80% of radius to get on the ring
                py = int(y + r * 0.8 * np.sin(angle))
                
                # Check if point is within image bounds
                if 0 <= px < self.pointcloud_data.width and 0 <= py < self.pointcloud_data.height:
                    point = pc_array[py, px]
                    if np.isfinite(point).all() and not np.isnan(point).any():
                        ring_points.append(point)
            
            # If we have enough points, compute the median position
            if len(ring_points) >= 3:
                ring_position = np.median(np.array(ring_points), axis=0)
                return ring_position
                
            return None
                
        except Exception as e:
            self.get_logger().error(f"Error extracting point cloud data: {e}")
            return None

    def is_hollow_ring(self, x, y, r, depth_map):
        """Check if ring is hollow by comparing center depth with perimeter depth."""
        try:
            # Create debug visualization
            debug_img = cv2.cvtColor(depth_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            points_img = np.zeros_like(debug_img)
            
            # Get center depth and mark it
            center_depth = depth_map[y, x]
            cv2.circle(points_img, (x, y), 2, (0, 0, 255), -1)  # Red dot for center
            
            # Get perimeter depths by sampling points around the circle
            num_points = 8
            perimeter_depths = []
            for angle in np.linspace(0, 2*np.pi, num_points, endpoint=False):
                px = int(x + r * np.cos(angle))
                py = int(y + r * np.sin(angle))
                if 0 <= px < self.depth_width and 0 <= py < self.depth_height:
                    depth = depth_map[py, px]
                    if depth > 0 and not np.isnan(depth):
                        perimeter_depths.append(depth)
                        # Mark perimeter points in green
                        cv2.circle(points_img, (px, py), 2, (0, 255, 0), -1)
            
            if len(perimeter_depths) < 4:
                return False
                    
            perimeter_depth = np.mean(perimeter_depths)
            depth_difference = center_depth - perimeter_depth
            
            # Draw the full circle and depth values
            cv2.circle(debug_img, (x, y), r, (255, 255, 255), 1)
            cv2.putText(debug_img, 
                        f"Center: {center_depth:.2f}m", 
                        (x+10, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 255), 
                        1)
            cv2.putText(debug_img, 
                        f"Perim: {perimeter_depth:.2f}m", 
                        (x+10, y+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        1)
            cv2.putText(debug_img, 
                        f"Diff: {depth_difference:.2f}m", 
                        (x+10, y+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        1)
            
            # Show debug windows
            cv2.imshow("Ring Debug", debug_img)
            cv2.imshow("Depth Points", points_img)
            cv2.waitKey(1)
            
            min_depth_diff = 0.1
            return depth_difference > min_depth_diff
                
        except Exception as e:
            self.get_logger().error(f"Error checking ring depth: {e}")
            return False

    def get_ring_color(self, frame, x, y, r):
        # Create multiple masks with different thicknesses to sample more effectively
        inner_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        outer_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Use two masks to better capture the ring's color
        cv2.circle(inner_mask, (x, y), int(r-1), 255, 3)  # Inner part of ring
        cv2.circle(outer_mask, (x, y), int(r+1), 255, 3)  # Outer part of ring
        combined_mask = cv2.bitwise_or(inner_mask, outer_mask)
        
        # Create debug visualization
        debug_color = frame.copy()
        cv2.circle(debug_color, (x, y), int(r), (0, 255, 255), 2)
        
        # Sample colors using the mask
        mean_bgr = cv2.mean(frame, mask=combined_mask)[:3]
        b, g, r = mean_bgr
        
        # Calculate RGB differences right away so they're available for all detection methods
        rg_diff = abs(r - g)
        rb_diff = abs(r - b)
        gb_diff = abs(g - b)
        
        # Convert to multiple color spaces for better detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Get color values in different spaces
        mean_hsv = cv2.mean(hsv_frame, mask=combined_mask)[:3]
        mean_lab = cv2.mean(lab_frame, mask=combined_mask)[:3]
        
        h, s, v = mean_hsv
        l, a, b_val = mean_lab
        
        # Show debug info with multiple color spaces
        cv2.putText(debug_color, f"HSV: {h:.0f},{s:.0f},{v:.0f}", 
                    (int(x + r + 10), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(debug_color, f"BGR: {b:.0f},{g:.0f},{r:.0f}", 
                    (int(x + r + 10), int(y+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(debug_color, f"LAB: {l:.0f},{a:.0f},{b_val:.0f}", 
                    (int(x + r + 10), int(y+40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Special debug for potentially black rings
        if max(r, g, b) < 100:
            cv2.putText(debug_color, "Potentially black ring", 
                        (int(x + r + 10), int(y+60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        
        # Show the mask and debug info
        cv2.imshow("Color Debug", debug_color)
        
        # Decision making using multiple color spaces
        # More aggressive black detection with multiple approaches
        # Approach 1: Low value in HSV
        if v < 50:  # Increased threshold from adaptive_v_threshold * 1.5
            return "black", (0, 0, 0)

        # Approach 2: Low saturation and value together
        if s < 40 and v < 80:
            return "black", (0, 0, 0)

        # Approach 3: All RGB channels are low and similar
        if max(r, g, b) < 80 and max(rg_diff, rb_diff, gb_diff) < 20:
            return "black", (0, 0, 0)
        
        # BLUE detection (particularly improved)
        # Blue in HSV has high H, and in Lab has negative b component
        if (85 <= h <= 140) and b_val < 128:  # Blue in HSV and Lab
            # Further check in BGR
            if b > max(r, g) + 5:  # B channel is stronger 
                return "blue", (255, 0, 0)
        
        # GREEN detection (use both HSV and Lab)
        if (35 <= h <= 85) and a < 128:  # Green in HSV and Lab
            if g > max(r, b) + 5:  # G channel is stronger
                return "green", (0, 255, 0)
                
        # RED detection (improved)
        if ((0 <= h <= 15) or (165 <= h <= 180)) and a > 128:
            if r > max(g, b) + 5:  # R channel is stronger
                return "red", (0, 0, 255)
                
        # YELLOW detection (improved)
        # if (15 <= h <= 35) and b_val > 128 and a > 128:
        #     if r > b + 10 and g > b + 10:  # Both R and G channels are stronger than B
        #         return "yellow", (0, 255, 255)
        
        # Fallback to RGB ratio analysis with more robust thresholds
        max_channel = max(r, g, b)
        if max_channel > 30:
            r_ratio = r / max_channel
            g_ratio = g / max_channel
            b_ratio = b / max_channel
            
            # Added feature: channel differences
            rg_diff = abs(r - g)
            rb_diff = abs(r - b)
            gb_diff = abs(g - b)
            
            # Define color by strongest channel with significant difference
            if b_ratio > 0.4 and b > r + 10 and b > g + 10:
                return "blue", (255, 0, 0)
            elif g_ratio > 0.4 and g > r + 10 and g > b + 10:
                return "green", (0, 255, 0)
            elif r_ratio > 0.4 and r > b + 10:
                if r_ratio > 0.5 and g_ratio > 0.5 and g > b + 10:
                    return "", (128, 128, 128)
                return "red", (0, 0, 255)
        
        # Final check for dark colors before returning unknown
        if max(r, g, b) < 60:  # Very low RGB values
            return "black", (0, 0, 0)

        # Final aggressive check for dark colors before returning unknown
        if max(r, g, b) < 80:  # Increased from 60
            return "black", (0, 0, 0)

        return "", (128, 128, 128)

    def get_3d_position_from_depth(self, x, y, depth_value):
        """Calculate 3D position from depth value and camera intrinsics."""
        # Camera intrinsic parameters (if not already defined in __init__)
        fx = 306.00787353515625
        fy = 306.00787353515625
        cx = 188.68125915527344
        cy = 105.0
        
        z = depth_value  # Depth value (already in meters)
        x_3d = (x - cx) * z / fx
        y_3d = (y - cy) * z / fy

        return np.array([-x_3d, -y_3d, z])  # Convert to camera frame (Z forward, X right, Y down)

    def image_callback(self, msg):
        try:
            # Convert the ROS image message to an OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply median blur for noise reduction
            blurred_image = cv2.medianBlur(gray_image, 5)
            
            # Apply Canny edge detection
            param1 = 50
            edges = cv2.Canny(blurred_image, param1 / 2, param1)
            cv2.imshow("Canny", edges)
            
            # Check if depth data is available
            if self.depth_data is None:
                self.get_logger().warn("No depth data available")
                return
                
            # Get the current depth map
            depth_map = self.depth_data
            
            # Use Hough Gradient Alternative method to detect circles
            circles = cv2.HoughCircles(
                blurred_image,
                cv2.HOUGH_GRADIENT_ALT,
                dp=1.5,
                minDist=10,
                param1=param1,
                param2=0.9,
                minRadius=5,
                maxRadius=100
            )
            
            # If circles are detected, process them
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    x, y, r = circle  # Extract circle center (x, y) and radius (r)
                    
                    # Check if this is a hollow ring
                    if self.is_hollow_ring(x, y, r, depth_map):
                        # Get ring color
                        color_name, color_bgr = self.get_ring_color(frame, x, y, r)
                        
                        # Skip if we couldn't determine a color
                        if not color_name:
                            continue
                        
                        # Draw the detected hollow ring in its detected color
                        cv2.circle(frame, (x, y), r, color_bgr, 2)  # Circle outline
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # Center point in red
                        
                        # Add color label below the ring
                        cv2.putText(frame, 
                                    color_name, 
                                    (x - 20, y + r + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color_bgr,
                                    2)
                        
                        # Get depth value from depth map for the ring center
                        try:
                            # Make sure x and y are within depth map bounds
                            if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                                depth_value = depth_map[y + r, x]
                                
                                # Check if depth value is valid
                                if depth_value > 0 and depth_value < 5.0:  # Reasonable depth range (0-5m)
                                    # Calculate 3D position from 2D point and depth
                                    position_3d = self.get_3d_position_from_depth(x, y, depth_value)
                                    
                                    self.get_logger().info(
                                        f"Ring at ({x},{y}) with depth {depth_value:.2f}m, 3D pos: {position_3d}"
                                    )
                                    
                                    # Transform to map frame
                                    map_position = self.transform_point_to_map(position_3d)
                                    self.get_logger().info(
                                        f"Transformed ring position to map frame: {map_position}"
                                    )
                                    if map_position is not None:
                                        self.get_logger().info(
                                            f"{color_name.upper()} hollow ring detected at "
                                            f"({x}, {y}) with radius {r}, map position: {map_position}, depth: {depth_value:.2f}m"
                                        )
                                        
                                        # Store the ring data
                                        self.update_ring(map_position, r, color_name, color_bgr)
                                else:
                                    self.get_logger().warn(
                                        f"Invalid depth value at ({x},{y}): {depth_value}"
                                    )
                            else:
                                self.get_logger().warn(
                                    f"Point ({x},{y}) outside depth map bounds: {depth_map.shape}"
                                )
                        except Exception as e:
                            self.get_logger().error(f"Error getting depth for point ({x},{y}): {e}")
                    else:
                        # Draw non-hollow circles in gray
                        cv2.circle(frame, (x, y), r, (128, 128, 128), 1)
            
            # Show the detected circles
            cv2.imshow("Detected Rings", frame)
            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def update_ring(self, position, radius_px, color_name, color_bgr):
        """Update ring data in storage, create new entry if needed"""
        # Check if this ring is already in our dictionary by checking if it's near an existing ring
        matched_hash = None
        for ring_hash, ring_data in self.rings.items():
            distance = np.linalg.norm(position - ring_data.position)
            if distance < self.ring_position_threshold:
                matched_hash = ring_hash
                break
        
        current_time = time.time()
        
        # If matching ring found, update its data but don't announce or create new markers
        if matched_hash:
            # Update position with some smoothing
            smoothing = 0.3
            self.rings[matched_hash].position = (1 - smoothing) * self.rings[matched_hash].position + smoothing * position
            self.rings[matched_hash].last_seen = current_time
            
            # Only announce if color changed (this is a significantly different observation)
            if self.rings[matched_hash].color_name != color_name:
                self.announce_ring_color(color_name)
                self.rings[matched_hash].color_name = color_name
                self.rings[matched_hash].color_bgr = color_bgr
        else:
            # This is a new ring - create new entry and announce it
            pos_hash = self.position_hash(position)
            self.rings[pos_hash] = RingData(
                position=position,
                radius=radius_px * 0.002,  # Convert pixels to approximate meters
                color_name=color_name,
                color_bgr=color_bgr,
                last_seen=current_time,
                announced=True  # Mark as announced when created
            )
            # Announce new ring
            self.announce_ring_color(color_name)

    def cleanup_old_rings(self):
        """Remove rings that haven't been seen recently"""
        current_time = time.time()
        keys_to_remove = []
        
        for ring_hash, ring_data in self.rings.items():
            if current_time - ring_data.last_seen > self.ring_timeout:
                keys_to_remove.append(ring_hash)
        
        for key in keys_to_remove:
            self.rings.pop(key)
            
        if keys_to_remove:
            self.get_logger().debug(f"Removed {len(keys_to_remove)} old rings")

    def publish_ring_markers(self):
        """Publish markers for all tracked rings, avoiding duplicates"""
        if not self.rings:
            return
            
        marker_array = MarkerArray()
        
        # Track positions we've seen to avoid duplicates
        published_positions = set()
        
        for ring_hash, ring_data in self.rings.items():
            # Create position key for deduplication (rounded to cm precision)
            pos_key = (round(ring_data.position[0], 2), 
                    round(ring_data.position[1], 2), 
                    round(ring_data.position[2], 2))
            
            # Skip if we've already published a marker at this position
            if pos_key in published_positions:
                continue
                
            # Add to set of published positions
            published_positions.add(pos_key)
            
            # Ring position marker (sphere)
            ring_marker = Marker()
            ring_marker.header.frame_id = "map"
            ring_marker.header.stamp = self.get_clock().now().to_msg()
            ring_marker.ns = "ring_positions"
            ring_marker.id = hash(ring_hash) % 10000  # Use hash for ID
            ring_marker.type = Marker.SPHERE
            ring_marker.action = Marker.ADD
            ring_marker.pose.position.x = ring_data.position[0]
            ring_marker.pose.position.y = ring_data.position[1]
            ring_marker.pose.position.z = ring_data.position[2]
            ring_marker.pose.orientation.w = 1.0
            ring_marker.scale.x = ring_marker.scale.y = ring_marker.scale.z = ring_data.radius * 10  # Diameter
            
            # Set color from BGR to RGB
            b, g, r = ring_data.color_bgr
            ring_marker.color.r = float(r) / 255.0
            ring_marker.color.g = float(g) / 255.0
            ring_marker.color.b = float(b) / 255.0
            ring_marker.color.a = 0.8
            
            # Set lifetime
            if self.marker_lifetime > 0:
                ring_marker.lifetime.sec = int(self.marker_lifetime)
                ring_marker.lifetime.nanosec = int((self.marker_lifetime % 1) * 1e9)
            
            marker_array.markers.append(ring_marker)
            
            # Text marker for color label
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "ring_colors"
            text_marker.id = hash(ring_hash) % 10000  # Use hash for ID
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = ring_data.position[0]
            text_marker.pose.position.y = ring_data.position[1]
            text_marker.pose.position.z = ring_data.position[2] + 0.1  # Above the sphere
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.20 # Text size
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.8
            text_marker.text = ring_data.color_name.upper()
            
            if self.marker_lifetime > 0:
                text_marker.lifetime.sec = int(self.marker_lifetime)
                text_marker.lifetime.nanosec = int((self.marker_lifetime % 1) * 1e9)
            
            marker_array.markers.append(text_marker)
        
        # Publish the marker array
        self.marker_pub.publish(marker_array)

def main():
    rclpy.init()
    detector = RingDetector()
    rclpy.spin(detector)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()