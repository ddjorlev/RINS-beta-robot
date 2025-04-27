#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
import numpy as np
import time
import math
from enum import Enum
from dataclasses import dataclass

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped, PoseArray
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

class RobotState(Enum):
    IDLE = 0
    UNDOCKING = 1
    WAITING_FOR_PATH = 2
    FOLLOWING_GLOBAL_PLAN = 3
    APPROACHING_PERSON = 4
    APPROACHING_RING = 5
    RETURNING_TO_PLAN = 6


@dataclass
class DetectedObject:
    position: np.ndarray  # [x, y, z]
    normal: np.ndarray = None  # Normal vector (for people)
    object_type: str = "ring"  # "ring" or "person"
    last_seen: float = 0.0


class RobotCommander(Node):
    def __init__(self):
        super().__init__('robot_commander')

        # Parameters
        self.declare_parameter('map_image_path', '/path/to/map_image.png')
        self.declare_parameter('approach_distance', 1)
        self.declare_parameter('detection_timeout', 5.0)
        self.declare_parameter('path_wait_timeout', 30.0)  # Timeout for waiting for path

        # Get parameters
        self.map_image_path = self.get_parameter('map_image_path').get_parameter_value().string_value
        self.approach_distance = self.get_parameter('approach_distance').get_parameter_value().double_value
        self.detection_timeout = self.get_parameter('detection_timeout').get_parameter_value().double_value
        self.path_wait_timeout = self.get_parameter('path_wait_timeout').get_parameter_value().double_value

        # State variables
        self.dock_status_received = False
        self.state = RobotState.IDLE
        self.global_path = None
        self.path_index = 0
        self.detected_objects = []  # List of DetectedObject
        self.last_position = None
        self.path_wait_start_time = None

        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = False  # Default to not docked
        
        # TF2 buffer and listener for transformation
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ROS2 publishers and subscribers
        self.create_subscription(DockStatus,
                                 'dock_status',
                                 self._dockCallback,
                                 qos_profile_sensor_data)
        
        self.path_pub = self.create_publisher(Path, '/global_plan', QoSReliabilityPolicy.BEST_EFFORT)
        self.goal_pub = self.create_publisher(PoseStamped, '/navigation_goal', QoSReliabilityPolicy.BEST_EFFORT)

        self.rings_sub = self.create_subscription(
            MarkerArray, '/ring_markers', self.rings_callback, QoSReliabilityPolicy.BEST_EFFORT
        )
        self.people_sub = self.create_subscription(
            MarkerArray, '/people_array', self.people_callback, QoSReliabilityPolicy.BEST_EFFORT
        )

        self.people_poses_sub = self.create_subscription(
            PoseArray, 
            '/detected_people_poses', 
            self.people_poses_callback, 
            QoSReliabilityPolicy.BEST_EFFORT
        )
        
        # Fix the naming conflict - use path_sub instead of path_pub for subscription
        self.path_sub = self.create_subscription(
            Path,
            '/global_path',
            self.path_callback,
            QoSProfile(
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
            )
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.2, self.control_loop)
        self.path_msg = None

        # Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')

        self.get_logger().info("Robot Commander initialized and ready!")

    def path_callback(self, msg: Path):
        """Callback to receive the global path."""
        self.global_path = [np.array([pose.pose.position.x, pose.pose.position.y]) for pose in msg.poses]
        self.path_index = 0
        self.get_logger().info(f"Received global path with {len(self.global_path)} waypoints.")
        
        # If we were waiting for the path, we can now start following it
        if self.state == RobotState.WAITING_FOR_PATH:
            self.path_index = self.find_closest_path_point()
            self.state = RobotState.FOLLOWING_GLOBAL_PLAN
            self.get_logger().info(f"Starting global plan from point {self.path_index}.")
    
    def rings_callback(self, msg: MarkerArray):
        """Process detected rings."""
        current_time = time.time()
        
        # Clear old objects first
        self.cleanup_detected_objects()
        
        for marker in msg.markers:
            if marker.ns == "ring_positions":
                position = np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
                self.detected_objects.append(DetectedObject(position=position, object_type="ring", last_seen=current_time))

    def people_callback(self, msg: MarkerArray):
        """Process detected people."""
        current_time = time.time()
        
        # Clear old objects first
        self.cleanup_detected_objects()
        
        for marker in msg.markers:
            if marker.ns == "face_positions":
                position = np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
                normal = np.array([marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z])
                self.detected_objects.append(DetectedObject(position=position, normal=normal, object_type="person", last_seen=current_time))

    def people_poses_callback(self, msg: PoseArray):
        """Process people positions from the dedicated topic"""
        current_time = time.time()
        
        # Clear old person objects
        self.detected_objects = [obj for obj in self.detected_objects if obj.object_type != "person"]
        
        for pose in msg.poses:
            position = np.array([pose.position.x, pose.position.y, pose.position.z])
            
            # Extract orientation quaternion
            qx = pose.orientation.x
            qy = pose.orientation.y
            qz = pose.orientation.z
            qw = pose.orientation.w
            
            # Convert quaternion to a forward-facing vector
            # Forward direction is along X-axis in quaternion space
            normal = np.array([
                1 - 2*(qy*qy + qz*qz),    # Forward vector x component
                2*(qx*qy + qw*qz),        # Forward vector y component
                2*(qx*qz - qw*qy)         # Forward vector z component
            ])
            
            # Only use X and Y components for 2D navigation
            normal_xy = normal[:2]
            # Normalize to unit length
            # norm = np.linalg.norm(normal_xy)
            # if norm > 0.001:  # Avoid division by near-zero
            #     normal_xy = normal_xy / norm
                
            # Replace Z component with 0 for 2D navigation
            normal = np.array([normal_xy[0], normal_xy[1], 0.0])
            
            self.get_logger().info(f"Person detected at {position[:2]}, normal direction: {normal_xy}")
            
            # Store as detected object
            self.detected_objects.append(DetectedObject(
                position=position, 
                normal=normal, 
                object_type="person", 
                last_seen=current_time
            ))

    def cleanup_detected_objects(self):
        """Remove old detected objects."""
        current_time = time.time()
        self.detected_objects = [obj for obj in self.detected_objects 
                                if (current_time - obj.last_seen) < self.detection_timeout]

    def control_loop(self):
        """Main control loop for the robot."""
        if not self.dock_status_received:
            self.get_logger().info("Waiting for dock status...")
            return

        if self.state == RobotState.IDLE:
            if self.is_docked:
                self.state = RobotState.UNDOCKING
                self.get_logger().info("Starting undocking sequence.")
            else:
                self.state = RobotState.WAITING_FOR_PATH
                self.path_wait_start_time = time.time()
                self.get_logger().info("Waiting for global path...")
                
        elif self.state == RobotState.UNDOCKING:
            self.perform_undocking()
            
        elif self.state == RobotState.WAITING_FOR_PATH:
            self.wait_for_global_path()

        elif self.state == RobotState.FOLLOWING_GLOBAL_PLAN:
            self.follow_global_plan()

        elif self.state == RobotState.APPROACHING_PERSON:
            self.approach_person()

        elif self.state == RobotState.APPROACHING_RING:
            self.approach_ring()

        elif self.state == RobotState.RETURNING_TO_PLAN:
            self.return_to_global_plan()

    def perform_undocking(self):
        """Undock the robot and transition to following the path."""
        if not hasattr(self, '_undocking_started'):
            self._undocking_started = True
            self.get_logger().info("Starting undocking process...")
            self.undock()
        
        # Check if undocking is complete
        if hasattr(self, 'undock_result_future') and self.isUndockComplete():
            self.get_logger().info("Undocking complete!")
            self._undocking_started = False  # Reset for next time
            
            # Clear existing path and wait for a new one
            # self.global_path = None
            self.state = RobotState.WAITING_FOR_PATH
            self.path_wait_start_time = time.time()

    def wait_for_global_path(self):
        """Wait for the global path to be received before starting navigation."""
        if self.global_path:
            # Find the closest point on the path to start from
            self.path_index = self.find_closest_path_point()
            self.state = RobotState.FOLLOWING_GLOBAL_PLAN
            self.get_logger().info(f"Starting global plan from point {self.path_index}.")
        else:
            # Check for timeout
            current_time = time.time()
            if self.path_wait_start_time and (current_time - self.path_wait_start_time) > self.path_wait_timeout:
                self.get_logger().error(f"Timed out waiting for global path after {self.path_wait_timeout} seconds.")
                self.state = RobotState.IDLE
            else:
                self.get_logger().warn("Still waiting for global path...")

    def find_closest_path_point(self):
        """Find the index of the closest point on the global path to current position."""
        if not self.global_path:
            return 0
            
        try:
            # # Log all the points in the global path
            # for idx, point in enumerate(self.global_path):
            #     self.get_logger().info(f"Path point {idx}: x={point[0]:.2f}, y={point[1]:.2f}")
            
            # Get current robot position - use latest available transform
            transform = self.tf_buffer.lookup_transform(
                "map", 
                "base_link",
                rclpy.time.Time(),  # Ask for the latest transform instead of a specific time
                rclpy.duration.Duration(seconds=1.0)
            )
            
            
            current_pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y
            ])
            self.get_logger().info(f"Current position: x={current_pos[0]:.2f}, y={current_pos[1]:.2f}")
            # Calculate distances to all path points
            distances = [np.linalg.norm(current_pos - path_point) for path_point in self.global_path]
            
            # Return the index of the closest point
            closest_idx = np.argmin(distances)
            self.get_logger().info(f"Closest path point is {closest_idx} at distance {distances[closest_idx]:.2f}m")
            
            return closest_idx
            
        except Exception as e:
            self.get_logger().error(f"Error finding closest path point: {e}")
            return 0  # Default to start of path

    def start_global_plan(self):
        """Start following the global plan."""
        if self.global_path:
            self.path_index = self.find_closest_path_point()
            self.state = RobotState.FOLLOWING_GLOBAL_PLAN
            self.get_logger().info(f"Starting global plan from point {self.path_index}.")
        else:
            self.get_logger().warn("No global plan available.")
            self.state = RobotState.WAITING_FOR_PATH
            self.path_wait_start_time = time.time()

    def follow_global_plan(self):
        """Follow the global plan."""
        if not self.global_path or self.path_index >= len(self.global_path):
            self.get_logger().info("Global plan completed.")
            self.state = RobotState.IDLE
            return

        # Check for detected objects
        for obj in self.detected_objects:
            if time.time() - obj.last_seen < self.detection_timeout:
                if obj.object_type == "person":
                    self.state = RobotState.APPROACHING_PERSON
                    self.last_position = self.global_path[self.path_index]
                    self.get_logger().info(f"Detected person, interrupting global plan.")
                    return
                elif obj.object_type == "ring":
                    self.state = RobotState.APPROACHING_RING
                    self.last_position = self.global_path[self.path_index]
                    self.get_logger().info(f"Detected ring, interrupting global plan.")
                    return

        # Check if a navigation goal is already in progress
        if hasattr(self, '_navigating') and self._navigating:
            return
        
        # Get current position
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0))
            current_pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y
            ])
        except Exception as e:
            self.get_logger().error(f"Error getting current position: {e}")
            return
            
        # Make sure we have a valid path index
        if self.path_index >= len(self.global_path):
            self.get_logger().info("Reached end of global plan.")
            self.state = RobotState.IDLE
            return
        
        # Get target position
        target = self.global_path[self.path_index]
        distance = np.linalg.norm(current_pos - target)
        
        # If we're already close enough to the current waypoint, move to the next one
        if distance < 0.3:  # 30cm threshold, adjust as needed
            self.path_index += 1
            self.get_logger().info(f"Already at waypoint {self.path_index-1}, moving to next one.")
            if self.path_index >= len(self.global_path):
                self.get_logger().info("Reached end of global plan.")
                self.state = RobotState.IDLE
            return
            
        # No goal in progress, start a new one
        self._navigating = True
        self.get_logger().info(f"Starting navigation to waypoint {self.path_index}")
        
        # Navigate to the target
        self.navigate_to(target)

    def approach_person(self):
        """Approach a detected person."""
        recent_people = [obj for obj in self.detected_objects 
                        if obj.object_type == "person" and 
                        time.time() - obj.last_seen < self.detection_timeout]
        
        if recent_people:
            # Get the closest person
            person = min(recent_people, 
                        key=lambda p: np.linalg.norm(p.position[:2]))
            
            # Get position in 2D (x,y)
            person_pos_2d = person.position[:2]
            
            # Get normal vector in 2D (x,y)
            normal_2d = person.normal[:2]
            norm = np.linalg.norm(normal_2d)
            
            if norm > 0.01:
                # Normal vector is the direction the person is facing
                # To approach from front, go opposite to normal
                approach_vec_2d = -normal_2d  # IMPORTANT: Flip direction to approach from front
            else:
                # Fallback if normal isn't usable
                self.get_logger().warn("Person normal vector too small, using position-based approach")
                try:
                    # Get robot position
                    transform = self.tf_buffer.lookup_transform(
                        "map", "base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0))
                    robot_pos = np.array([transform.transform.translation.x, transform.transform.translation.y])
                    
                    # Vector from person to robot
                    approach_vec_2d = robot_pos - person_pos_2d
                    
                    # Normalize
                    # norm = np.linalg.norm(approach_vec_2d)
                    # if norm > 0:
                    #     approach_vec_2d = approach_vec_2d / norm
                    # else:
                    #     approach_vec_2d = np.array([1.0, 0.0])  # Default if robot is at person position
                except Exception as e:
                    self.get_logger().error(f"Error getting robot position: {e}")
                    approach_vec_2d = np.array([1.0, 0.0])  # Default
            
            # Calculate the approach position by moving approach_distance opposite to normal direction
            target_pos = person_pos_2d ##+ approach_vec_2d * self.approach_distance
            
            # Log detailed approach information
            self.get_logger().info(f"Person position: {person_pos_2d}")
            self.get_logger().info(f"Person normal: {normal_2d}")
            self.get_logger().info(f"Approach vector: {approach_vec_2d}")
            self.get_logger().info(f"Approach target: {target_pos}, distance: {self.approach_distance}m")
            
            # Navigate to the calculated position
            self.navigate_to(target_pos)
            self.state = RobotState.RETURNING_TO_PLAN
        else:
            # If no valid person is found, return to the global plan
            self.get_logger().warn("Lost track of person, returning to plan.")
            self.state = RobotState.RETURNING_TO_PLAN
            

    def approach_ring(self):
        """Approach a detected ring."""
        recent_rings = [obj for obj in self.detected_objects 
                       if obj.object_type == "ring" and 
                       time.time() - obj.last_seen < self.detection_timeout]
        
        if recent_rings:
            # Get the closest ring
            ring = min(recent_rings, 
                       key=lambda r: np.linalg.norm(r.position[:2]))
            
            target_pos = ring.position[:2]  # Only use x, y coords
            self.navigate_to(target_pos)
            self.get_logger().info(f"Approaching ring at {target_pos}.")
            self.state = RobotState.RETURNING_TO_PLAN
        else:
            # If no valid ring is found, return to the global plan
            self.get_logger().warn("Lost track of ring, returning to plan.")
            self.state = RobotState.RETURNING_TO_PLAN

    def return_to_global_plan(self):
        """Return to the global plan after interacting with an object."""
        if self.global_path is None:
            self.state = RobotState.IDLE
            self.get_logger().warn("No global path available to return to")
            return
            
        # Find the closest point on the global path
        self.path_index = self.find_closest_path_point()
        
        if self.path_index < len(self.global_path):
            target = self.global_path[self.path_index]
            self.navigate_to(target)
            self.get_logger().info(f"Returning to global plan at point {self.path_index}.")
            self.state = RobotState.FOLLOWING_GLOBAL_PLAN
        else:
            self.get_logger().info("Reached end of global plan.")
            self.state = RobotState.IDLE

    def navigate_to(self, target):
        """Send a navigation goal to the robot."""
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('NavigateToPose action server not available')
            self._navigating = False
            return False
            
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(target[0])
        goal_msg.pose.pose.position.y = float(target[1])
        
        # Calculate orientation
        if self.global_path and self.path_index + 1 < len(self.global_path) and self.state == RobotState.FOLLOWING_GLOBAL_PLAN:
            next_point = self.global_path[self.path_index + 1]
            dx = next_point[0] - target[0]
            dy = next_point[1] - target[1]
            yaw = math.atan2(dy, dx)
            quat = self.YawToQuaternion(yaw)
            goal_msg.pose.pose.orientation = quat
            self.get_logger().info("Navigating to target with calculated orientation.")
        else:
            # Default orientation
            goal_msg.pose.pose.orientation.w = 1.0
        
        self.get_logger().info(f"Navigating to point ({goal_msg.pose.pose.position.x:.2f}, {goal_msg.pose.pose.position.y:.2f})")
        
        # Send the goal asynchronously
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg, feedback_callback=self._navigation_feedback_callback)
        send_goal_future.add_done_callback(self._navigation_goal_response_callback)
        return True

    def _navigation_goal_response_callback(self, future):
        """Process navigation goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal rejected')
            self._navigating = False
            return
            
        self.get_logger().info('Navigation goal accepted')
        self._get_navigation_result_future = goal_handle.get_result_async()
        self._get_navigation_result_future.add_done_callback(self._navigation_result_callback)

    def _navigation_result_callback(self, future):
        """Process navigation action result."""
        self._navigating = False  # Mark navigation as complete regardless of outcome
        try:
            result = future.result().result
            status = future.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('Navigation succeeded')
                # Don't increment path_index here, let follow_global_plan handle it
            else:
                self.get_logger().info(f'Navigation failed with status: {status}')
        except Exception as e:
            self.get_logger().error(f'Error processing navigation result: {e}')
    
    def _send_goal_and_wait(self, goal_msg):
        """Send goal and wait for completion or cancellation - synchronous version."""
        # Store the current state so we can check for interruptions
        starting_state = self.state
        self.get_logger().info(f"Sending navigation goal to {goal_msg.pose.pose.position.x:.2f}, {goal_msg.pose.pose.position.y:.2f}")
        
        # Wait for action server
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateToPose action server not available')
            return False

        # Send the goal
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg, feedback_callback=self._navigation_feedback_callback)
        
        # Wait for goal to be accepted
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal rejected')
            return False
            
        self.get_logger().info('Navigation goal accepted')
        
        # Get the result future
        result_future = goal_handle.get_result_async()
        
        # Set a timeout for goal completion
        start_time = time.time()
        timeout = 60.0  # 1 minute
        
        # Wait for result using a more direct approach with better debugging
        while not result_future.done():
            # Log status periodically
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:  # Log every 5 seconds
                self.get_logger().info(f"Waiting for navigation result... ({elapsed:.1f}s elapsed)")
            
            # Process more callbacks per iteration to ensure we catch the result
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.01)
            
            # Check for timeout
            if time.time() - start_time > timeout:
                self.get_logger().error(f'Navigation timed out after {timeout} seconds')
                cancel_future = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future)
                return False
                
            # Check if our state has changed
            if self.state != starting_state:
                self.get_logger().info('Cancelling navigation due to state change')
                cancel_future = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future)
                return False
        
        # Get the final result - add robust error handling and debugging
        try:
            result = result_future.result()
            status = result.status
            
            # Print the entire result for debugging
            self.get_logger().info(f"Navigation completed with result: {result}")
            self.get_logger().info(f"Status: {status} ({GoalStatus.STATUS_SUCCEEDED=})")
            
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('✅ Navigation goal SUCCEEDED!')
                return True
            else:
                self.get_logger().warn(f'❌ Navigation goal FAILED with status: {status}')
                return False
        except Exception as e:
            self.get_logger().error(f'Error processing navigation result: {e}')
            return False
        
    def _navigation_result_callback(self, future):
        """Process navigation action result."""
        self._navigating = False  # Mark navigation as complete regardless of outcome
        try:
            result = future.result().result
            status = future.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('Navigation succeeded')
                # Increment the path index here
                if self.state == RobotState.FOLLOWING_GLOBAL_PLAN:
                    self.path_index += 1
                    self.get_logger().info(f"Moving to next waypoint {self.path_index}")
            else:
                self.get_logger().info(f'Navigation failed with status: {status}')
        except Exception as e:
            self.get_logger().error(f'Error processing navigation result: {e}')

    def _navigation_feedback_callback(self, feedback_msg):
        """Process navigation feedback."""
        feedback = feedback_msg.feedback
        
        # You can log distance remaining, time elapsed, etc.
        self.get_logger().debug(f'Navigation feedback: {feedback}')
        return feedback
    
    def _dockCallback(self, msg):
        """Process dock status messages."""
        # Update the is_docked status based on the message
        self.is_docked = msg.is_docked
        self.dock_status_received = True 
        self.get_logger().debug(f"Dock status update: {'docked' if self.is_docked else 'undocked'}")

    def YawToQuaternion(self, yaw):
        """Convert a yaw angle to a quaternion."""
        return Quaternion(x=0.0, y=0.0, z=math.sin(yaw/2), w=math.cos(yaw/2))

    def undock(self):
        """Send undock action goal."""
        self.get_logger().info('Sending undock goal')
        if not self.undock_action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Undock action server not available')
            return
            
        goal_msg = Undock.Goal()
        self.undock_result_future = self.undock_action_client.send_goal_async(
            goal_msg, feedback_callback=self._undockFeedbackCallback)
        self.undock_result_future.add_done_callback(self._undockGoalResponseCallback)

    def _undockFeedbackCallback(self, feedback_msg):
        """Process undock action feedback."""
        self.get_logger().debug('Received undock feedback')

    def _undockGoalResponseCallback(self, future):
        """Process undock action goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Undock goal rejected')
            return
            
        self.get_logger().info('Undock goal accepted')
        self._get_undock_result_future = goal_handle.get_result_async()
        self._get_undock_result_future.add_done_callback(self._undockResultCallback)

    def _undockResultCallback(self, future):
        """Process undock action result."""
        result = future.result().result
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Undock succeeded')
        else:
            self.get_logger().info(f'Undock failed with status: {status}')

    def isUndockComplete(self):
        """Check if undock action is complete."""
        if hasattr(self, '_get_undock_result_future'):
            return self._get_undock_result_future.done()
        return False
    
    def waitUntilNav2Active(self):
        """Wait until Nav2 is active."""
        # Wait for lifecycle nodes to be active
        lifecycle_nodes = [
            'controller_server',
            'planner_server',
            'recoveries_server',
            'bt_navigator'
        ]
        
        for node in lifecycle_nodes:
            node_name = f'{node}'
            self.get_logger().info(f'Waiting for {node_name}...')
            
            client = self.create_client(GetState, f'{node_name}/get_state')
            
            if not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'{node_name} service not available, continuing...')
                continue
                
            future = client.call_async(GetState.Request())
            rclpy.spin_until_future_complete(self, future)
            
            if future.result() is not None:
                state = future.result().current_state.id
                if state != 3:  # 3 is ACTIVE state
                    self.get_logger().info(f'{node_name} is not active (state={state})')
                else:
                    self.get_logger().info(f'{node_name} is active')
            else:
                self.get_logger().info(f'Service call to {node_name} failed')
                
        self.get_logger().info('Nav2 is active and ready')

    # def undock(self):
    #     """Perform Undock action."""
    #     self.info('Undocking...')
    #     self.undock_send_goal()

    #     while not self.isUndockComplete():
    #         time.sleep(0.1)

    # def undock_send_goal(self):
    #     goal_msg = Undock.Goal()
    #     self.undock_action_client.wait_for_server()
    #     goal_future = self.undock_action_client.send_goal_async(goal_msg)

    #     rclpy.spin_until_future_complete(self, goal_future)

    #     self.undock_goal_handle = goal_future.result()

    #     if not self.undock_goal_handle.accepted:
    #         self.error('Undock goal rejected')
    #         return

    #     self.undock_result_future = self.undock_goal_handle.get_result_async()

    # def isUndockComplete(self):
    #     """
    #     Get status of Undock action.

    #     :return: ``True`` if undocked, ``False`` otherwise.
    #     """
    #     if self.undock_result_future is None or not self.undock_result_future:
    #         return True

    #     rclpy.spin_until_future_complete(self, self.undock_result_future, timeout_sec=0.1)

    #     if self.undock_result_future.result():
    #         self.undock_status = self.undock_result_future.result().status
    #         if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
    #             self.info(f'Goal with failed with status code: {self.status}')
    #             return True
    #     else:
    #         return False

    #     self.info('Undock succeeded')
    #     return True
    
    # def cancelTask(self):
    #     """Cancel pending task request of any type."""
    #     self.info('Canceling current task.')
    #     if self.result_future:
    #         future = self.goal_handle.cancel_goal_async()
    #         rclpy.spin_until_future_complete(self, future)
    #     return
    

    # def isTaskComplete(self):
    #     """Check if the task request of any type is complete yet."""
    #     if not self.result_future:
    #         # task was cancelled or completed
    #         return True
    #     rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
    #     if self.result_future.result():
    #         self.status = self.result_future.result().status
    #         if self.status != GoalStatus.STATUS_SUCCEEDED:
    #             self.debug(f'Task with failed with status code: {self.status}')
    #             return True
    #     else:
    #         # Timed out, still processing, not complete yet
    #         return False

    #     self.debug('Task succeeded!')
    #     return True


    # def waitUntilNav2Active(self, navigator='bt_navigator', localizer='amcl'):
    #     """Block until the full navigation system is up and running."""
    #     self._waitForNodeToActivate(localizer)
    #     if not self.initial_pose_received:
    #         time.sleep(1)
    #     self._waitForNodeToActivate(navigator)
    #     self.info('Nav2 is ready for use!')
    #     return

    # def _waitForNodeToActivate(self, node_name):
    #     # Waits for the node within the tester namespace to become active
    #     self.debug(f'Waiting for {node_name} to become active..')
    #     node_service = f'{node_name}/get_state'
    #     state_client = self.create_client(GetState, node_service)
    #     while not state_client.wait_for_service(timeout_sec=1.0):
    #         self.info(f'{node_service} service not available, waiting...')

    #     req = GetState.Request()
    #     state = 'unknown'
    #     while state != 'active':
    #         self.debug(f'Getting {node_name} state...')
    #         future = state_client.call_async(req)
    #         rclpy.spin_until_future_complete(self, future)
    #         if future.result() is not None:
    #             state = future.result().current_state.label
    #             self.debug(f'Result of get_state: {state}')
    #         time.sleep(2)
    #     return
    
    # def YawToQuaternion(self, angle_z = 0.):
    #     quat_tf = quaternion_from_euler(0, 0, angle_z)

    #     # Convert a list to geometry_msgs.msg.Quaternion
    #     quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
    #     return quat_msg

    # def _amclPoseCallback(self, msg):
    #     self.debug('Received amcl pose')
    #     self.initial_pose_received = True
    #     self.current_pose = msg.pose
    #     return

    # def _feedbackCallback(self, msg):
    #     self.debug('Received action feedback message')
    #     self.feedback = msg.feedback
    #     return
    
    # def getFeedback(self):
    #     """Get the pending action feedback message."""
    #     return self.feedback


    # def _dockCallback(self, msg: DockStatus):
    #     self.is_docked = msg.is_docked

    # def setInitialPose(self, pose):
    #     msg = PoseWithCovarianceStamped()
    #     msg.pose.pose = pose
    #     msg.header.frame_id = self.pose_frame_id
    #     msg.header.stamp = 0
    #     self.info('Publishing Initial Pose')
    #     self.initial_pose_pub.publish(msg)
    #     return

    # def info(self, msg):
    #     self.get_logger().info(msg)
    #     return

    # def warn(self, msg):
    #     self.get_logger().warn(msg)
    #     return

    # def error(self, msg):
    #     self.get_logger().error(msg)
    #     return

    # def debug(self, msg):
    #     self.get_logger().debug(msg)
    #     return
    
    # def spin(self, spin_dist=1.57, time_allowance=10):
    #     self.debug("Waiting for 'Spin' action server")
    #     while not self.spin_client.wait_for_server(timeout_sec=1.0):
    #         self.info("'Spin' action server not available, waiting...")
    #     goal_msg = Spin.Goal()
    #     goal_msg.target_yaw = spin_dist
    #     goal_msg.time_allowance = Duration(sec=time_allowance)

    #     self.info(f'Spinning to angle {goal_msg.target_yaw}....')
    #     send_goal_future = self.spin_client.send_goal_async(goal_msg, self._feedbackCallback)
    #     rclpy.spin_until_future_complete(self, send_goal_future)
    #     self.goal_handle = send_goal_future.result()

    #     if not self.goal_handle.accepted:
    #         self.error('Spin request was rejected!')
    #         return False

    #     self.result_future = self.goal_handle.get_result_async()
    #     return True
    
    # def destroyNode(self):
    #     self.nav_to_pose_client.destroy()
    #     super().destroy_node()   
        
def main(args=None):
    rclpy.init(args=args)
    node = RobotCommander()

    try:
        # Wait for the navigation stack to be active
        node.get_logger().info("Waiting for navigation system to become active...")
        node.waitUntilNav2Active()
        
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()