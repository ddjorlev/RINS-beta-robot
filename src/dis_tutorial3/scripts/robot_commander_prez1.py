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
    SPINNING_AFTER_UNDOCK = 2
    WAITING_FOR_PATH = 3
    FOLLOWING_GLOBAL_PLAN = 4
    APPROACHING_PERSON = 5
    APPROACHING_RING = 6
    RETURNING_TO_PLAN = 7
    PROCESSING_VISION = 8  # Add this new state
    RECOVERING = 9  # New recovery state
    VISITING_OBJECTS = 10


@dataclass
class DetectedObject:
    position: np.ndarray  # [x, y, z]
    normal: np.ndarray = None  # Normal vector (for people)
    object_type: str = "ring"  # "ring" or "person"
    last_seen: float = 0.0


class RobotCommander(Node):
    def __init__(self):
        super().__init__('robot_commander')
        self.last_state_change = time.time()
        self._spin_required = True  # Set to True if we want to do a 360 spin before path following
        self._spin_complete = False
        # Parameters
        self.declare_parameter('map_image_path', '/path/to/map_image.png')
        self.declare_parameter('approach_distance', 0.4)
        self.declare_parameter('detection_timeout', 5.0)
        self.declare_parameter('path_wait_timeout', 30.0)
        self.declare_parameter('person_approach_timeout', 10.0)

        # Get parameters
        self.map_image_path = self.get_parameter('map_image_path').get_parameter_value().string_value
        self.approach_distance = self.get_parameter('approach_distance').get_parameter_value().double_value
        self.detection_timeout = self.get_parameter('detection_timeout').get_parameter_value().double_value
        self.path_wait_timeout = self.get_parameter('path_wait_timeout').get_parameter_value().double_value
        self.person_approach_timeout = self.get_parameter('person_approach_timeout').get_parameter_value().double_value

        # State variables
        self.dock_status_received = False
        self.state = RobotState.IDLE
        self.global_path = None
        self.path_index = 0
        self.detected_objects = []
        self.last_position = None
        self.path_wait_start_time = None
        self.person_approach_start_time = None
        self.last_path_index_before_detour = 0
        self.ringe = []

        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = False
        
        # TF2 buffer and listener
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
        self.global_path.append(np.array([-1.21, 3.44]))
        self.global_path.append(np.array([-1.56, 3.41])) 
        self.global_path.append(np.array([-1.51, 3.99])) 
        #elf.global_path.append(np.array([-1.5, 4.44]))   

        self.get_logger().info(f"Received global path with {len(self.global_path)} waypoints.")
        
        if self.state == RobotState.WAITING_FOR_PATH:
            self.path_index = self.find_closest_path_point()
            self.state = RobotState.FOLLOWING_GLOBAL_PLAN
            self.get_logger().info(f"Starting global plan from point {self.path_index}.")
    
    def rings_callback(self, msg: MarkerArray):
        """Process detected rings."""
        current_time = time.time()
        self.cleanup_detected_objects()
        
        for marker in msg.markers:
            if marker.ns == "ring_positions":
                position = np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
                # self.detected_objects.append(DetectedObject(position, object_type="ring", last_seen=current_time))
                self.ringe.append(DetectedObject(position, object_type="ring", last_seen=current_time))

    def people_callback(self, msg: MarkerArray):
        """Process detected people."""
        current_time = time.time()
        self.cleanup_detected_objects()
        
        for marker in msg.markers:
            if marker.ns == "face_positions":
                position = np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
                normal = np.array([marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z])
                self.detected_objects.append(DetectedObject(position=position+normal*0.7, normal=normal, object_type="person", last_seen=current_time))

    def people_poses_callback(self, msg: PoseArray):
        """Process people positions from the dedicated topic"""
        current_time = time.time()
        self.detected_objects = [obj for obj in self.detected_objects if obj.object_type != "person"]
        
        for pose in msg.poses:
            position = np.array([pose.position.x, pose.position.y, pose.position.z])
            
            qx = pose.orientation.x
            qy = pose.orientation.y
            qz = pose.orientation.z
            qw = pose.orientation.w
            
            normal = np.array([
                1 - 2*(qy*qy + qz*qz),
                2*(qx*qy + qw*qz),
                2*(qx*qz - qw*qy)
            ])
            
            normal_2d = normal[:2]
            norm = np.linalg.norm(normal_2d)
            if norm > 0.001:
                normal_2d = normal_2d / norm
            
            normal = np.array([normal_2d[0], normal_2d[1], 0.0])
            
            self.detected_objects.append(DetectedObject(
                position=position,
                normal=normal,
                object_type="person",
                last_seen=current_time
            ))
            
            self.get_logger().info(f"Person detected at {position[:2]}, normal={normal_2d}")

    def cleanup_detected_objects(self):
        """Remove old detected objects."""
        current_time = time.time()
        self.detected_objects = [obj for obj in self.detected_objects 
                               if (current_time - obj.last_seen) < self.detection_timeout]

    def control_loop(self):
        """Improved control loop with recovery mechanisms."""
        try:
            # Add case for visiting objects
            if self.state == RobotState.VISITING_OBJECTS:
                self.handle_object_visits()
            elif self.state == RobotState.SPINNING_AFTER_UNDOCK:
                # Check if we've already sent the spin command
                if not hasattr(self, '_spin_sent') or not self._spin_sent:
                    self.get_logger().info("Initiating 360-degree spin...")
                    self.perform_360_spin()
                    self._spin_sent = True
                # Check if spin is complete
                elif hasattr(self, '_spin_complete') and self._spin_complete:
                    self.get_logger().info("Spin completed, now waiting for global path...")
                    self._set_state(RobotState.WAITING_FOR_PATH)
                    self.path_wait_start_time = time.time()     
            elif self.state == RobotState.IDLE:
                self.handle_idle_state()
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
            elif self.state == RobotState.PROCESSING_VISION:
                self.process_vision_data()
            elif self.state == RobotState.RECOVERING:
                self.recover_from_failure()
            
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
            self._set_state(RobotState.RECOVERING)

    def visit_next_object(self):
        """Process the next object in the visielf.detected_objects)"""
        
        # Sort queue by type and then by distance to current position
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0))
            robot_pos = np.array([transform.transform.translation.x, transform.transform.translation.y])
            
            # First sort by distance
            self.visit_queue.sort(key=lambda obj: np.linalg.norm(obj.position[:2] - robot_pos))
            
            # Flag that this is the second approach (for ring proximity detection)
            self._second_approach = True
            
            # Initialize visited_rings list if it doesn't exist
            if not hasattr(self, 'visited_rings'):
                self.visited_rings = []
            
            # Log the visit plan
            self.get_logger().info(f"💠 Planning to visit {len(self.visit_queue)} objects:")
            for i, obj in enumerate(self.visit_queue):
                self.get_logger().info(f"  {i+1}. {obj.object_type} at {obj.position[:2]}")
            
            # Start the visit process
            self.visit_next_object()
            
        except Exception as e:
            self.get_logger().error(f"Failed to start object visits: {e}")
            self._set_state(RobotState.IDLE)

    def visit_next_object(self):
        """Process the next object in the visit queue."""
        # Check if there are more objects to visit
        if not hasattr(self, 'visit_queue') or not self.visit_queue:
            self.get_logger().info("🏁 Object visits complete - all objects visited!")
            self._set_state(RobotState.IDLE)
            return
        
        # Get next object from queue
        obj = self.visit_queue.pop(0)
        object_type = obj.object_type
        
        # Log what we're visiting
        self.get_logger().info(f"🎯 Visiting {object_type} at {obj.position[:2]}")
        
        # Set object visit start time for timeout monitoring
        self.object_visit_start_time = time.time()
        
        # For rings, use the arc approach strategy
        target_position = None
        if object_type == "ring":
            # Calculate an approach point along an arc around the ring
            target_position = self.calculate_arc_approach_point(obj.position)
            self.get_logger().info(f"🔵 Using arc approach for ring at {obj.position[:2]}")
        else:
            # Use direct approach for other objects
            target_position = obj.position[:2]
        
        # Navigate to the object position
        if self.navigate_to(target_position):
            # Store the object type and original position for when we reach the position
            self.current_visit_object_type = object_type
            self.current_visit_object_position = obj.position[:2]
        else:
            self.get_logger().warn(f"Failed to navigate to {object_type} at {target_position}")
            # Continue with next object
            self.visit_next_object()

    def recover_from_failure(self):
        """System recovery after failures."""
        recovery_actions = [
            self.recovery_cancel_navigation,
            self.recovery_get_new_position,
            self.recovery_reset_path_index
        ]
        
        for action in recovery_actions:
            if action():
                self._set_state(RobotState.FOLLOWING_GLOBAL_PLAN)
                return
                
        # If all else fails
        self.get_logger().error("All recovery attempts failed")
        self._set_state(RobotState.IDLE)

    def recovery_cancel_navigation(self):
        """Cancel any ongoing navigation."""
        if hasattr(self, '_navigating') and self._navigating:
            self.cancel_navigation()
            return True
        return False

    def recovery_get_new_position(self):
        """Get fresh position estimate."""
        pos, _ = self.get_current_pose()
        if pos is not None:
            self.last_position = pos
            return True
        return False

    def recovery_reset_path_index(self):
        """Reset to closest path point."""
        if self.global_path:
            self.path_index = self.find_closest_path_point()
            return True
        return False
            
    def handle_idle_state(self):
        """Handle the IDLE state transitions."""
        if self.is_docked:
            self._set_state(RobotState.UNDOCKING)
            self.get_logger().info("Starting undocking sequence.")
        else:
            rings = [obj for obj in self.ringe if obj.object_type == "ring"]
            people = [obj for obj in self.detected_objects if obj.object_type == "person"]

            # Check if we have any detected objects that we can visit
            if len(rings) + len(people) >= 6:
                # Fixed the logger method name here - changed object_positioninfo to info
                self.get_logger().info(f"🔍 Found {len(rings)} rings and {len(people)} people - starting visit sequence")
                self._set_state(RobotState.VISITING_OBJECTS)
                self.start_object_visits()
                return

            # If robot is not docked but we still want to do a spin before path following
            if self._spin_required and not self._spin_complete:
                self._set_state(RobotState.SPINNING_AFTER_UNDOCK)
                self.get_logger().info("Starting initial 360-degree spin...")
                self.perform_360_spin()
            else:
                self._set_state(RobotState.WAITING_FOR_PATH)
                self.path_wait_start_time = time.time()
                self.get_logger().info("Waiting for global path...")

    def start_object_visits(self):
        """Začne obiskovanje vseh zaznanh objektov v preprostem načinu."""
        # Ustvari seznam za obisk iz vseh zaznanih objektov
        self.visit_queue = []
        
        # Dodaj osebe v seznam za obisk
        for obj in self.detected_objects:
            self.visit_queue.append(obj)
        
        # Dodaj še obroče v seznam za obisk
        for obj in self.ringe:
            self.visit_queue.append(obj)
        
        # Razvrsti po razdalji od trenutne pozicije
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0))
            robot_pos = np.array([transform.transform.translation.x, transform.transform.translation.y])
            
            # Razvrsti po razdalji - najbližji bodo prvi
            self.visit_queue.sort(key=lambda obj: np.linalg.norm(obj.position[:2] - robot_pos))
            
            # Izpiši kaj bomo obiskali
            self.get_logger().info(f"💠 Načrtujem obisk {len(self.visit_queue)} objektov:")
            for i, obj in enumerate(self.visit_queue):
                self.get_logger().info(f"  {i+1}. {obj.object_type} na {obj.position[:2]}")
            
            # Začni obisk prvega objekta
            self.object_visit_start_time = time.time()
            self.visit_next_object()
        
        except Exception as e:
            self.get_logger().error(f"Napaka pri začetku obiska objektov: {e}")
            self._set_state(RobotState.IDLE)

    def visit_next_object(self):
        """Obišči naslednji objekt iz seznama."""
        # Preveri, če je še kaj objektov v seznamu
        if not hasattr(self, 'visit_queue') or not self.visit_queue:
            self.get_logger().info("🏁 Vsi objekti obiskani!")
            self._set_state(RobotState.IDLE)
            return
        
        # Vzemi naslednji objekt s seznama
        obj = self.visit_queue.pop(0)
        object_type = obj.object_type
        
        # Izpiši informacijo
        self.get_logger().info(f"🎯 Obiskujem {object_type} na {obj.position[:2]}")
        
        # Začni merjenje časa za timeout
        self.object_visit_start_time = time.time()
        
        # Pelji se neposredno do objekta
        target_position = obj.position[:2]
        
        # Navigiraj do objekta
        if self.navigate_to(target_position):
            # Shrani tip objekta in pozicijo za ko prispemo
            self.current_visit_object_type = object_type
            self.current_visit_object_position = obj.position[:2]
        else:
            self.get_logger().warn(f"Nisem uspel navigirati do {object_type} na {target_position}")
            # Nadaljuj z naslednjim objektom
            self.visit_next_object()

    def handle_object_visits(self):
        """Handle the object visit state."""
        # Check if we have an active navigation
        if not hasattr(self, '_navigating') or not self._navigating:
            # If we were navigating but now we're not, we've either reached the position or failed
            if hasattr(self, 'current_visit_object_type'):
                object_type = self.current_visit_object_type
                object_position = self.current_visit_object_position
                
                # For "home" position, just proceed to next object without pausing
                if object_type == "home":
                    self.get_logger().info("✅ Reached home position, proceeding to objects")
                    delattr(self, 'current_visit_object_type')
                    delattr(self, 'current_visit_object_position')
                    self.visit_next_object()
                    return
                
                # Calculate pause duration based on object type (1s for rings, 5s for people)
                pause_duration = 5.0 if object_type == "person" else 1.0
                
                self.get_logger().info(f"🛑 Pausing for {pause_duration}s at {object_type}")
                
                # Use a non-blocking approach for pausing
                if not hasattr(self, 'visit_pause_start'):
                    self.visit_pause_start = time.time()
                    # Mark this object as visited
                    self.visited_objects.append(object_position)
                    return
                    
                # Check if we've paused long enough
                if time.time() - self.visit_pause_start < pause_duration:
                    # Still pausing - report progress once per second
                    elapsed = time.time() - self.visit_pause_start
                    if int(elapsed) != int(elapsed - 0.2):  # Only log once per second
                        self.get_logger().info(f"⏱️ Visiting {object_type}: {elapsed:.1f}s/{pause_duration:.1f}s")
                    return
                    
                # Pause complete, clean up
                self.get_logger().info(f"✅ Visit to {object_type} complete")
                delattr(self, 'visit_pause_start')
                delattr(self, 'current_visit_object_type')
                delattr(self, 'current_visit_object_position')
                
                # Move to next object
                self.visit_next_object()
            else:
                # No current object, move to next
                self.visit_next_object()
                    
        # Check for timeout
        if hasattr(self, 'object_visit_start_time') and time.time() - self.object_visit_start_time > 30.0:
            self.get_logger().warn("⏱️ Object visit timed out, moving to next object")
            # Clean up
            if hasattr(self, 'visit_pause_start'):
                delattr(self, 'visit_pause_start')
            if hasattr(self, 'current_visit_object_type'):
                delattr(self, 'current_visit_object_type')
            if hasattr(self, 'current_visit_object_position'):
                delattr(self, 'current_visit_object_position')
            # Move to next object
            self.visit_next_object()

            
    def _set_state(self, new_state):
        """Wrapper for state transitions with logging."""
        self.get_logger().info(f"State change: {self.state.name} -> {new_state.name}")
        self.state = new_state
        self.last_state_change = time.time()  # For state timeout tracking

    

    def process_vision_data(self):
        """Process vision data with proper state management."""
        try:
            # 1. Get current pose before processing
            current_pos, _ = self.get_current_pose()
            if current_pos is None:
                self.get_logger().warn("Can't get position for vision processing")
                self._set_state(RobotState.RECOVERING)
                return

            # 2. Process vision (time-limited)
            start_time = time.time()
            while time.time() - start_time < 1.5:  # Max 1.5 seconds
                # Your vision processing here
                pass

            # 3. Return to navigation
            self._set_state(RobotState.FOLLOWING_GLOBAL_PLAN)
            
        except Exception as e:
            self.get_logger().error(f"Vision processing failed: {e}")
            self._set_state(RobotState.RECOVERING)


    def perform_undocking(self):
        """Undock the robot and transition to spinning state."""
        if not hasattr(self, '_undocking_started'):
            self._undocking_started = True
            self.get_logger().info("Starting undocking process...")
            self.undock()
        

        if hasattr(self, '_get_undock_result_future') and self._get_undock_result_future.done():
            result = self._get_undock_result_future.result()
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info("Undocking complete! Starting 360-degree spin...")
                self._undocking_started = False
                self._set_state(RobotState.SPINNING_AFTER_UNDOCK)
                self.perform_360_spin()
            else:
                self.get_logger().error("Undocking failed!")
                self._set_state(RobotState.RECOVERING)


    def wait_for_global_path(self):
        """Wait for the global path to be received."""
        if self.global_path:
            # Only proceed if spin is completed or not required
            if not self._spin_required or self._spin_complete:
                self.path_index = self.find_closest_path_point()
                self._set_state(RobotState.FOLLOWING_GLOBAL_PLAN)
                self.get_logger().info(f"Starting global plan from point {self.path_index}")
            else:
                self._set_state(RobotState.SPINNING_AFTER_UNDOCK)
                self.get_logger().info("Need to complete 360-degree spin before following path")
                self.perform_360_spin()
        else:
            current_time = time.time()
            if self.path_wait_start_time and (current_time - self.path_wait_start_time) > self.path_wait_timeout:
                self.get_logger().error(f"Timed out waiting for global path after {self.path_wait_timeout} seconds.")
                self._set_state(RobotState.IDLE)

    def find_closest_path_point(self):
        """Find the index of the closest point on the global path."""
        if not self.global_path:
            return 0
            
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", 
                "base_link",
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
            
            current_pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y
            ])
            
            distances = [np.linalg.norm(current_pos - path_point) for path_point in self.global_path]
            closest_idx = np.argmin(distances)
            self.get_logger().info(f"Closest path point is {closest_idx} at distance {distances[closest_idx]:.2f}m")
            return closest_idx
            
        except Exception as e:
            self.get_logger().error(f"Error finding closest path point: {e}")
            return 0

    def follow_global_plan(self):
        """Follow the global path with comprehensive error handling and recovery.
        
        This function:
        - Checks for timeout conditions
        - Validates the current path and position
        - Handles waypoint progression
        - Manages navigation commands
        - Initiates recovery when needed
        """
        # 1. Timeout check (20 seconds max in this state)
        if time.time() - self.last_state_change > 70.0:
            self.get_logger().warn(f"Timeout in FOLLOWING_GLOBAL_PLAN state for waypoint {self.path_index}")
            self._set_state(RobotState.RECOVERING)
            return

        # 2. Validate we have a path to follow
        if not self.global_path:
            self.get_logger().error("No global path available!")
            self._set_state(RobotState.IDLE)
            return

        # 3. Check if we've completed all waypoints
        if self.path_index >= len(self.global_path):
            self.get_logger().info("Global plan completed - all waypoints reached!")
            self._set_state(RobotState.IDLE)
            return

        # 4. Get current position with fresh TF data
        current_pos, current_yaw = self.get_current_pose()
        if current_pos is None:
            self.get_logger().warn("Cannot get current position - attempting recovery")
            self._set_state(RobotState.RECOVERING)
            return

        # 5. Log current navigation status
        target = self.global_path[self.path_index]
        distance = np.linalg.norm(current_pos - target)
        self.get_logger().info(
            f"Navigating to waypoint {self.path_index}/{len(self.global_path)-1} | "
            f"Distance: {distance:.2f}m | "
            f"Position: ({current_pos[0]:.2f}, {current_pos[1]:.2f})"
        )

        # 6. Check if we've reached the current waypoint
        if distance < 0.3:  # Waypoint reached threshold
            self.get_logger().info(f"Reached waypoint {self.path_index}")
            self.path_index += 1
            
            if self.path_index >= len(self.global_path):
                self._set_state(RobotState.IDLE)
            else:
                # Immediately proceed to next waypoint
                self._set_state(RobotState.FOLLOWING_GLOBAL_PLAN)
            return

        # 7. Check if we're already navigating to this point
        if hasattr(self, '_navigating') and self._navigating:
            return  # Already navigating, wait for completion

        # 8. Send navigation command to current target
        self.get_logger().info(f"Initiating navigation to waypoint {self.path_index}")
        if not self.navigate_to(target):
            self.get_logger().warn("Navigation command failed - initiating recovery")
            self._set_state(RobotState.RECOVERING)

    def approach_person(self):
        """Approach a detected person with smarter vector calculation and pause at face."""
        # Check for timeout    def _spin_result_callback(self, future):
        """Handle spin result."""
        try:
            result = future.result().result
            status = future.result().status
            
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info("✅ Spin completed successfully!")
            else:
                self.get_logger().warn(f"Spin completed with status {status}")
                
            # Signal that spin is complete - control loop will handle state transition
            self._spin_complete = True
            
        except Exception as e:
            self.get_logger().error(f"Error processing spin result: {e}")
            self._set_state(RobotState.RECOVERING)
        if time.time() - self.person_approach_start_time > self.person_approach_timeout:
            self.get_logger().warn("Person approach timeout, returning to plan")
            self.state = RobotState.RETURNING_TO_PLAN
            self.return_to_global_plan()
            return

        recent_people = [obj for obj in self.detected_objects 
                        if obj.object_type == "person" and 
                        time.time() - obj.last_seen < self.detection_timeout]
        
        if not recent_people:
            self.get_logger().warn("Lost track of person, returning to plan")
            self.state = RobotState.RETURNING_TO_PLAN
            self.return_to_global_plan()
            return

        # Get the closest person
        person = min(recent_people, key=lambda p: np.linalg.norm(p.position[:2]))
        person_pos_2d = person.position[:2]
        
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0))
            robot_pos = np.array([transform.transform.translation.x, transform.transform.translation.y])
            
            robot_to_person = person_pos_2d - robot_pos
            dist_to_person = np.linalg.norm(robot_to_person)
            
            # Check if we're close enough to pause at the face
            if dist_to_person < self.approach_distance * 1.2:  # Slightly larger than approach distance
                self.get_logger().info("Reached face position - pausing for 1 second")
                time.sleep(1.0)  # Pause for 1 second
                self.state = RobotState.RETURNING_TO_PLAN
                self.return_to_global_plan()
                return
                
            if dist_to_person > 0.01:
                robot_to_person = robot_to_person / dist_to_person
            
            normal_2d = person.normal[:2]
            normal_norm = np.linalg.norm(normal_2d)
            
            if normal_norm > 0.01:
                normal_2d = normal_2d / normal_norm
                dot_product = np.dot(normal_2d, -robot_to_person)
                
                if dot_product > 0:
                    approach_vec_2d = -robot_to_person
                    self.get_logger().info("Person is facing robot, approaching from current position")
                else:
                    perp1 = np.array([-normal_2d[1], normal_2d[0]])
                    perp2 = np.array([normal_2d[1], -normal_2d[0]])
                    dot1 = np.dot(perp1, robot_to_person)
                    dot2 = np.dot(perp2, robot_to_person)
                    side_vec = perp1 if dot1 > dot2 else perp2
                    approach_vec_2d = 0.7 * side_vec - 0.3 * normal_2d
                    approach_vec_2d = approach_vec_2d / np.linalg.norm(approach_vec_2d)
                    self.get_logger().info("Person facing away, approaching from side-front")
            else:
                approach_vec_2d = -robot_to_person
                self.get_logger().info("Using robot-to-person vector for approach")
                
        except Exception as e:
            self.get_logger().error(f"Error calculating approach vector: {e}")
            approach_vec_2d = np.array([1.0, 0.0])
            
        distances_to_try = [self.approach_distance]
        
        for distance in distances_to_try:
            target_pos = person_pos_2d + approach_vec_2d * distance
            self.get_logger().info(f"Trying approach at distance {distance}m: {target_pos}")
            
            if self.navigate_to(target_pos):
                self.get_logger().info(f"Successfully approaching at distance {distance}m")
                return

        self.get_logger().warn("All approach attempts failed, returning to plan")
        self.state = RobotState.RETURNING_TO_PLAN
        self.return_to_global_plan()

    def approach_ring(self):
        """Approach a detected ring."""
        recent_rings = [obj for obj in self.detected_objects 
                       if obj.object_type == "ring" and 
                       time.time() - obj.last_seen < self.detection_timeout]
        
        if recent_rings:
            ring = min(recent_rings, key=lambda r: np.linalg.norm(r.position[:2]))
            target_pos = ring.position[:2]
            self.navigate_to(target_pos)
            self.state = RobotState.RETURNING_TO_PLAN
            self.return_to_global_plan()
        else:
            self.get_logger().warn("Lost track of ring, returning to plan.")
            self.state = RobotState.RETURNING_TO_PLAN
            self.return_to_global_plan()

    def return_to_global_plan(self, attempt=0):
        """Return to the global plan after interacting with an object."""
        if self.global_path is None:
            self.state = RobotState.IDLE
            self.get_logger().warn("No global path available to return to")
            return
        
        MAX_ATTEMPTS = 3
        
        # Find a suitable return point
        if attempt == 0:
            # First try: closest point after detour
            self.path_index = max(self.last_path_index_before_detour, 
                                self.find_closest_path_point())
        else:
            # Subsequent attempts: skip ahead in path
            self.path_index = min(self.path_index + 2, len(self.global_path) - 1)
        
        if attempt >= MAX_ATTEMPTS:
            self.get_logger().error("Max return attempts reached, resetting")
            self.state = RobotState.IDLE
            return
        
        if self.path_index < len(self.global_path):
            target = self.global_path[self.path_index]
            self.get_logger().info(f"Attempt {attempt+1}: Returning to point {self.path_index}")
            
            if hasattr(self, '_navigating') and self._navigating:
                self.cancel_navigation()
                
            if self.navigate_to(target):
                self.state = RobotState.FOLLOWING_GLOBAL_PLAN
            else:
                self.get_logger().warn(f"Failed attempt {attempt+1}, will retry")
                self.return_to_global_plan(attempt + 1)

    def cancel_navigation(self):
        """Cancel any current navigation goal."""
        if hasattr(self, '_get_navigation_result_future'):
            if not self._get_navigation_result_future.done():
                # The correct way to access the goal handle is different
                # We need to get the goal handle from the goal response
                if hasattr(self, 'goal_handle') and self.goal_handle is not None:
                    cancel_future = self.goal_handle.cancel_goal_async()
                    rclpy.spin_until_future_complete(self, cancel_future)
                    self.get_logger().info("Cancelled current navigation goal")
                else:
                    self.get_logger().warn("No goal handle available to cancel")
        self._navigating = False

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
        
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0))
            robot_pos = np.array([transform.transform.translation.x, transform.transform.translation.y])
            
            direction = target - robot_pos
            if np.linalg.norm(direction) > 0.001:
                yaw = math.atan2(direction[1], direction[0])
                quat = self.YawToQuaternion(yaw)
                goal_msg.pose.pose.orientation = quat
            else:
                goal_msg.pose.pose.orientation.w = 1.0
        except Exception as e:
            self.get_logger().warn(f"Error calculating orientation: {e}")
            goal_msg.pose.pose.orientation.w = 1.0
        
        self.get_logger().info(f"Navigating to ({goal_msg.pose.pose.position.x:.2f}, {goal_msg.pose.pose.position.y:.2f})")
        
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg, feedback_callback=self._navigation_feedback_callback)
        send_goal_future.add_done_callback(self._navigation_goal_response_callback)
        
        self._navigating = True
        return True

    def _navigation_goal_response_callback(self, future):
        """Process navigation goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal rejected')
            self._navigating = False
            return
            
        self.get_logger().info('Navigation goal accepted')
        # Store the goal handle for cancellation purposes
        self.goal_handle = goal_handle
        self._get_navigation_result_future = goal_handle.get_result_async()
        self._get_navigation_result_future.add_done_callback(self._navigation_result_callback)

    def _navigation_result_callback(self, future):
        """Process navigation action result."""
        self._navigating = False
        try:
            result = future.result().result
            status = future.result().status
            
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('✅ Navigation succeeded!')
                if self.state == RobotState.APPROACHING_PERSON:
                    self.state = RobotState.RETURNING_TO_PLAN
                    self.return_to_global_plan()
                elif self.state == RobotState.RETURNING_TO_PLAN:
                    self.state = RobotState.FOLLOWING_GLOBAL_PLAN
                elif self.state == RobotState.FOLLOWING_GLOBAL_PLAN:
                    self.path_index += 1
                    
            elif status == GoalStatus.STATUS_CANCELED:
                self.get_logger().info('Navigation was canceled')
                
            elif status == GoalStatus.STATUS_ABORTED:  # Status 6
                self.get_logger().warn('⚠️ Navigation aborted (Status 6)')
                
                # Try to get more detailed error information
                error_info = "Unknown reason"
                if hasattr(result, 'error_code'):  # Changed from result_msg to result
                    error_code = result.error_code
                    error_info = self._get_nav2_error_description(error_code)
                self.get_logger().warn(f'Abort reason: {error_info}')
                
                # Special handling based on error type
                self.handle_navigation_failure()  # Remove the parameter here
                
        except Exception as e:
            self.get_logger().error(f'Error processing navigation result: {e}')
            self.handle_navigation_failure()  # Remove the parameter here

    def get_current_pose(self):
        """Get current robot pose in map frame with proper time handling."""
        try:
            # Use the most recent available transform
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time(),  # Get latest available
                timeout=rclpy.duration.Duration(seconds=0.1))
            
            position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y
            ])
            
            # Get orientation as yaw angle
            q = transform.transform.rotation
            yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 
                            q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
            
            return position, yaw
            
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None, None

    def _get_nav2_error_description(self, error_code):
        """Convert Nav2 error codes to human-readable messages."""
        error_messages = {
            1: "Unknown error",
            2: "Failed to find valid path",
            3: "Controller failed to produce valid command",
            4: "Robot is stuck",
            5: "Goal is obstructed",
            6: "Invalid goal pose",
            7: "No valid control available",
            8: "Planner timeout",
            9: "No valid planner available"
        }
        return error_messages.get(error_code, f"Unknown error code: {error_code}")


    def find_better_return_point(self):
        """Find a better point to return to after a navigation failure."""
        if not self.global_path:
            return 0
        
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0))
            current_pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y
            ])
            
            # Find the closest point that's ahead of our last known good position
            distances = []
            for i, point in enumerate(self.global_path):
                if i >= self.last_path_index_before_detour:
                    dist = np.linalg.norm(current_pos - point)
                    distances.append((i, dist))
            
            if distances:
                # Sort by distance and return the index of the closest point
                distances.sort(key=lambda x: x[1])
                return distances[0][0]
                
        except Exception as e:
            self.get_logger().error(f"Error finding better return point: {e}")
        
        # Fallback: return the next point after last known good position
        return min(self.last_path_index_before_detour + 1, len(self.global_path) - 1)


    def handle_navigation_failure(self, error_info=None):
        """Handle navigation failures with appropriate recovery behavior."""
        self.get_logger().warn(f"Handling navigation failure: {error_info if error_info else 'Unknown reason'}")
        
        if self.state == RobotState.APPROACHING_PERSON:
            self.get_logger().info("Retrying person approach with alternative strategy")
            self.approach_person_with_alternative_strategy()
            
        elif self.state == RobotState.RETURNING_TO_PLAN:
            # Try to find a better point to return to
            self.path_index = self.find_better_return_point()
            
            if self.path_index < len(self.global_path):
                self.get_logger().info(f"Trying alternative return point {self.path_index}")
                self.return_to_global_plan()
            else:
                self.get_logger().error("No valid return points left")
                self.state = RobotState.IDLE
                
        elif self.state == RobotState.FOLLOWING_GLOBAL_PLAN:
            # Skip to next point in path
            self.path_index += 1
            if self.path_index < len(self.global_path):
                self.get_logger().info(f"Skipping to next waypoint {self.path_index}")
                self.follow_global_plan()
            else:
                self.get_logger().info("Reached end of global plan")
                self.state = RobotState.IDLE
        else:
            self.state = RobotState.IDLE

    def approach_person_with_alternative_strategy(self):
        """Try alternative approach vectors when initial approach fails."""
        recent_people = [obj for obj in self.detected_objects 
                        if obj.object_type == "person" and 
                        time.time() - obj.last_seen < self.detection_timeout]
        
        if not recent_people:
            self.get_logger().warn("Lost track of person during recovery")
            self.state = RobotState.RETURNING_TO_PLAN
            self.return_to_global_plan()
            return

        person = min(recent_people, key=lambda p: np.linalg.norm(p.position[:2]))
        person_pos_2d = person.position[:2]
        
        # Try different approach strategies
        approach_strategies = [
            {"name": "front", "weight": 0.0, "side_weight": 0.0},  # Direct approach
            {"name": "left", "weight": -0.5, "side_weight": 1.0},  # Approach from left
            {"name": "right", "weight": 0.5, "side_weight": 1.0},  # Approach from right
            {"name": "behind", "weight": 1.0, "side_weight": 0.0}  # Approach from behind
        ]
        
        for strategy in approach_strategies:
            try:
                transform = self.tf_buffer.lookup_transform(
                    "map", "base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0))
                robot_pos = np.array([transform.transform.translation.x, transform.transform.translation.y])
                
                robot_to_person = person_pos_2d - robot_pos
                dist_to_person = np.linalg.norm(robot_to_person)
                
                if dist_to_person > 0.01:
                    robot_to_person = robot_to_person / dist_to_person
                
                normal_2d = person.normal[:2] if person.normal is not None else np.array([1.0, 0.0])
                normal_norm = np.linalg.norm(normal_2d)
                
                if normal_norm > 0.01:
                    normal_2d = normal_2d / normal_norm
                    
                    # Calculate perpendicular vector
                    perp = np.array([-normal_2d[1], normal_2d[0]])  # 90 degree rotation
                    
                    # Blend approach vectors based on strategy
                    approach_vec_2d = (normal_2d * strategy["weight"] + 
                                    perp * strategy["side_weight"])
                    approach_vec_2d = approach_vec_2d / np.linalg.norm(approach_vec_2d)
                    
                    self.get_logger().info(f"Trying {strategy['name']} approach strategy")
                    
                    target_pos = person_pos_2d + approach_vec_2d * self.approach_distance
                    
                    if self.navigate_to(target_pos):
                        return  # Successfully started navigation
                    
            except Exception as e:
                self.get_logger().error(f"Error in alternative approach: {e}")
                continue
        
        # If all strategies fail
        self.get_logger().warn("All alternative approaches failed, returning to plan")
        self.state = RobotState.RETURNING_TO_PLAN
        self.return_to_global_plan()



    def _navigation_feedback_callback(self, feedback_msg):
        """Process navigation feedback."""
        self.get_logger().debug(f'Navigation feedback: {feedback_msg.feedback}')

    def _dockCallback(self, msg):
        """Process dock status messages."""
        self.is_docked = msg.is_docked
        self.dock_status_received = True 
        self.get_logger().debug(f"Dock status: {'docked' if self.is_docked else 'undocked'}")

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
        try:
            result = future.result().result
            status = future.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('Undock succeeded')
                # Don't transition the state here - let perform_undocking() handle it
            else:
                self.get_logger().info(f'Undock failed with status: {status}')
        except Exception as e:
            self.get_logger().error(f"Error in undock result callback: {e}")
    def isUndockComplete(self):
        """Check if undock action is complete."""
        if hasattr(self, '_get_undock_result_future'):
            return self._get_undock_result_future.done()
        return False
        
    def perform_360_spin(self):
        """Perform a full 360-degree spin."""
        self._spin_sent = True
        self._spin_complete = False
        
        if not self.spin_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Spin action server not available")
            # Mark spin as complete even though it failed
            self._spin_complete = True
            self._set_state(RobotState.WAITING_FOR_PATH)
            return
            
        spin_goal = Spin.Goal()
        spin_goal.target_yaw = 6.28  # 360 degrees in radians
        spin_goal.time_allowance = Duration(sec=30)  # 30 seconds max
        
        self._spin_send_goal_future = self.spin_client.send_goal_async(spin_goal)
        self._spin_send_goal_future.add_done_callback(self._spin_goal_response_callback)

    def _spin_goal_response_callback(self, future):
        """Handle spin goal response."""
        self._spin_goal_handle = future.result()
        if not self._spin_goal_handle.accepted:
            self.get_logger().error("Spin goal rejected")
            self._set_state(RobotState.WAITING_FOR_PATH)
            return
            
        self.get_logger().info("Spin goal accepted")
        self._spin_result_future = self._spin_goal_handle.get_result_async()
        self._spin_result_future.add_done_callback(self._spin_result_callback)
    
    
    def _spin_result_callback(self, future):
        """Handle spin result."""
        try:
            result = future.result().result
            status = future.result().status
            
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info("✅ Spin completed successfully!")
            else:
                self.get_logger().warn(f"Spin completed with status {status}")
                
            # Always mark spin as complete, regardless of success/failure
            self._spin_complete = True
            
        except Exception as e:
            self.get_logger().error(f"Error processing spin result: {e}")
            # Even with errors, mark as complete to avoid getting stuck
            self._spin_complete = True


    def is_position_valid(self, position):
        """Check if a position is likely to be navigable with more thorough checks."""
        try:
            # Get current robot position
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0))
            robot_pos = np.array([transform.transform.translation.x, transform.transform.translation.y])
            
            # Check distance from robot
            distance_to_robot = np.linalg.norm(position - robot_pos)
            if distance_to_robot > 5.0:  # Reduced from 10.0 to be more conservative
                self.get_logger().warn(f"Target position too far: {distance_to_robot:.2f}m")
                return False
                
            # Check if position is behind the robot (might be harder to reach)
            robot_orientation = transform.transform.rotation
            _, _, yaw = quaternion_from_euler(
                robot_orientation.x,
                robot_orientation.y,
                robot_orientation.z,
                robot_orientation.w
            )
            
            robot_to_target = position - robot_pos
            angle_to_target = math.atan2(robot_to_target[1], robot_to_target[0])
            angle_diff = abs(angle_to_target - yaw)
            
            if angle_diff > math.pi/2:  # More than 90 degrees from current orientation
                self.get_logger().warn(f"Target position is behind robot (angle: {math.degrees(angle_diff):.1f}°)")
                # Don't return False here - just warn as it might still be reachable
                
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error checking position validity: {e}")
            return False

    def waitUntilNav2Active(self):
        """Wait until Nav2 is active."""
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
                if state != 3:
                    self.get_logger().info(f'{node_name} is not active (state={state})')
                else:
                    self.get_logger().info(f'{node_name} is active')
            else:
                self.get_logger().info(f'Service call to {node_name} failed')
                
        self.get_logger().info('Nav2 is active and ready')

def main(args=None):
    rclpy.init(args=args)
    node = RobotCommander()

    try:
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